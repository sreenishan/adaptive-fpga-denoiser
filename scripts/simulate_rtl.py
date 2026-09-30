"""RTL-versus-golden co-simulation (the check CLAUDE.md requires of every filter).

Streams real noisy images through the complete ``fpga_denoiser_top`` in Icarus
Verilog — window generator, filter controller and all four filters — and
compares every output pixel against the Python golden filters in
``src/denoising/filters/``, within the tolerances in ``configs/hardware.yaml``
(median 0, gaussian 0, wiener 1).

This is different from ``tests/rtl``, which checks a hand-transcribed Python
*model* of the RTL and cannot see a syntax, elaboration or protocol error; and
from the ``rtl/tb`` unit benches, which compare against small references
written inside the benches. Here the SystemVerilog itself runs, on full frames,
and is judged by the golden model.

    python scripts/simulate_rtl.py            # 224x224, all cases
    python scripts/simulate_rtl.py --quick    # small frame, fewer cases

Icarus is found via $IVERILOG/$VVP, then PATH, then C:/iverilog/bin (the
default Windows install). Exit code 0 only if every frame is within tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from denoising.config import load_dataset_config, load_hardware_config  # noqa: E402
from denoising.dataset import synthetic_sources  # noqa: E402
from denoising.filters import (  # noqa: E402
    CONTROL_CODE,
    estimate_noise_variance,
    gaussian_filter,
    median_filter,
    wiener_filter,
)
from denoising.metrics import calculate_psnr  # noqa: E402
from denoising.noise import (  # noqa: E402
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
)

#: Fallback noise power, used only where an estimate makes no sense.
#: The RTL takes ``noise_var`` as a runtime port now, so each frame is run with
#: the value the host measured for that image — the same integer the golden
#: filter is given, which is the whole point of the port existing.
DEFAULT_NOISE_VAR = 100

#: Widest value the RTL's NV_W=16 port can carry; 255**2 = 65025 is the most a
#: real estimate can reach.
NV_MAX = 65535


def host_noise_var(image: np.ndarray) -> int:
    """The noise power a host would send with this frame.

    The software estimator returns a float; the port is an integer, so the host
    rounds. Both sides then use the SAME integer, so any residual difference is
    the filter's fixed-point arithmetic and not a disagreement about the input.
    """
    return int(min(max(round(estimate_noise_variance(image)), 0), NV_MAX))

BENCH = ["rtl/tb/tb_golden_image.sv"]
RTL = [line.strip() for line in (ROOT / "rtl" / "filelist.f").read_text().splitlines()
       if line.strip() and not line.lstrip().startswith("//")]


@dataclass
class FrameResult:
    case: str
    filter: str
    noise_var: int
    width: int
    height: int
    stall: bool
    pixels: int
    mismatched: int
    max_abs_error: int
    mean_abs_error: float
    tolerance: int
    passed: bool
    sim_seconds: float


def find_tool(name: str, env: str) -> str:
    candidates = [os.environ.get(env), shutil.which(name), f"C:/iverilog/bin/{name}.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise SystemExit(f"{name} not found: set ${env}, put it on PATH, or install to C:/iverilog")


def golden(image: np.ndarray, filter_name: str, noise_var: int) -> np.ndarray:
    if filter_name == "bypass":
        return image.copy()
    if filter_name == "median":
        return median_filter(image, 3)
    if filter_name == "gaussian":
        return gaussian_filter(image, 3, None, integer_kernel=True)
    if filter_name == "wiener":
        return wiener_filter(image, 3, noise_variance=float(noise_var))
    raise ValueError(filter_name)


def write_hex(path: Path, image: np.ndarray) -> None:
    path.write_text("\n".join(f"{v:02x}" for v in image.ravel()) + "\n", encoding="ascii")


def read_hex(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = [int(tok, 16) for tok in path.read_text(encoding="ascii").split()]
    if len(values) != shape[0] * shape[1]:
        raise RuntimeError(f"{path.name}: {len(values)} pixels, expected {shape[0] * shape[1]}")
    return np.array(values, dtype=np.uint8).reshape(shape)


def compile_bench(iverilog: str, width: int, height: int, out: Path) -> None:
    """Compile once per geometry. Noise power is a plusarg, not a parameter."""
    cmd = [iverilog, "-g2012", "-o", str(out),
           f"-Ptb_golden_image.W={width}", f"-Ptb_golden_image.H={height}",
           *RTL, *BENCH]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"iverilog failed:\n{proc.stdout}{proc.stderr}")


def build_cases(width: int, height: int, quick: bool) -> dict[str, np.ndarray]:
    ds = load_dataset_config()
    image_cfg = type(ds.image)(width, height, True)
    clean = synthetic_sources(1, image_cfg, 2026)[0].image
    cases = {"clean": clean}
    levels = ((0.10,), (0.10,), (0.10,)) if quick else (
        tuple(ds.noise.salt_pepper.amounts), tuple(ds.noise.gaussian.sigmas),
        tuple(ds.noise.speckle.variances))
    for a in levels[0]:
        cases[f"salt_pepper_{a:g}"] = add_salt_pepper_noise(clean, a, seed=11)
    for s in levels[1]:
        cases[f"gaussian_{s:g}"] = add_gaussian_noise(clean, 0.0, s, seed=12)
    for v in levels[2]:
        cases[f"speckle_{v:g}"] = add_speckle_noise(clean, v, seed=13)
    # Extremes the synthetic sources do not reach: every pixel at the rails,
    # which exercises saturation in the Gaussian adder and Wiener clamp.
    rng = np.random.default_rng(14)
    cases["rails"] = rng.choice(np.array([0, 255], np.uint8), size=(height, width))
    if quick:
        # The edge vectors docs/verification.md lists. Run on the small frame,
        # where the border is a large share of the image — every frame has all
        # four corners, and a 31x17 frame is ~20% border against ~2% at 224x224.
        cases["flat_0"] = np.zeros((height, width), np.uint8)
        cases["flat_255"] = np.full((height, width), 255, np.uint8)
        cases["flat_128"] = np.full((height, width), 128, np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]
        cases["gradient"] = ((xx * 255) // max(width - 1, 1)).astype(np.uint8)
        spike = np.full((height, width), 40, np.uint8)
        spike[height // 2, width // 2] = 255
        spike[0, 0] = spike[0, -1] = spike[-1, 0] = spike[-1, -1] = 255  # corner spikes
        cases["spikes"] = spike
    return cases


def cosimulate(width: int, height: int, quick: bool,
               workdir: Path | None = None, log=print) -> tuple[list[FrameResult], dict]:
    hw = load_hardware_config()
    tolerance = dict(hw.simulation.max_abs_error)
    tolerance.setdefault("bypass", 0)
    iverilog, vvp = find_tool("iverilog", "IVERILOG"), find_tool("vvp", "VVP")
    dirs = {k: (workdir / k if workdir else getattr(hw.simulation, f"{k}_dir"))
            for k in ("input", "expected", "output")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        sim = Path(tmp) / f"golden_{width}x{height}.vvp"
        compile_bench(iverilog, width, height, sim)
        cases = build_cases(width, height, quick)
        results: list[FrameResult] = []
        stall_case = next(iter(k for k in cases if k.startswith("gaussian")))
        for case, image in cases.items():
            in_hex = dirs["input"] / f"{case}_{width}x{height}.hex"
            write_hex(in_hex, image)
            noise_var = host_noise_var(image)
            for name, code in CONTROL_CODE.items():
                for stall in ((False, True) if case == stall_case else (False,)):
                    tag = f"{case}_{width}x{height}_{name}{'_stall' if stall else ''}"
                    expect = golden(image, name, noise_var)
                    write_hex(dirs["expected"] / f"{tag}.hex", expect)
                    out_hex = dirs["output"] / f"{tag}.hex"
                    started = time.perf_counter()
                    proc = subprocess.run(
                        [vvp, "-n", str(sim), f"+IN={in_hex.as_posix()}", f"+OUT={out_hex.as_posix()}",
                         f"+SEL={code}", f"+NV={noise_var}", f"+STALL={int(stall)}", "+SEED=7"],
                        cwd=ROOT, capture_output=True, text=True)
                    seconds = time.perf_counter() - started
                    if proc.returncode != 0:
                        raise SystemExit(f"simulation failed for {tag}:\n{proc.stdout}{proc.stderr}")
                    got = read_hex(out_hex, image.shape)
                    diff = np.abs(got.astype(int) - expect.astype(int))
                    r = FrameResult(case, name, noise_var, width, height, stall, int(diff.size),
                                    int((diff > 0).sum()), int(diff.max()),
                                    round(float(diff.mean()), 5), tolerance[name],
                                    bool(diff.max() <= tolerance[name]), round(seconds, 2))
                    results.append(r)
                    log(f"  {'PASS' if r.passed else 'FAIL'}  {tag:<42} nv {noise_var:>5}  "
                        f"mismatched {r.mismatched:>6}/{r.pixels}  max|err| {r.max_abs_error}  "
                        f"(tol {r.tolerance})  {seconds:5.1f}s")

    # What the runtime port bought, and what integer rounding costs.
    #   old_fixed   — what the hardware did when NOISE_VAR was compiled in
    #   host_int    — what it does now: the host's estimate, rounded to an int
    #   software    — the software pipeline's own float estimate
    divergence = {}
    clean = cases["clean"]
    for case, image in cases.items():
        if not case.startswith(("salt_pepper", "gaussian", "speckle")):
            continue    # PSNR against the clean source only means something for noisy copies of it
        nv = host_noise_var(image)
        old_fixed = wiener_filter(image, 3, noise_variance=float(DEFAULT_NOISE_VAR))
        host_int = wiener_filter(image, 3, noise_variance=float(nv))
        software = wiener_filter(image, 3, noise_variance=None)
        divergence[case] = {
            "host_noise_var": nv,
            "psnr_old_fixed_100": round(calculate_psnr(clean, old_fixed), 2),
            "psnr_host_integer": round(calculate_psnr(clean, host_int), 2),
            "psnr_software_float": round(calculate_psnr(clean, software), 2),
            "max_abs_diff_vs_software": int(np.abs(host_int.astype(int) - software.astype(int)).max()),
        }
    return results, divergence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="31x17 frame, one level per noise")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()
    hw = load_hardware_config()
    width = args.width or (31 if args.quick else hw.stream.image_width)
    height = args.height or (17 if args.quick else hw.stream.image_height)

    print(f"Co-simulating fpga_denoiser_top at {width}x{height}; "
          f"noise_var is sent per frame by the host")
    results, divergence = cosimulate(width, height, args.quick)
    failed = [r for r in results if not r.passed]

    print(f"\n{len(results) - len(failed)}/{len(results)} frames within tolerance")
    print("\nWiener, PSNR vs clean — old compile-time 100 | host integer (now) | software float:")
    for case, d in divergence.items():
        print(f"  {case:<18} nv {d['host_noise_var']:>5}   "
              f"{d['psnr_old_fixed_100']:6.2f} | {d['psnr_host_integer']:6.2f} | "
              f"{d['psnr_software_float']:6.2f} dB    max pixel diff vs software "
              f"{d['max_abs_diff_vs_software']}")

    out = ROOT / "results" / "rtl" / f"cosim_{width}x{height}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"width": width, "height": height,
                               "frames": [asdict(r) for r in results],
                               "wiener_divergence": divergence}, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
