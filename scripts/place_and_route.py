"""Place-and-route the RTL on a Lattice ECP5-25k using oss-cad-suite.

Runs the full vendor-neutral ECP5 flow:

  1. yosys synth_ecp5  — maps to ECP5 LUT4s and infers BRAM/DSP primitives
  2. nextpnr-ecp5       — places, routes, reports timing at 100 MHz
  3. ecppack            — packs the bitstream (proves routing is complete)

Timing, utilisation and any hold/setup violations are written to
``results/rtl/pnr_ecp5.json`` and printed in a summary table.

    python scripts/place_and_route.py [--freq MHZ] [--package PKG] [--seed N]

Defaults: 100 MHz, CABGA381, seed 1.

Tools are found from PATH first, then the oss-cad-suite install under
C:/oss-cad-suite/oss-cad-suite (the path on this machine).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "fpga_denoiser_top"
DEVICE = "LFE5U-25F"
SUITE = Path("C:/oss-cad-suite/oss-cad-suite")


# ── tool discovery ────────────────────────────────────────────────────────────

def _find(name: str) -> tuple[str, dict[str, str]]:
    """Return (binary_path, env) with oss-cad-suite lib on PATH."""
    env = dict(os.environ)
    for candidate in (
        os.environ.get(name.upper().replace("-", "_")),
        shutil.which(name),
        str(SUITE / "bin" / f"{name}.exe"),
        str(SUITE / "bin" / name),
    ):
        if candidate and Path(candidate).exists():
            binary = Path(candidate)
            suite = binary.parent.parent
            if (suite / "lib").is_dir():
                env["PATH"] = f"{binary.parent};{suite / 'lib'};{env.get('PATH', '')}"
            return str(binary), env
    raise SystemExit(
        f"{name} not found. Add oss-cad-suite/bin to PATH or set "
        f"${name.upper().replace('-', '_')}."
    )


def rtl_sources() -> list[str]:
    return [line.strip() for line in (ROOT / "rtl" / "filelist.f").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")]


# ── synthesis (yosys synth_ecp5) ──────────────────────────────────────────────

def run_synth(out: Path) -> tuple[str, dict]:
    yosys, env = _find("yosys")
    json_path = out / "synth_ecp5.json"
    script = (
        f"read_verilog -sv {' '.join(rtl_sources())}; "
        f"synth_ecp5 -top {TOP} -json {json_path.as_posix()}"
    )
    proc = subprocess.run([yosys, "-p", script], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out / "synth_ecp5.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0 or re.search(r"^ERROR", log, re.M):
        errors = "\n".join(re.findall(r"^ERROR.*", log, re.M)) or log[-3000:]
        raise SystemExit(f"yosys synth_ecp5 failed:\n{errors}")

    # Extract utilisation counts from yosys log
    util: dict[str, int] = {}
    for pattern, key in (
        (r"LUT4\s+(\d+)", "lut4"),
        (r"TRELLIS_FF\s+(\d+)", "ff"),
        (r"TRELLIS_BRAM\s+(\d+)", "bram18"),
        (r"TRELLIS_DPR16X4\s+(\d+)", "lutram"),
        (r"MULT18X18D\s+(\d+)", "dsp"),
    ):
        m = re.search(pattern, log)
        util[key] = int(m.group(1)) if m else 0

    print(f"  yosys synth_ecp5 done — {json_path.name} written")
    return str(json_path), util


# ── place and route (nextpnr-ecp5) ───────────────────────────────────────────

def run_nextpnr(netlist: str, out: Path, freq: float, package: str, seed: int) -> dict:
    nextpnr, env = _find("nextpnr-ecp5")
    lpf = ROOT / "fpga" / "constraints" / "ecp5_25k.lpf"
    report = out / "pnr_ecp5_report.json"
    bitstream = out / "fpga_denoiser_top.bit"

    cmd = [
        nextpnr,
        "--25k",
        "--package", package,
        "--json", netlist,
        "--lpf", str(lpf),
        "--lpf-allow-unconstrained",
        "--freq", str(freq),
        "--seed", str(seed),
        "--write", str(out / "pnr_ecp5.json"),
        "--report", str(report),
        "--routed-svg", str(out / "pnr_ecp5.svg"),
        "--log", str(out / "pnr_ecp5.log"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out / "nextpnr_stdout.log").write_text(log, encoding="utf-8")

    # nextpnr exits 1 when timing fails but routing is complete; that is a
    # timing warning, not a routing failure.  A real routing failure produces
    # no routed JSON, so check for that instead.
    routed_json = out / "pnr_ecp5.json"
    if proc.returncode != 0 and not routed_json.exists():
        print(log[-3000:], file=sys.stderr)
        raise SystemExit("nextpnr-ecp5 failed (no routed JSON; see results/rtl/nextpnr_stdout.log)")

    # Parse the JSON timing report nextpnr writes with --report
    timing: dict = {}
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            timing = data
        except json.JSONDecodeError:
            pass

    # Also scrape stdout for the MHz summary line
    fmax_m = re.search(r"Max frequency for clock[^:]*:\s*([\d.]+)\s*MHz", log)
    wns_m  = re.search(r"Critical path.*?(-?[\d.]+)\s*ns", log)
    timing["fmax_mhz"] = float(fmax_m.group(1)) if fmax_m else None
    timing["wns_ns"]   = float(wns_m.group(1))  if wns_m  else None

    return timing


# ── pack bitstream (ecppack) ──────────────────────────────────────────────────

def run_ecppack(out: Path) -> None:
    ecppack, env = _find("ecppack")
    pnr_json = out / "pnr_ecp5.json"
    bit_out  = out / "fpga_denoiser_top.bit"
    proc = subprocess.run([ecppack, "--input", str(pnr_json), "--bit", str(bit_out)],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out / "ecppack.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"ecppack failed:\n{log[-2000:]}")
    size = bit_out.stat().st_size if bit_out.exists() else 0
    print(f"  ecppack done — {bit_out.name} ({size // 1024} KiB)")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(util: dict, timing: dict, freq: float) -> None:
    print()
    print("=" * 52)
    print(f"  ECP5-25k place-and-route summary  (target {freq} MHz)")
    print("=" * 52)
    resources = [
        ("LUT4",       util.get("lut4",   0), 24288),
        ("TRELLIS_FF", util.get("ff",     0), 24288),
        ("BRAM18",     util.get("bram18", 0),    56),
        ("MULT18X18D", util.get("dsp",    0),    28),
    ]
    for name, used, avail in resources:
        pct = 100 * used / avail if avail else 0
        print(f"  {name:<14} {used:>6} / {avail:<6}  ({pct:.1f} %)")
    print()
    fmax = timing.get("fmax_mhz")
    wns  = timing.get("wns_ns")
    if fmax is not None:
        met = "PASS" if fmax >= freq else "FAIL"
        print(f"  Fmax   {fmax:.2f} MHz   (target {freq} MHz — {met})")
    if wns is not None:
        print(f"  WNS    {wns:.3f} ns")
    print("=" * 52)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq",    type=float, default=100.0, help="clock target in MHz")
    ap.add_argument("--package", default="CABGA381",        help="ECP5 package code")
    ap.add_argument("--seed",    type=int,   default=1,     help="nextpnr placement seed")
    args = ap.parse_args()

    out = ROOT / "results" / "rtl"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] yosys synth_ecp5 ...")
    netlist, util = run_synth(out)

    print(f"[2/3] nextpnr-ecp5 (target {args.freq} MHz, seed {args.seed}) ...")
    timing = run_nextpnr(netlist, out, args.freq, args.package, args.seed)

    print(f"[3/3] ecppack ...")
    run_ecppack(out)

    print_summary(util, timing, args.freq)

    result = {"device": f"{DEVICE}-{args.package}", "freq_target_mhz": args.freq,
              "utilisation": util, "timing": timing}
    report_path = out / "pnr_ecp5.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Full report written to {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
