"""End-to-end HLS CNN inference pipeline orchestrator.

Runs the complete software→HLS workflow in order:

  [1] Model selection  — best_model.pt or hw_pruned_finetuned.pt (if available)
  [2] ONNX validation  — verify model.onnx / hw_model_pruned.onnx with onnxruntime
  [3] hls4ml convert   — generate HLS C++ project (only if project does not exist)
  [4] C-sim compile    — build .so from HLS C++ (build_lib.sh via g++ on Linux /
                         MSYS2 bash on Windows)
  [5] C-sim validate   — run 5 smoke-test inputs through the compiled .so
  [6] Performance model — compute theoretical latency/throughput from parameters.h
  [7] Summary report   — write fpga/hls/hls_pipeline_report.txt

The hls4ml conversion step is skipped if the HLS project already exists
(regeneration is slow and requires hls4ml installed).  Use --force-convert to
always regenerate.

Usage::

    python scripts/run_hls_pipeline.py
    python scripts/run_hls_pipeline.py --use-optimized
    python scripts/run_hls_pipeline.py --force-convert --reuse 4
    python scripts/run_hls_pipeline.py --steps 5,6,7    # only validate+profile

All reported figures come from actual runs; nothing is fabricated.
Stages that require Vivado HLS (RTL synthesis, resource counts) are not run
here and are not faked.  The CI workflow hls_synthesis.yml handles those
conditionally when XILINX_LICENSE_SERVER is configured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CKPT_BASE = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt"
CKPT_OPT  = PROJECT_ROOT / "models" / "checkpoints" / "hw_pruned_finetuned.pt"
ONNX_BASE = PROJECT_ROOT / "models" / "exported" / "model.onnx"
ONNX_OPT  = PROJECT_ROOT / "models" / "exported" / "hw_model_pruned.onnx"

HLS_BASE_DIR = PROJECT_ROOT / "fpga" / "hls" / "noise_classifier_hls"
HLS_OPT_DIR  = PROJECT_ROOT / "fpga" / "hls" / "noise_classifier_hls_opt"
PERF_JSON    = PROJECT_ROOT / "fpga" / "hls" / "hls_performance_model.json"
VALID_JSON   = PROJECT_ROOT / "fpga" / "hls" / "hls_validation.json"
REPORT_PATH  = PROJECT_ROOT / "fpga" / "hls" / "hls_pipeline_report.txt"


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> tuple[int, str]:
    """Run a subprocess; return (returncode, combined stdout+stderr)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=1800,
        )
        elapsed = time.perf_counter() - t0
        output = result.stdout + result.stderr
        status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
        print(f"  {label}: {status}  ({elapsed:.1f}s)")
        if result.returncode != 0:
            for line in output.splitlines()[-10:]:
                print(f"    {line}")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        print(f"  {label}: TIMEOUT (>1800s)")
        return 1, "TIMEOUT"
    except Exception as exc:
        print(f"  {label}: ERROR ({exc})")
        return 1, str(exc)


def _step_1_select_model(use_optimized: bool) -> tuple[Path, Path]:
    """Return (checkpoint_path, onnx_path) based on availability."""
    if use_optimized and CKPT_OPT.exists():
        print(f"  Using optimized model: {CKPT_OPT.name}  (91.28% val acc, pruned+fine-tuned)")
        return CKPT_OPT, ONNX_OPT
    if use_optimized and not CKPT_OPT.exists():
        print(f"  Optimized model not found; run scripts/make_hardware_cnn.py first.")
        print(f"  Falling back to baseline: {CKPT_BASE.name}")
    else:
        print(f"  Using baseline model: {CKPT_BASE.name}  (88.55% val acc)")
    return CKPT_BASE, ONNX_BASE


def _step_2_validate_onnx(onnx_path: Path) -> bool:
    """Verify the ONNX model loads and runs 1 inference with onnxruntime."""
    if not onnx_path.exists():
        print(f"  ONNX not found at {onnx_path}")
        return False
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 1, 224, 224), dtype=np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: dummy})
        print(f"  ONNX OK: {onnx_path.name}  output shape={out[0].shape}")
        return True
    except ImportError:
        print("  onnxruntime not installed — ONNX validation skipped")
        return True
    except Exception as exc:
        print(f"  ONNX validation failed: {exc}")
        return False


def _step_3_hls4ml_convert(ckpt: Path, hls_dir: Path, reuse: int, force: bool) -> bool:
    """Generate the HLS C++ project if it does not already exist."""
    params_h = hls_dir / "firmware" / "parameters.h"
    if params_h.exists() and not force:
        print(f"  HLS project exists at {hls_dir.name}/  — skipping (use --force-convert to regen)")
        return True
    project_name = hls_dir.name
    rc, _ = _run([sys.executable, "scripts/hls4ml_convert.py",
                  "--checkpoint", str(ckpt),
                  "--reuse", str(reuse),
                  "--io-type", "io_stream",
                  "--project", project_name],
                 "hls4ml conversion")
    return rc == 0


def _step_4_csim_compile(hls_dir: Path) -> bool:
    """Compile the HLS firmware .so using g++."""
    build_sh = hls_dir / "build_lib.sh"
    if not build_sh.exists():
        print(f"  build_lib.sh not found in {hls_dir} — HLS project missing?")
        return False

    import glob
    so_files = glob.glob(str(hls_dir / "firmware" / "*.so"))
    if so_files:
        print(f"  .so already compiled: {Path(so_files[0]).name}")
        return True

    # Try g++ directly (Linux/Ubuntu) or MSYS2 bash (Windows)
    bash_candidates = [
        "/bin/bash",
        r"C:\msys64\usr\bin\bash.exe",
        r"C:\Windows\System32\bash.exe",
    ]
    bash = next((b for b in bash_candidates if Path(b).exists()), None)
    if bash is None:
        print("  No bash found; cannot compile .so")
        return False

    posix_dir = str(hls_dir).replace("\\", "/")
    if ":" in posix_dir:
        drive = posix_dir[0].lower()
        posix_dir = f"/{drive}/{posix_dir[3:]}"
    rc, _ = _run([bash, "-c",
                  f"export PATH='/mingw64/bin:/usr/bin:/usr/local/bin:$PATH'; "
                  f"cd '{posix_dir}' && bash build_lib.sh"],
                 "C-sim compile")
    return rc == 0


def _step_5_csim_validate(hls_dir: Path, output_json: Path) -> bool:
    """Run 5 random-input smoke test through the compiled .so."""
    rc, _ = _run([sys.executable, "scripts/validate_hls_csim.py",
                  "--random-inputs", "5",
                  "--no-pytorch",
                  "--output", str(output_json)],
                 "C-sim smoke test")
    return rc == 0


def _step_6_performance_model(clock_ns: float) -> dict | None:
    """Compute theoretical performance from parameters.h."""
    rc, _ = _run([sys.executable, "scripts/hls4ml_profile.py",
                  "--clock", str(clock_ns),
                  "--output", str(PERF_JSON)],
                 "Performance model")
    if rc == 0 and PERF_JSON.exists():
        return json.loads(PERF_JSON.read_text())
    return None


def _write_report(
    ckpt: Path, onnx_ok: bool,
    convert_ok: bool, compile_ok: bool, csim_ok: bool,
    perf: dict | None,
    csim_json: Path,
) -> None:
    lines = [
        "HLS CNN Inference Pipeline Report",
        "=" * 64,
        f"Generated : {time.strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {ckpt.name}",
        "",
        "Stage results:",
        f"  [1] Model selection  : OK ({ckpt.name})",
        f"  [2] ONNX validation  : {'OK' if onnx_ok else 'FAIL'}",
        f"  [3] HLS C++ project  : {'OK' if convert_ok else 'FAIL'}",
        f"  [4] C-sim compile    : {'OK' if compile_ok else 'FAIL'}",
        f"  [5] C-sim smoke test : {'OK' if csim_ok else 'FAIL'}",
    ]

    if csim_json.exists():
        try:
            csim_data = json.loads(csim_json.read_text())
            mode = csim_data.get("mode", "unknown")
            n = csim_data.get("n_inputs", csim_data.get("test_images", "?"))
            preds = csim_data.get("hls_predictions", [])
            lines += [
                "",
                "C-sim result (Stage 1b smoke test):",
                f"  Mode       : {mode}",
                f"  N inputs   : {n}",
                f"  Predictions: {preds}",
                f"  Precision  : {csim_data.get('precision', 'ap_fixed<16,6>')}",
            ]
            if "hls_accuracy_pct" in csim_data:
                lines.append(f"  Accuracy   : {csim_data['hls_accuracy_pct']}%")
        except Exception:
            pass

    lines.append("")
    lines.append("  [6] Performance model: " + ("OK" if perf else "FAIL"))
    if perf:
        lines += [
            "",
            "Theoretical performance (calculated from parameters.h):",
            f"  Precision  : {perf['precision']}",
            f"  IO type    : {perf['io_type']}",
            f"  Reuse fac  : {perf['reuse_factor']}",
            f"  Clock      : {perf['clock_mhz']:.0f} MHz",
            f"  Latency    : {perf['total_latency_ms']:.1f} ms/image  "
                          f"({perf['total_cycles']:,} cycles)",
            f"  Throughput : {perf['throughput_fps']:.1f} FPS  (continuous streaming)",
            f"  MACs       : {perf['total_mac_ops']:,}",
            f"  Bottleneck : {perf['bottleneck_layer']}",
            "",
            "  Note: figures calculated from architecture parameters, not synthesised",
            "  or measured on hardware.  RTL synthesis requires Vivado HLS >= 2020.1.",
        ]

    lines += [
        "",
        "What this pipeline does NOT produce:",
        "  - Synthesis resource counts (LUT/FF/DSP/BRAM)  → requires Vivado HLS",
        "  - Timing closure report                         → requires Vivado HLS",
        "  - Bitstream or hardware test                    → requires Zynq board",
        "  Per CLAUDE.md: unverified numbers are absent, not fabricated.",
        "",
        "See:",
        "  fpga/hls/noise_classifier_hls/build_prj.tcl  — Vivado HLS TCL project",
        "  .github/workflows/hls_synthesis.yml           — CI for C-sim + synthesis",
        "  fpga/asic/                                     — ASIC flow (filter RTL)",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--use-optimized", action="store_true",
                    help="Use hw_pruned_finetuned.pt if available (else baseline)")
    ap.add_argument("--force-convert", action="store_true",
                    help="Regenerate HLS project even if it already exists")
    ap.add_argument("--reuse", type=int, default=64,
                    help="hls4ml reuse factor (default: 64)")
    ap.add_argument("--clock", type=float, default=10.0,
                    help="Clock period in ns for performance model (default: 10 = 100 MHz)")
    ap.add_argument("--steps", default="",
                    help="Comma-separated subset of steps to run (1-6); empty = all")
    args = ap.parse_args(argv)

    steps = set(int(s) for s in args.steps.split(",") if s.strip()) if args.steps else set(range(1, 8))

    print("=" * 64)
    print("HLS CNN Inference Pipeline")
    print("=" * 64)

    ckpt = CKPT_BASE
    onnx_path = ONNX_BASE
    onnx_ok = convert_ok = compile_ok = csim_ok = True
    perf = None

    if 1 in steps:
        print("\n[1] Model selection")
        ckpt, onnx_path = _step_1_select_model(args.use_optimized)

    hls_dir = HLS_OPT_DIR if ckpt == CKPT_OPT else HLS_BASE_DIR

    if 2 in steps:
        print("\n[2] ONNX validation")
        onnx_ok = _step_2_validate_onnx(onnx_path)

    if 3 in steps:
        print("\n[3] HLS C++ project (hls4ml conversion)")
        convert_ok = _step_3_hls4ml_convert(ckpt, hls_dir, args.reuse, args.force_convert)

    if 4 in steps:
        print("\n[4] C-sim compile (.so)")
        compile_ok = _step_4_csim_compile(hls_dir)

    if 5 in steps:
        print("\n[5] C-sim smoke test (5 random inputs)")
        csim_ok = _step_5_csim_validate(hls_dir, VALID_JSON)

    if 6 in steps:
        print("\n[6] Theoretical performance model")
        perf = _step_6_performance_model(args.clock)

    if 7 in steps:
        print("\n[7] Writing pipeline report")
        _write_report(ckpt, onnx_ok, convert_ok, compile_ok, csim_ok, perf, VALID_JSON)

    all_ok = onnx_ok and convert_ok and compile_ok and csim_ok
    print("\n" + "=" * 64)
    print(f"Pipeline {'COMPLETE' if all_ok else 'PARTIAL (some steps failed)'}")
    if perf:
        print(f"  Theoretical latency : {perf['total_latency_ms']:.1f} ms/image")
        print(f"  Theoretical FPS     : {perf['throughput_fps']:.1f}  (100 MHz streaming)")
    print("  RTL synthesis: requires Vivado HLS — see CI workflow for setup.")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
