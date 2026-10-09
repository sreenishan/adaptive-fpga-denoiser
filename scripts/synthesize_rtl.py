"""Vendor-neutral synthesis of the RTL with yosys.

Proves the design is synthesisable and reports how much logic each module
costs. It maps to generic 6-input LUTs and flip-flops, which is a reasonable
area proxy for most FPGAs but is **not** a vendor fitter result:

  · no timing, so no Fmax and no timing closure
  · no vendor primitives, so shift registers (SRL/ALTSHIFT_TAPS), block RAM and
    DSP blocks are NOT inferred — the line buffer lands as discrete flip-flops
    here and may map far smaller on a real part
  · no placement or routing, so no power figure

Those need a board and its toolchain, and `docs/hardware.md` stays TBD for them
until a real run produces them.

    python scripts/synthesize_rtl.py

yosys is found via $YOSYS, then PATH, then the oss-cad-suite install under
~/tools. Exit code 0 only if synthesis completes with no errors.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "fpga_denoiser_top"
LUT_SIZE = 6

#: Where oss-cad-suite lands by default on this machine. Its binaries need the
#: suite's own lib/ on PATH — yosys.exe fails with a missing libreadline8.dll
#: otherwise, which looks like a broken install rather than a PATH problem.
SUITE = Path.home() / "tools" / "oss-cad-suite"


def find_yosys() -> tuple[str, dict[str, str]]:
    env = dict(os.environ)
    for candidate in (os.environ.get("YOSYS"), shutil.which("yosys"), SUITE / "bin" / "yosys.exe"):
        if candidate and Path(candidate).exists():
            binary = Path(candidate)
            suite = binary.parent.parent
            if (suite / "lib").is_dir():
                env["PATH"] = f"{binary.parent};{suite / 'lib'};{env.get('PATH', '')}"
            return str(binary), env
    raise SystemExit(
        "yosys not found. Set $YOSYS, put it on PATH, or install oss-cad-suite "
        f"to {SUITE} (https://github.com/YosysHQ/oss-cad-suite-build/releases)."
    )


def rtl_sources() -> list[str]:
    return [line.strip() for line in (ROOT / "rtl" / "filelist.f").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")]


def short(module: str) -> str:
    """Strip yosys's $paramod mangling down to the module name.

    A parameterised module arrives as ``$paramod\\median_filter\\DEPTH=s32'...``,
    so the last backslash-separated segment is the parameter, not the name.
    Match against the known sources instead.
    """
    known = {Path(s).stem for s in rtl_sources()}
    for part in re.split(r"[\\$]", module):
        if part in known:
            return part
    return module.split("\\")[-1]


def synthesize() -> tuple[dict, str]:
    yosys, env = find_yosys()
    out = ROOT / "results" / "rtl"
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "synth_stats.json"
    script = (f"read_verilog -sv {' '.join(rtl_sources())}; "
              f"synth -top {TOP} -lut {LUT_SIZE}; "
              f"tee -o {stats_path.as_posix()} stat -json")
    proc = subprocess.run([yosys, "-p", script], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (out / "synth.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0 or re.search(r"^ERROR", log, re.M):
        errors = "\n".join(re.findall(r"^ERROR.*", log, re.M)) or log[-2000:]
        raise SystemExit(f"synthesis failed:\n{errors}")
    return json.loads(stats_path.read_text(encoding="utf-8")), log


def main() -> int:
    stats, log = synthesize()
    modules = stats.get("modules", {})
    rows = []
    for name, data in modules.items():
        cells = data.get("num_cells_by_type", {})
        luts = sum(v for k, v in cells.items() if k.startswith("$lut"))
        ffs = sum(v for k, v in cells.items() if "DFF" in k.upper())
        rows.append((short(name), luts, ffs, data.get("num_cells", 0)))
    rows.sort(key=lambda r: -(r[1] + r[2]))

    print(f"Generic synthesis of {TOP} ({LUT_SIZE}-LUT mapping, no vendor primitives)\n")
    print(f"{'module':<20}{'LUTs':>8}{'FFs':>8}{'cells':>8}")
    for name, luts, ffs, cells in rows:
        print(f"{name:<20}{luts:>8}{ffs:>8}{cells:>8}")
    print(f"{'TOTAL':<20}{sum(r[1] for r in rows):>8}{sum(r[2] for r in rows):>8}")

    warnings = len(re.findall(r"^Warning", log, re.M))
    print(f"\nyosys warnings: {warnings}; full log in results/rtl/synth.log")
    print("No timing, no vendor primitives, no power — see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
