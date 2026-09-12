"""The RTL must still be synthesisable, and must not infer latches.

Simulation cannot see either. `median_filter` swapped through a module-level
temporary that was assigned only inside `if` branches; Icarus ran it happily
for as long as it existed, while synthesis inferred a latch and refused the
design. This runs yosys's elaboration and process conversion — under a second,
because it stops before technology mapping — and fails on any latch.

Full area numbers come from `scripts/synthesize_rtl.py`, which takes minutes
and is not run here.

Without yosys this SKIPS and says so. It must never be made to pass without
synthesising, for the same reason the co-simulation test must not.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOP = "fpga_denoiser_top"


def _yosys_available() -> bool:
    suite = Path.home() / "tools" / "oss-cad-suite" / "bin" / "yosys.exe"
    return bool(os.environ.get("YOSYS") or shutil.which("yosys") or suite.exists())


pytestmark = pytest.mark.skipif(
    not _yosys_available(),
    reason="yosys not found (set $YOSYS, add to PATH, or install oss-cad-suite) — "
           "the RTL was NOT synthesised in this run",
)


@pytest.fixture(scope="module")
def elaboration() -> str:
    spec = importlib.util.spec_from_file_location(
        "synthesize_rtl", ROOT / "scripts" / "synthesize_rtl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yosys, env = module.find_yosys()
    sources = " ".join(module.rtl_sources())
    proc = subprocess.run(
        [yosys, "-p", f"read_verilog -sv {sources}; hierarchy -top {TOP}; proc; check -assert"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    return proc.stdout + proc.stderr + f"\n__returncode__ {proc.returncode}"


def test_rtl_elaborates_and_passes_yosys_check(elaboration: str) -> None:
    assert "__returncode__ 0" in elaboration, elaboration[-3000:]
    assert "Found and reported 0 problems" in elaboration, elaboration[-3000:]


def test_no_latch_is_inferred(elaboration: str) -> None:
    """A latch in combinational logic is a hardware bug a simulator cannot show."""
    latches = re.findall(r"^ERROR: Latch inferred.*", elaboration, re.M)
    assert not latches, latches


def test_every_module_in_the_filelist_is_elaborated(elaboration: str) -> None:
    spec = importlib.util.spec_from_file_location(
        "synthesize_rtl", ROOT / "scripts" / "synthesize_rtl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for source in module.rtl_sources():
        name = Path(source).stem
        assert name in elaboration, f"{name} never elaborated — is it still instantiated?"
