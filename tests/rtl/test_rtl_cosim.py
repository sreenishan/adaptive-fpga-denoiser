"""The SystemVerilog itself, simulated, judged by the Python golden filters.

``test_rtl_golden`` checks a Python transcription of the RTL; it cannot see a
syntax, elaboration or streaming-protocol error. This runs the real
``fpga_denoiser_top`` in Icarus Verilog on small real frames (quick mode of
``scripts/simulate_rtl.py``) and holds every pixel to ``configs/hardware.yaml``.

Without Icarus this SKIPS, and says so in the skip reason. It must never be
turned into a pass: a green run that simulated nothing is how this project once
described its RTL as testbench-verified when no simulator had run.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _icarus_available() -> bool:
    for name, env in (("iverilog", "IVERILOG"), ("vvp", "VVP")):
        found = os.environ.get(env) or shutil.which(name)
        if not (found and Path(found).exists()) and not Path(f"C:/iverilog/bin/{name}.exe").exists():
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _icarus_available(),
    reason="Icarus Verilog not found (set $IVERILOG/$VVP, add to PATH, or install to "
           "C:/iverilog) — the RTL was NOT simulated in this run",
)


def _cosim_module():
    spec = importlib.util.spec_from_file_location("simulate_rtl", ROOT / "scripts" / "simulate_rtl.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves its module through sys.modules; a file loaded by path
    # is not registered there until we do it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cosim(tmp_path_factory):
    module = _cosim_module()
    results, divergence = module.cosimulate(31, 17, quick=True,
                                            workdir=tmp_path_factory.mktemp("cosim"),
                                            log=lambda *_: None)
    return results, divergence


def test_every_filter_on_every_frame_is_within_tolerance(cosim) -> None:
    results, _ = cosim
    failed = [f"{r.case}/{r.filter}{' stall' if r.stall else ''}: max|err| {r.max_abs_error} "
              f"> {r.tolerance}" for r in results if not r.passed]
    assert not failed, failed


def test_bit_exact_filters_really_are_bit_exact(cosim) -> None:
    results, _ = cosim
    for r in results:
        if r.filter in ("bypass", "median", "gaussian"):
            assert r.mismatched == 0, f"{r.case}/{r.filter}: {r.mismatched} pixels differ"


def test_all_four_codes_and_a_stalled_stream_were_exercised(cosim) -> None:
    results, _ = cosim
    assert {r.filter for r in results} == {"bypass", "median", "gaussian", "wiener"}
    assert any(r.stall for r in results), "no frame ran with input stalls"
    assert all(r.pixels == 31 * 17 for r in results)
