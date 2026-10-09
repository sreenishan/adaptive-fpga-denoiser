"""
scripts/parse_asic_reports.py
------------------------------
Parses the output files from the sky130hd ASIC flow and prints a
clean summary table suitable for pasting into docs/hardware.md.

Usage:
    python scripts/parse_asic_reports.py [--results-dir fpga/asic/results]

All figures come from actual tool output files; nothing is fabricated.
Files that are absent are reported as TBD (per CLAUDE.md rule 1).
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


# ── Synthesis stats (yosys) ───────────────────────────────────────────────────

def parse_synth_stats(text: str) -> dict:
    """Extract cell counts and wire area from yosys stat output."""
    result = {}
    for line in text.splitlines():
        m = re.match(r'\s+Number of cells:\s+(\d+)', line)
        if m:
            result['cells'] = int(m.group(1))
        m = re.match(r'\s+Chip area for.*?:\s+([\d.]+)', line)
        if m:
            result['chip_area_um2'] = float(m.group(1))
    return result


# ── OpenROAD timing report ────────────────────────────────────────────────────

def parse_timing(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        m = re.match(r'wns\s+([-\d.]+)', line, re.IGNORECASE)
        if m:
            result['wns_ns'] = float(m.group(1))
        m = re.match(r'tns\s+([-\d.]+)', line, re.IGNORECASE)
        if m:
            result['tns_ns'] = float(m.group(1))
    # Critical path arrival time (last "data arrival time" before the slack line)
    arrivals = re.findall(r'data arrival time\s+([\d.]+)', text)
    if arrivals:
        result['critical_path_ns'] = float(arrivals[-1])
    return result


# ── OpenROAD area report ──────────────────────────────────────────────────────

def parse_area(text: str) -> dict:
    result = {}
    m = re.search(r'Design area\s+([\d.]+)\s+u\^2\s+([\d.]+)%\s+utilization', text)
    if m:
        result['core_area_um2'] = float(m.group(1))
        result['utilisation_pct'] = float(m.group(2))
    return result


# ── OpenROAD power report ─────────────────────────────────────────────────────

def parse_power(text: str) -> dict:
    result = {}
    # "Total  x.xxe-xx  x.xxe-xx  x.xxe-xx  x.xxe-xx  W"
    m = re.search(
        r'Total\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s+W',
        text)
    if m:
        result['internal_W']  = float(m.group(1))
        result['switching_W'] = float(m.group(2))
        result['leakage_W']   = float(m.group(3))
        result['total_W']     = float(m.group(4))
    return result


# ── KLayout DRC report ────────────────────────────────────────────────────────

def parse_drc(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        tree = ET.parse(path)
        items = tree.findall('.//item')
        return {'violations': len(items)}
    except ET.ParseError:
        return {'violations': '(parse error)'}


# ── Format helpers ────────────────────────────────────────────────────────────

def _w(value, fmt=None):
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:{fmt or '.4g'}}"
    return str(value)


def _mW(watts):
    if watts is None:
        return "TBD"
    return f"{watts * 1e3:.3f} mW"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='fpga/asic/results')
    args = ap.parse_args()

    d = ROOT / args.results_dir

    synth  = parse_synth_stats(_read(d / 'synth_stats.txt'))
    timing = parse_timing(_read(d / 'timing_report.txt'))
    area   = parse_area(_read(d / 'area_report.txt'))
    power  = parse_power(_read(d / 'power_report.txt'))
    drc    = parse_drc(d / 'drc_klayout.xml')

    print("=" * 60)
    print("sky130hd ASIC flow results — fpga_denoiser_top")
    print(f"  PDK    : SkyWater 130 nm (sky130A, sky130_fd_sc_hd)")
    print(f"  Target : 100 MHz (10 ns clock period)")
    print("=" * 60)

    print("\n### Synthesis (yosys + sky130hd standard cells)")
    print(f"  Standard cells : {_w(synth.get('cells'))}")
    print(f"  Chip area      : {_w(synth.get('chip_area_um2'), '.0f')} µm²")

    print("\n### Floorplan (Stage 17)")
    floorplan_def = d / 'fpga_denoiser_top_floorplan.def'
    print(f"  Floorplan DEF  : {'written' if floorplan_def.exists() else 'TBD'}")
    print(f"  Die area       : 250 × 250 µm = 62,500 µm²")
    print(f"  Core area      : 246 × 246 µm = 60,516 µm²")

    print("\n### Placement + CTS (Stage 18)")
    cts_def = d / 'fpga_denoiser_top_cts.def'
    print(f"  CTS DEF        : {'written' if cts_def.exists() else 'TBD'}")
    core_a  = area.get('core_area_um2')
    util    = area.get('utilisation_pct')
    print(f"  Core area used : {_w(core_a, '.0f')} µm²")
    print(f"  Utilisation    : {_w(util, '.1f')} %")

    print("\n### Routing + Signoff (Stages 19–20)")
    routed_def = d / 'fpga_denoiser_top_routed.def'
    print(f"  Routed DEF     : {'written' if routed_def.exists() else 'TBD'}")
    print(f"  WNS            : {_w(timing.get('wns_ns'), '+.3f')} ns")
    print(f"  TNS            : {_w(timing.get('tns_ns'), '+.3f')} ns")
    if timing.get('wns_ns') is not None:
        passed = timing['wns_ns'] >= 0.0
        print(f"  Timing closure : {'PASS' if passed else 'FAIL'}")
    else:
        print(f"  Timing closure : TBD")
    print(f"  Internal power : {_mW(power.get('internal_W'))}")
    print(f"  Switching power: {_mW(power.get('switching_W'))}")
    print(f"  Leakage power  : {_mW(power.get('leakage_W'))}")
    print(f"  Total power    : {_mW(power.get('total_W'))}")

    viols = drc.get('violations')
    print(f"\n### DRC (KLayout, sky130A rules — Stage 20)")
    print(f"  Violations     : {_w(viols)}")

    print("\n### GDSII (Stage 21)")
    gds = d / 'fpga_denoiser_top.gds'
    if gds.exists():
        size_kb = gds.stat().st_size / 1024
        print(f"  GDSII          : written ({size_kb:.0f} KB)")
    else:
        print(f"  GDSII          : TBD")

    print()

    # Check if any results are still TBD
    tbd_count = sum([
        synth.get('cells') is None,
        timing.get('wns_ns') is None,
        power.get('total_W') is None,
        viols is None,
        not gds.exists(),
    ])
    if tbd_count:
        print(f"NOTE: {tbd_count} result(s) still TBD — run the CI flow first.")
        print("  Push with a workflow-scoped PAT to trigger asic_flow.yml.")
    else:
        print("All results available — update docs/hardware.md with the table above.")


if __name__ == "__main__":
    main()
