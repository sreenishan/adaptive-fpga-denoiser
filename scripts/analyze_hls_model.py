"""
scripts/analyze_hls_model.py
----------------------------
Reports exact facts about the hls4ml CNN project that can be derived
from the generated source files without running Vitis HLS synthesis.

Outputs:
  - Layer structure (sizes, reuse factor, strategy)
  - Weight storage in bytes (exact for ap_fixed<16,6> = 2 bytes/value)
  - Theoretical minimum multiply-accumulate operations per inference

Resource figures (LUT / FF / DSP48E / BRAM) require Vitis HLS synthesis
and are NOT reported here.  Run:
  vitis_hls -f fpga/hls/noise_classifier_hls/build_prj.tcl csim=0 synth=1
to get them.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS_H = ROOT / "fpga/hls/noise_classifier_hls/firmware/parameters.h"
WEIGHTS_DIR = ROOT / "fpga/hls/noise_classifier_hls/firmware/weights"

BYTES_PER_WEIGHT = 2  # ap_fixed<16,6>


def parse_weight_files():
    """Return {name: element_count} by counting values in each .txt file."""
    counts = {}
    for f in sorted(WEIGHTS_DIR.glob("*.txt")):
        values = [v for v in f.read_text().split() if v.strip()]
        counts[f.stem] = len(values)
    return counts


def parse_layers(params_text):
    """Extract conv2d and dense layer configs from parameters.h."""
    layers = []

    # Conv2D layers: find config structs with n_filt and n_chan
    conv_pattern = re.compile(
        r'struct\s+(config\d+)\s*:\s*nnet::conv2d_config\s*\{([^}]+)\}',
        re.DOTALL)
    for m in conv_pattern.finditer(params_text):
        name, body = m.group(1), m.group(2)

        def val(key):
            hit = re.search(rf'static const unsigned {key}\s*=\s*(\d+)', body)
            return int(hit.group(1)) if hit else None

        layers.append({
            'type': 'Conv2D',
            'name': name,
            'n_chan': val('n_chan'),
            'n_filt': val('n_filt'),
            'filt_h': val('filt_height'),
            'filt_w': val('filt_width'),
            'out_h': val('out_height'),
            'out_w': val('out_width'),
            'reuse': val('reuse_factor'),
            'partitions': val('n_partitions'),
        })

    # Dense (classifier) layer
    dense_pattern = re.compile(
        r'struct\s+(config19)\s*:\s*nnet::dense_config\s*\{([^}]+)\}',
        re.DOTALL)
    for m in dense_pattern.finditer(params_text):
        name, body = m.group(1), m.group(2)

        def val(key):
            hit = re.search(rf'static const unsigned {key}\s*=\s*(\d+)', body)
            return int(hit.group(1)) if hit else None

        layers.append({
            'type': 'Dense',
            'name': name,
            'n_in': val('n_in'),
            'n_out': val('n_out'),
            'reuse': val('reuse_factor'),
            'n_nonzeros': val('n_nonzeros'),
        })

    return layers


def main():
    if not PARAMS_H.exists():
        print(f"ERROR: {PARAMS_H} not found", file=sys.stderr)
        sys.exit(1)

    params_text = PARAMS_H.read_text()
    weight_counts = parse_weight_files()
    layers = parse_layers(params_text)

    print("=" * 68)
    print("hls4ml CNN model analysis")
    print(f"  Source : {PARAMS_H.relative_to(ROOT)}")
    print(f"  Target : xc7z020clg400-1  (from project.tcl)")
    print(f"  Clock  : 10.0 ns / 100 MHz")
    print(f"  Type   : ap_fixed<16,6>  ({BYTES_PER_WEIGHT} bytes/value)")
    print("=" * 68)

    print("\nLayer structure:")
    print(f"  {'Layer':<14} {'Type':<8} {'Shape in→out':<28} {'RF':>5}  {'MACs/inf':>12}")
    print(f"  {'-'*14} {'-'*8} {'-'*28} {'-'*5}  {'-'*12}")

    total_macs = 0
    for L in layers:
        if L['type'] == 'Conv2D':
            ksize = L['filt_h'] * L['filt_w']
            macs = ksize * L['n_chan'] * L['n_filt'] * L['out_h'] * L['out_w']
            shape = f"{L['n_chan']}×{L['filt_h']}×{L['filt_w']}→{L['n_filt']}  {L['out_h']}×{L['out_w']}"
            rf_str = str(L['reuse'])
            print(f"  {L['name']:<14} {'Conv2D':<8} {shape:<28} {rf_str:>5}  {macs:>12,}")
            total_macs += macs
        elif L['type'] == 'Dense':
            macs = L['n_in'] * L['n_out']
            shape = f"{L['n_in']}→{L['n_out']}"
            rf_str = str(L['reuse'])
            print(f"  {L['name']:<14} {'Dense':<8} {shape:<28} {rf_str:>5}  {macs:>12,}")
            total_macs += macs

    print(f"\n  Total MACs per inference: {total_macs:,}")

    print("\nWeight storage (exact, ap_fixed<16,6> = 2 bytes each):")
    total_weights = 0
    total_bytes = 0
    for stem in sorted(weight_counts):
        n = weight_counts[stem]
        b = n * BYTES_PER_WEIGHT
        kind = "bias" if stem.startswith('b') else "weight"
        print(f"  {stem:<6}  {n:>7,} {kind}s  =  {b:>8,} bytes")
        total_weights += n
        total_bytes += b

    print(f"  {'Total':<6}  {total_weights:>7,} values  =  {total_bytes:>8,} bytes"
          f"  ({total_bytes / 1024:.1f} KB)")

    print("\nSynthesis resource figures:")
    print("  LUT      : TBD  (run vitis_hls -f build_prj.tcl synth=1)")
    print("  FF       : TBD")
    print("  DSP48E   : TBD")
    print("  BRAM_18K : TBD")
    print("  Latency  : TBD")
    print()


if __name__ == "__main__":
    main()
