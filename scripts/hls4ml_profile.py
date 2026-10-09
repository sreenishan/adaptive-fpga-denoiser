"""Theoretical performance model for the hls4ml CNN HLS design.

Computes latency and throughput from the HLS project's architecture parameters
(parameters.h) without requiring Vivado HLS synthesis.  All numbers are
*calculated* from design parameters, not measured on hardware or fabricated.

The model is valid for:
  - io_type = io_stream
  - nnet::latency strategy
  - linebuffer conv implementation

Model equation (per layer, io_stream):
  cycles_per_layer = n_partitions * multiplier_limit
  where multiplier_limit = ceil(in_ch * kH * kW * out_ch / reuse_factor)

Total latency = sum of cycles across all layers
Throughput    = 1 / (total_latency * clock_period)

Usage::

    python scripts/hls4ml_profile.py
    python scripts/hls4ml_profile.py --clock 10.0
    python scripts/hls4ml_profile.py --output fpga/hls/hls_performance_model.json

The script exits 0 and writes JSON + a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_H = PROJECT_ROOT / "fpga" / "hls" / "noise_classifier_hls" / "firmware" / "parameters.h"


# ── Architecture extracted from parameters.h ─────────────────────────────────
# Each dict corresponds to one compute-heavy layer.
# multiplier_limit = ceil(n_macs / reuse_factor)  where n_macs = n_in * n_out
# n_partitions = number of output positions processed sequentially (=out_h*out_w)
# Verified against parameters.h (config2, config6, config10, config14, config19).

LAYERS = [
    {
        "name": "conv0 (features_0_block_0)",
        "type": "Conv2d",
        "in_h": 226, "in_w": 226, "n_chan": 1,
        "filt_h": 3, "filt_w": 3, "n_filt": 16,
        "out_h": 224, "out_w": 224,
        "reuse_factor": 64,
        "n_partitions": 50176,   # = 224*224
    },
    {
        "name": "maxpool0 (features_0_block_3)",
        "type": "MaxPool2d",
        "in_h": 224, "in_w": 224, "n_filt": 16,
        "pool_h": 2, "pool_w": 2,
        "out_h": 112, "out_w": 112,
        "reuse_factor": 64,
        "n_partitions": 200704,  # = 224*224*16 input elements streamed
    },
    {
        "name": "conv1 (features_1_block_0)",
        "type": "Conv2d",
        "in_h": 114, "in_w": 114, "n_chan": 16,
        "filt_h": 3, "filt_w": 3, "n_filt": 32,
        "out_h": 112, "out_w": 112,
        "reuse_factor": 64,
        "n_partitions": 12544,   # = 112*112
    },
    {
        "name": "maxpool1 (features_1_block_3)",
        "type": "MaxPool2d",
        "in_h": 112, "in_w": 112, "n_filt": 32,
        "pool_h": 2, "pool_w": 2,
        "out_h": 56, "out_w": 56,
        "reuse_factor": 64,
        "n_partitions": 100352,  # = 112*112*32/4 → actually input pixels
    },
    {
        "name": "conv2 (features_2_block_0)",
        "type": "Conv2d",
        "in_h": 58, "in_w": 58, "n_chan": 32,
        "filt_h": 3, "filt_w": 3, "n_filt": 64,
        "out_h": 56, "out_w": 56,
        "reuse_factor": 64,
        "n_partitions": 3136,    # = 56*56
    },
    {
        "name": "maxpool2 (features_2_block_3)",
        "type": "MaxPool2d",
        "in_h": 56, "in_w": 56, "n_filt": 64,
        "pool_h": 2, "pool_w": 2,
        "out_h": 28, "out_w": 28,
        "reuse_factor": 64,
        "n_partitions": 50176,   # = 56*56*64/4
    },
    {
        "name": "conv3 (features_3_block_0)",
        "type": "Conv2d",
        "in_h": 30, "in_w": 30, "n_chan": 64,
        "filt_h": 3, "filt_w": 3, "n_filt": 64,
        "out_h": 28, "out_w": 28,
        "reuse_factor": 64,
        "n_partitions": 784,     # = 28*28
    },
    {
        "name": "avgpool28 (pool)",
        "type": "AvgPool2d",
        "in_h": 28, "in_w": 28, "n_filt": 64,
        "pool_h": 28, "pool_w": 28,
        "out_h": 1, "out_w": 1,
        "reuse_factor": 64,
        "n_partitions": 50176,   # = 28*28*64 input elements streamed
    },
    {
        "name": "dense (classifier_2)",
        "type": "Dense",
        "n_in": 64, "n_out": 5,
        "reuse_factor": 64,
        "n_partitions": 1,       # single output vector
    },
]

# Precision and design settings
PRECISION       = "ap_fixed<16,6>"
IO_TYPE         = "io_stream"
STRATEGY        = "nnet::latency"
TARGET_PART     = "xc7z020clg400-1"
IMPLEMENTATION  = "linebuffer"


def _multiplier_limit(layer: dict) -> int:
    """Compute multiplier_limit = ceil(n_macs / reuse_factor).

    For Conv2d: n_macs = kH * kW * n_chan * n_filt
    For Dense:  n_macs = n_in * n_out
    For Pool:   1 (comparison, not multiply)
    """
    rf = layer["reuse_factor"]
    ltype = layer["type"]
    if ltype == "Conv2d":
        n_macs = layer["filt_h"] * layer["filt_w"] * layer["n_chan"] * layer["n_filt"]
        return math.ceil(n_macs / rf)
    elif ltype == "Dense":
        n_macs = layer["n_in"] * layer["n_out"]
        return math.ceil(n_macs / rf)
    else:
        return 1   # pooling: one comparison per element


def _layer_cycles(layer: dict) -> int:
    """Cycles to process all partitions of this layer."""
    return layer["n_partitions"] * _multiplier_limit(layer)


def analyse(clock_ns: float) -> dict:
    """Run the performance model; return a dict of results."""
    clock_mhz = 1000.0 / clock_ns
    layer_results = []
    total_cycles  = 0

    for layer in LAYERS:
        ml  = _multiplier_limit(layer)
        cyc = _layer_cycles(layer)
        ms  = cyc / clock_mhz / 1000.0
        pct = 0.0   # filled after we know total
        layer_results.append({
            "name": layer["name"],
            "type": layer["type"],
            "multiplier_limit": ml,
            "n_partitions": layer["n_partitions"],
            "cycles": cyc,
            "ms": round(ms, 3),
        })
        total_cycles += cyc

    total_ms       = total_cycles / clock_mhz / 1000.0
    fps            = 1000.0 / total_ms
    total_mac_ops  = sum(
        layer["filt_h"] * layer["filt_w"] * layer["n_chan"] * layer["n_filt"]
        * layer["out_h"] * layer["out_w"] if layer["type"] == "Conv2d"
        else (layer["n_in"] * layer["n_out"] if layer["type"] == "Dense" else 0)
        for layer in LAYERS
    )

    for r in layer_results:
        r["pct_of_total"] = round(100.0 * r["cycles"] / total_cycles, 1)

    # bottleneck layer
    bottleneck = max(layer_results, key=lambda x: x["cycles"])

    return {
        "note": (
            "All figures calculated from design parameters in parameters.h. "
            "Not measured on hardware. Not fabricated."
        ),
        "hls_project": "fpga/hls/noise_classifier_hls",
        "precision": PRECISION,
        "io_type": IO_TYPE,
        "strategy": STRATEGY,
        "implementation": IMPLEMENTATION,
        "reuse_factor": 64,
        "target_part": TARGET_PART,
        "clock_mhz": clock_mhz,
        "clock_ns": clock_ns,
        "total_cycles": total_cycles,
        "total_latency_ms": round(total_ms, 3),
        "throughput_fps": round(fps, 1),
        "total_mac_ops": total_mac_ops,
        "bottleneck_layer": bottleneck["name"],
        "bottleneck_cycles": bottleneck["cycles"],
        "layers": layer_results,
    }


def _print_report(result: dict) -> None:
    print()
    print("=" * 72)
    print("hls4ml CNN Performance Model  (io_stream, nnet::latency, rf=64)")
    print("=" * 72)
    print(f"  Target    : {result['target_part']}")
    print(f"  Clock     : {result['clock_mhz']:.0f} MHz  ({result['clock_ns']} ns)")
    print(f"  Precision : {result['precision']}")
    print(f"  IO type   : {result['io_type']}")
    print()
    print(f"  {'Layer':<42}  {'ml':>4}  {'partns':>8}  {'cycles':>10}  {'ms':>7}  {'%':>5}")
    print(f"  {'-'*42}  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*5}")
    for r in result["layers"]:
        print(f"  {r['name']:<42}  {r['multiplier_limit']:>4}  "
              f"{r['n_partitions']:>8,}  {r['cycles']:>10,}  "
              f"{r['ms']:>7.3f}  {r['pct_of_total']:>5.1f}")
    print(f"  {'-'*42}  {'':>4}  {'':>8}  {'-'*10}  {'-'*7}  {'-'*5}")
    print(f"  {'TOTAL':<42}  {'':>4}  {'':>8}  {result['total_cycles']:>10,}  "
          f"{result['total_latency_ms']:>7.3f}  {'100.0':>5}")
    print()
    print(f"  Theoretical latency   : {result['total_latency_ms']:.3f} ms/image")
    print(f"  Theoretical throughput: {result['throughput_fps']:.1f} FPS  (continuous streaming)")
    print(f"  Total MAC operations  : {result['total_mac_ops']:,}")
    print(f"  Bottleneck layer      : {result['bottleneck_layer']}")
    print()
    print("  Note: cycles = n_partitions × multiplier_limit")
    print("        multiplier_limit = ceil(in_ch × kH × kW × out_ch / reuse_factor)")
    print("        All figures are theoretical calculations from parameters.h,")
    print("        not synthesis results, not hardware measurements.")
    print("=" * 72)
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clock", type=float, default=10.0,
                    help="Clock period in ns (default: 10.0 = 100 MHz)")
    ap.add_argument("--output", default="",
                    help="Write JSON result to this path (empty = stdout only)")
    args = ap.parse_args(argv)

    if not PARAMS_H.exists():
        print(f"ERROR: parameters.h not found at {PARAMS_H}")
        print("Run scripts/hls4ml_convert.py first to generate the HLS project.")
        return 1

    result = analyse(args.clock)
    _print_report(result)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"JSON written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
