#!/usr/bin/env bash
# run_tb.sh — compile and run all RTL testbenches with Icarus Verilog.
#
# Usage:
#   cd <repo-root>
#   bash rtl/tb/run_tb.sh          # run all testbenches
#   bash rtl/tb/run_tb.sh median   # run only tb_median_filter
#
# Icarus Verilog must be on PATH.  On Windows with the default installer:
#   export PATH="/c/iverilog/bin:$PATH"
#
# Exit code: 0 if all testbenches pass, 1 if any fail.

set -euo pipefail

IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
IVFLAGS="-g2012 -Wall"

RTL_SRCS=(
    rtl/line_buffer.sv
    rtl/window_gen.sv
    rtl/median_filter.sv
    rtl/gaussian_filter.sv
    rtl/wiener_filter.sv
    rtl/filter_controller.sv
    rtl/fpga_denoiser_top.sv
)

TBS=(
    tb_median_filter
    tb_gaussian_filter
    tb_wiener_filter
    tb_window_gen
    tb_filter_controller
    tb_fpga_denoiser_top
)

# Filter to requested testbench if an argument is given.
if [[ $# -ge 1 ]]; then
    TBS=("tb_${1#tb_}")          # accept "median" or "tb_median_filter"
    TBS=("${TBS[@]/%/_filter/}")  # add _filter suffix if missing
    TBS[0]="tb_${1}"             # simpler: just prefix with tb_
fi

PASS=0
FAIL=0
OUTDIR="${TMPDIR:-/tmp}/fdeg_tb"
mkdir -p "$OUTDIR"

for tb in "${TBS[@]}"; do
    sv="rtl/tb/${tb}.sv"
    if [[ ! -f "$sv" ]]; then
        echo "SKIP: $sv not found"
        continue
    fi

    out="$OUTDIR/${tb}.vvp"
    echo "──────────────────────────────────────────"
    echo "COMPILE: $tb"
    if ! $IVERILOG $IVFLAGS -o "$out" "${RTL_SRCS[@]}" "$sv" 2>&1; then
        echo "COMPILE FAIL: $tb"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo "RUN: $tb"
    if $VVP "$out" 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo "══════════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
