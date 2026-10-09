#!/usr/bin/env bash
# run_flow.sh — RTL-to-GDSII for fpga_denoiser_top using sky130hd
#
# Requires Docker (any platform) or a Linux env with OpenROAD + sky130 PDK.
#
# Usage:
#   bash fpga/asic/run_flow.sh                  # Docker (auto-pulls image)
#   bash fpga/asic/run_flow.sh --local          # no Docker, local tools
#   bash fpga/asic/run_flow.sh --pdk /path/sky  # override PDK path
#
# Outputs land in fpga/asic/results/:
#   *_sky130.v       gate-level netlist
#   synth_stats.txt  cell counts and area
#   timing_report.txt WNS / TNS after PnR
#   power_report.txt  static + dynamic power estimate
#   *_routed.def     post-route DEF
#   *.gds            GDSII layout (if KLayout available)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/fpga/asic/results"
mkdir -p "$OUT_DIR"

# OpenROAD-flow-scripts Docker image (includes OpenROAD, yosys, KLayout, PDK)
ORFS_IMAGE="openroad/flow-ubuntu22.04-builder:latest"
# PDK inside the ORFS image
ORFS_PDK="/OpenROAD-flow-scripts/flow/platforms"

USE_DOCKER=1
PDK_ROOT="$ORFS_PDK"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)    USE_DOCKER=0 ;;
        --pdk)      PDK_ROOT="$2"; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Run flow ──────────────────────────────────────────────────────────────────
FLOW_CMDS="
set -euo pipefail
export PDK_ROOT='$PDK_ROOT'
cd /repo
mkdir -p fpga/asic/results

echo '=== Step 1/4: Synthesis (yosys + sky130hd) ==='
yosys fpga/asic/synth_sky130.tcl

echo '=== Step 2/4: Floorplan (Stage 17) ==='
openroad -exit fpga/asic/floorplan.tcl

echo '=== Step 3/4: Placement + CTS (Stage 18) ==='
openroad -exit fpga/asic/place_cts.tcl

echo '=== Step 4/4: Routing + Signoff + GDSII (Stages 19-21) ==='
openroad -exit fpga/asic/route_signoff.tcl

echo ''
echo '=== Flow complete ==='
echo 'Results in fpga/asic/results/'
ls -lh fpga/asic/results/
"

if [[ "$USE_DOCKER" -eq 1 ]]; then
    echo "Pulling $ORFS_IMAGE (first run downloads ~2 GB) …"
    docker pull "$ORFS_IMAGE"
    docker run --rm \
        -v "$REPO_ROOT:/repo" \
        -e PDK_ROOT="$PDK_ROOT" \
        "$ORFS_IMAGE" \
        bash -c "$FLOW_CMDS"
else
    # Local run — requires openroad and yosys on PATH, PDK_ROOT set
    if [[ -z "${PDK_ROOT:-}" ]]; then
        echo "ERROR: set PDK_ROOT to the sky130A PDK path"
        exit 1
    fi
    eval "$FLOW_CMDS"
fi
