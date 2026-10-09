# OpenROAD-flow-scripts (ORFS) platform configuration
# Run with the ORFS Docker image — see run_docker.sh
#
# Technology : sky130hd  (SkyWater 130 nm high-density standard cells)
# Top module : fpga_denoiser_top
# Target Fmax: 100 MHz  (10 ns clock period)
#
# Area estimate (from yosys ECP5 synthesis proxy):
#   ~2,049 LUT4 → ~6,000 sky130hd cells @ 40% utilisation
#   → core ≈ 250 μm × 250 μm

export DESIGN_NAME    = fpga_denoiser_top
export PLATFORM       = sky130hd

# ── Source files ────────────────────────────────────────────────────────────
# Path is relative to the ORFS flow/ root inside the container.
# The repo is mounted at /repo; see run_docker.sh.
export VERILOG_FILES  = $(sort $(wildcard /repo/rtl/*.sv))
export SDC_FILE       = /repo/fpga/asic/constraint.sdc

# ── Floorplan ────────────────────────────────────────────────────────────────
export CORE_UTILIZATION  = 40
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN       = 2.0

# ── Placement ────────────────────────────────────────────────────────────────
export PLACE_DENSITY     = 0.60

# ── CTS ──────────────────────────────────────────────────────────────────────
export CTS_CLUSTER_SIZE      = 30
export CTS_CLUSTER_DIAMETER  = 100

# ── Global routing ──────────────────────────────────────────────────────────
export ROUTING_LAYER_ADJUSTMENT = 0.5

# ── Signoff ──────────────────────────────────────────────────────────────────
# Hold slack target 0 ns; setup already constrained by SDC.
export TNS_END_PERCENT = 100
