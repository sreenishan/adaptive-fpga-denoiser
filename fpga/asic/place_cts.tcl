# fpga/asic/place_cts.tcl — Stage 18: Placement + CTS (sky130hd)
#
# Reads the floorplan checkpoint and runs:
#   global placement → detailed placement → clock tree synthesis →
#   post-CTS hold repair → final detailed placement
#
# Outputs:
#   fpga/asic/results/fpga_denoiser_top_cts.def
#   fpga/asic/results/fpga_denoiser_top_cts.odb
#
# Run inside the ORFS Docker container:
#   openroad -exit fpga/asic/place_cts.tcl

set TOP       "fpga_denoiser_top"
set PDK_ROOT  $env(PDK_ROOT)
set PLATFORM  "sky130A"
set SCL       "sky130_fd_sc_hd"

set LIB_TYP   "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lib/${SCL}__tt_025C_1v80.lib"
set LEF_TECH  "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/techlef/${SCL}__nom.tlef"
set LEF_CELLS "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lef/${SCL}.lef"

set SDC_FILE  "fpga/asic/constraint.sdc"
set ODB_IN    "fpga/asic/results/${TOP}_floorplan.odb"
set OUT_DIR   "fpga/asic/results"

# ── Restore floorplan checkpoint ──────────────────────────────────────────────
read_lef  $LEF_TECH
read_lef  $LEF_CELLS
read_lib  $LIB_TYP
read_db   $ODB_IN
read_sdc  $SDC_FILE

# ── Global placement ──────────────────────────────────────────────────────────
global_placement \
    -density  0.60 \
    -pad_left 2 -pad_right 2

repair_design

# ── Detailed placement ────────────────────────────────────────────────────────
detailed_placement
check_placement -verbose

write_def $OUT_DIR/${TOP}_placed.def
write_db  $OUT_DIR/${TOP}_placed.odb
puts "  Checkpoint: detailed placement written"

# ── Clock tree synthesis ──────────────────────────────────────────────────────
# Note: inside TCL {}, backslash is literal — keep buf_list on one logical line.
clock_tree_synthesis \
    -root_buf  sky130_fd_sc_hd__buf_12 \
    -buf_list  {sky130_fd_sc_hd__buf_4 sky130_fd_sc_hd__buf_8 sky130_fd_sc_hd__buf_12} \
    -wire_unit 20

# ── Post-CTS timing repair ────────────────────────────────────────────────────
set_propagated_clock [all_clocks]
repair_timing -hold -slack_margin 0.1

detailed_placement
check_placement -verbose

# ── Save CTS checkpoint ───────────────────────────────────────────────────────
write_def $OUT_DIR/${TOP}_cts.def
write_db  $OUT_DIR/${TOP}_cts.odb

puts "\n*** Stage 18 complete: placement + CTS written ***"
puts "  DEF : $OUT_DIR/${TOP}_cts.def"
puts "  ODB : $OUT_DIR/${TOP}_cts.odb"
