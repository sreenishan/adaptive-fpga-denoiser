# fpga/asic/route_signoff.tcl — Stage 19+20+21: Routing + Signoff + GDSII (sky130hd)
#
# Reads the CTS checkpoint and runs:
#   global routing → detailed routing → antenna check →
#   timing signoff → power report → DEF/GDSII export
#
# Outputs:
#   fpga/asic/results/fpga_denoiser_top_routed.def
#   fpga/asic/results/fpga_denoiser_top_final.odb
#   fpga/asic/results/timing_report.txt
#   fpga/asic/results/power_report.txt
#   fpga/asic/results/fpga_denoiser_top.gds  (if KLayout is available)
#
# Run inside the ORFS Docker container:
#   openroad -exit fpga/asic/route_signoff.tcl

set TOP       "fpga_denoiser_top"
set PDK_ROOT  $env(PDK_ROOT)
set PLATFORM  "sky130A"
set SCL       "sky130_fd_sc_hd"

set LIB_TYP   "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lib/${SCL}__tt_025C_1v80.lib"
set LEF_TECH  "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/techlef/${SCL}__nom.tlef"
set LEF_CELLS "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lef/${SCL}.lef"

set SDC_FILE  "fpga/asic/constraint.sdc"
set ODB_IN    "fpga/asic/results/${TOP}_cts.odb"
set OUT_DIR   "fpga/asic/results"

# ── Restore CTS checkpoint ────────────────────────────────────────────────────
read_lef  $LEF_TECH
read_lef  $LEF_CELLS
read_lib  $LIB_TYP
read_db   $ODB_IN
read_sdc  $SDC_FILE

# ── Stage 19a: Global routing ─────────────────────────────────────────────────
global_route \
    -guide_file              $OUT_DIR/route.guide \
    -congestion_iterations   30 \
    -verbose                 1

puts "  Stage 19a: global routing complete"

# ── Stage 19b: Detailed routing ───────────────────────────────────────────────
detailed_route \
    -input_guide           $OUT_DIR/route.guide \
    -bottom_routing_layer  met1 \
    -top_routing_layer     met5 \
    -verbose               1

check_antennas
puts "  Stage 19b: detailed routing complete"

# ── Stage 20: Signoff checks ──────────────────────────────────────────────────
# Timing — min/max paths with full clock expansion; WNS + TNS summary
report_checks \
    -path_delay min_max \
    -format     full_clock_expanded \
    > $OUT_DIR/timing_report.txt
report_tns >> $OUT_DIR/timing_report.txt
report_wns >> $OUT_DIR/timing_report.txt

puts "  Stage 20a: timing report written"

# Area — core utilisation and cell count
report_design_area                          >> $OUT_DIR/area_report.txt
report_cell_usage -verbose                  >> $OUT_DIR/area_report.txt

puts "  Stage 20b: area report written"

# Power — switching activity estimate (20% toggle, 50% duty)
# A VCD from simulation would give accurate switching; this is a static estimate.
set_power_activity -input -activity 0.2 -duty 0.5
report_power > $OUT_DIR/power_report.txt

puts "  Stage 20c: power report written"

# Geometric DRC via OpenROAD (catches spacing/width violations before GDS)
if {[catch {check_drc -output_file $OUT_DIR/drc_openroad.rpt} err]} {
    puts "  Stage 20d: OpenROAD DRC not available in this build: $err"
} else {
    puts "  Stage 20d: OpenROAD DRC report written"
}

# ── Stage 21: Write final DEF + ODB ──────────────────────────────────────────
write_def $OUT_DIR/${TOP}_routed.def
write_db  $OUT_DIR/${TOP}_final.odb

# GDSII stream-out via KLayout using the sky130 DEF-to-GDS converter
if {[catch {
    exec klayout -zz \
        -rd input=$OUT_DIR/${TOP}_routed.def \
        -rd output=$OUT_DIR/${TOP}.gds \
        -r $PDK_ROOT/$PLATFORM/libs.tech/klayout/def2stream.py
} err]} {
    puts "WARNING: KLayout DEF→GDS failed: $err"
    puts "  Run manually after this script completes:"
    puts "  klayout -zz -rd input=$OUT_DIR/${TOP}_routed.def \\"
    puts "    -rd output=$OUT_DIR/${TOP}.gds \\"
    puts "    -r \$PDK_ROOT/sky130A/libs.tech/klayout/def2stream.py"
} else {
    puts "  Stage 21: GDSII written: $OUT_DIR/${TOP}.gds"
}

puts "\n*** Stages 19-21 complete ***"
puts "  DEF     : $OUT_DIR/${TOP}_routed.def"
puts "  Timing  : $OUT_DIR/timing_report.txt"
puts "  Area    : $OUT_DIR/area_report.txt"
puts "  Power   : $OUT_DIR/power_report.txt"
puts "  GDS     : $OUT_DIR/${TOP}.gds  (if KLayout succeeded)"
