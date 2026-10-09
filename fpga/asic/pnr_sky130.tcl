# OpenROAD place-and-route script — sky130hd
#
# Runs the complete physical design flow:
#   read netlist → floorplan → I/O placement → power grid →
#   global placement → CTS → routing → signoff → GDSII
#
# Usage (inside OpenROAD or the ORFS container):
#
#   openroad fpga/asic/pnr_sky130.tcl
#
# Environment variables required:
#   PDK_ROOT   — path to an installed sky130A PDK
#                (e.g. /root/pdks or /openroad-flow-scripts/flow/platforms)
#
# Outputs (written to fpga/asic/results/):
#   *.def, *.odb, *.gds, timing_report.txt, power_report.txt

# ── Configuration ─────────────────────────────────────────────────────────────
set TOP        "fpga_denoiser_top"
set PDK_ROOT   $env(PDK_ROOT)
set PLATFORM   "sky130A"
set SCL        "sky130_fd_sc_hd"

set LIB_TYPICAL "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lib/${SCL}__tt_025C_1v80.lib"
set LEF_TECH    "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/techlef/${SCL}__nom.tlef"
set LEF_CELLS   "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lef/${SCL}.lef"

set NETLIST     "fpga/asic/results/${TOP}_sky130.v"
set SDC_FILE    "fpga/asic/constraint.sdc"
set OUT_DIR     "fpga/asic/results"

# Die/core area (μm).  250×250 die, 2 μm margin → 246×246 core.
set DIE_AREA   "0 0 250 250"
set CORE_AREA  "2 2 248 248"

# ── Load libraries ────────────────────────────────────────────────────────────
read_lef  $LEF_TECH
read_lef  $LEF_CELLS
read_lib  $LIB_TYPICAL

# ── Read netlist and constraints ──────────────────────────────────────────────
read_verilog  $NETLIST
link_design   $TOP
read_sdc      $SDC_FILE

# ── Floorplan ─────────────────────────────────────────────────────────────────
initialize_floorplan \
    -die_area  $DIE_AREA \
    -core_area $CORE_AREA \
    -site      unithd

# I/O placement: auto ring around perimeter
place_pins -hor_layers met3 -ver_layers met4

# Power grid (VDD/VSS stripes on met1/met4)
add_global_connection -net VDD -pin_pattern {^VPWR$} -power
add_global_connection -net VDD -pin_pattern {^VPB$}  -power
add_global_connection -net VSS -pin_pattern {^VGND$} -ground
add_global_connection -net VSS -pin_pattern {^VNB$}  -ground

set_voltage_domain -power VDD -ground VSS

define_pdn_grid -name stdcell_grid -starts_with POWER -voltage_domains {CORE}
add_pdn_stripe -grid stdcell_grid -layer met1 -width 0.48 -followpins
add_pdn_stripe -grid stdcell_grid -layer met4 -width 1.6  -pitch 50.0 -offset 12.5
add_pdn_stripe -grid stdcell_grid -layer met5 -width 1.6  -pitch 50.0 -offset 12.5
add_pdn_connect -grid stdcell_grid -layers {met1 met4}
add_pdn_connect -grid stdcell_grid -layers {met4 met5}

pdngen

# ── Stage 18a: Global placement ──────────────────────────────────────────────
global_placement \
    -density 0.60 \
    -pad_left 2 -pad_right 2

repair_design
write_def $OUT_DIR/${TOP}_global_placed.def
puts "*** Stage 17/18a complete: floorplan + global placement DEF written ***"

# ── Stage 18b: Detailed placement ────────────────────────────────────────────
detailed_placement
check_placement -verbose
write_def $OUT_DIR/${TOP}_detailed_placed.def
puts "*** Stage 18b complete: detailed placement DEF written ***"

# ── Stage 18c: Clock tree synthesis ──────────────────────────────────────────
clock_tree_synthesis \
    -root_buf "sky130_fd_sc_hd__buf_12" \
    -buf_list {"sky130_fd_sc_hd__buf_4" "sky130_fd_sc_hd__buf_8" "sky130_fd_sc_hd__buf_12"} \
    -wire_unit 20

repair_clock_skew

set_propagated_clock [all_clocks]
repair_timing -hold -slack_margin 0.1
detailed_placement
check_placement -verbose
write_def $OUT_DIR/${TOP}_cts.def
puts "*** Stage 18c complete: CTS DEF written ***"

# ── Stage 19a: Global routing ─────────────────────────────────────────────────
global_route \
    -guide_file $OUT_DIR/route.guide \
    -congestion_iterations 30 \
    -verbose 1
puts "*** Stage 19a complete: global routing guide written ***"

# ── Stage 19b: Detailed routing ───────────────────────────────────────────────
detailed_route \
    -input_guide $OUT_DIR/route.guide \
    -bottom_routing_layer met1 \
    -top_routing_layer    met5 \
    -verbose 1

check_antennas

# ── Signoff timing ────────────────────────────────────────────────────────────
report_checks -path_delay min_max -format full_clock_expanded \
    > $OUT_DIR/timing_report.txt

report_tns >> $OUT_DIR/timing_report.txt
report_wns >> $OUT_DIR/timing_report.txt

# ── Power estimation ─────────────────────────────────────────────────────────
# Activity from VCD would give accurate numbers; switching activity estimate used here.
set_power_activity -input -activity 0.2 -duty 0.5
report_power > $OUT_DIR/power_report.txt

# ── Write DEF and GDS ────────────────────────────────────────────────────────
write_def   $OUT_DIR/${TOP}_routed.def
write_db    $OUT_DIR/${TOP}_final.odb

# GDSII via KLayout merge (requires klayout on PATH)
if {[catch {exec klayout -zz \
    -rd input=$OUT_DIR/${TOP}_routed.def \
    -rd output=$OUT_DIR/${TOP}.gds \
    -r $PDK_ROOT/$PLATFORM/libs.tech/klayout/def2stream.py} err]} {
    puts "WARNING: KLayout GDSII export failed: $err"
    puts "Run manually: klayout -rd input=... -rd output=... -r def2stream.py"
} else {
    puts "\n*** GDSII written: $OUT_DIR/${TOP}.gds ***\n"
}

puts "\n*** PnR complete ***"
puts "  Timing : $OUT_DIR/timing_report.txt"
puts "  Power  : $OUT_DIR/power_report.txt"
puts "  DEF    : $OUT_DIR/${TOP}_routed.def"
puts "  DB     : $OUT_DIR/${TOP}_final.odb"
