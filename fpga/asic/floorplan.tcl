# fpga/asic/floorplan.tcl — Stage 17: Floorplan (sky130hd)
#
# Reads the synthesised gate-level netlist and timing constraints,
# places the die/core boundary, pins, and power grid.
#
# Outputs:
#   fpga/asic/results/fpga_denoiser_top_floorplan.def
#
# Run inside the ORFS Docker container:
#   openroad -exit fpga/asic/floorplan.tcl

set TOP       "fpga_denoiser_top"
set PDK_ROOT  $env(PDK_ROOT)
set PLATFORM  "sky130A"
set SCL       "sky130_fd_sc_hd"

set LIB_TYP   "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lib/${SCL}__tt_025C_1v80.lib"
set LEF_TECH  "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/techlef/${SCL}__nom.tlef"
set LEF_CELLS "$PDK_ROOT/$PLATFORM/libs.ref/$SCL/lef/${SCL}.lef"

set NETLIST   "fpga/asic/results/${TOP}_sky130.v"
set SDC_FILE  "fpga/asic/constraint.sdc"
set OUT_DIR   "fpga/asic/results"

# ── Libraries ─────────────────────────────────────────────────────────────────
read_lef  $LEF_TECH
read_lef  $LEF_CELLS
read_lib  $LIB_TYP

# ── Netlist + constraints ──────────────────────────────────────────────────────
read_verilog $NETLIST
link_design  $TOP
read_sdc     $SDC_FILE

# ── Die and core (1000×1000 µm, 10 µm margin → 980×980 µm core ≈ 45% util) ──
# GPL-0301: 250×250 µm die yielded 707% utilisation; cell area ≈ 428 000 µm².
# 980×980 µm core = 960 400 µm² → ≈ 45% utilisation at target density 0.60.
initialize_floorplan \
    -die_area  "0 0 1000 1000" \
    -core_area "10 10 990 990" \
    -site      unithd

# Populate routing track database from the tech LEF (required before place_pins)
make_tracks

# I/O pins — auto ring on met3 (H) / met4 (V)
place_pins -hor_layers met3 -ver_layers met4

# ── Power distribution network ────────────────────────────────────────────────
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

# ── Save checkpoint ───────────────────────────────────────────────────────────
write_def $OUT_DIR/${TOP}_floorplan.def
write_db  $OUT_DIR/${TOP}_floorplan.odb

puts "\n*** Stage 17 complete: floorplan written ***"
puts "  DEF : $OUT_DIR/${TOP}_floorplan.def"
puts "  ODB : $OUT_DIR/${TOP}_floorplan.odb"
