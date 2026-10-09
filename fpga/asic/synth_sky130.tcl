# Yosys synthesis script — sky130hd standard cells
#
# Usage (inside ORFS container or a Linux env with sky130 PDK):
#
#   yosys fpga/asic/synth_sky130.tcl
#
# Requires:
#   $PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
#   (Set PDK_ROOT in environment, or edit LIB_FILE below.)
#
# Outputs:
#   fpga/asic/results/fpga_denoiser_top_sky130.v   — mapped gate-level netlist
#   fpga/asic/results/synth_stats.txt              — cell counts and area

set TOP       "fpga_denoiser_top"
set LIB_FILE  "$env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set OUT_DIR   "fpga/asic/results"

# ── Read RTL ──────────────────────────────────────────────────────────────────
foreach f {
    rtl/line_buffer.sv
    rtl/window_gen.sv
    rtl/median_filter.sv
    rtl/adaptive_median_filter.sv
    rtl/gaussian_filter.sv
    rtl/wiener_filter.sv
    rtl/filter_controller.sv
    rtl/fpga_denoiser_top.sv
} {
    read_verilog -sv $f
}

# ── Synthesise ────────────────────────────────────────────────────────────────
hierarchy -check -top $TOP

# Standard synth_design flow:
#   proc → opt → fsm → opt → memory → opt → techmap → abc → opt
synth -top $TOP

# ── Technology mapping to sky130hd ─────────────────────────────────────────
read_liberty -lib $LIB_FILE

# Map flip-flops first, then combinational cells
dfflibmap -liberty $LIB_FILE
abc -liberty $LIB_FILE -D 10000 -script "+strash;ifraig;scorr;dc2;dretime;strash;&get -n;&dch -f;&nf{16};&put;buffer;upsize;dnsize;stime -p"

# Clean up after mapping
setundef -zero
splitnets
opt_clean -purge

# ── Reports ───────────────────────────────────────────────────────────────────
tee -o $OUT_DIR/synth_stats.txt stat -liberty $LIB_FILE

# ── Write netlist ─────────────────────────────────────────────────────────────
write_verilog -noattr -noexpr -nohex -nodec \
    $OUT_DIR/${TOP}_sky130.v

puts "\n*** Synthesis complete — netlist: $OUT_DIR/${TOP}_sky130.v ***\n"
puts "*** Stats: $OUT_DIR/synth_stats.txt ***\n"
