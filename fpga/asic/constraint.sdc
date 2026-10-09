# Timing constraints for fpga_denoiser_top — sky130hd ASIC flow
# Target: 100 MHz (10 ns period)
#
# The design has one clock domain.  All filter paths are pipelined so the
# critical path is a single register-to-register combinational hop.

# ── Clock ────────────────────────────────────────────────────────────────────
create_clock -name clk -period 10.0 [get_ports clk]

# ── I/O timing ───────────────────────────────────────────────────────────────
# Assume 20% of the clock period for input arrival and output departure.
set_input_delay  2.0 -clock clk [all_inputs]
set_output_delay 2.0 -clock clk [all_outputs]

# Remove clock from I/O delay constraints
set_input_delay  0.0 -clock clk [get_ports clk]

# ── Resets are asynchronous ───────────────────────────────────────────────────
set_false_path -from [get_ports rst_n]
