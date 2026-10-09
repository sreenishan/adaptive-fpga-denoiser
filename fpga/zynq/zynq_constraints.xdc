# fpga/zynq/zynq_constraints.xdc
#
# Minimal XDC for the Zynq denoiser block design.
# Target: xc7z020clg400-1 (PYNQ-Z1, PYNQ-Z2, Arty-Z7-20)
#
# The pixel clock comes from PS FCLK_CLK0 (100 MHz, configured in create_project.tcl).
# No external user clock is needed for the denoiser pipeline.
#
# Board-specific I/O (HDMI, buttons, LEDs) is NOT included here.
# Add board constraints from the board's master XDC when needed.
#
# No board has been programmed — these constraints are for synthesis/P&R only.

# PS FCLK_CLK0 is internal to the PS block; Vivado derives the 100 MHz constraint
# automatically from the PS7 IP configuration.  If it is not derived, add:
#   create_clock -period 10.000 -name fclk0 [get_nets {zynq_ps/inst/FCLKCLK[0]}]

# False paths on asynchronous reset inputs (handled by proc_sys_reset IP)
set_false_path -from [get_ports FCLK_RESET0_N]

# Maximum delay for AXI-Lite register outputs (relaxed — not timing-critical)
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {NAME =~ *reg_filter_sel*}] 20.0
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {NAME =~ *reg_noise_var*}]  20.0
