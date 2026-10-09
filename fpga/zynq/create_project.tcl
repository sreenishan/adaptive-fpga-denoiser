# fpga/zynq/create_project.tcl
#
# Creates a Vivado 2020.1+ project with the Zynq denoiser block design.
#
# Target: xc7z020clg400-1 (PYNQ-Z1, PYNQ-Z2, Arty-Z7-20)
#
# System architecture:
#   PS FCLK_CLK0 (100 MHz) → clocks everything
#   PS M_AXI_GP0 → SmartConnect → {AXI DMA ctrl, Denoiser ctrl}
#   PS S_AXI_HP0 ← SmartConnect ← {AXI DMA MM2S, AXI DMA S2MM}
#   AXI DMA MM2S → S_AXIS → Denoiser → M_AXIS → AXI DMA S2MM
#
# Address map (GP0):
#   AXI DMA     0x40400000  (64 KB)
#   Denoiser    0x43C00000  (64 KB)
#
# Usage (Vivado Tcl console, from repo root):
#   source fpga/zynq/create_project.tcl
#
# No board has been programmed — this script creates the design only.

set REPO_ROOT [file normalize [file dirname [info script]]/../..]
set PART      "xc7z020clg400-1"
set OUT_DIR   "$REPO_ROOT/fpga/vivado"

# ── Create project ────────────────────────────────────────────────────────
create_project zynq_denoiser $OUT_DIR/zynq_denoiser -part $PART -force

set_property target_language SystemVerilog [current_project]

# ── Add RTL sources ───────────────────────────────────────────────────────
add_files -norecurse [list \
    $REPO_ROOT/rtl/line_buffer.sv            \
    $REPO_ROOT/rtl/window_gen.sv             \
    $REPO_ROOT/rtl/median_filter.sv          \
    $REPO_ROOT/rtl/adaptive_median_filter.sv \
    $REPO_ROOT/rtl/gaussian_filter.sv        \
    $REPO_ROOT/rtl/wiener_filter.sv          \
    $REPO_ROOT/rtl/filter_controller.sv      \
    $REPO_ROOT/rtl/fpga_denoiser_top.sv      \
    $REPO_ROOT/fpga/zynq/denoiser_axi_wrapper.sv \
]
update_compile_order -fileset sources_1

# ── Add constraints ───────────────────────────────────────────────────────
add_files -fileset constrs_1 $REPO_ROOT/fpga/zynq/zynq_constraints.xdc

# ── Create block design ───────────────────────────────────────────────────
create_bd_design "system"
update_compile_order -fileset sources_1

# ── Zynq PS ───────────────────────────────────────────────────────────────
create_bd_cell -type ip \
    -vlnv xilinx.com:ip:processing_system7:5.5 \
    zynq_ps

# Bare-minimum PS config: 100 MHz PL clock, HP0, GP0, UART0
apply_bd_automation \
    -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} \
    [get_bd_cells zynq_ps]

set_property -dict {
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ   100
    CONFIG.PCW_USE_FABRIC_INTERRUPT        1
    CONFIG.PCW_IRQ_F2P_INTR               1
    CONFIG.PCW_USE_S_AXI_HP0              1
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH       32
    CONFIG.PCW_USE_M_AXI_GP0              1
    CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP  1
} [get_bd_cells zynq_ps]

# ── Reset ─────────────────────────────────────────────────────────────────
create_bd_cell -type ip \
    -vlnv xilinx.com:ip:proc_sys_reset:5.0 \
    proc_sys_reset_0

# ── AXI DMA ───────────────────────────────────────────────────────────────
create_bd_cell -type ip \
    -vlnv xilinx.com:ip:axi_dma:7.1 \
    axi_dma_0

# Simple DMA mode (no scatter-gather), 8-bit stream, 23-bit length register
set_property -dict {
    CONFIG.c_include_sg                0
    CONFIG.c_sg_include_stscntrl_strm  0
    CONFIG.c_m_axi_mm2s_data_width     32
    CONFIG.c_m_axis_mm2s_tdata_width   8
    CONFIG.c_s_axis_s2mm_tdata_width   8
    CONFIG.c_m_axi_s2mm_data_width     32
    CONFIG.c_sg_length_width           23
    CONFIG.c_include_mm2s_dre          1
    CONFIG.c_include_s2mm_dre          1
} [get_bd_cells axi_dma_0]

# ── Denoiser module reference ─────────────────────────────────────────────
# Vivado infers AXI4-Lite from s_axi_* ports and AXI4-Stream from s/m_axis_*.
create_bd_cell -type module \
    -reference denoiser_axi_wrapper \
    denoiser_0

# ── AXI SmartConnect: GP0 → AXI-Lite slaves ──────────────────────────────
create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:1.0 \
    axi_smc_gp0

set_property CONFIG.NUM_SI 1 [get_bd_cells axi_smc_gp0]
set_property CONFIG.NUM_MI 2 [get_bd_cells axi_smc_gp0]

# ── AXI SmartConnect: DMA masters → HP0 ──────────────────────────────────
create_bd_cell -type ip \
    -vlnv xilinx.com:ip:smartconnect:1.0 \
    axi_smc_hp0

set_property CONFIG.NUM_SI 2 [get_bd_cells axi_smc_hp0]
set_property CONFIG.NUM_MI 1 [get_bd_cells axi_smc_hp0]

# ── Clocks and resets ─────────────────────────────────────────────────────
connect_bd_net \
    [get_bd_pins zynq_ps/FCLK_CLK0] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins denoiser_0/aclk]            \
    [get_bd_pins axi_smc_gp0/aclk]          \
    [get_bd_pins axi_smc_hp0/aclk]          \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins zynq_ps/M_AXI_GP0_ACLK]   \
    [get_bd_pins zynq_ps/S_AXI_HP0_ACLK]

connect_bd_net \
    [get_bd_pins zynq_ps/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

connect_bd_net \
    [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn]    \
    [get_bd_pins denoiser_0/aresetn]      \
    [get_bd_pins axi_smc_gp0/aresetn]    \
    [get_bd_pins axi_smc_hp0/aresetn]

# ── AXI-Lite connections (GP0 master → DMA ctrl and Denoiser ctrl) ────────
connect_bd_intf_net \
    [get_bd_intf_pins zynq_ps/M_AXI_GP0] \
    [get_bd_intf_pins axi_smc_gp0/S00_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_smc_gp0/M00_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net \
    [get_bd_intf_pins axi_smc_gp0/M01_AXI] \
    [get_bd_intf_pins denoiser_0/S_AXI]

# ── HP0 connections (DMA read/write masters → PS DDR) ─────────────────────
connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_smc_hp0/S00_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_smc_hp0/S01_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins axi_smc_hp0/M00_AXI] \
    [get_bd_intf_pins zynq_ps/S_AXI_HP0]

# ── AXI4-Stream: DMA → Denoiser → DMA ────────────────────────────────────
connect_bd_intf_net \
    [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins denoiser_0/S_AXIS]

connect_bd_intf_net \
    [get_bd_intf_pins denoiser_0/M_AXIS] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# ── Interrupt: DMA → PS IRQ ───────────────────────────────────────────────
# mm2s_introut and s2mm_introut concatenated to IRQ_F2P[1:0]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property CONFIG.NUM_PORTS 2 [get_bd_cells xlconcat_0]

connect_bd_net \
    [get_bd_pins axi_dma_0/mm2s_introut] \
    [get_bd_pins xlconcat_0/In0]

connect_bd_net \
    [get_bd_pins axi_dma_0/s2mm_introut] \
    [get_bd_pins xlconcat_0/In1]

connect_bd_net \
    [get_bd_pins xlconcat_0/dout] \
    [get_bd_pins zynq_ps/IRQ_F2P]

# ── Address assignment ─────────────────────────────────────────────────────
assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] \
    -range 64K -offset 0x40400000 -force

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs denoiser_0/S_AXI/Reg] \
    -range 64K -offset 0x43C00000 -force

# HP0 sees all of DDR (1 GB on PYNQ-Z2)
assign_bd_address -target_address_space /axi_dma_0/Data_MM2S \
    [get_bd_addr_segs zynq_ps/S_AXI_HP0/HP0_DDR_LOWOCM] \
    -range 1G -offset 0x00000000 -force

assign_bd_address -target_address_space /axi_dma_0/Data_S2MM \
    [get_bd_addr_segs zynq_ps/S_AXI_HP0/HP0_DDR_LOWOCM] \
    -range 1G -offset 0x00000000 -force

# ── Validate and generate HDL wrapper ────────────────────────────────────
validate_bd_design
save_bd_design

make_wrapper -files [get_files system.bd] -top
add_files -norecurse $OUT_DIR/zynq_denoiser/zynq_denoiser.srcs/sources_1/bd/system/hdl/system_wrapper.v
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts ""
puts "Block design 'system' created."
puts "Address map:"
puts "  AXI DMA ctrl  : 0x40400000"
puts "  Denoiser ctrl : 0x43C00000"
puts ""
puts "To synthesise:  launch_runs synth_1"
puts "To implement:   launch_runs impl_1 -to_step write_bitstream"
puts ""
puts "No board has been programmed — synthesis results are TBD."
