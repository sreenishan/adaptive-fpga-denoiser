// fpga/zynq/tb_denoiser_axi_wrapper.sv
//
// Icarus-compatible testbench for denoiser_axi_wrapper.
// Uses a tiny 8x8 frame so the sim completes quickly.
//
// Run:
//   iverilog -g2012 -o tb_wrap.vvp \
//       fpga/zynq/tb_denoiser_axi_wrapper.sv \
//       fpga/zynq/denoiser_axi_wrapper.sv \
//       rtl/fpga_denoiser_top.sv \
//       rtl/filter_controller.sv \
//       rtl/window_gen.sv \
//       rtl/line_buffer.sv \
//       rtl/median_filter.sv \
//       rtl/adaptive_median_filter.sv \
//       rtl/gaussian_filter.sv \
//       rtl/wiener_filter.sv
//   vvp tb_wrap.vvp

`default_nettype none
`timescale 1ns/1ps

module tb_denoiser_axi_wrapper;

    // Small frame to keep sim fast
    localparam int W  = 8;
    localparam int H  = 8;
    localparam int N  = W * H;      // 64 pixels
    localparam int PS = 18;         // PIPE_STAGES
    localparam int FLUSH_CYC = W + 2 + PS;  // 28

    logic        aclk    = 0;
    logic        aresetn = 0;

    // AXI4-Lite
    logic [3:0]  s_axi_awaddr  = 0;
    logic        s_axi_awvalid = 0;
    logic        s_axi_awready;
    logic [31:0] s_axi_wdata   = 0;
    logic [3:0]  s_axi_wstrb   = 4'hf;
    logic        s_axi_wvalid  = 0;
    logic        s_axi_wready;
    logic [1:0]  s_axi_bresp;
    logic        s_axi_bvalid;
    logic        s_axi_bready  = 1;
    logic [3:0]  s_axi_araddr  = 0;
    logic        s_axi_arvalid = 0;
    logic        s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic [1:0]  s_axi_rresp;
    logic        s_axi_rvalid;
    logic        s_axi_rready  = 1;

    // AXI4-Stream slave
    logic [7:0]  s_axis_tdata  = 0;
    logic        s_axis_tvalid = 0;
    logic        s_axis_tready;
    logic        s_axis_tlast  = 0;

    // AXI4-Stream master
    logic [7:0]  m_axis_tdata;
    logic        m_axis_tvalid;
    logic        m_axis_tready = 1;
    logic        m_axis_tlast;

    denoiser_axi_wrapper #(
        .IMG_WIDTH   (W),
        .IMG_HEIGHT  (H),
        .PIPE_STAGES (PS)
    ) dut (.*);

    always #5 aclk = ~aclk;   // 100 MHz

    // AXI4-Lite write helper
    task axi_write(input [3:0] addr, input [31:0] data);
        @(posedge aclk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1;
        s_axi_wdata   <= data;
        s_axi_wvalid  <= 1;
        @(posedge aclk);
        s_axi_awvalid <= 0;
        s_axi_wvalid  <= 0;
        @(posedge aclk);
    endtask

    // AXI4-Lite read helper
    task axi_read(input [3:0] addr, output [31:0] data);
        @(posedge aclk);
        s_axi_araddr  <= addr;
        s_axi_arvalid <= 1;
        @(posedge aclk);
        s_axi_arvalid <= 0;
        while (!s_axi_rvalid) @(posedge aclk);
        data = s_axi_rdata;
        @(posedge aclk);
    endtask

    // Send one frame of N pixels (values 0..N-1)
    task send_frame();
        int i;
        for (i = 0; i < N; i++) begin
            @(posedge aclk);
            s_axis_tdata  <= i[7:0];
            s_axis_tvalid <= 1;
            s_axis_tlast  <= (i == N-1);
            wait (s_axis_tready);
        end
        @(posedge aclk);
        s_axis_tvalid <= 0;
        s_axis_tlast  <= 0;
    endtask

    // ── Main test ─────────────────────────────────────────────────────────
    integer out_count;
    integer tlast_count;
    integer errors;
    logic [31:0] rdata;

    initial begin
        errors = 0;
        // Release reset after 4 cycles
        repeat (4) @(posedge aclk);
        aresetn = 1;
        repeat (2) @(posedge aclk);

        // ── Test 1: register read/write ────────────────────────────────
        // Write filter_sel = 2 (gaussian)
        axi_write(4'h0, 32'd2);
        axi_read(4'h0, rdata);
        if (rdata[2:0] !== 3'd2) begin
            $display("FAIL: filter_sel readback expected 2, got %0d", rdata[2:0]);
            errors++;
        end else $display("PASS: filter_sel readback = %0d", rdata[2:0]);

        // Write noise_var = 100
        axi_write(4'h4, 32'd100);
        axi_read(4'h4, rdata);
        if (rdata[15:0] !== 16'd100) begin
            $display("FAIL: noise_var readback expected 100, got %0d", rdata[15:0]);
            errors++;
        end else $display("PASS: noise_var readback = %0d", rdata[15:0]);

        // frame_cnt should be 0 at start
        axi_read(4'h8, rdata);
        if (rdata[31:16] !== 16'd0) begin
            $display("FAIL: frame_cnt expected 0, got %0d", rdata[31:16]);
            errors++;
        end else $display("PASS: frame_cnt initial = %0d", rdata[31:16]);

        // ── Test 2: full frame — count outputs and verify TLAST ────────
        // filter_sel = 0 (bypass) so output == input shifted by LATENCY
        axi_write(4'h0, 32'd0);

        out_count   = 0;
        tlast_count = 0;
        fork
            // Sender
            begin
                send_frame();
            end
            // Receiver: count output pixels and TLAST
            begin
                while (out_count < N) begin
                    @(posedge aclk);
                    if (m_axis_tvalid && m_axis_tready) begin
                        out_count++;
                        if (m_axis_tlast) tlast_count++;
                    end
                end
            end
        join

        if (out_count !== N) begin
            $display("FAIL: expected %0d output pixels, got %0d", N, out_count);
            errors++;
        end else $display("PASS: received %0d output pixels", out_count);

        if (tlast_count !== 1) begin
            $display("FAIL: expected 1 TLAST, got %0d", tlast_count);
            errors++;
        end else $display("PASS: TLAST asserted exactly once");

        // Wait for flush to complete + some margin
        repeat (FLUSH_CYC + 10) @(posedge aclk);

        // ── Test 3: frame_cnt increments after flush ───────────────────
        axi_read(4'h8, rdata);
        if (rdata[31:16] !== 16'd1) begin
            $display("FAIL: frame_cnt expected 1 after one frame, got %0d", rdata[31:16]);
            errors++;
        end else $display("PASS: frame_cnt = 1 after first frame");

        // ── Test 4: second frame back-to-back ─────────────────────────
        out_count   = 0;
        tlast_count = 0;
        fork
            begin send_frame(); end
            begin
                while (out_count < N) begin
                    @(posedge aclk);
                    if (m_axis_tvalid && m_axis_tready) begin
                        out_count++;
                        if (m_axis_tlast) tlast_count++;
                    end
                end
            end
        join

        repeat (FLUSH_CYC + 10) @(posedge aclk);

        axi_read(4'h8, rdata);
        if (rdata[31:16] !== 16'd2) begin
            $display("FAIL: frame_cnt expected 2 after two frames, got %0d", rdata[31:16]);
            errors++;
        end else $display("PASS: frame_cnt = 2 after second frame");

        // ── Summary ────────────────────────────────────────────────────
        repeat (4) @(posedge aclk);
        if (errors == 0)
            $display("ALL TESTS PASSED");
        else
            $display("%0d TEST(S) FAILED", errors);

        $finish;
    end

    // Timeout watchdog
    initial begin
        #2_000_000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
