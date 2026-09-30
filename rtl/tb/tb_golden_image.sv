// tb_golden_image.sv — stream one real image through fpga_denoiser_top.
//
// The golden-model co-simulation bench, driven by scripts/simulate_rtl.py.
//
// It reads a raster image from a hex file, streams it through the complete top
// module (window generator + filter controller) with the requested filter_sel,
// and writes every output pixel to a hex file. It does NOT judge the pixel
// values: scripts/simulate_rtl.py compares them against the Python golden
// filters in src/denoising/filters/ — the reference CLAUDE.md requires every
// RTL filter to be checked against. A second reference written in SystemVerilog
// here would be one more implementation free to drift.
//
// What this bench does fail on is the protocol: an output count other than
// IMG_WIDTH*IMG_HEIGHT, m_overflow, or s_ready dropping.
//
// Plusargs:  +IN=<hex>  +OUT=<hex>  +SEL=<0..3>  [+NV=<n> +STALL=1 +SEED=<n>]
//   NV is the Wiener noise power the host writes for this frame (squared grey
//   levels). It is a runtime port now, so the driver passes the same estimate
//   the software filter used instead of recompiling per value.
//   STALL=1 inserts random idle cycles (s_valid low) between pixels, which the
//   top module's header says it must hold through.
// Geometry is a parameter, set at compile time with -P; noise power is not.

`timescale 1ns/1ps
`default_nettype none

module tb_golden_image;

    parameter int W         = 224;
    parameter int H         = 224;
    parameter int NV_W      = 16;
    parameter int DEPTH     = 8;
    // Must match fpga_denoiser_top's PIPE_STAGES.
    localparam int PIPE_STAGES = 8;
    localparam int N = W * H;

    logic             clk = 1'b0;
    logic             rst_n = 1'b0;
    logic [1:0]       filter_sel = 2'b00;
    logic             s_valid = 1'b0;
    logic [DEPTH-1:0] s_pixel = '0;
    logic             s_flush = 1'b0;
    logic             m_ready = 1'b1;
    logic [NV_W-1:0]  noise_var = 16'd100;
    wire              s_ready;
    wire              m_valid;
    wire  [DEPTH-1:0] m_pixel;
    wire              m_overflow;

    fpga_denoiser_top #(
        .IMG_WIDTH  (W),
        .IMG_HEIGHT (H),
        .NV_W       (NV_W),
        .DEPTH      (DEPTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .filter_sel (filter_sel),
        .noise_var  (noise_var),
        .s_valid    (s_valid),
        .s_pixel    (s_pixel),
        .s_flush    (s_flush),
        .s_ready    (s_ready),
        .m_valid    (m_valid),
        .m_pixel    (m_pixel),
        .m_ready    (m_ready),
        .m_overflow (m_overflow)
    );

    always #5 clk = ~clk;

    logic [DEPTH-1:0] image [0:N-1];
    string in_path, out_path;
    integer sel, stall, seed, gap, fd, nv;
    integer out_count, errors;

    // Called once per clock, just after the rising edge.
    task automatic sample();
        if (m_valid) begin
            if (out_count < N) $fwrite(fd, "%02x\n", m_pixel);
            out_count++;
        end
        if (m_overflow) begin
            $display("FAIL: m_overflow asserted after %0d outputs", out_count);
            errors++;
        end
    endtask

    initial begin
        if (!$value$plusargs("IN=%s",  in_path))  $fatal(1, "tb_golden_image: missing +IN=");
        if (!$value$plusargs("OUT=%s", out_path)) $fatal(1, "tb_golden_image: missing +OUT=");
        if (!$value$plusargs("SEL=%d", sel))      $fatal(1, "tb_golden_image: missing +SEL=");
        if (!$value$plusargs("NV=%d", nv))        nv = 100;
        if (nv < 0 || nv > 65535) $fatal(1, "tb_golden_image: +NV=%0d outside NV_W", nv);
        noise_var = NV_W'(nv);
        if (!$value$plusargs("STALL=%d", stall))  stall = 0;
        if (!$value$plusargs("SEED=%d", seed))    seed = 1;
        if (sel < 0 || sel > 3) $fatal(1, "tb_golden_image: +SEL=%0d out of range", sel);

        $readmemh(in_path, image);
        fd = $fopen(out_path, "w");
        if (fd == 0) $fatal(1, "tb_golden_image: cannot open %s", out_path);

        errors = 0;
        out_count = 0;
        filter_sel = sel[1:0];    // fixed for the whole frame; see the top's header

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (int i = 0; i < N; i++) begin
            if (stall != 0) begin
                gap = (($random(seed) & 3) == 0) ? (($random(seed) & 1) + 1) : 0;
                repeat (gap) begin
                    @(negedge clk); s_valid = 1'b0; s_flush = 1'b0;
                    @(posedge clk); #1; sample();
                end
            end
            @(negedge clk); s_valid = 1'b1; s_flush = 1'b0; s_pixel = image[i];
            @(posedge clk); #1; sample();
            if (!s_ready) begin
                $display("FAIL: s_ready low at input pixel %0d", i);
                errors++;
            end
        end

        // Flush FLUSH_CYCLES, per the top module's protocol, then drain. The
        // Wiener divider is pipelined, so this is W+2+PIPE_STAGES now; flushing
        // for W+2 would leave PIPE_STAGES pixels stuck in the pipe.
        for (int i = 0; i < W + 2 + PIPE_STAGES; i++) begin
            @(negedge clk); s_valid = 1'b0; s_flush = 1'b1;
            @(posedge clk); #1; sample();
        end
        @(negedge clk); s_valid = 1'b0; s_flush = 1'b0;
        repeat (W + 4) begin
            @(posedge clk); #1; sample();
        end
        $fclose(fd);

        if (out_count != N) begin
            $display("FAIL: %0d output pixels, expected %0d", out_count, N);
            errors++;
        end
        if (errors != 0) $fatal(1, "tb_golden_image: FAIL (%0d protocol errors)", errors);
        $display("tb_golden_image: %0d pixels, sel=%0d, nv=%0d, stall=%0d",
                 out_count, sel, nv, stall);
        $finish;
    end

endmodule

`default_nettype wire
