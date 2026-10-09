// tb_golden_image_cascade.sv — stream one image through fpga_denoiser_cascade.
//
// Two-pass co-simulation bench; driven by scripts/simulate_rtl.py --cascade.
// Identical interface to tb_golden_image.sv but routes through the cascade
// module, which applies the same filter twice in series.  The Python script
// compares the output against two sequential applications of the golden filter.
//
// Plusargs:  +IN=<hex>  +OUT=<hex>  +SEL=<0..4>  [+NV=<n>]
//   NV  — Wiener noise power, passed identically to both passes.
//   No STALL: stall behaviour is covered by tb_golden_image.sv single-pass.
// Geometry: compile-time parameter W / H (set with iverilog -P).

`timescale 1ns/1ps
`default_nettype none

module tb_golden_image_cascade;

    parameter int W         = 224;
    parameter int H         = 224;
    parameter int NV_W      = 16;
    parameter int DEPTH     = 8;
    // Must match fpga_denoiser_cascade's PIPE_STAGES.
    localparam int PIPE_STAGES   = 18;
    localparam int SINGLE_FLUSH  = W + 2 + PIPE_STAGES;
    // +3: gap(1) + 2 extra cycles pass 2 needs — see fpga_denoiser_cascade.sv.
    localparam int CASCADE_FLUSH = 2 * SINGLE_FLUSH + 3;
    localparam int N = W * H;

    logic             clk = 1'b0;
    logic             rst_n = 1'b0;
    logic [2:0]       filter_sel = 3'b000;
    logic             s_valid = 1'b0;
    logic [DEPTH-1:0] s_pixel = '0;
    logic             s_flush = 1'b0;
    logic             m_ready = 1'b1;
    logic [NV_W-1:0]  noise_var = 16'd100;
    wire              s_ready;
    wire              m_valid;
    wire  [DEPTH-1:0] m_pixel;
    wire              m_overflow;

    fpga_denoiser_cascade #(
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
    integer sel, fd, nv;
    integer out_count, errors;

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
        if (!$value$plusargs("IN=%s",  in_path))  $fatal(1, "tb_golden_image_cascade: missing +IN=");
        if (!$value$plusargs("OUT=%s", out_path)) $fatal(1, "tb_golden_image_cascade: missing +OUT=");
        if (!$value$plusargs("SEL=%d", sel))      $fatal(1, "tb_golden_image_cascade: missing +SEL=");
        if (!$value$plusargs("NV=%d", nv))        nv = 100;
        if (nv < 0 || nv > 65535) $fatal(1, "tb_golden_image_cascade: +NV=%0d outside NV_W", nv);
        noise_var = NV_W'(nv);
        if (sel < 0 || sel > 4) $fatal(1, "tb_golden_image_cascade: +SEL=%0d out of range", sel);

        $readmemh(in_path, image);
        fd = $fopen(out_path, "w");
        if (fd == 0) $fatal(1, "tb_golden_image_cascade: cannot open %s", out_path);

        errors    = 0;
        out_count = 0;
        filter_sel = sel[2:0];

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // Input phase: stream all W*H pixels.
        for (int i = 0; i < N; i++) begin
            @(negedge clk); s_valid = 1'b1; s_flush = 1'b0; s_pixel = image[i];
            @(posedge clk); #1; sample();
            if (!s_ready) begin
                $display("FAIL: s_ready low at input pixel %0d", i);
                errors++;
            end
        end
        // Flush phase: start immediately after last input (no extra idle cycle).
        for (int i = 0; i < CASCADE_FLUSH; i++) begin
            @(negedge clk); s_valid = 1'b0; s_flush = 1'b1;
            @(posedge clk); #1; sample();
        end
        @(negedge clk); s_valid = 1'b0; s_flush = 1'b0;

        // Drain: a few extra cycles in case any pixel trails after the flush.
        repeat (W + 4) begin
            @(posedge clk); #1; sample();
        end

        $fclose(fd);

        if (out_count != N) begin
            $display("FAIL: %0d output pixels, expected %0d", out_count, N);
            errors++;
        end
        if (errors != 0)
            $fatal(1, "tb_golden_image_cascade: FAIL (%0d protocol errors)", errors);
        $display("tb_golden_image_cascade: %0d pixels, sel=%0d, nv=%0d",
                 out_count, sel, nv);
        $finish;
    end

endmodule

`default_nettype wire
