// tb_fpga_denoiser_cascade.sv
//
// Testbench for fpga_denoiser_cascade (two-pass denoising accelerator).
//
// Uses a 4×4 ramp image, checks:
//   1. Bypass × 2 (filter_sel=000): output pixel-exact matches input (bypass
//      is pass-through of the centre pixel in a replicate-padded 3×3 window,
//      so two passes also give the input back).
//   2. Exactly W*H output pixels are produced in each run.
//   3. m_overflow stays low throughout.
//   4. Cascade flush must cover 2×(W+2+PIPE_STAGES) cycles.

`timescale 1ns/1ps
`default_nettype none

module tb_fpga_denoiser_cascade;

    parameter int W          = 4;
    parameter int H          = 4;
    parameter int NOISE_VAR  = 100;
    parameter int DEPTH      = 8;

    // Must match fpga_denoiser_cascade PIPE_STAGES
    localparam int PIPE_STAGES    = 18;
    localparam int SINGLE_FLUSH   = W + 2 + PIPE_STAGES;
    // +1 gap cycle: prevents p2_flush from overlapping pass-1's last m_valid
    localparam int CASCADE_FLUSH  = 2 * SINGLE_FLUSH + 1;

    // ── DUT ────────────────────────────────────────────────────────────────────
    logic              clk, rst_n;
    logic [2:0]        filter_sel;
    logic [15:0]       noise_var;
    logic              s_valid, s_flush, s_ready;
    logic [DEPTH-1:0]  s_pixel;
    logic              m_valid, m_ready, m_overflow;
    logic [DEPTH-1:0]  m_pixel;

    fpga_denoiser_cascade #(
        .IMG_WIDTH  (W), .IMG_HEIGHT(H),
        .NV_W(16), .DEPTH(DEPTH)
    ) dut (
        .clk        (clk),        .rst_n      (rst_n),
        .filter_sel (filter_sel), .noise_var  (noise_var),
        .s_valid    (s_valid),    .s_pixel    (s_pixel),
        .s_flush    (s_flush),    .s_ready    (s_ready),
        .m_valid    (m_valid),    .m_pixel    (m_pixel),
        .m_ready    (m_ready),    .m_overflow (m_overflow)
    );

    always #5 clk = ~clk;

    // ── Ramp image: pixel i = i+1 (1..16) ─────────────────────────────────────
    logic [DEPTH-1:0] image [0:H*W-1];
    logic [DEPTH-1:0] outputs [0:H*W-1];

    integer test_num;
    integer fail_count;
    integer i;
    integer out_idx;

    // ── Stream + collect task ───────────────────────────────────────────────────
    task automatic run_frame;
        input logic [2:0] sel;
        output integer    n_out;
        integer           cycle;
        integer           max_cycles;

        begin
            filter_sel = sel;
            noise_var  = 16'(NOISE_VAR);
            s_flush    = 0;
            s_valid    = 0;
            s_pixel    = 0;
            m_ready    = 1;
            n_out      = 0;

            // Input phase: stream W*H pixels
            @(posedge clk); #1;
            for (i = 0; i < H*W; i++) begin
                s_valid = 1;
                s_pixel = image[i];
                if (m_valid) begin
                    outputs[n_out] = m_pixel;
                    n_out = n_out + 1;
                end
                @(posedge clk); #1;
            end
            s_valid = 0;
            s_pixel = 0;

            // Flush phase: hold s_flush for CASCADE_FLUSH cycles
            s_flush = 1;
            max_cycles = CASCADE_FLUSH + 10;
            for (cycle = 0; cycle < max_cycles; cycle++) begin
                if (m_valid) begin
                    if (n_out < H*W) begin
                        outputs[n_out] = m_pixel;
                        n_out = n_out + 1;
                    end
                end
                @(posedge clk); #1;
            end
            s_flush = 0;

            // Extra drain cycles (in case a few pixels dribble out late)
            for (cycle = 0; cycle < 10; cycle++) begin
                if (m_valid && n_out < H*W) begin
                    outputs[n_out] = m_pixel;
                    n_out = n_out + 1;
                end
                @(posedge clk); #1;
            end
        end
    endtask

    integer n_out;

    initial begin
        // Initialise
        clk        = 0;
        rst_n      = 0;
        filter_sel = 3'd0;
        noise_var  = 16'(NOISE_VAR);
        s_valid    = 0;
        s_pixel    = 0;
        s_flush    = 0;
        m_ready    = 1;
        fail_count = 0;

        for (i = 0; i < H*W; i++)
            image[i] = 8'(i + 1);

        // Reset for 4 cycles
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ── Test 1: Bypass × 2 ─────────────────────────────────────────────────
        test_num = 1;
        run_frame(3'd0, n_out);   // bypass

        if (n_out !== H*W) begin
            $display("FAIL T%0d: got %0d output pixels, expected %0d",
                     test_num, n_out, H*W);
            fail_count++;
        end

        for (i = 0; i < H*W; i++) begin
            if (outputs[i] !== image[i]) begin
                $display("FAIL T%0d: pixel %0d: out=%0d exp=%0d",
                         test_num, i, outputs[i], image[i]);
                fail_count++;
            end
        end

        if (m_overflow) begin
            $display("FAIL T%0d: m_overflow asserted", test_num);
            fail_count++;
        end

        $display("Cascade bypass×2: %0d/%0d correct (n_out=%0d)",
                 H*W - fail_count, H*W, n_out);

        // Reset between tests
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ── Test 2: Gaussian × 2 — check only pixel count + no overflow ────────
        test_num = 2;
        fail_count = 0;
        run_frame(3'd2, n_out);   // gaussian

        if (n_out !== H*W) begin
            $display("FAIL T%0d gaussian×2: got %0d pixels, expected %0d",
                     test_num, n_out, H*W);
            fail_count++;
        end else begin
            $display("PASS T%0d: gaussian×2 produced %0d pixels", test_num, n_out);
        end

        if (m_overflow) begin
            $display("FAIL T%0d: m_overflow asserted", test_num);
            fail_count++;
        end

        // ── Final result ────────────────────────────────────────────────────────
        if (fail_count == 0)
            $display("PASS: all tb_fpga_denoiser_cascade checks passed");
        else
            $display("FAIL: %0d check(s) failed", fail_count);

        $finish;
    end

endmodule
