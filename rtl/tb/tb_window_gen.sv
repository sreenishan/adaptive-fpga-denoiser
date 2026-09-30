// tb_window_gen.sv
//
// Self-checking testbench for window_gen + line_buffer.
//
// Icarus Verilog 12 limitations worked around:
//   - Unpacked array output ports read as x → use flat logic [N*N*D-1:0] connected
//     to the DUT's descending-packed port; element [r][c] = win_flat[(r*3+c)*D +: D]
//   - Packed array variable indexing fails → use +: on the flat 1D signal
//   - No always-block monitor (avoids packed-index issue and timing races)
//
// Frame: WIDTH=6, HEIGHT=4. Tests:
//   1. Ramp image (pixel = raster index 1..24)
//   2. Checkerboard (pixel = (r+c)%2 * 200)
//   3. Back-to-back frames (verifies that `emitting` clears properly)

`timescale 1ns/1ps
`default_nettype none

module tb_window_gen;

    parameter int W     = 6;
    parameter int H     = 4;
    parameter int DEPTH = 8;
    parameter int TOTAL = W * H;
    parameter int WF    = 3*3*DEPTH;   // 72 bits for a 3×3 window

    // ── DUT ────────────────────────────────────────────────────────────────
    logic              clk, rst_n;
    logic [DEPTH-1:0]  pixel_in;
    logic              we, flush;
    // Flat 1D signal connected to descending-packed 2D port:
    // win_flat[(r*3+c)*DEPTH +: DEPTH] == DUT internal win_out[r][c]
    logic [WF-1:0]     win_flat;
    logic              valid_out;

    window_gen #(
        .WIDTH  (W),
        .HEIGHT (H),
        .SIZE   (3),
        .DEPTH  (DEPTH)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .pixel_in (pixel_in),
        .we       (we),
        .flush    (flush),
        .win_out  (win_flat),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    // ── Images (flat W*H arrays) ───────────────────────────────────────────
    logic [DEPTH-1:0] img [0:TOTAL-1];   // flat image: img[r*W+c]

    // Replicate-clamped pixel access
    function automatic [DEPTH-1:0] img_px(input int r_raw, c_raw);
        int r, c;
        r = (r_raw < 0) ? 0 : ((r_raw >= H) ? H-1 : r_raw);
        c = (c_raw < 0) ? 0 : ((c_raw >= W) ? W-1 : c_raw);
        return img[r*W + c];
    endfunction

    // ── Expected window at (cr, cc) for current `img` ─────────────────────
    logic [DEPTH-1:0] exp_w [0:2][0:2];

    task automatic compute_expected(input int cr, cc);
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++)
                exp_w[dr+1][dc+1] = img_px(cr+dr, cc+dc);
    endtask

    // ── Output buffer — internal unpacked array, no port issue ────────────
    logic [DEPTH-1:0] out_buf [0:TOTAL-1][0:2][0:2];
    int win_count;

    // ── Error counter ─────────────────────────────────────────────────────
    int errors;

    // Helper: capture the current win_flat into out_buf[win_count], then inc
    task automatic capture_window();
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                out_buf[win_count][row][col] = win_flat[(row*3+col)*DEPTH +: DEPTH];
        win_count++;
    endtask

    // ── Stimulus + checker task ────────────────────────────────────────────
    task automatic stream_and_check(input string lbl);
        int mismatches;
        mismatches = 0;
        win_count  = 0;

        // Send pixels
        for (int i = 0; i < TOTAL; i++) begin
            @(negedge clk);
            we       = 1'b1;
            flush    = 1'b0;
            pixel_in = img[i];
            @(posedge clk); #1;
            if (valid_out) capture_window();
        end

        // Flush for WIDTH+2 cycles
        for (int i = 0; i < W+2; i++) begin
            @(negedge clk);
            we    = 1'b0;
            flush = 1'b1;
            @(posedge clk); #1;
            if (valid_out && win_count < TOTAL) capture_window();
            else if (valid_out) win_count++;  // count overruns
        end

        // De-assert flush; drain any tail
        @(negedge clk); we = 0; flush = 0;
        repeat (W+4) begin
            @(posedge clk); #1;
            if (valid_out && win_count < TOTAL) capture_window();
            else if (valid_out) win_count++;
        end

        // Count check
        if (win_count !== TOTAL) begin
            $display("FAIL %s: emitted %0d windows, expected %0d", lbl, win_count, TOTAL);
            mismatches++;
        end

        // Content check
        for (int idx = 0; idx < TOTAL; idx++) begin
            int cr, cc;
            cr = idx / W;
            cc = idx % W;
            compute_expected(cr, cc);
            for (int row = 0; row < 3; row++) begin
                for (int col = 0; col < 3; col++) begin
                    if (out_buf[idx][row][col] !== exp_w[row][col]) begin
                        if (mismatches == 0)
                            $display("FAIL %s (%0d,%0d) win[%0d][%0d]: got %0d want %0d",
                                lbl, cr, cc, row, col,
                                out_buf[idx][row][col], exp_w[row][col]);
                        mismatches++;
                    end
                end
            end
        end

        errors += mismatches;
        if (mismatches == 0)
            $display("  %s: OK (%0d windows correct)", lbl, TOTAL);
        else
            $display("  %s: %0d mismatches", lbl, mismatches);
    endtask

    // ── Main ───────────────────────────────────────────────────────────────
    initial begin
        clk  = 0; rst_n = 0;
        we   = 0; flush = 0;
        pixel_in = '0;
        errors   = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        // Test 1: ramp
        for (int i = 0; i < TOTAL; i++) img[i] = 8'(i + 1);
        $display("tb_window_gen: ramp (%0dx%0d)", H, W);
        stream_and_check("ramp");

        repeat (4) @(posedge clk);

        // Test 2: checkerboard
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                img[r*W+c] = ((r+c) % 2 == 0) ? 8'd0 : 8'd200;
        $display("tb_window_gen: checkerboard (%0dx%0d)", H, W);
        stream_and_check("checker");

        repeat (4) @(posedge clk);

        // Test 3: back-to-back frames
        $display("tb_window_gen: back-to-back frames");
        for (int i = 0; i < TOTAL; i++) img[i] = 8'(i + 1);
        for (int i = 0; i < TOTAL; i++) begin
            @(negedge clk); we=1; flush=0; pixel_in=img[i];
        end
        for (int i = 0; i < W+2; i++) begin
            @(negedge clk); we=0; flush=1;
        end
        @(negedge clk); we=0; flush=0;
        @(negedge clk); we=0; flush=0;
        repeat(2) @(posedge clk);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                img[r*W+c] = ((r+c) % 2 == 0) ? 8'd0 : 8'd200;
        stream_and_check("back-to-back/frame2");

        if (errors == 0)
            $display("tb_window_gen: PASS");
        else
            $fatal(1, "tb_window_gen: FAIL (%0d mismatches)", errors);

        $finish;
    end

endmodule

`default_nettype wire
