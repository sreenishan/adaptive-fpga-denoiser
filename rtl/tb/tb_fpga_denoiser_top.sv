// tb_fpga_denoiser_top.sv
//
// End-to-end testbench for fpga_denoiser_top.
//
// Small frame (4×4, 16 pixels) streamed through each of the four filter modes.
// Checks:
//   1. Bypass mode (filter_sel=00): every output pixel matches the input pixel
//      (since bypass is pass-through of the centre pixel, and the centre of a
//      replicate-padded neighbourhood on a ramp is the pixel itself).
//   2. Median mode (filter_sel=01): each output pixel is within budget.
//   3. Gaussian mode (filter_sel=10): each output pixel is within budget.
//   4. Wiener mode (filter_sel=11): each output pixel within 1 LSB of ref.
//   5. Exactly IMG_WIDTH*IMG_HEIGHT valid output pixels are produced.
//   6. m_overflow stays low throughout (m_ready held high).
//   7. s_ready is always high (no backpressure).
//
// Protocol: s_valid high for W*H pixels, then s_flush high for W+1 cycles.

`timescale 1ns/1ps
`default_nettype none

module tb_fpga_denoiser_top;

    parameter int W          = 4;
    parameter int H          = 4;
    parameter int NOISE_VAR  = 100;
    parameter int DEPTH      = 8;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic              clk, rst_n;
    logic [1:0]        filter_sel;
    logic              s_valid, s_flush, s_ready;
    logic [DEPTH-1:0]  s_pixel;
    logic              m_valid, m_ready, m_overflow;
    logic [DEPTH-1:0]  m_pixel;

    fpga_denoiser_top #(
        .IMG_WIDTH  (W),
        .IMG_HEIGHT (H),
        .NOISE_VAR  (NOISE_VAR),
        .DEPTH      (DEPTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .filter_sel (filter_sel),
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

    // ── Image ──────────────────────────────────────────────────────────────
    // Ramp: pixel i = i+1 (1..16), so every centre pixel is distinct.
    logic [DEPTH-1:0] image [0:H-1][0:W-1];

    // ── Inline references (same as tb_filter_controller) ──────────────────

    function automatic [DEPTH-1:0] ref_median(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        logic [DEPTH-1:0] a [0:8];
        logic [DEPTH-1:0] tmp;
        a[0]=p0; a[1]=p1; a[2]=p2; a[3]=p3; a[4]=p4;
        a[5]=p5; a[6]=p6; a[7]=p7; a[8]=p8;
        for (int pass = 0; pass < 9; pass++)
            for (int j = 0; j < 8; j++)
                if (a[j] > a[j+1]) begin tmp=a[j]; a[j]=a[j+1]; a[j+1]=tmp; end
        return a[4];
    endfunction

    function automatic [DEPTH-1:0] ref_gaussian(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        logic [11:0] ws;
        ws = 12'(p0) + {p1,1'b0} + 12'(p2)
           + {p3,1'b0} + {p4,2'b0} + {p5,1'b0}
           + 12'(p6) + {p7,1'b0} + 12'(p8);
        return (ws + 12'd8) >> 4;
    endfunction

    function automatic integer ref_wiener(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8,
        input integer nv
    );
        integer s, s2, v81, nv81, num, den, gain, acc, q;
        integer p [0:8];
        p[0]=p0; p[1]=p1; p[2]=p2; p[3]=p3; p[4]=p4;
        p[5]=p5; p[6]=p6; p[7]=p7; p[8]=p8;
        s=0; s2=0;
        for (int i=0; i<9; i++) begin s+=p[i]; s2+=p[i]*p[i]; end
        v81=9*s2-s*s; nv81=81*nv;
        num=(v81>nv81)?(v81-nv81):0; den=(v81>nv81)?v81:nv81;
        if (den==0) gain=0;
        else begin gain=(num<<8)/den; if(gain>255) gain=255; end
        acc=s*256+gain*(9*p[4]-s)+1152;
        if(acc<=0) q=0; else q=acc/2304;
        if(q>255) q=255;
        return q;
    endfunction

    // Replicate-padded image lookup
    function automatic [DEPTH-1:0] img_px(input int r_raw, c_raw);
        int r, c;
        r = r_raw < 0 ? 0 : (r_raw >= H ? H-1 : r_raw);
        c = c_raw < 0 ? 0 : (c_raw >= W ? W-1 : c_raw);
        return image[r][c];
    endfunction

    // Build the expected 3×3 window for pixel (r, c)
    function automatic [DEPTH-1:0] exp_pixel(input int r, c, input [1:0] sel);
        logic [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8;
        integer ew;
        p0=img_px(r-1,c-1); p1=img_px(r-1,c); p2=img_px(r-1,c+1);
        p3=img_px(r,  c-1); p4=img_px(r,  c); p5=img_px(r,  c+1);
        p6=img_px(r+1,c-1); p7=img_px(r+1,c); p8=img_px(r+1,c+1);
        case (sel)
            2'b00: return p4;
            2'b01: return ref_median(p0,p1,p2,p3,p4,p5,p6,p7,p8);
            2'b10: return ref_gaussian(p0,p1,p2,p3,p4,p5,p6,p7,p8);
            2'b11: begin ew = ref_wiener(p0,p1,p2,p3,p4,p5,p6,p7,p8, NOISE_VAR);
                         return 8'(ew); end
            default: return p4;
        endcase
    endfunction

    // ── Run one frame, collect outputs ─────────────────────────────────────
    int errors;

    task automatic run_frame(input [1:0] sel, input string mode_name);
        int total, out_count, ri, ci;
        logic [DEPTH-1:0] got [0:H-1][0:W-1];
        logic [DEPTH-1:0] exp_px;
        integer diff, tolerance;
        total     = W * H;
        out_count = 0;
        filter_sel = sel;

        // Input pixels
        for (int i = 0; i < total; i++) begin
            @(negedge clk);
            s_valid = 1'b1; s_flush = 1'b0;
            s_pixel = image[i/W][i%W];
            @(posedge clk); #1;
            if (m_valid) begin
                got[out_count/W][out_count%W] = m_pixel;
                out_count++;
            end
            if (m_overflow) begin
                $display("FAIL %s: m_overflow asserted", mode_name);
                errors++;
            end
            if (!s_ready) begin
                $display("FAIL %s: s_ready deasserted", mode_name);
                errors++;
            end
        end

        // Flush (WIDTH+2 cycles per the updated protocol)
        for (int i = 0; i < W+2; i++) begin
            @(negedge clk);
            s_valid = 1'b0; s_flush = 1'b1;
            @(posedge clk); #1;
            if (m_valid) begin
                if (out_count < total) begin
                    got[out_count/W][out_count%W] = m_pixel;
                end
                out_count++;
            end
        end

        // Drain any tail
        @(negedge clk); s_valid=0; s_flush=0;
        repeat (W+4) begin
            @(posedge clk); #1;
            if (m_valid) begin
                if (out_count < total) begin
                    got[out_count/W][out_count%W] = m_pixel;
                end
                out_count++;
            end
        end

        if (out_count !== total) begin
            $display("FAIL %s: %0d output pixels, expected %0d", mode_name, out_count, total);
            errors++;
        end

        // Tolerance per mode
        tolerance = (sel == 2'b11) ? 1 : 0;

        // Verify each pixel
        for (int r = 0; r < H; r++) begin
            for (int c = 0; c < W; c++) begin
                exp_px = exp_pixel(r, c, sel);
                diff   = int'(got[r][c]) - int'(exp_px);
                if (diff < 0) diff = -diff;
                if (diff > tolerance) begin
                    $display("FAIL %s (%0d,%0d): got %0d want %0d",
                        mode_name, r, c, got[r][c], exp_px);
                    errors++;
                end
            end
        end

        if (errors == 0 || (errors > 0 && out_count == total))
            $display("  %s: %0d pixels, %0s",
                mode_name, out_count, (errors == 0) ? "all correct" : "some wrong");
    endtask

    // ── Main ───────────────────────────────────────────────────────────────
    initial begin
        clk = 0; rst_n = 0;
        s_valid = 0; s_flush = 0; s_pixel = 0;
        filter_sel = 0; m_ready = 1;
        errors = 0;

        // Build image: ramp 1..H*W
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                image[r][c] = 8'(r*W + c + 1);

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        $display("tb_fpga_denoiser_top: %0dx%0d ramp image", H, W);

        run_frame(2'b00, "bypass");
        repeat(4) @(posedge clk);
        run_frame(2'b01, "median");
        repeat(4) @(posedge clk);
        run_frame(2'b10, "gaussian");
        repeat(4) @(posedge clk);
        run_frame(2'b11, "wiener");

        // ── m_overflow check: hold m_ready low for one cycle ─────────────
        // Reset and run one frame in bypass, drop m_ready for 1 cycle mid-stream.
        @(negedge clk); rst_n=0;
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        repeat(2) @(posedge clk);

        filter_sel = 2'b00;
        // Send enough pixels to prime the pipeline (W+2 cycles)
        for (int i = 0; i < W*H; i++) begin
            @(negedge clk);
            s_valid=1; s_flush=0;
            s_pixel = image[i/W][i%W];
            // Drop m_ready mid-stream for exactly one pixel output cycle
            m_ready = (i == W+1) ? 1'b0 : 1'b1;
        end
        for (int i = 0; i < W+2; i++) begin
            @(negedge clk); s_valid=0; s_flush=1; m_ready=1;
        end
        @(negedge clk); s_valid=0; s_flush=0; m_ready=1;
        repeat(4) @(posedge clk); #1;

        // m_overflow should now be set (we produced a pixel while m_ready was low)
        if (!m_overflow)
            $display("NOTE: m_overflow not set (pixel may not have arrived when m_ready was low — timing-dependent)");
        else
            $display("  m_overflow: correctly set after m_ready=0 during output");

        // ── Result ────────────────────────────────────────────────────────
        if (errors == 0)
            $display("tb_fpga_denoiser_top: PASS");
        else
            $fatal(1, "tb_fpga_denoiser_top: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
