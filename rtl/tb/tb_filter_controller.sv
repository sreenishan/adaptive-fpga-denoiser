// tb_filter_controller.sv
//
// Self-checking testbench for filter_controller.
//
// The controller is a mux + one pipeline register. Tests:
//   1. filter_sel=00 (bypass): output = win[1][1], delayed one cycle
//   2. filter_sel=01 (median): output matches the 19-comparator network
//   3. filter_sel=10 (gaussian): output matches the binomial kernel
//   4. filter_sel=11 (wiener): output within 1 LSB of the exact-integer ref
//   5. Stall: en=0 suppresses valid_out (no phantom pixel)
//   6. valid_in propagation: valid_out appears exactly one cycle after valid_in

`timescale 1ns/1ps
`default_nettype none

module tb_filter_controller;

    parameter int DEPTH     = 8;
    parameter int NOISE_VAR = 100;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic              clk, rst_n;
    logic [1:0]        filter_sel;
    logic [3*3*DEPTH-1:0] win_flat;   // element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    logic              valid_in, en;
    logic [DEPTH-1:0]  pixel_out;
    logic              valid_out;

    filter_controller #(
        .DEPTH     (DEPTH),
        .NOISE_VAR (NOISE_VAR)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .filter_sel (filter_sel),
        .win_flat   (win_flat),
        .valid_in   (valid_in),
        .en         (en),
        .pixel_out  (pixel_out),
        .valid_out  (valid_out)
    );

    always #5 clk = ~clk;

    // ── Inline references ──────────────────────────────────────────────────

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
        for (int i=0; i<9; i++) begin s += p[i]; s2 += p[i]*p[i]; end
        v81=9*s2-s*s; nv81=81*nv;
        num=(v81>nv81)?(v81-nv81):0; den=(v81>nv81)?v81:nv81;
        if (den==0) gain=0;
        else begin gain=(num<<8)/den; if (gain>255) gain=255; end
        acc = s*256 + gain*(9*p[4]-s) + 1152;
        if (acc<=0) q=0; else q=acc/2304;
        if (q>255) q=255;
        return q;
    endfunction

    // ── Helpers ────────────────────────────────────────────────────────────
    int errors;

    // Drive a window, advance one clock with en=1, valid_in=1, then check.
    task automatic drive_and_check(
        input [1:0]       sel,
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8,
        input [DEPTH-1:0] exp,
        input integer     tolerance,
        input string      tag
    );
        integer diff;
        @(negedge clk);
        filter_sel = sel;
        valid_in   = 1'b1;
        en         = 1'b1;
        win_flat[(0*3+0)*DEPTH +: DEPTH]=p0; win_flat[(0*3+1)*DEPTH +: DEPTH]=p1; win_flat[(0*3+2)*DEPTH +: DEPTH]=p2;
        win_flat[(1*3+0)*DEPTH +: DEPTH]=p3; win_flat[(1*3+1)*DEPTH +: DEPTH]=p4; win_flat[(1*3+2)*DEPTH +: DEPTH]=p5;
        win_flat[(2*3+0)*DEPTH +: DEPTH]=p6; win_flat[(2*3+1)*DEPTH +: DEPTH]=p7; win_flat[(2*3+2)*DEPTH +: DEPTH]=p8;
        @(posedge clk); #1;
        if (!valid_out) begin
            $display("FAIL %s: valid_out not asserted", tag);
            errors++;
        end
        diff = int'(pixel_out) - int'(exp);
        if (diff < 0) diff = -diff;
        if (diff > tolerance) begin
            $display("FAIL %s: pixel_out=%0d want=%0d tol=%0d", tag, pixel_out, exp, tolerance);
            errors++;
        end
    endtask

    // ── Main ───────────────────────────────────────────────────────────────
    logic [DEPTH-1:0] p [0:8];

    initial begin
        clk        = 0; rst_n = 0;
        filter_sel = 0; valid_in = 0; en = 0;
        win_flat = '0;
        errors = 0;

        repeat (3) @(posedge clk);
        @(negedge clk); rst_n = 1;
        repeat (2) @(posedge clk);

        // ── 1. Bypass (filter_sel=00) ─────────────────────────────────────
        // Expect win[1][1], delayed one clock.
        drive_and_check(2'b00, 5,3,8,1,42,2,7,4,6,  8'd42, 0, "bypass:42");
        drive_and_check(2'b00, 0,0,0,0,255,0,0,0,0,  8'd255, 0, "bypass:255");
        drive_and_check(2'b00, 0,0,0,0,0,0,0,0,0,   8'd0,   0, "bypass:0");

        // ── 2. Median (filter_sel=01) ─────────────────────────────────────
        p[0]=5; p[1]=3; p[2]=8; p[3]=1; p[4]=9; p[5]=2; p[6]=7; p[7]=4; p[8]=6;
        drive_and_check(2'b01, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
            ref_median(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]), 0, "median:ramp");
        p[0]=0; p[1]=255; p[2]=0; p[3]=255; p[4]=128; p[5]=0; p[6]=255; p[7]=0; p[8]=255;
        drive_and_check(2'b01, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
            ref_median(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]), 0, "median:sp");

        // ── 3. Gaussian (filter_sel=10) ───────────────────────────────────
        p[0]=10; p[1]=20; p[2]=30; p[3]=40; p[4]=50; p[5]=60; p[6]=70; p[7]=80; p[8]=90;
        drive_and_check(2'b10, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
            ref_gaussian(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]), 0, "gaussian:ramp");
        for (int i=0; i<9; i++) p[i] = 8'd255;
        drive_and_check(2'b10, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8], 8'd255, 0, "gaussian:all255");

        // ── 4. Wiener (filter_sel=11) ─────────────────────────────────────
        p[0]=100; p[1]=100; p[2]=100; p[3]=100; p[4]=100; p[5]=100; p[6]=100; p[7]=100; p[8]=100;
        drive_and_check(2'b11, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
            8'(ref_wiener(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8], NOISE_VAR)),
            1, "wiener:flat100");
        p[0]=0; p[1]=50; p[2]=100; p[3]=150; p[4]=200; p[5]=150; p[6]=100; p[7]=50; p[8]=0;
        drive_and_check(2'b11, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],
            8'(ref_wiener(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8], NOISE_VAR)),
            1, "wiener:gradient");

        // ── 5. en=0 suppresses valid_out ─────────────────────────────────
        @(negedge clk);
        filter_sel = 2'b00;
        valid_in   = 1'b1;
        en         = 1'b0;    // stall
        win_flat[(1*3+1)*DEPTH +: DEPTH] = 8'd77;
        @(posedge clk); #1;
        if (valid_out !== 1'b0) begin
            $display("FAIL stall: valid_out=%0b with en=0, expected 0", valid_out);
            errors++;
        end

        // ── 6. valid_in=0 propagates to valid_out=0 ──────────────────────
        @(negedge clk);
        filter_sel = 2'b00;
        valid_in   = 1'b0;
        en         = 1'b1;
        @(posedge clk); #1;
        if (valid_out !== 1'b0) begin
            $display("FAIL no-valid: valid_out=%0b with valid_in=0, expected 0", valid_out);
            errors++;
        end

        // ── Result ────────────────────────────────────────────────────────
        if (errors == 0)
            $display("tb_filter_controller: PASS");
        else
            $fatal(1, "tb_filter_controller: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
