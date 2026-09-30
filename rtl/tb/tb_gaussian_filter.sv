// tb_gaussian_filter.sv
//
// Self-checking testbench for gaussian_filter. Reference computed inline:
//
//   weighted_sum = 1*c00 + 2*c01 + 1*c02
//                + 2*c10 + 4*c11 + 2*c12
//                + 1*c20 + 2*c21 + 1*c22
//   out = (weighted_sum + 8) >> 4
//
// gaussian_filter is pipelined (1-cycle latency): stimulus is driven at
// negedge, the result is sampled at posedge + #1.
//
// Tests:
//   1. Uniform inputs  (out must equal input)
//   2. Impulse at each position (expected computed from the weights)
//   3. 2000 pseudorandom windows cross-checked against the inline reference

`timescale 1ns/1ps
`default_nettype none

module tb_gaussian_filter;

    parameter int DEPTH = 8;

    // ── Clock ──────────────────────────────────────────────────────────────
    logic clk   = 1'b0;
    logic rst_n = 1'b0;
    logic en    = 1'b1;
    always #5 clk = ~clk;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic [DEPTH-1:0] win [0:2][0:2];
    logic [DEPTH-1:0] gaussian_out;

    logic [3*3*DEPTH-1:0] win_flat;
    genvar gr, gc;
    generate
        for (gr = 0; gr < 3; gr++) begin : g_row
            for (gc = 0; gc < 3; gc++) begin : g_col
                assign win_flat[(gr*3+gc)*DEPTH +: DEPTH] = win[gr][gc];
            end
        end
    endgenerate

    gaussian_filter #(.DEPTH(DEPTH)) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (en),
        .win_flat   (win_flat),
        .gaussian_out(gaussian_out)
    );

    // ── Reference ──────────────────────────────────────────────────────────
    function automatic [DEPTH-1:0] ref_gaussian(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        logic [11:0] ws;
        ws = 12'(p0) + {p1,1'b0} + 12'(p2)
           + {p3,1'b0} + {p4,2'b0} + {p5,1'b0}
           + 12'(p6) + {p7,1'b0} + 12'(p8);
        return (ws + 12'd8) >> 4;
    endfunction

    // ── Drive and check ────────────────────────────────────────────────────
    int errors;
    logic [DEPTH-1:0] exp;

    // Present stimulus at negedge; sample gaussian_out one cycle later.
    task automatic check9(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        @(negedge clk);
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        @(posedge clk); #1;
        exp = ref_gaussian(p0,p1,p2,p3,p4,p5,p6,p7,p8);
        if (gaussian_out !== exp) begin
            $display("FAIL: gaussian(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                p0,p1,p2,p3,p4,p5,p6,p7,p8, gaussian_out, exp);
            errors++;
        end
    endtask

    integer seed;

    initial begin
        errors = 0;
        seed   = 7;

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ── 1. Uniform windows ───────────────────────────────────────────
        check9(  0,  0,  0,  0,  0,  0,  0,  0,  0);
        check9(255,255,255,255,255,255,255,255,255);
        check9(128,128,128,128,128,128,128,128,128);
        check9( 64, 64, 64, 64, 64, 64, 64, 64, 64);

        // ── 2. Impulse at each position ──────────────────────────────────
        check9(255,0,0,0,0,0,0,0,0);   // corner → 16
        check9(0,255,0,0,0,0,0,0,0);   // edge   → 32
        check9(0,0,0,0,255,0,0,0,0);   // centre → 64

        // ── 3. Accumulator overflow guard ────────────────────────────────
        check9(255,255,255,255,255,255,255,255,255);

        // ── 4. Pseudorandom sweep ────────────────────────────────────────
        for (int i = 0; i < 2000; i++) begin
            logic [DEPTH-1:0] p [0:8];
            for (int k = 0; k < 9; k++)
                p[k] = $urandom(seed) % 256;
            check9(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]);
        end

        if (errors == 0)
            $display("tb_gaussian_filter: PASS (all tests passed)");
        else
            $fatal(1, "tb_gaussian_filter: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
