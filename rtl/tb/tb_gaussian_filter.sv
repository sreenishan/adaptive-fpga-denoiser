// tb_gaussian_filter.sv
//
// Self-checking testbench for gaussian_filter. Reference computed inline:
//
//   weighted_sum = 1*c00 + 2*c01 + 1*c02
//                + 2*c10 + 4*c11 + 2*c12
//                + 1*c20 + 2*c21 + 1*c22
//   out = (weighted_sum + 8) >> 4
//
// Tests:
//   1. Uniform inputs  (out must equal input)
//   2. Impulse at each position (expected computed from the weights)
//   3. 2000 pseudorandom windows cross-checked against the inline reference

`timescale 1ns/1ps
`default_nettype none

module tb_gaussian_filter;

    parameter int DEPTH = 8;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic [DEPTH-1:0] win [0:2][0:2];
    logic [DEPTH-1:0] gaussian_out;

    gaussian_filter #(.DEPTH(DEPTH)) dut (
        .win(win),
        .gaussian_out(gaussian_out)
    );

    // ── Reference ──────────────────────────────────────────────────────────
    function automatic [DEPTH-1:0] ref_gaussian(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        // Weights: corners=1, edges=2, centre=4.  Total=16.
        // +8 rounds half-up before >>4.
        logic [11:0] ws;
        ws = 12'(p0) + {p1,1'b0} + 12'(p2)
           + {p3,1'b0} + {p4,2'b0} + {p5,1'b0}
           + 12'(p6) + {p7,1'b0} + 12'(p8);
        return (ws + 12'd8) >> 4;
    endfunction

    // ── Drive and check ────────────────────────────────────────────────────
    int errors;
    logic [DEPTH-1:0] exp;

    task automatic check9(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        #1;
        exp = ref_gaussian(p0,p1,p2,p3,p4,p5,p6,p7,p8);
        if (gaussian_out !== exp) begin
            $display("FAIL: gaussian(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                p0,p1,p2,p3,p4,p5,p6,p7,p8, gaussian_out, exp);
            errors++;
        end
    endtask

    integer seed;
    int w;

    initial begin
        errors = 0;
        seed   = 7;

        // ── 1. Uniform windows ───────────────────────────────────────────
        // A uniform window of value k:  weighted_sum = k*16, (k*16+8)>>4 = k.
        check9(  0,  0,  0,  0,  0,  0,  0,  0,  0);
        check9(255,255,255,255,255,255,255,255,255);
        check9(128,128,128,128,128,128,128,128,128);
        check9( 64, 64, 64, 64, 64, 64, 64, 64, 64);

        // ── 2. Impulse at each position ──────────────────────────────────
        // Weight map:  1 2 1 / 2 4 2 / 1 2 1.  Impulse at position p with
        // value 255 → (255*w + 8) >> 4 where w is the weight of position p.
        // Weights:
        //   corners (w=1): (255+8)/16 = 16
        //   edges   (w=2): (510+8)/16 = 32
        //   centre  (w=4): (1020+8)/16 = 64
        check9(255,0,0,0,0,0,0,0,0);   // corner → 16
        check9(0,255,0,0,0,0,0,0,0);   // edge   → 32
        check9(0,0,0,0,255,0,0,0,0);   // centre → 64

        // ── 3. Accumulator overflow guard ────────────────────────────────
        // Maximum: all 255, weighted_sum=255*16=4080, +8=4088 < 4096 (12 bits).
        check9(255,255,255,255,255,255,255,255,255);

        // ── 4. Pseudorandom sweep ────────────────────────────────────────
        for (int i = 0; i < 2000; i++) begin
            logic [DEPTH-1:0] p [0:8];
            for (int k = 0; k < 9; k++)
                p[k] = $urandom(seed) % 256;
            check9(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]);
        end

        // ── Result ───────────────────────────────────────────────────────
        if (errors == 0)
            $display("tb_gaussian_filter: PASS (all tests passed)");
        else
            $fatal(1, "tb_gaussian_filter: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
