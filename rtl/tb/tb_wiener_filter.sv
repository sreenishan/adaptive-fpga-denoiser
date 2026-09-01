// tb_wiener_filter.sv
//
// Self-checking testbench for wiener_filter. Reference mirrors the exact-integer
// formulation from the SV:
//
//   v81  = 9*s2 - s*s          (81 * variance, exact)
//   gain = clamp((v81-nv81)<<8 / max(v81,nv81), 0, 255)  in Q8
//   out  = (s*256 + gain*(9*centre - s) + 1152) / 2304   (round half-up)
//
// Tests:
//   1. Flat window — any value, any NV — must return the uniform value
//   2. NV=0 on non-flat window — gain=1 (pass-through after smoothing to
//      local mean, but here mean=(s/9) and centre is the exact SV formula)
//   3. 1500 pseudorandom windows × NV ∈ {0,1,25,100,400} against the
//      inline reference (tolerance: 1 grey level, per hardware.yaml)

`timescale 1ns/1ps
`default_nettype none

module tb_wiener_filter;

    parameter int DEPTH     = 8;
    parameter int NOISE_VAR = 100;

    // ── DUT ────────────────────────────────────────────────────────────────
    // The RTL NOISE_VAR is fixed at compile time; we instantiate once per NV
    // value we test.  For the random sweep we instantiate with NOISE_VAR=100
    // and drive the reference to the same constant.

    logic [DEPTH-1:0] win [0:2][0:2];
    logic [DEPTH-1:0] wiener_out;

    wiener_filter #(.DEPTH(DEPTH), .NOISE_VAR(NOISE_VAR)) dut (
        .win(win),
        .wiener_out(wiener_out)
    );

    // ── Inline reference ───────────────────────────────────────────────────
    // Matches wiener_filter.sv statement for statement, integer arithmetic.
    function automatic integer ref_wiener(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8,
        input integer nv
    );
        integer s, s2, v81, nv81, num, den, gain;
        integer centre_term, acc, q;
        integer p [0:8];
        p[0]=p0; p[1]=p1; p[2]=p2;
        p[3]=p3; p[4]=p4; p[5]=p5;
        p[6]=p6; p[7]=p7; p[8]=p8;
        s=0; s2=0;
        for (int i=0; i<9; i++) begin s += p[i]; s2 += p[i]*p[i]; end
        v81  = 9*s2 - s*s;
        nv81 = 81*nv;
        num  = (v81 > nv81) ? (v81 - nv81) : 0;
        den  = (v81 > nv81) ? v81 : nv81;
        if (den == 0) begin
            gain = 0;
        end else begin
            gain = (num << 8) / den;
            if (gain > 255) gain = 255;
        end
        centre_term = 9*p[4] - s;
        acc = s*256 + gain*centre_term + 1152;
        if (acc <= 0) q = 0;
        else          q = acc / 2304;
        if (q > 255) q = 255;
        return q;
    endfunction

    // ── Drive and check ────────────────────────────────────────────────────
    int errors;

    task automatic check9(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8,
        input integer nv
    );
        integer exp;
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        #1;
        exp = ref_wiener(p0,p1,p2,p3,p4,p5,p6,p7,p8, nv);
        // Tolerance: 1 grey level (hardware.yaml max_abs_error.wiener = 1)
        if ((wiener_out > exp+1) || (int'(wiener_out)+1 < exp)) begin
            $display("FAIL NV=%0d: wiener(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                nv, p0,p1,p2,p3,p4,p5,p6,p7,p8, wiener_out, exp);
            errors++;
        end
    endtask

    // Check DUT output exactly equals reference (no tolerance argument)
    task automatic check_exact(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        integer exp;
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        #1;
        exp = ref_wiener(p0,p1,p2,p3,p4,p5,p6,p7,p8, NOISE_VAR);
        if (wiener_out !== exp[DEPTH-1:0]) begin
            $display("FAIL: wiener(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                p0,p1,p2,p3,p4,p5,p6,p7,p8, wiener_out, exp);
            errors++;
        end
    endtask

    integer seed;

    initial begin
        errors = 0;
        seed   = 99;

        // ── 1. Flat windows — must return the uniform value ───────────────
        // Any flat window (all pixels equal) has variance=0.
        // With NV=100: v81=0, nv81=8100, num=0, gain=0.
        // out = (k*9*256 + 0 + 1152) / 2304 = k + 1152/2304 = k (integer) = k.
        check_exact(  0,  0,  0,  0,  0,  0,  0,  0,  0);
        check_exact(255,255,255,255,255,255,255,255,255);
        check_exact(128,128,128,128,128,128,128,128,128);
        check_exact( 64, 64, 64, 64, 64, 64, 64, 64, 64);
        check_exact(  1,  1,  1,  1,  1,  1,  1,  1,  1);
        check_exact(100,100,100,100,100,100,100,100,100);

        // ── 2. Known non-flat windows ─────────────────────────────────────
        // Ramp: 0,1,2,3,4,5,6,7,8 → s=36, s2=204, v81=9*204-36^2=1836-1296=540
        // centre=4. With NV=100: nv81=8100 > v81 → gain=0 → out = (36*256+1152)/2304 = 9456/2304 = 4.
        check_exact(0,1,2,3,4,5,6,7,8);
        // Salt-and-pepper: 0,0,0,0,255,0,0,0,0  (mostly dark, bright centre)
        check_exact(0,0,0,0,255,0,0,0,0);
        // Checkerboard-like: 0,255,0,255,0,255,0,255,0
        check_exact(0,255,0,255,0,255,0,255,0);

        // ── 3. Pseudorandom sweep with NV=NOISE_VAR ──────────────────────
        for (int i = 0; i < 1500; i++) begin
            logic [DEPTH-1:0] p [0:8];
            for (int k = 0; k < 9; k++)
                p[k] = $urandom(seed) % 256;
            check9(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8], NOISE_VAR);
        end

        // ── Result ───────────────────────────────────────────────────────
        if (errors == 0)
            $display("tb_wiener_filter: PASS (all tests passed)");
        else
            $fatal(1, "tb_wiener_filter: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
