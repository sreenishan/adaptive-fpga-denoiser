// tb_median_filter.sv
//
// Self-checking testbench for median_filter. No simulator golden file required.
// Reference: bubble-sort of the nine input pixels; result must match v[4].
//
// median_filter is pipelined (4-cycle latency): stimulus is driven at negedge,
// the result is sampled four posedges later + #1.
//
// Tests:
//   1. Corner cases (uniform windows, extremes)
//   2. Known permutations with a precomputed median
//   3. 2000 pseudorandom windows cross-checked against the bubble-sort reference
//
// Pass/fail printed at end. Exits non-zero (via $fatal) on any mismatch.

`timescale 1ns/1ps
`default_nettype none

module tb_median_filter;

    parameter int DEPTH = 8;

    // ── Clock ──────────────────────────────────────────────────────────────
    logic clk   = 1'b0;
    logic rst_n = 1'b0;
    logic en    = 1'b1;
    always #5 clk = ~clk;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic [DEPTH-1:0] win [0:2][0:2];
    logic [DEPTH-1:0] median_out;

    logic [3*3*DEPTH-1:0] win_flat;
    genvar gr, gc;
    generate
        for (gr = 0; gr < 3; gr++) begin : g_row
            for (gc = 0; gc < 3; gc++) begin : g_col
                assign win_flat[(gr*3+gc)*DEPTH +: DEPTH] = win[gr][gc];
            end
        end
    endgenerate

    median_filter #(.DEPTH(DEPTH)) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (en),
        .win_flat   (win_flat),
        .median_out (median_out)
    );

    // ── Bubble-sort reference ───────────────────────────────────────────────
    function automatic [DEPTH-1:0] ref_median(
        input [DEPTH-1:0] a0, a1, a2, a3, a4, a5, a6, a7, a8
    );
        logic [DEPTH-1:0] a [0:8];
        logic [DEPTH-1:0] tmp;
        a[0]=a0; a[1]=a1; a[2]=a2;
        a[3]=a3; a[4]=a4; a[5]=a5;
        a[6]=a6; a[7]=a7; a[8]=a8;
        for (int pass = 0; pass < 9; pass++)
            for (int j = 0; j < 8; j++)
                if (a[j] > a[j+1]) begin tmp=a[j]; a[j]=a[j+1]; a[j+1]=tmp; end
        return a[4];
    endfunction

    // ── Drive and check ────────────────────────────────────────────────────
    int errors;
    logic [DEPTH-1:0] exp;

    // Present stimulus at negedge; sample median_out three cycles later.
    task automatic check9(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        @(negedge clk);
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk); #1;
        exp = ref_median(p0,p1,p2,p3,p4,p5,p6,p7,p8);
        if (median_out !== exp) begin
            $display("FAIL: median(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                p0,p1,p2,p3,p4,p5,p6,p7,p8, median_out, exp);
            errors++;
        end
    endtask

    integer seed;

    initial begin
        errors = 0;
        seed   = 42;

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ── 1. Uniform windows ───────────────────────────────────────────
        check9( 0, 0, 0, 0, 0, 0, 0, 0, 0);
        check9(255,255,255,255,255,255,255,255,255);
        check9(128,128,128,128,128,128,128,128,128);

        // ── 2. Known permutations ────────────────────────────────────────
        check9(5,3,8,1,9,2,7,4,6);          // sorted: 1..9, median=5
        check9(1,2,3,4,5,6,7,8,9);
        check9(9,8,7,6,5,4,3,2,1);
        check9(0,255,0,255,128,0,255,0,255); // salt-and-pepper → 128
        check9(50,50,50,50,50,50,50,50,255); // median = 50
        check9( 0,50,50,50,50,50,50,50,50);  // median = 50
        check9(0,0,0,0,0,255,255,255,255);   // median = 0
        check9(0,0,0,0,255,255,255,255,255); // median = 255
        check9(10,10,20,10,10,20,30,30,40);  // median = 20
        check9(100,100,100,100,200,200,200,200,150); // median = 150

        // ── 3. Pseudorandom sweep ────────────────────────────────────────
        for (int i = 0; i < 2000; i++) begin
            logic [DEPTH-1:0] p [0:8];
            for (int k = 0; k < 9; k++)
                p[k] = $urandom(seed) % 256;
            check9(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]);
        end

        if (errors == 0)
            $display("tb_median_filter: PASS (all tests passed)");
        else
            $fatal(1, "tb_median_filter: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
