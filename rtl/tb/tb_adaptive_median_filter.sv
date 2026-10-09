// tb_adaptive_median_filter.sv
//
// Self-checking testbench for adaptive_median_filter. No simulator golden file
// required. Reference: the Hwang-Haddad AMF (max_size=3) logic computed in
// SystemVerilog.
//
// adaptive_median_filter is pipelined (6-cycle latency): stimulus is driven
// at negedge, the result is sampled six posedges later + #1.
//
// Tests:
//   1. All-same window (no impulse) → output = center = common value
//   2. Single impulse in center → replaced by median
//   3. Single impulse not in center → center kept
//   4. All impulses (min==max==med) → output = median
//   5. 1000 pseudorandom windows — cross-checked against in-simulation reference

`timescale 1ns/1ps
`default_nettype none

module tb_adaptive_median_filter;

    parameter int DEPTH = 8;

    // ── Clock ──────────────────────────────────────────────────────────────
    logic clk   = 1'b0;
    logic rst_n = 1'b0;
    logic en    = 1'b1;
    always #5 clk = ~clk;

    // ── DUT ────────────────────────────────────────────────────────────────
    logic [DEPTH-1:0] win [0:2][0:2];
    logic [DEPTH-1:0] adaptive_out;
    logic [3*3*DEPTH-1:0] win_flat;

    genvar gr, gc;
    generate
        for (gr = 0; gr < 3; gr++) begin : g_row
            for (gc = 0; gc < 3; gc++) begin : g_col
                assign win_flat[(gr*3+gc)*DEPTH +: DEPTH] = win[gr][gc];
            end
        end
    endgenerate

    adaptive_median_filter #(.DEPTH(DEPTH)) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .en          (en),
        .win_flat    (win_flat),
        .adaptive_out(adaptive_out)
    );

    // ── Reference model ─────────────────────────────────────────────────────
    // Computes the Hwang-Haddad AMF with max_size=3:
    //   z_min = min of 9 pixels
    //   z_max = max of 9 pixels
    //   z_med = median of 9 pixels (bubble sort)
    //   usable      = (z_min < z_med) && (z_med < z_max)
    //   not_impulse = (z_min < center) && (center < z_max)
    //   output      = (usable && not_impulse) ? center : z_med
    function automatic [DEPTH-1:0] ref_adaptive(
        input [DEPTH-1:0] a0,a1,a2, a3,a4,a5, a6,a7,a8
    );
        logic [DEPTH-1:0] a [0:8];
        logic [DEPTH-1:0] tmp;
        logic [DEPTH-1:0] z_min, z_max, z_med, center;
        logic usable, not_impulse;
        a[0]=a0; a[1]=a1; a[2]=a2;
        a[3]=a3; a[4]=a4; a[5]=a5;
        a[6]=a6; a[7]=a7; a[8]=a8;
        center = a4;
        // Bubble sort for min/max/median
        for (int pass = 0; pass < 9; pass++)
            for (int j = 0; j < 8; j++)
                if (a[j] > a[j+1]) begin tmp=a[j]; a[j]=a[j+1]; a[j+1]=tmp; end
        z_min = a[0];
        z_med = a[4];
        z_max = a[8];
        usable      = (z_min < z_med) && (z_med < z_max);
        not_impulse = (z_min < center) && (center < z_max);
        return (usable && not_impulse) ? center : z_med;
    endfunction

    // ── Drive and check ────────────────────────────────────────────────────
    int errors;
    logic [DEPTH-1:0] exp;

    task automatic check9(
        input [DEPTH-1:0] p0,p1,p2,p3,p4,p5,p6,p7,p8
    );
        @(negedge clk);
        win[0][0]=p0; win[0][1]=p1; win[0][2]=p2;
        win[1][0]=p3; win[1][1]=p4; win[1][2]=p5;
        win[2][0]=p6; win[2][1]=p7; win[2][2]=p8;
        @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk); @(posedge clk); #1;
        exp = ref_adaptive(p0,p1,p2,p3,p4,p5,p6,p7,p8);
        if (adaptive_out !== exp) begin
            $display("FAIL: adaptive(%0d,%0d,%0d, %0d,%0d,%0d, %0d,%0d,%0d) = %0d, want %0d",
                p0,p1,p2,p3,p4,p5,p6,p7,p8, adaptive_out, exp);
            errors++;
        end
    endtask

    integer seed;

    initial begin
        errors = 0;
        seed   = 99;

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ── 1. Uniform windows (no noise) ────────────────────────────────
        // All values equal: z_min == z_med == z_max, usable=false → output=z_med=center
        check9(100,100,100, 100,100,100, 100,100,100);  // expect 100
        check9(  0,  0,  0,   0,  0,  0,   0,  0,  0);  // expect 0
        check9(255,255,255, 255,255,255, 255,255,255);  // expect 255

        // ── 2. Center impulse (255 = salt) ────────────────────────────────
        // Neighbors are clean (50); center is an impulse → replaced by median=50
        check9( 50, 50, 50,  50,255, 50,  50, 50, 50);  // center=255, expect 50
        check9( 50, 50, 50,  50,  0, 50,  50, 50, 50);  // center=0, expect 50

        // ── 3. Impulse NOT in center ─────────────────────────────────────
        // Corner has 255, center is clean (50) → center kept
        check9(255, 50, 50,  50, 50, 50,  50, 50, 50);  // center=50, expect 50
        check9(  0, 50, 50,  50, 50, 50,  50, 50, 50);  // center=50, expect 50

        // ── 4. Heavy impulse noise (many extremes) ────────────────────────
        // Salt-pepper majority: median may itself be an extreme
        check9(  0,255,  0, 255,  0,255,   0,255,  0);  // 5 zeros, 4 255s → median=0, all extreme
        check9(255,  0,255,   0,255,  0, 255,  0,255);  // 5 255s, 4 zeros → median=255

        // ── 5. Mixed: center clean, surroundings varied ───────────────────
        check9(10,200, 30, 190, 50, 40, 170, 60, 80);  // center=50
        check9( 5, 10, 15,  20,128, 30,  25, 40, 45);  // center=128 (not extreme)

        // ── 6. Pseudorandom sweep ─────────────────────────────────────────
        for (int i = 0; i < 1000; i++) begin
            logic [DEPTH-1:0] pv [0:8];
            for (int k = 0; k < 9; k++)
                pv[k] = $urandom(seed) % 256;
            check9(pv[0],pv[1],pv[2],pv[3],pv[4],pv[5],pv[6],pv[7],pv[8]);
        end

        if (errors == 0)
            $display("tb_adaptive_median_filter: PASS (all tests passed)");
        else
            $fatal(1, "tb_adaptive_median_filter: FAIL (%0d errors)", errors);

        $finish;
    end

endmodule

`default_nettype wire
