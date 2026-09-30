// median_filter.sv — Phase 16
//
// Pipelined 3×3 median filter: the 19-comparator median-of-9 network
// (Paeth, "Median Finding on a 3x3 Grid", Graphics Gems I), split into
// two combinational stages separated by a pipeline register.
//
// Stage 1 (comb): sort each of the three columns — steps 1-9, 3 comparators deep.
// Pipeline register: captures the nine sorted-column values.
// Stage 2 (comb): find the median of the sorted columns — steps 10-19, 5 deep.
//
// Latency: 1 clock cycle.  filter_controller.sv's Stage G absorbed the old
// median_r register; moving it here keeps the pipeline depth unchanged
// (PIPE_STAGES = 13 throughout the design).
//
// WHY THIS SPLIT
// ---------------
// The original single-stage always_comb was 8 comparators deep through v[4]
// (~27 ns on ECP5 after win-input register, measured critical path).  The
// column-sort / median-selection boundary is a natural cut: after step 9 the
// three columns are independently sorted, and the 10 remaining comparators
// operate on those sorted triples.  Stage 1: 3 deep; stage 2: 5 deep.
//
// WHY NO TEMPORARY / NO LATCH
// ----------------------------
// A module-level temporary assigned only inside `if` branches causes yosys to
// infer a latch and refuse synthesis.  Concatenated conditional swaps assign
// BOTH sides unconditionally on every evaluation, so no latch is inferred.
// 19 is the minimum correct comparator count; do not remove one.
//
// Each step is a compare-exchange: after CS(i,j) the smaller value is in v[i]
// and the larger in v[j].  Steps 17 and 19 are written (4,2) and step 18 is
// (6,4) — the descending index order is deliberate, not a typo.

`default_nettype none

module median_filter #(
    parameter int DEPTH = 8
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               en,
    input  logic [3*3*DEPTH-1:0] win_flat,
    output logic [DEPTH-1:0]     median_out
);
    logic [DEPTH-1:0] p [0:8];
    assign p[0] = win_flat[(0*3+0)*DEPTH +: DEPTH];
    assign p[1] = win_flat[(0*3+1)*DEPTH +: DEPTH];
    assign p[2] = win_flat[(0*3+2)*DEPTH +: DEPTH];
    assign p[3] = win_flat[(1*3+0)*DEPTH +: DEPTH];
    assign p[4] = win_flat[(1*3+1)*DEPTH +: DEPTH];
    assign p[5] = win_flat[(1*3+2)*DEPTH +: DEPTH];
    assign p[6] = win_flat[(2*3+0)*DEPTH +: DEPTH];
    assign p[7] = win_flat[(2*3+1)*DEPTH +: DEPTH];
    assign p[8] = win_flat[(2*3+2)*DEPTH +: DEPTH];

    // ── Stage 1 (comb): sort each column (steps 1-9, 3 comparators deep) ───
    // Columns are independent so they are processed in column order rather than
    // the interleaved order in the original — the hardware is identical.
    logic [DEPTH-1:0] s [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) s[i] = p[i];
        // Column 0: p[0],p[1],p[2] → s[0]≤s[1]≤s[2]
        {s[1], s[2]} = (s[1] > s[2]) ? {s[2], s[1]} : {s[1], s[2]};
        {s[0], s[1]} = (s[0] > s[1]) ? {s[1], s[0]} : {s[0], s[1]};
        {s[1], s[2]} = (s[1] > s[2]) ? {s[2], s[1]} : {s[1], s[2]};
        // Column 1: p[3],p[4],p[5] → s[3]≤s[4]≤s[5]
        {s[4], s[5]} = (s[4] > s[5]) ? {s[5], s[4]} : {s[4], s[5]};
        {s[3], s[4]} = (s[3] > s[4]) ? {s[4], s[3]} : {s[3], s[4]};
        {s[4], s[5]} = (s[4] > s[5]) ? {s[5], s[4]} : {s[4], s[5]};
        // Column 2: p[6],p[7],p[8] → s[6]≤s[7]≤s[8]
        {s[7], s[8]} = (s[7] > s[8]) ? {s[8], s[7]} : {s[7], s[8]};
        {s[6], s[7]} = (s[6] > s[7]) ? {s[7], s[6]} : {s[6], s[7]};
        {s[7], s[8]} = (s[7] > s[8]) ? {s[8], s[7]} : {s[7], s[8]};
    end

    // ── Pipeline register: capture sorted-column values ───────────────────
    logic [DEPTH-1:0] q [0:8];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) q[i] <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) q[i] <= s[i];
        end
    end

    // ── Stage 2 (comb): median of sorted columns (steps 10-19, 5 deep) ────
    logic [DEPTH-1:0] v [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) v[i] = q[i];
        // 10..12: discard the impossible extremes
        {v[0], v[3]} = (v[0] > v[3]) ? {v[3], v[0]} : {v[0], v[3]};
        {v[5], v[8]} = (v[5] > v[8]) ? {v[8], v[5]} : {v[5], v[8]};
        {v[4], v[7]} = (v[4] > v[7]) ? {v[7], v[4]} : {v[4], v[7]};
        // 13..15
        {v[3], v[6]} = (v[3] > v[6]) ? {v[6], v[3]} : {v[3], v[6]};
        {v[1], v[4]} = (v[1] > v[4]) ? {v[4], v[1]} : {v[1], v[4]};
        {v[2], v[5]} = (v[2] > v[5]) ? {v[5], v[2]} : {v[2], v[5]};
        // 16..19: converge on the 5th order statistic
        {v[4], v[7]} = (v[4] > v[7]) ? {v[7], v[4]} : {v[4], v[7]};
        {v[4], v[2]} = (v[4] > v[2]) ? {v[2], v[4]} : {v[4], v[2]};
        {v[6], v[4]} = (v[6] > v[4]) ? {v[4], v[6]} : {v[6], v[4]};
        {v[4], v[2]} = (v[4] > v[2]) ? {v[2], v[4]} : {v[4], v[2]};
    end

    assign median_out = v[4];

endmodule

`default_nettype wire
