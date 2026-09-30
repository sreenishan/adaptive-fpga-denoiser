// median_filter.sv — Phase 16
//
// Pipelined 3×3 median filter: the 19-comparator median-of-9 network
// (Paeth, "Median Finding on a 3x3 Grid", Graphics Gems I), split into
// three combinational stages separated by two pipeline registers.
//
// Stage 1  (comb): sort each of the three columns — steps 1-9, 3 deep.
// Register q[0..8]: captures the nine sorted-column values.
// Stage 2a (comb): steps 10-16 — discard extremes + one converge step, 3 deep.
// Register r[0..8]: captures all nine values after steps 10-16.
// Stage 2b (comb): steps 17-19 — final convergence, 3 deep.
//
// Latency: 2 clock cycles.  filter_controller adds Stage G2 to delay
// gaussian_px and centre_r by one extra cycle so all paths meet at cycle 3
// from win_flat (PIPE_STAGES = 15, total pipeline latency = 16).
//
// WHY THIS THREE-WAY SPLIT
// -------------------------
// Stage 1 cut the original 8-deep network to 5-deep (~20 ns on ECP5), which
// had become the critical path after the Wiener variance stage was pipelined.
// The five-comparator Stage 2 (steps 10-19) is now split at step 16:
//   Stage 2a (steps 10-16): 3 comparators deep from q[].
//   Stage 2b (steps 17-19): 3 comparators deep from r[].
// Both halves are ~12 ns, versus ~20 ns for the original five-deep stage.
// Adding one more clock cycle to filter_controller (Stage G2) aligns the
// paths.
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

    // ── Stage 2a (comb): steps 10-16 — discard extremes + first converge ──
    // 3 comparators deep from q[]: layers {10,11,12} → {13,14,15} → {16}.
    logic [DEPTH-1:0] v [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) v[i] = q[i];
        // 10..12: discard the impossible extremes (layer 1)
        {v[0], v[3]} = (v[0] > v[3]) ? {v[3], v[0]} : {v[0], v[3]};
        {v[5], v[8]} = (v[5] > v[8]) ? {v[8], v[5]} : {v[5], v[8]};
        {v[4], v[7]} = (v[4] > v[7]) ? {v[7], v[4]} : {v[4], v[7]};
        // 13..15 (layer 2)
        {v[3], v[6]} = (v[3] > v[6]) ? {v[6], v[3]} : {v[3], v[6]};
        {v[1], v[4]} = (v[1] > v[4]) ? {v[4], v[1]} : {v[1], v[4]};
        {v[2], v[5]} = (v[2] > v[5]) ? {v[5], v[2]} : {v[2], v[5]};
        // 16 (layer 3)
        {v[4], v[7]} = (v[4] > v[7]) ? {v[7], v[4]} : {v[4], v[7]};
    end

    // ── Stage 2a register: captures all nine values after steps 10-16 ──────
    logic [DEPTH-1:0] r [0:8];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) r[i] <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) r[i] <= v[i];
        end
    end

    // ── Stage 2b (comb): steps 17-19 — final convergence ───────────────────
    // 3 comparators deep from r[]: {17} → {18} → {19}.
    // Only r[2], r[4], r[6] are read; the others are registered for symmetry.
    logic [DEPTH-1:0] w [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) w[i] = r[i];
        // 17..19: converge on the 5th order statistic
        {w[4], w[2]} = (w[4] > w[2]) ? {w[2], w[4]} : {w[4], w[2]};
        {w[6], w[4]} = (w[6] > w[4]) ? {w[4], w[6]} : {w[6], w[4]};
        {w[4], w[2]} = (w[4] > w[2]) ? {w[2], w[4]} : {w[4], w[2]};
    end

    assign median_out = w[4];

endmodule

`default_nettype wire
