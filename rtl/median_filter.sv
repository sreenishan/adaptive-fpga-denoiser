// median_filter.sv — Phase 16
//
// Pipelined 3×3 median filter: the 19-comparator median-of-9 network
// (Paeth, "Median Finding on a 3x3 Grid", Graphics Gems I), split into
// three combinational stages separated by two pipeline registers.
//
// Stage 1a (comb): column-sort layer 1 — CS(1,2)/CS(4,5)/CS(7,8), 1 deep.
// Register s1_r[0..8]: captures the layer-1 output.
// Stage 1b (comb): column-sort layers 2-3 — CS(0,1)/CS(3,4)/CS(6,7) then
//                  CS(1,2)/CS(4,5)/CS(7,8), 2 deep.
// Register q[0..8]: captures the nine fully sorted-column values.
// Stage 2a (comb): steps 10-16 — discard extremes + one converge step, 3 deep.
// Register r[0..8]: captures all nine values after steps 10-16.
// Stage 2b-a (comb): step 17 — CS(4,2), 1 deep.
// Register w_r[0..8]: captures the step-17 output.
// Stage 2b-b (comb): steps 18-19 — CS(6,4) then CS(4,2), 2 deep.
//
// Latency: 4 clock cycles.  filter_controller adds Stages G2, G3 and G4 to
// delay gaussian_px and centre_r by three extra cycles so all paths meet at
// cycle 5 from win_flat (PIPE_STAGES = 17, total pipeline latency = 18).
//
// WHY THIS FIVE-WAY SPLIT
// -------------------------
// Stage 1 cut the original 8-deep network to 5-deep (~20 ns on ECP5), which
// had become the critical path after the Wiener variance stage was pipelined.
// The five-comparator Stage 2 (steps 10-19) was split at step 16:
//   Stage 2a (steps 10-16): 3 comparators deep from q[].
//   Stage 2b (steps 17-19): 3 comparators deep from r[].
// The 3-layer column-sort was then the new critical path (~10.25 ns), split
// after layer 1 (Stage 1a/1b). The new critical path was stage-2b (~10.45 ns),
// so stage-2b is now split after step 17:
//   Stage 2b-a (step 17): 1 comparator deep from r[] (~3 ns).
//   Stage 2b-b (steps 18-19): 2 comparators deep from w_r[] (~7 ns).
// Adding one more clock cycle to filter_controller (Stage G4) aligns the paths.
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

    // ── Stage 1a (comb): column-sort layer 1 (1 comparator deep) ────────────
    // CS(1,2), CS(4,5), CS(7,8) — three independent column bottom-pair swaps.
    logic [DEPTH-1:0] s1 [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) s1[i] = p[i];
        {s1[1], s1[2]} = (s1[1] > s1[2]) ? {s1[2], s1[1]} : {s1[1], s1[2]};
        {s1[4], s1[5]} = (s1[4] > s1[5]) ? {s1[5], s1[4]} : {s1[4], s1[5]};
        {s1[7], s1[8]} = (s1[7] > s1[8]) ? {s1[8], s1[7]} : {s1[7], s1[8]};
    end

    // ── Stage 1a register ─────────────────────────────────────────────────────
    // Breaks win_r Q → 3-layer column-sort → q[] setup (~10.25 ns on ECP5-25k)
    // into win_r Q → layer 1 (~3 ns) → s1_r; s1_r Q → layers 2-3 → q[].
    logic [DEPTH-1:0] s1_r [0:8];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) s1_r[i] <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) s1_r[i] <= s1[i];
        end
    end

    // ── Stage 1b (comb): column-sort layers 2-3 (2 comparators deep) ─────────
    // CS(0,1), CS(3,4), CS(6,7) then CS(1,2), CS(4,5), CS(7,8).
    logic [DEPTH-1:0] s [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) s[i] = s1_r[i];
        // Column 0 layers 2-3
        {s[0], s[1]} = (s[0] > s[1]) ? {s[1], s[0]} : {s[0], s[1]};
        {s[1], s[2]} = (s[1] > s[2]) ? {s[2], s[1]} : {s[1], s[2]};
        // Column 1 layers 2-3
        {s[3], s[4]} = (s[3] > s[4]) ? {s[4], s[3]} : {s[3], s[4]};
        {s[4], s[5]} = (s[4] > s[5]) ? {s[5], s[4]} : {s[4], s[5]};
        // Column 2 layers 2-3
        {s[6], s[7]} = (s[6] > s[7]) ? {s[7], s[6]} : {s[6], s[7]};
        {s[7], s[8]} = (s[7] > s[8]) ? {s[8], s[7]} : {s[7], s[8]};
    end

    // ── Stage 1b register: capture sorted-column values ──────────────────────
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

    // ── Stage 2b-a (comb): step 17 — CS(4,2), 1 comparator deep ─────────────
    logic [DEPTH-1:0] wa [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) wa[i] = r[i];
        {wa[4], wa[2]} = (wa[4] > wa[2]) ? {wa[2], wa[4]} : {wa[4], wa[2]};
    end

    // ── Stage 2b-a register ───────────────────────────────────────────────────
    // Breaks r[] Q → 3-comparator stage-2b → comb_r setup (~10.45 ns on ECP5-25k)
    // into r[] Q → step 17 (~3 ns) → w_r; w_r Q → steps 18-19 (~7 ns) → comb_r.
    logic [DEPTH-1:0] w_r [0:8];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) w_r[i] <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) w_r[i] <= wa[i];
        end
    end

    // ── Stage 2b-b (comb): steps 18-19 — final convergence (2 deep) ──────────
    logic [DEPTH-1:0] w [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) w[i] = w_r[i];
        {w[6], w[4]} = (w[6] > w[4]) ? {w[4], w[6]} : {w[6], w[4]};
        {w[4], w[2]} = (w[4] > w[2]) ? {w[2], w[4]} : {w[4], w[2]};
    end

    assign median_out = w[4];

endmodule

`default_nettype wire
