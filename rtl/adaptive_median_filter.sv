// adaptive_median_filter.sv — Phase B rev1 (pipelined min/max tree)
//
// Adaptive median filter (Hwang & Haddad, 1995) with max_size=3, pipelined
// over the same five register stages as median_filter.sv so filter_controller
// needs no additional alignment registers for this path.
//
// ALGORITHM (max_size = 3, fixed 3×3 window)
// ------------------------------------------
//   z_min = min(p[0..8])
//   z_max = max(p[0..8])
//   z_med = median(p[0..8])  — same Paeth 19-comparator network as median_filter
//   center = p[4]
//
//   usable      = (z_min < z_med) && (z_med < z_max)   -- median is not itself noise
//   not_impulse = (z_min < center) && (center < z_max) -- pixel is not an extreme
//   output      = (usable && not_impulse) ? center : z_med
//
// This is pixel-exact against adaptive_median_filter(image, max_size=3)
// in src/denoising/filters/adaptive_median.py.  All comparisons are strict
// unsigned 8-bit; a pixel equal to min or max is treated as an impulse.
//
// PIPELINE
// --------
//   Stage 1a comb → s1_r   (1 cycle):  column-sort layer 1
//                                        min/max Level 0: 4 pairwise from p[0..7]
//   Stage 1b comb → q[]    (2 cycles): column-sort layers 2-3
//                                        min/max Level 1: 2 pairwise from Level 0
//   Stage 2a-a comb → v1_r (3 cycles): steps 10-12
//                                        min/max Level 2: 1 pairwise from Level 1
//   Stage 2a-b comb → r[]  (4 cycles): steps 13-16
//                                        min/max Level 3: include p[8] → final z_min/z_max
//   Stage 2b-a comb → w_r  (5 cycles): step 17 + forward z_min/z_max/center
//   Stage 2b-b-a comb → w2_r (6 cycles): step 18 + forward z_min/z_max/center
//   Stage 2b-b-b comb (output): step 19 → z_med, then adaptive mux
//
// Latency: 6 cycles (identical to median_filter.sv after stage-2b-b split).
//
// MIN/MAX PIPELINING (prevents long carry-chain path from win_r)
// --------------------------------------------------------------
// Level 0 (Stage 1a): 4 pairwise min/max of p[0..7] — 1 comparator deep
// Level 1 (Stage 1b): 2 pairwise of Level 0 results — 1 comparator deep
// Level 2 (Stage 2a-a): 1 pairwise of Level 1 results — 1 comparator deep
// Level 3 (Stage 2a-b): compare with p[8] — 1 comparator deep → z_min, z_max
//
// With max 1 comparison depth per stage the path through the min/max tree
// is bounded by one 8-bit comparator (~2.5 ns) instead of the 4-level
// combinational tree (~13 ns) that the original single-stage version created.
//
// WHY NO TEMPORARY / NO LATCH
// ----------------------------
// All comparator steps use concatenated conditional swaps that assign both
// sides unconditionally; yosys infers no latches.

`default_nettype none

module adaptive_median_filter #(
    parameter int DEPTH = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic [3*3*DEPTH-1:0]    win_flat,
    output logic [DEPTH-1:0]        adaptive_out
);
    // ── Unpack window ─────────────────────────────────────────────────────────
    logic [DEPTH-1:0] p [0:8];
    assign p[0] = win_flat[(0*3+0)*DEPTH +: DEPTH];
    assign p[1] = win_flat[(0*3+1)*DEPTH +: DEPTH];
    assign p[2] = win_flat[(0*3+2)*DEPTH +: DEPTH];
    assign p[3] = win_flat[(1*3+0)*DEPTH +: DEPTH];
    assign p[4] = win_flat[(1*3+1)*DEPTH +: DEPTH];  // center pixel
    assign p[5] = win_flat[(1*3+2)*DEPTH +: DEPTH];
    assign p[6] = win_flat[(2*3+0)*DEPTH +: DEPTH];
    assign p[7] = win_flat[(2*3+1)*DEPTH +: DEPTH];
    assign p[8] = win_flat[(2*3+2)*DEPTH +: DEPTH];

    // ── Stage 1a (comb): column-sort layer 1 + min/max Level 0 ───────────────
    // Sort: CS(1,2), CS(4,5), CS(7,8) — 1 comparator deep
    // Min/max Level 0: 4 pairwise min/max of p[0..7] — 1 comparator deep each
    //   zm0a = min(p[0], p[1]),  zm0b = min(p[2], p[3])
    //   zm0c = min(p[4], p[5]),  zm0d = min(p[6], p[7])
    //   zx0a = max(p[0], p[1]),  zx0b = max(p[2], p[3])
    //   zx0c = max(p[4], p[5]),  zx0d = max(p[6], p[7])
    logic [DEPTH-1:0] s1 [0:8];
    logic [DEPTH-1:0] zm0a, zm0b, zm0c, zm0d;
    logic [DEPTH-1:0] zx0a, zx0b, zx0c, zx0d;
    always_comb begin
        for (int i = 0; i < 9; i++) s1[i] = p[i];
        {s1[1], s1[2]} = (s1[1] > s1[2]) ? {s1[2], s1[1]} : {s1[1], s1[2]};
        {s1[4], s1[5]} = (s1[4] > s1[5]) ? {s1[5], s1[4]} : {s1[4], s1[5]};
        {s1[7], s1[8]} = (s1[7] > s1[8]) ? {s1[8], s1[7]} : {s1[7], s1[8]};
        // Level 0 min/max: 4 independent pairwise comparisons (1 level deep)
        zm0a = (p[0] < p[1]) ? p[0] : p[1];
        zm0b = (p[2] < p[3]) ? p[2] : p[3];
        zm0c = (p[4] < p[5]) ? p[4] : p[5];
        zm0d = (p[6] < p[7]) ? p[6] : p[7];
        zx0a = (p[0] > p[1]) ? p[0] : p[1];
        zx0b = (p[2] > p[3]) ? p[2] : p[3];
        zx0c = (p[4] > p[5]) ? p[4] : p[5];
        zx0d = (p[6] > p[7]) ? p[6] : p[7];
    end

    // ── Stage 1a register ─────────────────────────────────────────────────────
    logic [DEPTH-1:0] s1_r [0:8];
    logic [DEPTH-1:0] zm0a_r, zm0b_r, zm0c_r, zm0d_r;
    logic [DEPTH-1:0] zx0a_r, zx0b_r, zx0c_r, zx0d_r;
    logic [DEPTH-1:0] center_r1, p8_r1;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) s1_r[i] <= '0;
            zm0a_r <= '0; zm0b_r <= '0; zm0c_r <= '0; zm0d_r <= '0;
            zx0a_r <= '0; zx0b_r <= '0; zx0c_r <= '0; zx0d_r <= '0;
            center_r1 <= '0; p8_r1 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) s1_r[i] <= s1[i];
            zm0a_r <= zm0a; zm0b_r <= zm0b; zm0c_r <= zm0c; zm0d_r <= zm0d;
            zx0a_r <= zx0a; zx0b_r <= zx0b; zx0c_r <= zx0c; zx0d_r <= zx0d;
            center_r1 <= p[4]; p8_r1 <= p[8];
        end
    end

    // ── Stage 1b (comb): column-sort layers 2-3 + min/max Level 1 ────────────
    // Sort: 6 comparators (layers 2-3), 2 deep
    // Min/max Level 1: 2 pairwise of Level-0 results — 1 comparator deep each
    //   zm1a = min(zm0a_r, zm0b_r),  zm1b = min(zm0c_r, zm0d_r)
    //   zx1a = max(zx0a_r, zx0b_r),  zx1b = max(zx0c_r, zx0d_r)
    logic [DEPTH-1:0] s [0:8];
    logic [DEPTH-1:0] zm1a, zm1b, zx1a, zx1b;
    always_comb begin
        for (int i = 0; i < 9; i++) s[i] = s1_r[i];
        {s[0], s[1]} = (s[0] > s[1]) ? {s[1], s[0]} : {s[0], s[1]};
        {s[1], s[2]} = (s[1] > s[2]) ? {s[2], s[1]} : {s[1], s[2]};
        {s[3], s[4]} = (s[3] > s[4]) ? {s[4], s[3]} : {s[3], s[4]};
        {s[4], s[5]} = (s[4] > s[5]) ? {s[5], s[4]} : {s[4], s[5]};
        {s[6], s[7]} = (s[6] > s[7]) ? {s[7], s[6]} : {s[6], s[7]};
        {s[7], s[8]} = (s[7] > s[8]) ? {s[8], s[7]} : {s[7], s[8]};
        // Level 1 min/max: 2 independent pairwise comparisons (1 level deep)
        zm1a = (zm0a_r < zm0b_r) ? zm0a_r : zm0b_r;
        zm1b = (zm0c_r < zm0d_r) ? zm0c_r : zm0d_r;
        zx1a = (zx0a_r > zx0b_r) ? zx0a_r : zx0b_r;
        zx1b = (zx0c_r > zx0d_r) ? zx0c_r : zx0d_r;
    end

    // ── Stage 1b register ─────────────────────────────────────────────────────
    logic [DEPTH-1:0] q [0:8];
    logic [DEPTH-1:0] zm1a_r, zm1b_r, zx1a_r, zx1b_r;
    logic [DEPTH-1:0] center_r2, p8_r2;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) q[i] <= '0;
            zm1a_r <= '0; zm1b_r <= '0; zx1a_r <= '0; zx1b_r <= '0;
            center_r2 <= '0; p8_r2 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) q[i] <= s[i];
            zm1a_r <= zm1a; zm1b_r <= zm1b;
            zx1a_r <= zx1a; zx1b_r <= zx1b;
            center_r2 <= center_r1; p8_r2 <= p8_r1;
        end
    end

    // ── Stage 2a-a (comb): steps 10-12 + min/max Level 2 ─────────────────────
    // Sort: CS(0,3), CS(5,8), CS(4,7) — 1 comparator deep
    // Min/max Level 2: 1 pairwise of Level-1 results — 1 comparator deep
    //   zm2 = min(zm1a_r, zm1b_r),  zx2 = max(zx1a_r, zx1b_r)
    logic [DEPTH-1:0] v1 [0:8];
    logic [DEPTH-1:0] zm2, zx2;
    always_comb begin
        for (int i = 0; i < 9; i++) v1[i] = q[i];
        {v1[0], v1[3]} = (v1[0] > v1[3]) ? {v1[3], v1[0]} : {v1[0], v1[3]};
        {v1[5], v1[8]} = (v1[5] > v1[8]) ? {v1[8], v1[5]} : {v1[5], v1[8]};
        {v1[4], v1[7]} = (v1[4] > v1[7]) ? {v1[7], v1[4]} : {v1[4], v1[7]};
        // Level 2 min/max: 1 pairwise (1 level deep)
        zm2 = (zm1a_r < zm1b_r) ? zm1a_r : zm1b_r;
        zx2 = (zx1a_r > zx1b_r) ? zx1a_r : zx1b_r;
    end

    // ── Stage 2a-a register ───────────────────────────────────────────────────
    logic [DEPTH-1:0] v1_r [0:8];
    logic [DEPTH-1:0] zm2_r, zx2_r;
    logic [DEPTH-1:0] center_r3, p8_r3;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) v1_r[i] <= '0;
            zm2_r <= '0; zx2_r <= '0;
            center_r3 <= '0; p8_r3 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) v1_r[i] <= v1[i];
            zm2_r <= zm2; zx2_r <= zx2;
            center_r3 <= center_r2; p8_r3 <= p8_r2;
        end
    end

    // ── Stage 2a-b (comb): steps 13-16 + min/max Level 3 ─────────────────────
    // Sort: CS(3,6), CS(1,4), CS(2,5), CS(4,7) — 2 comparators deep
    // Min/max Level 3: include p[8] — 1 comparator deep → final z_min, z_max
    logic [DEPTH-1:0] v [0:8];
    logic [DEPTH-1:0] zmin_c4, zmax_c4;
    always_comb begin
        for (int i = 0; i < 9; i++) v[i] = v1_r[i];
        {v[3], v[6]} = (v[3] > v[6]) ? {v[6], v[3]} : {v[3], v[6]};
        {v[1], v[4]} = (v[1] > v[4]) ? {v[4], v[1]} : {v[1], v[4]};
        {v[2], v[5]} = (v[2] > v[5]) ? {v[5], v[2]} : {v[2], v[5]};
        {v[4], v[7]} = (v[4] > v[7]) ? {v[7], v[4]} : {v[4], v[7]};
        // Level 3 min/max: include p[8] (1 comparator deep)
        zmin_c4 = (zm2_r < p8_r3) ? zm2_r : p8_r3;
        zmax_c4 = (zx2_r > p8_r3) ? zx2_r : p8_r3;
    end

    // ── Stage 2a-b register ───────────────────────────────────────────────────
    logic [DEPTH-1:0] r [0:8];
    logic [DEPTH-1:0] zmin_r4, zmax_r4;
    logic [DEPTH-1:0] center_r4;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) r[i] <= '0;
            zmin_r4 <= '0; zmax_r4 <= '0; center_r4 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) r[i] <= v[i];
            zmin_r4 <= zmin_c4; zmax_r4 <= zmax_c4;
            center_r4 <= center_r3;
        end
    end

    // ── Stage 2b-a (comb): step 17 — CS(4,2), 1 comparator deep ─────────────
    logic [DEPTH-1:0] wa [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) wa[i] = r[i];
        {wa[4], wa[2]} = (wa[4] > wa[2]) ? {wa[2], wa[4]} : {wa[4], wa[2]};
    end

    // ── Stage 2b-a register ───────────────────────────────────────────────────
    logic [DEPTH-1:0] w_r [0:8];
    logic [DEPTH-1:0] zmin_r5, zmax_r5;
    logic [DEPTH-1:0] center_r5;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) w_r[i] <= '0;
            zmin_r5 <= '0; zmax_r5 <= '0; center_r5 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) w_r[i] <= wa[i];
            zmin_r5 <= zmin_r4; zmax_r5 <= zmax_r4;
            center_r5 <= center_r4;
        end
    end

    // ── Stage 2b-b-a (comb): step 18 — CS(6,4), 1 comparator deep ───────────
    logic [DEPTH-1:0] wa2 [0:8];
    always_comb begin
        for (int i = 0; i < 9; i++) wa2[i] = w_r[i];
        {wa2[6], wa2[4]} = (wa2[6] > wa2[4]) ? {wa2[4], wa2[6]} : {wa2[6], wa2[4]};
    end

    // ── Stage 2b-b-a register ─────────────────────────────────────────────────
    // Breaks w_r Q → 2-comparator stage-2b-b → comb_r setup (~10.28 ns on ECP5-25k)
    // into w_r Q → step 18 (~3 ns) → w2_r; w2_r Q → step 19 + adaptive mux (~7 ns).
    // zmin/zmax/center forwarded one extra cycle to stay aligned with w2_r.
    logic [DEPTH-1:0] w2_r [0:8];
    logic [DEPTH-1:0] zmin_r6, zmax_r6;
    logic [DEPTH-1:0] center_r6;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) w2_r[i] <= '0;
            zmin_r6 <= '0; zmax_r6 <= '0; center_r6 <= '0;
        end else if (en) begin
            for (int i = 0; i < 9; i++) w2_r[i] <= wa2[i];
            zmin_r6 <= zmin_r5; zmax_r6 <= zmax_r5;
            center_r6 <= center_r5;
        end
    end

    // ── Stage 2b-b-b (comb): step 19 → z_med + adaptive mux ─────────────────
    // z_med = w[4] (median of the 9 pixels, Paeth network)
    // usable      = (zmin_r6 < z_med) && (z_med < zmax_r6)
    // not_impulse = (zmin_r6 < center_r6) && (center_r6 < zmax_r6)
    // output      = (usable && not_impulse) ? center_r6 : z_med
    logic [DEPTH-1:0] w [0:8];
    logic usable, not_impulse;
    always_comb begin
        for (int i = 0; i < 9; i++) w[i] = w2_r[i];
        {w[4], w[2]} = (w[4] > w[2]) ? {w[2], w[4]} : {w[4], w[2]};
        usable      = (zmin_r6 < w[4]) && (w[4] < zmax_r6);
        not_impulse = (zmin_r6 < center_r6) && (center_r6 < zmax_r6);
        adaptive_out = (usable && not_impulse) ? center_r6 : w[4];
    end

endmodule

`default_nettype wire
