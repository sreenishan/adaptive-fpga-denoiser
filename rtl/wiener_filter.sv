// wiener_filter.sv — Phase 18
//
// 3×3 adaptive Wiener filter, exact-integer formulation.
//
//   out = mu + max(0, var - NV) / max(var, NV) * (centre - mu)
//
// WHY THE ARITHMETIC IS SHAPED THIS WAY
// -------------------------------------
// The obvious implementation — compute mu = S/9, then var = S2/9 - mu*mu — is
// numerically hopeless in integers, and the previous version of this file was
// exactly that. Rounding mu to a whole grey level puts an error of up to 0.5 in
// mu, and squaring it puts an error of roughly mu in var. With mu near 128 that
// is an error of ~128 against a NOISE_VAR of 100: the gain term is then
// dominated by noise from the rounding, not by the image. Measured against the
// software reference, that formulation was 19 grey levels out even after its
// two slice bugs were repaired, versus a budget of 1.
//
// The fix is to never form mu*mu from a rounded mean. For a 9-pixel window
//
//     81 * var  =  9 * SUM(x^2)  -  (SUM x)^2
//
// which is EXACT in integers — no division, no rounding, no cancellation. The
// factor 81 then cancels in the gain ratio, so it never has to be divided out:
//
//     n = max(0, v81 - 81*NV)          d = max(v81, 81*NV)
//     gain = (n << 8) / d                            (Q8, in [0,255])
//
// and the mean is likewise kept unrounded by scaling the whole output by 9:
//
//     out = ( S*2^8 + gain*(9*centre - S) + 9*2^7 ) / (9*2^8)
//
// where the +9*2^7 is round-half-up. Verified against the software reference
// over six image types (flat 0, flat 255, ramp, checkerboard, uniform random,
// noisy ramp) crossed with NOISE_VAR in {0, 1, 25, 100, 400, 4000}: worst-case
// error 1 grey level across all 36 combinations, which is the budget
// configs/hardware.yaml sets (max_abs_error.wiener = 1).
//
// SEMANTIC GAP, DELIBERATELY LEFT VISIBLE
// ---------------------------------------
// NOISE_VAR here is a compile-time parameter. The software reference estimates
// the noise variance per image (the mean of the local variances) whenever
// configs/inference.yaml leaves it null. A parameter cannot do that. Until a
// host writes this value per frame over a control interface, the RTL is the
// fixed-variance form of the filter and comparisons must pin the reference to
// the same constant. This is a known difference, not an approximation.
//
// Latency: fully combinational.

`default_nettype none

module wiener_filter #(
    parameter int DEPTH     = 8,
    parameter int NOISE_VAR = 100   // squared grey levels; ~sigma=10
) (
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    output logic [DEPTH-1:0]  wiener_out
);
    // ── Window sums ────────────────────────────────────────────────────────
    localparam int S_W  = DEPTH + 4;      // 12b: max 9*255 = 2295
    localparam int S2_W = 2*DEPTH + 4;    // 20b: max 9*255^2 = 585225

    logic [S_W-1:0]  s;        // SUM x
    logic [S2_W-1:0] s2;       // SUM x^2

    // Extract pixels into a local unpacked array to avoid 2D packed port indexing.
    logic [DEPTH-1:0] wp [0:8];
    for (genvar gi = 0; gi < 9; gi++) begin : gen_wp
        assign wp[gi] = win_flat[gi*DEPTH +: DEPTH];
    end

    always_comb begin
        s  = '0;
        s2 = '0;
        for (int i = 0; i < 9; i++) begin
            s  = s  + S_W'(wp[i]);
            s2 = s2 + S2_W'(wp[i]) * S2_W'(wp[i]);
        end
    end

    // ── 81*variance, exact ─────────────────────────────────────────────────
    // 9*s2 needs 24 bits (max 5267025); s*s needs 23 (max 5267025). By
    // Cauchy-Schwarz 9*s2 >= s*s always, so v81 is never negative.
    localparam int V_W = 2*DEPTH + 8;     // 24b
    logic [V_W-1:0] v81;
    logic [V_W-1:0] nv81;                 // 81 * NOISE_VAR

    assign nv81 = V_W'(81 * NOISE_VAR);

    always_comb begin
        v81 = (V_W'(s2) * V_W'(9)) - (V_W'(s) * V_W'(s));
    end

    // ── Gain in Q8, clamped to [0,255] ─────────────────────────────────────
    logic [V_W-1:0]   num_v, den_v;
    logic [V_W+8-1:0] num_shifted;
    logic [V_W+8-1:0] quotient;
    logic [7:0]       gain_q8;

    always_comb begin
        num_v = (v81 > nv81) ? (v81 - nv81) : '0;
        den_v = (v81 > nv81) ? v81 : nv81;

        num_shifted = {num_v, 8'b0};      // << 8

        if (den_v == '0) begin
            // Flat window AND zero noise: nothing to attenuate and nothing to
            // preserve. The reference floors the denominator at epsilon, so the
            // gain goes to 0 and the output is the local mean. Returning full
            // gain here (as an earlier version did) inverts that.
            gain_q8  = 8'd0;
            quotient = '0;
        end else begin
            quotient = num_shifted / den_v;
            gain_q8  = (quotient > (V_W+8)'(255)) ? 8'd255 : 8'(quotient);
        end
    end

    // ── out = (s*256 + gain*(9*centre - s) + 1152) / 2304 ──────────────────
    // 9*centre - s is in [-2295, 2295]           -> 13b signed
    // gain*(that)  is in [-585225, 585225]       -> 21b signed
    // s*256        is in [0, 587520]             -> 20b unsigned
    localparam int A_W = 2*DEPTH + 8;     // 24b signed accumulator

    // Every operand is widened to A_W BEFORE any shift or multiply. Relying on
    // cast-around-expression here would be a width bug waiting to happen: the
    // self-determined width of `s <<< 8` is s's own 12 bits, so the top eight
    // bits would be lost before the surrounding cast ever widened anything.
    logic signed [A_W-1:0] s_ext;         // sum, zero-extended
    logic signed [A_W-1:0] centre_ext;    // centre pixel, zero-extended
    logic signed [A_W-1:0] gain_ext;      // Q8 gain, zero-extended
    logic signed [A_W-1:0] centre_term;   // 9*centre - s
    logic signed [A_W-1:0] acc;           // biased dividend
    logic signed [A_W-1:0] quot;

    assign s_ext      = A_W'({{(A_W-S_W){1'b0}}, s});
    assign centre_ext = A_W'({{(A_W-DEPTH){1'b0}}, wp[4]});   // wp[4] = element [1][1]
    assign gain_ext   = A_W'({{(A_W-8){1'b0}}, gain_q8});

    always_comb begin
        centre_term = (centre_ext * A_W'(9)) - s_ext;
        acc = (s_ext <<< 8)
            + (gain_ext * centre_term)
            + A_W'(1152);                 // 9 << 7, round half up

        // The dividend is forced non-negative before the divide so that the
        // synthesised division is unsigned floor, matching the reference model.
        // A signed divide would truncate toward zero and disagree below zero.
        if (acc <= 0) begin
            quot = '0;
        end else begin
            quot = acc / A_W'(2304);      // 9 << 8
        end
    end

    assign wiener_out = (quot > A_W'(255)) ? '1 : DEPTH'(quot);

endmodule

`default_nettype wire
