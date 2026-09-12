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
// That divide is eight restoring steps rather than a 32/24-bit divider; see
// the comment at the gain computation for why the two are bit-identical.
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
// NOISE POWER IS A RUNTIME INPUT
// -----------------------------
// `noise_var` used to be a compile-time parameter, which made this the
// fixed-variance form of the filter while the software reference estimates the
// variance per image (configs/inference.yaml leaves it null). Co-simulation
// measured the cost of that gap at up to 6.1 dB and 109 grey levels, so the
// two were not the same filter and a software Wiener result did not describe
// what the FPGA would output.
//
// It is now an input port, written by the host alongside filter_sel: software
// estimates the noise power for the frame and hands the hardware the same
// number it used itself. Like filter_sel it is combinational, so change it
// between frames — see the skew note in fpga_denoiser_top.sv.
//
// Units are squared grey levels, so the largest meaningful value is 255^2 =
// 65025 and NV_W = 16 covers it.
//
// PIPELINED, NOT ITERATIVE
// -----------------------
// The eight restoring steps are chained: each needs the previous remainder.
// Combinationally that is eight 25-bit compare-subtracts in series behind the
// window sums and two multipliers, and in front of another multiply and the
// final divide — one pixel's entire computation in a single cycle. Placed and
// routed on an ECP5-25k that closed at 7.97 MHz.
//
// Each step now ends in a register, so eight pixels are in flight at once and
// throughput stays one pixel per cycle. It costs STAGES cycles of latency, and
// the caller must hold the stream open that much longer — see the flush
// contract in fpga_denoiser_top.sv, which is written in terms of this depth.
//
// The value is unchanged: the same eight steps in the same order, just with
// registers between them. `s` and the centre pixel ride alongside because the
// output arithmetic needs them when the quotient emerges.
//
// Latency: STAGES cycles (was combinational).

`default_nettype none

module wiener_filter #(
    parameter int DEPTH  = 8,
    parameter int NV_W   = 16,      // width of noise_var; 255^2 = 65025 fits
    parameter int STAGES = 8        // one per restoring step; also the latency
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              en,        // advance; hold the pipeline when low
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    input  logic [NV_W-1:0]   noise_var,     // squared grey levels, per frame
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
    logic [V_W-1:0] nv81;                 // 81 * noise_var

    // 81 * 65025 = 5267025, which needs 23 bits and so fits V_W (24). The
    // multiply is evaluated at 32 bits (the width of the integer literal)
    // before the cast, so nothing is lost on the way in.
    assign nv81 = V_W'(81 * noise_var);

    always_comb begin
        v81 = (V_W'(s2) * V_W'(9)) - (V_W'(s) * V_W'(s));
    end

    // ── Gain in Q8: eight-step restoring division ──────────────────────────
    //
    // This used to be `(num << 8) / den` — a 32/24-bit division by a VARIABLE,
    // evaluated combinationally for every pixel. Generic synthesis put it at
    // ~4200 LUTs, 87% of the whole design, and it was the obvious critical
    // path. It also computed 32 quotient bits when the result is clamped to 8.
    //
    // Eight restoring steps produce those eight bits directly, and the result
    // is IDENTICAL — not an approximation, so the 1 grey level budget in
    // configs/hardware.yaml is unchanged and still describes the fixed-point
    // output rounding rather than this divide:
    //
    //   num <= den always. num = v81 - nv81 < v81 = den when v81 > nv81, and
    //   num = 0 otherwise; they are equal only when nv81 = 0. So the true
    //   quotient floor(num*256/den) is at most 256.
    //     · num < den  -> quotient <= 255, and eight steps are exactly that.
    //     · num = den  -> the true value is 256; eight steps saturate at 255,
    //                     which is precisely what the old `> 255` clamp did.
    //
    //   Checked against the old expression over 302,091 (num, den) pairs —
    //   exhaustive for small values, random across the full 24-bit range, plus
    //   the num = den and den = 0 edges: zero mismatches.
    //
    // Every signal below is assigned unconditionally on every iteration. A
    // value written only inside a branch would infer a latch, which is how the
    // median network's temporary was caught.
    logic [V_W-1:0] num_v, den_v;
    logic [7:0]     gain_q8;

    always_comb begin
        num_v = (v81 > nv81) ? (v81 - nv81) : '0;
        den_v = (v81 > nv81) ? v81 : nv81;
    end

    // Pipeline registers, index 1..STAGES: the value after that many stages.
    // `den`, `s` and the centre pixel ride along because the remaining steps
    // and the output arithmetic still need them when this pixel's quotient
    // emerges STAGES cycles later.
    //
    // Each array is driven ONLY by its always_ff. Mixing a continuous assign
    // for element 0 with clocked writes to the rest makes Icarus reject the
    // whole array ("cannot be driven by primitives or continuous assignment"),
    // so the first stage is written out separately from the generated rest.
    logic [V_W:0]     rem_r [1:STAGES];
    logic [V_W-1:0]   den_r [1:STAGES];
    logic [7:0]       q_r   [1:STAGES];
    logic [S_W-1:0]   s_r   [1:STAGES];
    logic [DEPTH-1:0] c_r   [1:STAGES];

    // One restoring step, combinational, per stage boundary.
    logic [V_W:0] sh_c  [1:STAGES-1];
    logic         bit_c [1:STAGES-1];
    logic [V_W:0] nxt_c [1:STAGES-1];

    // Step 1 works on the combinational front end (sums, v81, num/den).
    logic [V_W:0] sh_0, nxt_0;
    logic         bit_0;
    assign sh_0  = {1'b0, num_v} << 1;
    assign bit_0 = (sh_0 >= {1'b0, den_v});
    assign nxt_0 = bit_0 ? (sh_0 - {1'b0, den_v}) : sh_0;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rem_r[1] <= '0;
            den_r[1] <= '0;
            q_r[1]   <= '0;
            s_r[1]   <= '0;
            c_r[1]   <= '0;
        end else if (en) begin
            rem_r[1] <= nxt_0;
            q_r[1]   <= {7'b0, bit_0};      // MSB first; seven bits still to come
            den_r[1] <= den_v;
            s_r[1]   <= s;
            c_r[1]   <= wp[4];              // wp[4] = element [1][1], the centre
        end
    end

    // Steps 2..STAGES. genvar, not a loop variable: an unpacked array indexed
    // by a loop variable is not a constant index to Icarus.
    for (genvar gr = 1; gr < STAGES; gr++) begin : g_step
        assign sh_c[gr]  = rem_r[gr] << 1;
        assign bit_c[gr] = (sh_c[gr] >= {1'b0, den_r[gr]});
        assign nxt_c[gr] = bit_c[gr] ? (sh_c[gr] - {1'b0, den_r[gr]}) : sh_c[gr];

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                rem_r[gr+1] <= '0;
                den_r[gr+1] <= '0;
                q_r[gr+1]   <= '0;
                s_r[gr+1]   <= '0;
                c_r[gr+1]   <= '0;
            end else if (en) begin
                rem_r[gr+1] <= nxt_c[gr];
                q_r[gr+1]   <= {q_r[gr][6:0], bit_c[gr]};
                den_r[gr+1] <= den_r[gr];
                s_r[gr+1]   <= s_r[gr];
                c_r[gr+1]   <= c_r[gr];
            end
        end
    end

    // Flat window AND zero noise: nothing to attenuate and nothing to preserve.
    // The reference floors the denominator at epsilon, so the gain goes to 0
    // and the output is the local mean. Returning full gain here (as an earlier
    // version did) inverts that.
    assign gain_q8 = (den_r[STAGES] == '0) ? 8'd0 : q_r[STAGES];

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


    // The delayed copies, so the sum and centre belong to the same pixel as the
    // quotient that just came out of the pipeline.
    assign s_ext      = A_W'({{(A_W-S_W){1'b0}}, s_r[STAGES]});
    assign centre_ext = A_W'({{(A_W-DEPTH){1'b0}}, c_r[STAGES]});
    assign gain_ext   = A_W'({{(A_W-8){1'b0}}, gain_q8});

    always_comb begin
        centre_term = (centre_ext * A_W'(9)) - s_ext;
        acc = (s_ext <<< 8)
            + (gain_ext * centre_term)
            + A_W'(1152);                 // 9 << 7, round half up

        // The dividend is forced non-negative before the divide so that the
        // synthesised division is unsigned floor, matching the reference model.
        // A signed divide would truncate toward zero and disagree below zero.
        //
        // This one stays a division. Replacing it with the exact reciprocal
        // multiply (acc * 233017) >> 29 — verified for every acc in the
        // reachable 0..1173897 — measured 129 LUTs WORSE under generic
        // mapping, because a 24x18 multiply becomes LUT logic when there are
        // no DSP blocks. On a part with DSPs it should win; re-measure with
        // the vendor tool before adopting it, rather than assuming.
        if (acc <= 0) begin
            quot = '0;
        end else begin
            quot = acc / A_W'(2304);      // 9 << 8
        end
    end

    assign wiener_out = (quot > A_W'(255)) ? '1 : DEPTH'(quot);

endmodule

`default_nettype wire
