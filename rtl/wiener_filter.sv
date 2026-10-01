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
// Total pipeline latency: STAGES + 4 cycles.
//
//   Pre-stage (1 cycle): registers the window sums (s, s2), centre pixel and
//   noise_var before the variance computation. Breaks the long combinational
//   path from the nine-pixel sum accumulators into a shorter first hop.
//
//   Variance stage (1 cycle): registers num_v (= max(0, v81-81*NV)) and den_v
//   (= max(v81, 81*NV)) after the MULT18X18D that computes s_pre*s_pre.
//   Breaks the path: s_pre Q → s_pre*s_pre (DSP) → v81 subtract → den_v
//   compare → rem_r[1] setup, which was ~20 ns on ECP5-25k.
//
//   Restoring-division stages (STAGES = 8 cycles): each step ends in a
//   register; eight pixels are in flight at once, throughput one pixel/cycle.
//   s and centre ride alongside so the output arithmetic has them when the
//   quotient emerges.
//
//   Gain-multiply stage (1 cycle): registers centre_term (= c_r[STAGES]*9 - s)
//   alongside gain_ext and s_ext before the gain × centre_term multiply.
//   Breaks the two-MULT cascade (c_r[STAGES] → first MULT → second MULT →
//   acc_r setup, ~17.6 ns on ECP5-25k) into two separate ~8 ns hops.
//   Not an approximation — the result is identical.
//
//   Output-accumulator stage (1 cycle): registers `acc` before the final
//   divide. Breaks the path from ct_r through the second MULT18X18D (gain ×
//   centre_term) and the accumulation to the output register.
//
//   The final divide is (acc_r * RECIP) >> RECIP_SHR, an exact reciprocal
//   multiply that replaces the `acc / 2304` carry-chain division: on ECP5 it
//   maps to two MULT18X18D blocks, removing the carry chain from this stage.
//   It was 129 LUTs WORSE under the generic (no-DSP) flow — see the comment at
//   the gain computation — but wins on this part. Do not adopt it for a
//   LUT-only flow without re-measuring.

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
    // ── Window sums (combinational) ────────────────────────────────────────
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

    // ── Pre-stage register — breaks the long path from window inputs ────────
    // Without this, the combinational chain s/s2 → v81 → den_v → step-0 →
    // rem_r[1] was ~25 ns on ECP5-25k, and the nine-pixel accumulations alone
    // consumed ~10 ns before the two multiplies.
    logic [S_W-1:0]   s_pre;
    logic [S2_W-1:0]  s2_pre;
    logic [DEPTH-1:0] c_pre;
    logic [NV_W-1:0]  nv_pre;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_pre  <= '0;  s2_pre <= '0;
            c_pre  <= '0;  nv_pre <= '0;
        end else if (en) begin
            s_pre  <= s;
            s2_pre <= s2;
            c_pre  <= wp[4];
            nv_pre <= noise_var;
        end
    end

    // ── 81*variance, exact (from pre-registered sums) ──────────────────────
    // 9*s2 needs 24 bits (max 5267025); s*s needs 23 (max 5267025). By
    // Cauchy-Schwarz 9*s2 >= s*s always, so v81 is never negative.
    localparam int V_W = 2*DEPTH + 8;     // 24b

    // ── Variance stage register — registers the two products before the ─────
    // subtraction.  Both p1 = 9*s2_pre (shift+add carry chain) and
    // p2 = s_pre*s_pre (MULT18X18D) are only ~5-8 ns from the pre-stage
    // registers; the full path through the subtraction and den_v comparison
    // was ~20 ns on ECP5-25k.  After registration, v81 = p1_r - p2_r is a
    // single 24-bit subtraction (~5 ns), leaving budget for step-0 and
    // rem_r[1] setup.
    //
    // nv81_r = 81*nv_pre is also registered here so it stays aligned with p1_r.
    logic [V_W-1:0]  p1_r;    // 9*s2_pre, registered
    logic [V_W-1:0]  p2_r;    // s_pre*s_pre, registered
    logic [V_W-1:0]  nv81_r;  // 81*nv_pre, registered; 81*65025=5267025 fits 24b
    logic [S_W-1:0]  s_r0;    // sum, latched alongside
    logic [DEPTH-1:0] c_r0;   // centre pixel, latched alongside

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            p1_r   <= '0; p2_r   <= '0; nv81_r <= '0;
            s_r0   <= '0; c_r0   <= '0;
        end else if (en) begin
            p1_r   <= V_W'(V_W'(s2_pre) * V_W'(9));
            p2_r   <= V_W'(V_W'(s_pre)  * V_W'(s_pre));
            nv81_r <= V_W'(81 * nv_pre);
            s_r0   <= s_pre;
            c_r0   <= c_pre;
        end
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
    logic [V_W-1:0] v81;     // 9*s2 - s^2, combinational from variance-stage regs
    logic [V_W-1:0] num_v, den_v;
    logic [7:0]     gain_q8;

    always_comb begin
        v81   = p1_r - p2_r;                           // fast subtraction
        num_v = (v81 > nv81_r) ? (v81 - nv81_r) : '0;
        den_v = (v81 > nv81_r) ? v81 : nv81_r;
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

    // Step 1 works from the variance-stage register outputs (v81 = p1_r - p2_r,
    // num_v, den_v) — all fast combinational from registered products.
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
            den_r[1] <= den_v;              // from variance-stage register output
            s_r[1]   <= s_r0;              // from variance-stage register
            c_r[1]   <= c_r0;              // from variance-stage register
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
    logic signed [A_W-1:0] centre_term;   // 9*centre - s (first MULT18X18D output)

    // The delayed copies, so the sum and centre belong to the same pixel as the
    // quotient that just came out of the pipeline.
    assign s_ext      = A_W'({{(A_W-S_W){1'b0}}, s_r[STAGES]});
    assign centre_ext = A_W'({{(A_W-DEPTH){1'b0}}, c_r[STAGES]});
    assign gain_ext   = A_W'({{(A_W-8){1'b0}}, gain_q8});

    assign centre_term = (centre_ext * A_W'(9)) - s_ext;

    // ── Gain-multiply stage register ──────────────────────────────────────────
    // centre_term = c_r[STAGES] × 9 − s uses one MULT18X18D (~8 ns from
    // c_r[STAGES] Q).  Without this register, gain × centre_term (second
    // MULT18X18D) was chained directly, giving ~17.6 ns from c_r[8] Q to
    // acc_r setup.  Registering here splits that into two ~8 ns hops.
    // Not an approximation — registers only move when arithmetic runs, not what
    // it produces.  Wiener total latency: STAGES + 3 → STAGES + 4 cycles.
    logic signed [A_W-1:0] ct_r;     // centre_term registered
    logic signed [A_W-1:0] gain_r;   // gain_ext registered
    logic signed [A_W-1:0] sext_r;   // s_ext registered

    always_ff @(posedge clk) begin
        if (!rst_n) begin ct_r <= '0; gain_r <= '0; sext_r <= '0; end
        else if (en)  begin ct_r <= centre_term; gain_r <= gain_ext; sext_r <= s_ext; end
    end

    // Second-stage accumulator: inputs now arrive from ct_r/gain_r/sext_r registers.
    logic signed [A_W-1:0] acc;
    always_comb begin
        acc = (sext_r <<< 8)
            + (gain_r * ct_r)
            + A_W'(1152);               // 9 << 7, round half up
    end

    // ── Output-accumulator stage register — breaks the path from ct_r ────────
    // Registering acc here isolates the reciprocal divide in its own stage.
    //
    // acc is clamped to 0 on the way in: a negative dividend produces quot=0
    // regardless, and the clamp avoids carrying a signed value into the
    // unsigned reciprocal multiply.
    logic [A_W-1:0] acc_r;               // unsigned; clamped to 0 if acc <= 0

    always_ff @(posedge clk) begin
        if (!rst_n) acc_r <= '0;
        else if (en) acc_r <= (acc > 0) ? A_W'(acc) : '0;
    end

    // ── Reciprocal multiply for acc / 2304 ────────────────────────────────
    // floor(acc / 2304) == (acc * RECIP) >> RECIP_SHR  for all acc in
    // [0, 1_173_897] (the reachable range given max s=2295, max gain=255).
    // Verified exhaustively in Python. Not an approximation; the 1 grey-level
    // budget still describes the fixed-point rounding, not this step.
    //
    // RECIP = floor(2^29 / 2304) = 233017. On ECP5 (synth_ecp5) this maps to
    // two MULT18X18D blocks (21-bit × 18-bit operands), replacing a 24-bit
    // carry-chain divider. It measured 129 LUTs worse under generic mapping
    // (no DSPs) — do not use it there.
    localparam int RECIP     = 233017;
    localparam int RECIP_SHR = 29;

    // 43-bit intermediate: max product = 1_173_897 × 233_017 ≈ 2^38.2 < 2^43.
    // SV truncates a * b to the width of the LHS assignment, so 43 bits is
    // wide enough and nothing is lost.
    logic [42:0] recip_prod;
    logic [A_W-1:0] quot;

    always_comb begin
        recip_prod = 43'(acc_r) * 43'(RECIP);
        quot       = A_W'(recip_prod >> RECIP_SHR);
    end

    assign wiener_out = (quot > A_W'(255)) ? '1 : DEPTH'(quot);

endmodule

`default_nettype wire
