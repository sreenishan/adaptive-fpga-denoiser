// gaussian_filter.sv — Phase 17
//
// 3×3 Gaussian blur using the binomial approximation kernel:
//
//   1  2  1
//   2  4  2   ÷ 16
//   1  2  1
//
// Pipelined (1 cycle latency): row sums are registered between the two
// adder stages to shorten the carry-chain depth.
//
// Stage 1 (combinational → register):
//   r0 = p00 + 2·p01 + p02       max 1020, 10 bits, 2 additions in series
//   r1 = 2·p10 + 4·p11 + 2·p12  max 2040, 11 bits, 2 additions in series
//   r2 = p20 + 2·p21 + p22       max 1020, 10 bits (same as r0)
//
// Stage 2 (combinational from r*_q):
//   weighted_sum = r0_q + r1_q + r2_q   max 4080, 12 bits, 2 additions
//   gaussian_out = (weighted_sum + 8) >> 4
//
// The arithmetic is unchanged — r0+r1+r2 equals the former flat weighted_sum
// exactly (addition is associative over integers within range) — so the
// 1-cycle output is bit-identical to the former combinational one.
// max_abs_error.gaussian = 0.
//
// Requires clk / rst_n (synchronous active-low) / en (clock enable).

`default_nettype none

module gaussian_filter #(
    parameter int DEPTH = 8
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               en,
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    output logic [DEPTH-1:0]     gaussian_out
);
    localparam int SUM_W = DEPTH + 4;  // 12 bits covers 255*16=4080
    localparam int ROW_W = DEPTH + 3;  // 11 bits covers row1 max 2040

    logic [DEPTH-1:0] p00, p01, p02, p10, p11, p12, p20, p21, p22;
    assign p00 = win_flat[(0*3+0)*DEPTH +: DEPTH];
    assign p01 = win_flat[(0*3+1)*DEPTH +: DEPTH];
    assign p02 = win_flat[(0*3+2)*DEPTH +: DEPTH];
    assign p10 = win_flat[(1*3+0)*DEPTH +: DEPTH];
    assign p11 = win_flat[(1*3+1)*DEPTH +: DEPTH];
    assign p12 = win_flat[(1*3+2)*DEPTH +: DEPTH];
    assign p20 = win_flat[(2*3+0)*DEPTH +: DEPTH];
    assign p21 = win_flat[(2*3+1)*DEPTH +: DEPTH];
    assign p22 = win_flat[(2*3+2)*DEPTH +: DEPTH];

    // ── Stage 1: row sums (combinational) ─────────────────────────────────
    logic [ROW_W-1:0] r0, r1, r2;
    always_comb begin
        r0 = ROW_W'(p00) + ROW_W'({p01, 1'b0}) + ROW_W'(p02);
        r1 = ROW_W'({p10, 1'b0}) + ROW_W'({p11, 2'b0}) + ROW_W'({p12, 1'b0});
        r2 = ROW_W'(p20) + ROW_W'({p21, 1'b0}) + ROW_W'(p22);
    end

    // ── Pipeline register: row sums ────────────────────────────────────────
    logic [ROW_W-1:0] r0_q, r1_q, r2_q;
    always_ff @(posedge clk) begin
        if (!rst_n) begin r0_q <= '0; r1_q <= '0; r2_q <= '0; end
        else if (en) begin r0_q <= r0; r1_q <= r1; r2_q <= r2; end
    end

    // ── Stage 2: final sum, round, shift (combinational from registers) ────
    logic [SUM_W-1:0] weighted_sum;
    assign weighted_sum = SUM_W'(r0_q) + SUM_W'(r1_q) + SUM_W'(r2_q);
    assign gaussian_out = DEPTH'((weighted_sum + SUM_W'(8)) >> 4);

endmodule

`default_nettype wire
