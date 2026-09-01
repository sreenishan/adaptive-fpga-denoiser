// gaussian_filter.sv — Phase 17
//
// 3×3 Gaussian blur using the binomial approximation kernel:
//
//   1  2  1
//   2  4  2   ÷ 16
//   1  2  1
//
// All arithmetic is integer.  The weighted sum fits in SUM_W = DEPTH+4 = 12
// bits (max = 255×16 = 4080, and 4080+8 = 4088 < 4095, so the rounding add
// cannot overflow either).  Division by 16 is a 4-bit right shift.
// Rounding: add 8 before the shift (round-half-up).
//
// This module is verified bit-exact against denoising.filters.gaussian_filter
// with integer_kernel=True: the three concatenation shapes were checked to
// scale by exactly 1, 2 and 4 at all 256 pixel values with no truncation, and
// the shifted result provably cannot exceed 255, so the assignment to the
// 8-bit output never loses a bit.  max_abs_error.gaussian = 0 is met.
//
// Latency: fully combinational.

`default_nettype none

module gaussian_filter #(
    parameter int DEPTH = 8
) (
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    output logic [DEPTH-1:0] gaussian_out
);
    localparam int SUM_W = DEPTH + 4;  // 12 bits covers 255*16=4080

    logic [SUM_W-1:0] weighted_sum;

    // Extract each pixel into a wire so the always_comb is not a packed-select.
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

    always_comb begin
        // Corner weight 1, edge weight 2, centre weight 4.
        weighted_sum =
            ( {4'b0, p00} +
              {3'b0, p01, 1'b0} +
              {4'b0, p02} +
              {3'b0, p10, 1'b0} +
              {2'b0, p11, 2'b0} +
              {3'b0, p12, 1'b0} +
              {4'b0, p20} +
              {3'b0, p21, 1'b0} +
              {4'b0, p22} );
    end

    // Divide by 16 with round-half-up.
    assign gaussian_out = (weighted_sum + SUM_W'(8)) >> 4;

endmodule

`default_nettype wire
