// filter_controller.sv — Phase 19
//
// Routes a 3×3 window to one of four filter cores and muxes the output.
//
// filter_sel encoding (matches FILTER_FOR_CLASS in Python):
//   2'b00 — bypass   (output = centre pixel, win[1][1])
//   2'b01 — median
//   2'b10 — gaussian
//   2'b11 — wiener
//
// pixel_out is valid one cycle after filter_sel and the window arrive,
// though all three filter cores are combinational; the register adds one
// cycle of clean timing margin.

`default_nettype none

module filter_controller #(
    parameter int DEPTH     = 8,
    parameter int NOISE_VAR = 100
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [1:0]         filter_sel,
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    input  logic               valid_in,
    // Register enable. The output register must advance ONLY on the cycles the
    // window generator advances, otherwise a stalled stream (s_valid low) makes
    // this block re-register the held window and emit the same pixel again with
    // m_valid still high — a silent duplicate for every stall cycle.
    input  logic               en,
    output logic [DEPTH-1:0]   pixel_out,
    output logic               valid_out
);
    // ── Filter core outputs ────────────────────────────────────────────────
    logic [DEPTH-1:0] median_px;
    logic [DEPTH-1:0] gaussian_px;
    logic [DEPTH-1:0] wiener_px;

    median_filter  #(.DEPTH(DEPTH))
        u_median  (.win_flat(win_flat), .median_out(median_px));

    gaussian_filter #(.DEPTH(DEPTH))
        u_gaussian (.win_flat(win_flat), .gaussian_out(gaussian_px));

    wiener_filter  #(.DEPTH(DEPTH), .NOISE_VAR(NOISE_VAR))
        u_wiener  (.win_flat(win_flat), .wiener_out(wiener_px));

    // ── Output mux + pipeline register ────────────────────────────────────
    logic [DEPTH-1:0] mux_out;

    // Centre pixel win[1][1] = win_flat[(1*3+1)*DEPTH +: DEPTH] = win_flat[4*DEPTH +: DEPTH].
    logic [DEPTH-1:0] centre;
    assign centre = win_flat[4*DEPTH +: DEPTH];

    always_comb begin
        case (filter_sel)
            2'b00:   mux_out = centre;
            2'b01:   mux_out = median_px;
            2'b10:   mux_out = gaussian_px;
            2'b11:   mux_out = wiener_px;
            default: mux_out = centre;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pixel_out <= '0;
            valid_out <= 1'b0;
        end else if (en) begin
            pixel_out <= mux_out;
            valid_out <= valid_in;
        end else begin
            valid_out <= 1'b0;   // no new pixel this cycle
        end
    end

endmodule

`default_nettype wire
