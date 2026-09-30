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
    parameter int NV_W = 16
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [1:0]         filter_sel,
    input  logic [NV_W-1:0]    noise_var,  // Wiener noise power, squared grey levels
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

    // The Wiener divider is pipelined, so its answer for this window arrives
    // DIV_STAGES cycles from now. This must equal wiener_filter's STAGES.
    localparam int DIV_STAGES = 8;

    wiener_filter  #(.DEPTH(DEPTH), .NV_W(NV_W), .STAGES(DIV_STAGES))
        u_wiener  (.clk(clk), .rst_n(rst_n), .en(en),
                   .win_flat(win_flat), .noise_var(noise_var), .wiener_out(wiener_px));

    // ── Output mux + pipeline register ────────────────────────────────────
    logic [DEPTH-1:0] mux_out;

    // Centre pixel win[1][1] = win_flat[(1*3+1)*DEPTH +: DEPTH] = win_flat[4*DEPTH +: DEPTH].
    logic [DEPTH-1:0] centre;
    assign centre = win_flat[4*DEPTH +: DEPTH];

    // Bypass, median and Gaussian are combinational: their answer for this
    // window is ready now, while the Wiener path is DIV_STAGES cycles behind.
    // Muxing them together directly would combine results from different
    // pixels, so the combinational answer, the selector and the valid all get
    // delayed by the same depth. Delaying the 8-bit result rather than the
    // 72-bit window is what keeps this cheap.
    logic [DEPTH-1:0] comb_px;
    always_comb begin
        case (filter_sel)
            2'b01:   comb_px = median_px;
            2'b10:   comb_px = gaussian_px;
            default: comb_px = centre;   // bypass; the wiener code is handled below
        endcase
    end

    // Driven only by their always_ff blocks; see the note in wiener_filter.sv
    // about mixing continuous and clocked drivers on one array.
    logic [DEPTH-1:0] comb_d [1:DIV_STAGES];
    logic [1:0]       sel_d  [1:DIV_STAGES];
    logic             vld_d  [1:DIV_STAGES];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            comb_d[1] <= '0;
            sel_d[1]  <= 2'b00;
            vld_d[1]  <= 1'b0;
        end else if (en) begin
            comb_d[1] <= comb_px;
            sel_d[1]  <= filter_sel;
            vld_d[1]  <= valid_in;
        end
    end

    for (genvar gd = 1; gd < DIV_STAGES; gd++) begin : g_delay
        always_ff @(posedge clk) begin
            if (!rst_n) begin
                comb_d[gd+1] <= '0;
                sel_d[gd+1]  <= 2'b00;
                vld_d[gd+1]  <= 1'b0;
            end else if (en) begin
                comb_d[gd+1] <= comb_d[gd];
                sel_d[gd+1]  <= sel_d[gd];
                vld_d[gd+1]  <= vld_d[gd];
            end
        end
    end

    // Everything here belongs to the same pixel: the Wiener pipeline output and
    // the delayed combinational one, chosen by the equally delayed selector.
    always_comb begin
        mux_out = (sel_d[DIV_STAGES] == 2'b11) ? wiener_px : comb_d[DIV_STAGES];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pixel_out <= '0;
            valid_out <= 1'b0;
        end else if (en) begin
            pixel_out <= mux_out;
            valid_out <= vld_d[DIV_STAGES];
        end else begin
            valid_out <= 1'b0;   // no new pixel this cycle
        end
    end

endmodule

`default_nettype wire
