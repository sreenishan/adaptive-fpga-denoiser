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
// Pipeline structure (total latency = DIV_STAGES + 2 cycles):
//   Stage 0  — pre-register: comb_px / filter_sel / valid_in → comb_r/sel_r/vld_r
//              (breaks the Gaussian carry-chain → comb_d[1] critical path, ~33 ns)
//   Stages 1..DIV_STAGES — delay chain aligning comb/sel/vld with the Wiener path
//   wiener_r  — one extra register on the Wiener output to match the pre-register
//   Output    — mux then pixel_out / valid_out register

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

    // wiener_filter latency: STAGES(8) + pre-stage(1) + output-acc(1) = 10.
    // An extra wiener_r register here adds 1 more cycle so all paths (comb,
    // sel, vld, wiener) arrive at the mux DIV_STAGES+1 cycles after the window,
    // then the output register adds one more — total latency = DIV_STAGES + 2.
    localparam int WIENER_STAGES = 8;   // restoring-division stages (wiener_filter param)
    localparam int DIV_STAGES    = 10;  // wiener_filter latency; also delay-chain depth

    wiener_filter  #(.DEPTH(DEPTH), .NV_W(NV_W), .STAGES(WIENER_STAGES))
        u_wiener  (.clk(clk), .rst_n(rst_n), .en(en),
                   .win_flat(win_flat), .noise_var(noise_var), .wiener_out(wiener_px));

    // ── Output mux + pipeline register ────────────────────────────────────
    logic [DEPTH-1:0] mux_out;

    // Centre pixel win[1][1] = win_flat[(1*3+1)*DEPTH +: DEPTH] = win_flat[4*DEPTH +: DEPTH].
    logic [DEPTH-1:0] centre;
    assign centre = win_flat[4*DEPTH +: DEPTH];

    // Bypass, median and Gaussian are combinational: their answer for this
    // window is ready now, while the Wiener path is DIV_STAGES cycles behind.
    // Muxing them directly would combine results from different pixels, so the
    // combinational answer, selector and valid are delayed to match.
    logic [DEPTH-1:0] comb_px;
    always_comb begin
        case (filter_sel)
            2'b01:   comb_px = median_px;
            2'b10:   comb_px = gaussian_px;
            default: comb_px = centre;   // bypass; wiener handled below
        endcase
    end

    // Pre-register: comb_px comes through the Gaussian carry chain, which was
    // the 33 ns critical path on ECP5-25k.  Registering it here cuts the path
    // to roughly one FF clock-to-Q + routing.  sel_r and vld_r move in lockstep
    // so the delay chain starting at comb_d[1] stays aligned.
    logic [DEPTH-1:0] comb_r;
    logic [1:0]       sel_r;
    logic             vld_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin comb_r <= '0; sel_r <= 2'b00; vld_r <= 1'b0; end
        else if (en) begin comb_r <= comb_px; sel_r <= filter_sel; vld_r <= valid_in; end
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
            comb_d[1] <= comb_r;      // was: comb_px (direct)
            sel_d[1]  <= sel_r;       // was: filter_sel (direct)
            vld_d[1]  <= vld_r;       // was: valid_in (direct)
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

    // wiener_r adds the cycle that matches wiener_filter's DIV_STAGES output to
    // the pre-registered comb/sel/vld paths.  All three paths meet at the mux
    // DIV_STAGES+1 cycles after the window arrives.
    logic [DEPTH-1:0] wiener_r;
    always_ff @(posedge clk) begin
        if (!rst_n) wiener_r <= '0;
        else if (en) wiener_r <= wiener_px;
    end

    // Everything here belongs to the same pixel: the Wiener pipeline output and
    // the delayed combinational one, chosen by the equally delayed selector.
    always_comb begin
        mux_out = (sel_d[DIV_STAGES] == 2'b11) ? wiener_r : comb_d[DIV_STAGES];
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
