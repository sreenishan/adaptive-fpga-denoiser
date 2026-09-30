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
// Pipeline structure (total latency = DIV_STAGES + 3 cycles):
//   Stage W   — win-input register: win_flat / filter_sel / valid_in → win_r/sel_wr/vld_wr
//               (cuts the window_gen counter → gaussian accumulator path, ~33 ns on ECP5)
//   Stage C   — comb pre-register: comb_px/sel_wr/vld_wr → comb_r/sel_r/vld_r
//   Stages 1..DIV_STAGES — delay chain aligning comb/sel/vld with the Wiener path
//   wiener_r  — one extra register on the Wiener output to match the two pre-registers
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
    // ── Win-input register ────────────────────────────────────────────────
    // Registers win_flat, filter_sel and valid_in one cycle before presenting
    // them to the filter cores.  This cuts the window_gen counter propagation
    // (routing + LUT mux logic, ~3.5 ns) out of the critical path so the new
    // critical path starts from win_r's Q rather than window_gen's counter FF.
    localparam int WIENER_STAGES = 8;   // restoring-division stages (wiener_filter param)
    localparam int DIV_STAGES    = 10;  // wiener_filter latency; also delay-chain depth

    logic [3*3*DEPTH-1:0] win_r;
    logic [1:0]           sel_wr;
    logic                 vld_wr;

    always_ff @(posedge clk) begin
        if (!rst_n) begin win_r <= '0; sel_wr <= 2'b00; vld_wr <= 1'b0; end
        else if (en) begin win_r <= win_flat; sel_wr <= filter_sel; vld_wr <= valid_in; end
    end

    // ── Filter core outputs ────────────────────────────────────────────────
    logic [DEPTH-1:0] median_px;
    logic [DEPTH-1:0] gaussian_px;
    logic [DEPTH-1:0] wiener_px;

    median_filter  #(.DEPTH(DEPTH))
        u_median  (.win_flat(win_r), .median_out(median_px));

    gaussian_filter #(.DEPTH(DEPTH))
        u_gaussian (.win_flat(win_r), .gaussian_out(gaussian_px));

    // wiener_filter latency: STAGES(8) + pre-stage(1) + output-acc(1) = 10.
    // win_r delays the window by 1 cycle, so wiener_px arrives at cycle 11 from
    // win_flat.  wiener_r adds 1 more = cycle 12, matching comb_d[10].
    // Output register adds one more — total latency = DIV_STAGES + 3.
    wiener_filter  #(.DEPTH(DEPTH), .NV_W(NV_W), .STAGES(WIENER_STAGES))
        u_wiener  (.clk(clk), .rst_n(rst_n), .en(en),
                   .win_flat(win_r), .noise_var(noise_var), .wiener_out(wiener_px));

    // ── Output mux + pipeline register ────────────────────────────────────
    logic [DEPTH-1:0] mux_out;

    // Centre pixel win[1][1] from the registered window.
    logic [DEPTH-1:0] centre;
    assign centre = win_r[4*DEPTH +: DEPTH];

    // Bypass, median and Gaussian are combinational from win_r; the Wiener path
    // is DIV_STAGES cycles behind win_r.  Delay comb/sel/vld to match.
    logic [DEPTH-1:0] comb_px;
    always_comb begin
        case (sel_wr)
            2'b01:   comb_px = median_px;
            2'b10:   comb_px = gaussian_px;
            default: comb_px = centre;   // bypass; wiener handled below
        endcase
    end

    // Comb pre-register: comb_px still passes through the Gaussian carry chain.
    // Registering here keeps the path from win_r Q to comb_d[1] to one FF hop.
    // sel_r and vld_r move in lockstep with sel_wr/vld_wr.
    logic [DEPTH-1:0] comb_r;
    logic [1:0]       sel_r;
    logic             vld_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin comb_r <= '0; sel_r <= 2'b00; vld_r <= 1'b0; end
        else if (en) begin comb_r <= comb_px; sel_r <= sel_wr; vld_r <= vld_wr; end
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

    // wiener_r aligns the Wiener output with comb_d[10].
    // wiener_filter takes win_r (cycle 1 from win_flat) and produces wiener_px
    // 10 cycles later = cycle 11 from win_flat.  wiener_r adds 1 more = cycle 12.
    // comb_d[10] is also cycle 12 (win_r→comb_r→comb_d[1..10]).  ✓
    // All paths meet at the mux DIV_STAGES+2 cycles after win_flat arrived.
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
