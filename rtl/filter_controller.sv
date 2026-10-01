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
// Pipeline structure (total latency = DIV_STAGES + 7 = 19 cycles):
//   Stage W   — win-input register: win_flat/filter_sel/valid_in → win_r/sel_wr/vld_wr
//               (cuts the window_gen counter → filter accumulator path, ~33 ns on ECP5)
//   Stage G1  — layer-1 register (inside median_filter, s1_r[0..8]) +
//               gaussian row-sum register (inside gaussian_filter, 3×11 bits) +
//               matching centre_r / sel_wr2 / vld_wr2
//               gaussian_px combinational at cycle 2 from win_flat
//   Stage G2  — layers-2-3 register (inside median_filter, q[0..8]) +
//               gaussian_r2 / centre_r2 / sel_wr3 / vld_wr3
//   Stage G3  — stage-2a register (inside median_filter, r[0..8]) +
//               gaussian_r3 / centre_r3 / sel_wr4 / vld_wr4
//   Stage G4  — stage-2b-a register (inside median_filter, w_r[0..8]) →
//               median_px combinational at cycle 5 from win_flat;
//               gaussian_r4 / centre_r4 / sel_wr5 / vld_wr5 delay gaussian/bypass
//               to cycle 5 so all comb paths arrive together
//   Stage C   — comb pre-register: comb_px/sel_wr5/vld_wr5 → comb_r/sel_r/vld_r
//   Stages 1..DIV_STAGES — delay chain aligning comb/sel/vld with the Wiener path
//   Output    — mux then pixel_out / valid_out register
//   (wiener_px arrives at cycle 17 = comb_d[11]; no extra wiener_r needed)

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
    localparam int DIV_STAGES    = 12;  // wiener_filter latency (STAGES+9); chain shortened by G3+G4 stages

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

    // median_filter is pipelined (4-cycle latency): s1_r fires at cycle 2; q[] at cycle 3;
    // r[] at cycle 4; w_r[] at cycle 5; median_px is Stage 2b-b comb output at cycle 5.
    // gaussian_px and centre_r are delayed by Stages G2+G3+G4 (below) to match cycle 5.
    median_filter  #(.DEPTH(DEPTH))
        u_median  (.clk(clk), .rst_n(rst_n), .en(en),
                   .win_flat(win_r), .median_out(median_px));

    // gaussian_filter is pipelined (1-cycle latency): its row-sum register fires
    // one cycle after win_r is presented, so gaussian_px is available at cycle 2
    // from win_flat.  centre_r/sel_wr2/vld_wr2 delay the bypass path to match
    // (stage G below).
    gaussian_filter #(.DEPTH(DEPTH))
        u_gaussian (.clk(clk), .rst_n(rst_n), .en(en),
                    .win_flat(win_r), .gaussian_out(gaussian_px));

    // wiener_filter latency: STAGES(8) + squaring(1) + row-partial(1) + pre-stage(1) + variance-stage(1) + den_v(1) + gain-multiply(1) + gain-product(1) + output-acc(1) + recip-product(1) = 17.
    // win_r delays the window by 1 cycle (cycle 1 from win_flat).
    // wiener_px arrives at cycle 18 from win_flat.
    // The comb path: W(1) + G1(1) + G2(1) + G3(1) + G4(1) + C(1) + chain(12) = comb_d[12] at cycle 18.
    // wiener_px (18) = comb_d[12] (18) — no alignment register needed.  ✓
    // Output register adds one more — total latency = DIV_STAGES + 7.
    wiener_filter  #(.DEPTH(DEPTH), .NV_W(NV_W), .STAGES(WIENER_STAGES))
        u_wiener  (.clk(clk), .rst_n(rst_n), .en(en),
                   .win_flat(win_r), .noise_var(noise_var), .wiener_out(wiener_px));

    // ── Stage G1: align bypass / sel / vld with gaussian_px ─────────────────
    // gaussian_filter is pipelined (1-cycle latency): gaussian_px at cycle 2.
    // Centre and sel/vld are combinational from win_r (cycle 1); one register
    // delays them to cycle 2 to match gaussian_px.

    logic [DEPTH-1:0] centre;       // combinational from win_r
    assign centre = win_r[4*DEPTH +: DEPTH];

    logic [DEPTH-1:0] centre_r;
    logic [1:0]       sel_wr2;
    logic             vld_wr2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            centre_r <= '0;
            sel_wr2  <= 2'b00; vld_wr2 <= 1'b0;
        end else if (en) begin
            centre_r <= centre;
            sel_wr2  <= sel_wr; vld_wr2 <= vld_wr;
        end
    end

    // ── Stage G2: advance gaussian_px and centre_r to cycle 3 ────────────────
    // median_filter's s1_r fires at cycle 2; q[] fires at cycle 3.
    // Delay gaussian_px and centre_r one more cycle to reach cycle 3.

    logic [DEPTH-1:0] gaussian_r2;
    logic [DEPTH-1:0] centre_r2;
    logic [1:0]       sel_wr3;
    logic             vld_wr3;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            gaussian_r2 <= '0; centre_r2 <= '0;
            sel_wr3 <= 2'b00; vld_wr3 <= 1'b0;
        end else if (en) begin
            gaussian_r2 <= gaussian_px;
            centre_r2   <= centre_r;
            sel_wr3 <= sel_wr2; vld_wr3 <= vld_wr2;
        end
    end

    // ── Stage G3: advance gaussian_r2 and centre_r2 to cycle 4 ──────────────
    // median_filter's q[] fires at cycle 3; r[] fires at cycle 4.
    // Delay gaussian_r2 and centre_r2 one more cycle to reach cycle 4.

    logic [DEPTH-1:0] gaussian_r3;
    logic [DEPTH-1:0] centre_r3;
    logic [1:0]       sel_wr4;
    logic             vld_wr4;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            gaussian_r3 <= '0; centre_r3 <= '0;
            sel_wr4 <= 2'b00; vld_wr4 <= 1'b0;
        end else if (en) begin
            gaussian_r3 <= gaussian_r2;
            centre_r3   <= centre_r2;
            sel_wr4 <= sel_wr3; vld_wr4 <= vld_wr3;
        end
    end

    // ── Stage G4: advance gaussian_r3 and centre_r3 to cycle 5 ──────────────
    // median_filter's r[] fires at cycle 4; w_r[] fires at cycle 5.
    // Delay gaussian_r3 and centre_r3 one more cycle to reach cycle 5.

    logic [DEPTH-1:0] gaussian_r4;
    logic [DEPTH-1:0] centre_r4;
    logic [1:0]       sel_wr5;
    logic             vld_wr5;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            gaussian_r4 <= '0; centre_r4 <= '0;
            sel_wr5 <= 2'b00; vld_wr5 <= 1'b0;
        end else if (en) begin
            gaussian_r4 <= gaussian_r3;
            centre_r4   <= centre_r3;
            sel_wr5 <= sel_wr4; vld_wr5 <= vld_wr4;
        end
    end

    // ── Output mux + pipeline register ────────────────────────────────────
    logic [DEPTH-1:0] mux_out;

    // All three comb paths arrive at cycle 5 from win_flat:
    //   median_px   — 4-cycle pipelined median_filter output (stage-2b-b comb)
    //   gaussian_r4 — gaussian_px delayed three extra cycles by Stages G2+G3+G4
    //   centre_r4   — bypass pixel delayed four extra cycles by G1+G2+G3+G4
    logic [DEPTH-1:0] comb_px;
    always_comb begin
        case (sel_wr5)
            2'b01:   comb_px = median_px;    // cycle 5 from win_flat ✓
            2'b10:   comb_px = gaussian_r4;  // cycle 5 from win_flat ✓
            default: comb_px = centre_r4;    // cycle 5 from win_flat ✓
        endcase
    end

    // Comb pre-register: comb_px emerges from stage-2b-b comb (2 comparators).
    // Registering here keeps the path to comb_d[1] to one FF hop.
    // sel_r and vld_r move in lockstep with sel_wr5/vld_wr5.
    logic [DEPTH-1:0] comb_r;
    logic [1:0]       sel_r;
    logic             vld_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin comb_r <= '0; sel_r <= 2'b00; vld_r <= 1'b0; end
        else if (en) begin comb_r <= comb_px; sel_r <= sel_wr5; vld_r <= vld_wr5; end
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

    // wiener_px arrives at cycle 18 from win_flat (win_r at cycle 1 + STAGES+9 = 17 cycles + 1).
    // comb_d[12] also arrives at cycle 18: W(1)+G1(1)+G2(1)+G3(1)+G4(1)+C(1)+chain(12) = 18.
    // They are in phase — no alignment register needed.
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
