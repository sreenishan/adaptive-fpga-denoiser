// fpga_denoiser_top.sv — Phase 20
//
// Top-level FPGA denoising accelerator.
//
// Data path:
//   pixel_in (raster order) → window_gen → filter_controller → pixel_out
//
// PROTOCOL
//   Assert s_valid for exactly IMG_WIDTH*IMG_HEIGHT pixels in raster order,
//   then hold s_flush for FLUSH_CYCLES cycles to drain the pipeline. m_valid is
//   then high for exactly IMG_WIDTH*IMG_HEIGHT cycles in total, one per input
//   pixel, each carrying the replicate-padded 3×3 result for that pixel.
//   The stream may stall (s_valid low) at any point; the pipeline holds.
//
//   FLUSH_CYCLES is IMG_WIDTH + 2 + PIPE_STAGES, both localparams below. It
//   was IMG_WIDTH+2 while every filter was combinational; the Wiener divider is
//   pipelined now, so a caller that still flushes for IMG_WIDTH+2 will come up
//   PIPE_STAGES pixels short. Read the localparam rather than hardcoding it.
//
// LATENCY
//   window_gen primes for IMG_WIDTH+2 advances, filter_controller's Wiener
//   divider adds PIPE_STAGES, and its output register one more, so the first
//   m_valid appears IMG_WIDTH+3+PIPE_STAGES advances after the first pixel.
//   (An older header claimed IMG_WIDTH*(SIZE/2) + SIZE/2, which was both the
//   wrong formula and two cycles short.)
//
//   Throughput is unaffected: the divider is pipelined, not iterative, so one
//   pixel still enters and leaves per advance.
//
// filter_sel, noise_var
//   The filter cores are combinational, so this control applies to whichever
//   window is in the generator that cycle — which is the window centred
//   IMG_WIDTH+1 pixels BEHIND the pixel currently entering s_pixel. Changing
//   either of them mid-frame therefore takes effect on output pixels from that
//   cycle onward, not on input pixels from that cycle onward. Change them only
//   between frames unless that skew is what you want; the previous header
//   called this "safe" without qualifying it. noise_var is the Wiener noise
//   power the host measured for this frame, in squared grey levels.
//
// m_ready
//   This design has no elastic buffering and cannot stall its output. m_ready
//   is accepted and checked only so a downstream deassertion is reported on
//   m_overflow rather than silently dropping a pixel; it does NOT create
//   back-pressure. configs/hardware.yaml declares backpressure: false, so the
//   honest interface is one that says loudly when the contract is broken.
//   Tie m_ready high if the sink is always ready.

`default_nettype none

module fpga_denoiser_top #(
    parameter int IMG_WIDTH  = 224,
    parameter int IMG_HEIGHT = 224,
    parameter int NV_W       = 16,
    parameter int DEPTH      = 8,
    parameter int WIN_SIZE   = 3     // neighbourhood size (must be 3 here)
) (
    input  logic              clk,
    input  logic              rst_n,
    // Control
    input  logic [1:0]        filter_sel,
    input  logic [NV_W-1:0]   noise_var,   // Wiener noise power, squared grey levels
    // AXI-Stream source (input pixels)
    input  logic              s_valid,
    input  logic [DEPTH-1:0]  s_pixel,
    input  logic              s_flush,   // hold IMG_WIDTH+1 cycles after a frame
    output logic              s_ready,
    // AXI-Stream sink (denoised pixels)
    output logic              m_valid,
    output logic [DEPTH-1:0]  m_pixel,
    input  logic              m_ready,
    output logic              m_overflow // sticky: a pixel was produced while
                                         // m_ready was low and has been lost
);
    // Pipeline depth of the Wiener divider in filter_controller. These two must
    // agree with DIV_STAGES there; the flush contract in the header is written
    // in terms of them, and the testbenches read them rather than hardcoding.
    localparam int PIPE_STAGES   = 8;
    localparam int LATENCY       = IMG_WIDTH + 3 + PIPE_STAGES;
    localparam int FLUSH_CYCLES  = IMG_WIDTH + 2 + PIPE_STAGES;

    // No internal buffering, so the source is never stalled.
    assign s_ready = 1'b1;

    // The controller runs for the whole flush: the last window still has to
    // travel the Wiener divider's PIPE_STAGES plus the output register.
    logic advance;
    assign advance = s_valid | s_flush;

    // The window generator must NOT. It finishes the frame after IMG_WIDTH+2
    // flush advances and then re-primes for the next one, so the extra
    // PIPE_STAGES advances the controller needs would prime it partway into a
    // frame that has not started and emit spurious valid pixels at the head of
    // the next one. (Symptom, before this counter existed: the first frame came
    // out right and every later frame was over-long by the surplus.)
    localparam int WIN_FLUSH = IMG_WIDTH + 2;
    localparam int FCW       = $clog2(FLUSH_CYCLES + 1);

    logic [FCW-1:0] flush_cnt;
    logic           win_flush;

    always_ff @(posedge clk) begin
        if (!rst_n)              flush_cnt <= '0;
        else if (!s_flush)       flush_cnt <= '0;        // between frames
        else if (flush_cnt < FCW'(FLUSH_CYCLES))
                                 flush_cnt <= flush_cnt + FCW'(1);
    end

    assign win_flush = s_flush && (flush_cnt < FCW'(WIN_FLUSH));

    // ── Window generator ──────────────────────────────────────────────────
    logic [WIN_SIZE*WIN_SIZE*DEPTH-1:0] win_flat;
    logic             win_valid;

    window_gen #(
        .WIDTH  (IMG_WIDTH),
        .HEIGHT (IMG_HEIGHT),
        .SIZE   (WIN_SIZE),
        .DEPTH  (DEPTH)
    ) u_win (
        .clk       (clk),
        .rst_n     (rst_n),
        .pixel_in  (s_pixel),
        .we        (s_valid),
        .flush     (win_flush),
        .win_out   (win_flat),
        .valid_out (win_valid)
    );

    // ── Filter controller ─────────────────────────────────────────────────
    // valid_in is win_valid alone. The previous version gated it with the
    // CURRENT-cycle s_valid, but win_valid already describes the window the
    // generator is presenting this cycle; ANDing the two shifted the qualifier
    // one cycle ahead of the data it qualified and silently dropped one output
    // pixel per stream. Stall handling belongs on the register enable, not on
    // the valid.
    filter_controller #(
        .DEPTH     (DEPTH),
        .NV_W      (NV_W)
    ) u_ctrl (
        .clk        (clk),
        .rst_n      (rst_n),
        .filter_sel (filter_sel),
        .noise_var  (noise_var),
        .win_flat   (win_flat),
        .valid_in   (win_valid),
        .en         (advance),
        .pixel_out  (m_pixel),
        .valid_out  (m_valid)
    );

    // ── Dropped-output detector ───────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n)                     m_overflow <= 1'b0;
        else if (m_valid && !m_ready)   m_overflow <= 1'b1;
    end

endmodule

`default_nettype wire
