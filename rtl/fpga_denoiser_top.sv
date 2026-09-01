// fpga_denoiser_top.sv — Phase 20
//
// Top-level FPGA denoising accelerator.
//
// Data path:
//   pixel_in (raster order) → window_gen → filter_controller → pixel_out
//
// PROTOCOL
//   Assert s_valid for exactly IMG_WIDTH*IMG_HEIGHT pixels in raster order,
//   then hold s_flush for IMG_WIDTH+2 cycles to drain the pipeline. m_valid is
//   then high for exactly IMG_WIDTH*IMG_HEIGHT cycles in total, one per input
//   pixel, each carrying the replicate-padded 3×3 result for that pixel.
//   The stream may stall (s_valid low) at any point; the pipeline holds.
//
// LATENCY
//   window_gen primes for IMG_WIDTH+2 advances, and filter_controller adds one
//   register stage, so the first m_valid appears IMG_WIDTH+3 advances after the
//   first pixel. (The previous header claimed IMG_WIDTH*(SIZE/2) + SIZE/2,
//   which was both the wrong formula and two cycles short.)
//
// filter_sel
//   The filter cores are combinational, so filter_sel applies to whichever
//   window is in the generator that cycle — which is the window centred
//   IMG_WIDTH+1 pixels BEHIND the pixel currently entering s_pixel. Changing
//   filter_sel mid-frame therefore takes effect on output pixels from that
//   cycle onward, not on input pixels from that cycle onward. Change it only
//   between frames unless that skew is what you want; the previous header
//   called this "safe" without qualifying it.
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
    parameter int NOISE_VAR  = 100,
    parameter int DEPTH      = 8,
    parameter int WIN_SIZE   = 3     // neighbourhood size (must be 3 here)
) (
    input  logic              clk,
    input  logic              rst_n,
    // Control
    input  logic [1:0]        filter_sel,
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
    // No internal buffering, so the source is never stalled.
    assign s_ready = 1'b1;

    logic advance;
    assign advance = s_valid | s_flush;

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
        .flush     (s_flush),
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
        .NOISE_VAR (NOISE_VAR)
    ) u_ctrl (
        .clk        (clk),
        .rst_n      (rst_n),
        .filter_sel (filter_sel),
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
