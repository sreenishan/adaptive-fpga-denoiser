// fpga_denoiser_cascade.sv — two-pass FPGA denoising accelerator
//
// Chains two fpga_denoiser_top instances in series: the output of pass 1
// feeds directly into the input of pass 2, applying the same filter twice.
// This is the hardware implementation of the rtl_multipass path.
//
// PROTOCOL (caller perspective)
//   Same as fpga_denoiser_top except the flush window is slightly longer:
//
//     s_flush must be held high for CASCADE_FLUSH cycles:
//       = 2 * (IMG_WIDTH + 2 + PIPE_STAGES) + 1
//
//   The +1 gap cycle separates pass-1's last m_valid from pass-2's s_flush.
//   Pass 1 consumes the first SINGLE_FLUSH cycles; one gap cycle; then pass 2
//   consumes the next SINGLE_FLUSH cycles.
//
// TIMING
//   Pass 1 last m_valid fires on flush-cycle SINGLE_FLUSH-1 (the last p1_flush
//   cycle).  The gap (flushcnt == SINGLE_FLUSH) lets that pixel propagate to
//   pass-2's s_valid before p2_flush activates.  Without the gap, window_gen's
//   lb_in = flush ? row[1] : pixel_in would pick the replicated row over the
//   last real pixel.
//
// CONTROL
//   filter_sel and noise_var are applied identically to both passes.
//   Change them only between frames; the same constraint as the single-pass top.

`default_nettype none

module fpga_denoiser_cascade #(
    parameter int IMG_WIDTH  = 224,
    parameter int IMG_HEIGHT = 224,
    parameter int NV_W       = 16,
    parameter int DEPTH      = 8,
    parameter int WIN_SIZE   = 3
) (
    input  logic              clk,
    input  logic              rst_n,
    // Control — same as fpga_denoiser_top
    input  logic [2:0]        filter_sel,
    input  logic [NV_W-1:0]   noise_var,
    // AXI-Stream source
    input  logic              s_valid,
    input  logic [DEPTH-1:0]  s_pixel,
    input  logic              s_flush,   // hold CASCADE_FLUSH_CYCLES after last s_valid
    output logic              s_ready,
    // AXI-Stream sink
    output logic              m_valid,
    output logic [DEPTH-1:0]  m_pixel,
    input  logic              m_ready,
    output logic              m_overflow
);
    // Single-pass constants (must match fpga_denoiser_top)
    localparam int PIPE_STAGES    = 18;
    localparam int SINGLE_FLUSH   = IMG_WIDTH + 2 + PIPE_STAGES;
    // Pass 1's last m_valid fires on the same posedge that flushcnt would reach
    // SINGLE_FLUSH, so p2_flush must start one cycle later (at SINGLE_FLUSH+1)
    // to avoid pass-2's window_gen seeing both s_valid and s_flush simultaneously.
    // CASCADE_FLUSH = p1_flush(SINGLE_FLUSH) + gap(1) + p2_flush(SINGLE_FLUSH).
    localparam int CASCADE_FLUSH  = 2 * SINGLE_FLUSH + 1;
    localparam int FCW            = $clog2(CASCADE_FLUSH + 1);

    // ── Intermediate ──────────────────────────────────────────────────────────
    logic              p1_valid;
    logic [DEPTH-1:0]  p1_pixel;
    logic              p1_overflow;

    // ── Flush splitter ────────────────────────────────────────────────────────
    // flushcnt counts cycles while s_flush is high.
    // First SINGLE_FLUSH cycles → p1_flush  (drains pass 1).
    // Next  SINGLE_FLUSH cycles → p2_flush  (drains pass 2).
    logic [FCW-1:0] flushcnt;

    always_ff @(posedge clk) begin
        if (!rst_n)
            flushcnt <= '0;
        else if (!s_flush)
            flushcnt <= '0;
        else if (flushcnt < FCW'(CASCADE_FLUSH))
            flushcnt <= flushcnt + FCW'(1);
    end

    logic p1_flush, p2_flush;
    assign p1_flush = s_flush && (flushcnt <  FCW'(SINGLE_FLUSH));
    // Gap at flushcnt == SINGLE_FLUSH: neither flush active.
    // This ensures pass-2 sees the last s_valid (p1_valid) at flushcnt==SINGLE_FLUSH-1
    // before s_flush (p2_flush) activates at flushcnt==SINGLE_FLUSH+1.
    assign p2_flush = s_flush && (flushcnt >= FCW'(SINGLE_FLUSH + 1))
                               && (flushcnt <  FCW'(CASCADE_FLUSH));

    // ── Pass 1 ────────────────────────────────────────────────────────────────
    fpga_denoiser_top #(
        .IMG_WIDTH (IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NV_W      (NV_W),      .DEPTH     (DEPTH),
        .WIN_SIZE  (WIN_SIZE)
    ) u_pass1 (
        .clk        (clk),        .rst_n      (rst_n),
        .filter_sel (filter_sel), .noise_var  (noise_var),
        .s_valid    (s_valid),    .s_pixel    (s_pixel),
        .s_flush    (p1_flush),
        .s_ready    (/* tied 1'b1 inside */),
        .m_valid    (p1_valid),   .m_pixel    (p1_pixel),
        .m_ready    (1'b1),       .m_overflow (p1_overflow)
    );

    // ── Pass 2 ────────────────────────────────────────────────────────────────
    fpga_denoiser_top #(
        .IMG_WIDTH (IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .NV_W      (NV_W),      .DEPTH     (DEPTH),
        .WIN_SIZE  (WIN_SIZE)
    ) u_pass2 (
        .clk        (clk),        .rst_n      (rst_n),
        .filter_sel (filter_sel), .noise_var  (noise_var),
        .s_valid    (p1_valid),   .s_pixel    (p1_pixel),
        .s_flush    (p2_flush),
        .s_ready    (/* tied 1'b1 inside */),
        .m_valid    (m_valid),    .m_pixel    (m_pixel),
        .m_ready    (m_ready),    .m_overflow (m_overflow)
    );

    assign s_ready = 1'b1;   // no back-pressure in either pass

endmodule
