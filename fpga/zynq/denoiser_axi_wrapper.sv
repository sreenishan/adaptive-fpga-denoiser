// fpga/zynq/denoiser_axi_wrapper.sv
//
// AXI4-Lite + AXI4-Stream wrapper around fpga_denoiser_top for Zynq-7000.
//
// Register map (byte addresses, 32-bit registers):
//   0x00  ctrl    [2:0] filter_sel   — 0=bypass 1=median 2=gaussian
//                                       3=wiener 4=adaptive_median
//   0x04  noise   [15:0] noise_var   — Wiener noise power (squared grey levels)
//   0x08  status  [0]   overflow     — sticky; write any value to clear
//                [31:16] frame_cnt   — frames processed since reset
//
// Data flow:
//   1. PS writes filter_sel + noise_var via AXI4-Lite before each frame.
//   2. PS DMA sends N=IMG_WIDTH*IMG_HEIGHT bytes to S_AXIS; TLAST on last byte.
//   3. Wrapper feeds denoiser, handles flush, asserts TLAST on last output byte.
//   4. PS DMA captures M_AXIS output into DDR (denoised frame).
//
// Back-pressure: fpga_denoiser_top has none (s_ready is always 1, m_ready is
// advisory).  Downstream stalls on M_AXIS cause m_overflow (visible in status).
//
// AXI4-Lite write note: this implementation requires AW and W to be presented
// in the same cycle (standard for Xilinx AXI-Lite masters).

`default_nettype none

module denoiser_axi_wrapper #(
    parameter int IMG_WIDTH   = 224,
    parameter int IMG_HEIGHT  = 224,
    parameter int PIPE_STAGES = 18   // must match fpga_denoiser_top localparam
) (
    input  logic        aclk,
    input  logic        aresetn,

    // ── AXI4-Lite slave (PS configuration, GP0 port) ─────────────────────
    input  logic [3:0]  s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [3:0]  s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    // ── AXI4-Stream slave (pixels in, from DMA MM2S) ─────────────────────
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    // ── AXI4-Stream master (pixels out, to DMA S2MM) ─────────────────────
    output logic [7:0]  m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast
);

    // ── Derived constants ─────────────────────────────────────────────────
    localparam int N           = IMG_WIDTH * IMG_HEIGHT;
    localparam int FLUSH_CYC   = IMG_WIDTH + 2 + PIPE_STAGES;
    localparam int IN_W        = $clog2(N);
    localparam int OUT_W       = $clog2(N);
    localparam int FLUSH_W     = $clog2(FLUSH_CYC + 1);

    // ── AXI4-Lite registers ───────────────────────────────────────────────
    logic [2:0]  reg_filter_sel;
    logic [15:0] reg_noise_var;
    logic [15:0] reg_frame_cnt;

    // Write path — AW and W accepted together (Xilinx master sends simultaneously)
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready  <= 1'b1;
            s_axi_wready   <= 1'b1;
            s_axi_bvalid   <= 1'b0;
            s_axi_bresp    <= 2'b00;
            reg_filter_sel <= 3'd0;
            reg_noise_var  <= 16'd0;
        end else begin
            s_axi_awready <= 1'b1;
            s_axi_wready  <= 1'b1;

            if (s_axi_awvalid && s_axi_awready &&
                s_axi_wvalid  && s_axi_wready) begin
                case (s_axi_awaddr[3:2])
                    2'h0: reg_filter_sel <= s_axi_wdata[2:0];
                    2'h1: reg_noise_var  <= s_axi_wdata[15:0];
                    // 0x08 write clears overflow (handled below)
                    default: ;
                endcase
                s_axi_bvalid <= 1'b1;
            end

            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    // Read path
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b1;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            s_axi_arready <= 1'b1;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[3:2])
                    2'h0: s_axi_rdata <= {29'b0, reg_filter_sel};
                    2'h1: s_axi_rdata <= {16'b0, reg_noise_var};
                    2'h2: s_axi_rdata <= {reg_frame_cnt, 15'b0, overflow_r};
                    default: s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    // ── Overflow register (sticky; cleared by any write to 0x08) ─────────
    logic overflow_r;
    logic m_overflow_w;  // from denoiser

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            overflow_r <= 1'b0;
        end else begin
            if (m_overflow_w)
                overflow_r <= 1'b1;
            if (s_axi_awvalid && s_axi_awready &&
                s_axi_wvalid  && s_axi_wready  &&
                s_axi_awaddr[3:2] == 2'h2)
                overflow_r <= 1'b0;
        end
    end

    // ── Pixel state machine ───────────────────────────────────────────────
    // ST_READY : accepting pixels from S_AXIS
    // ST_FLUSH : flushing pipeline after last pixel
    typedef enum logic { ST_READY = 1'b0, ST_FLUSH = 1'b1 } state_t;
    state_t state;

    logic [IN_W-1:0]    in_cnt;
    logic [OUT_W-1:0]   out_cnt;
    logic [FLUSH_W-1:0] flush_cnt;

    // Pixel accepted this cycle
    logic pix_accept;
    assign pix_accept = s_axis_tvalid && s_axis_tready;

    // Denoiser inputs
    logic d_valid, d_flush;
    logic [7:0] d_pixel;

    always_comb begin
        s_axis_tready = (state == ST_READY);
        d_valid       = pix_accept;
        d_flush       = (state == ST_FLUSH);
        d_pixel       = s_axis_tdata;
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            state     <= ST_READY;
            in_cnt    <= '0;
            flush_cnt <= '0;
            reg_frame_cnt <= '0;
        end else begin
            case (state)
                ST_READY: begin
                    if (pix_accept) begin
                        if (in_cnt == IN_W'(N - 1)) begin
                            // Last pixel — start flush
                            state     <= ST_FLUSH;
                            flush_cnt <= '0;
                            in_cnt    <= '0;
                        end else begin
                            in_cnt <= in_cnt + 1'd1;
                        end
                    end
                end

                ST_FLUSH: begin
                    if (flush_cnt == FLUSH_W'(FLUSH_CYC - 1)) begin
                        state         <= ST_READY;
                        flush_cnt     <= '0;
                        reg_frame_cnt <= reg_frame_cnt + 1'd1;
                    end else begin
                        flush_cnt <= flush_cnt + 1'd1;
                    end
                end
            endcase
        end
    end

    // ── Output pixel counter → TLAST ─────────────────────────────────────
    logic d_m_valid;
    logic [7:0] d_m_pixel;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            out_cnt <= '0;
        end else begin
            if (d_m_valid) begin
                if (out_cnt == OUT_W'(N - 1))
                    out_cnt <= '0;
                else
                    out_cnt <= out_cnt + 1'd1;
            end
        end
    end

    assign m_axis_tdata  = d_m_pixel;
    assign m_axis_tvalid = d_m_valid;
    assign m_axis_tlast  = d_m_valid && (out_cnt == OUT_W'(N - 1));

    // ── Denoiser core ─────────────────────────────────────────────────────
    fpga_denoiser_top #(
        .IMG_WIDTH  (IMG_WIDTH),
        .IMG_HEIGHT (IMG_HEIGHT)
    ) u_core (
        .clk        (aclk),
        .rst_n      (aresetn),
        .filter_sel (reg_filter_sel),
        .noise_var  (reg_noise_var),
        .s_valid    (d_valid),
        .s_pixel    (d_pixel),
        .s_flush    (d_flush),
        .s_ready    (/* always 1 */),
        .m_valid    (d_m_valid),
        .m_pixel    (d_m_pixel),
        .m_ready    (m_axis_tready),
        .m_overflow (m_overflow_w)
    );

endmodule
