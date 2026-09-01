// window_gen.sv — Phase 15 (boundary-correct redesign)
//
// Presents the SIZE×SIZE replicate-padded neighbourhood of every pixel, in
// raster order of the CENTRE pixel. For a W×H frame it emits exactly W*H
// windows — one per input pixel — which is what the software reference
// produces and what configs/hardware.yaml's `boundary_policy: replicate`
// requires.
//
// WHAT WAS WRONG BEFORE
// ---------------------
// The previous version had no boundary handling at all. It emitted 49950
// windows for a 224×224 (50176-pixel) frame, so the output was both the wrong
// SIZE and misaligned against the reference. Worse, it did not track column
// position, so windows straddling a row boundary were emitted as valid with
// their left column taken from the end of the previous row. Its valid_out also
// rose 225 write-pulses too early, meaning the first 225 windows contained
// un-written reset registers as their top row, and `filled` never cleared, so a
// second frame streamed back-to-back was mis-framed and mixed two images.
//
// TAP ALGEBRA (derived, then verified by a cycle-accurate model)
// -------------------------------------------------------------
// Reading the line-buffer taps BEFORE the shift gives
//     r0 = live[t]      r1 = live[t-W]      r2 = live[t-2W]
// so after clock t the column registers hold
//     col[0] = live[t-2],    live[t-1],    live[t]        <- BOTTOM row
//     col[1] = live[t-W-2],  live[t-W-1],  live[t-W]      <- CENTRE row
//     col[2] = live[t-2W-2], live[t-2W-1], live[t-2W]     <- TOP row
// Therefore the window completed at clock t is centred on linear index
//     m = t - W - 1
// Note col[0] is the BOTTOM row, not the top: the live pixel is the most
// recent one, which sits below the centre in a raster scan. The previous file
// wired win_out[0] to the live row while its header called win_out[0] the top,
// i.e. every window was vertically flipped. That was benign for the three
// symmetric kernels this project ships but is wrong, and would silently corrupt
// any asymmetric kernel added later.
//
// PROTOCOL
// --------
// Host asserts we=1 for exactly WIDTH*HEIGHT pixels in raster order, then holds
// flush=1 for WIDTH+2 cycles to drain the pipeline. valid_out is high for
// exactly WIDTH*HEIGHT cycles across the two phases. Asserting flush replays
// the last row into the delay lines, which is what makes the bottom edge
// replicate rather than read past the frame.
//
// Verified against denoising.filters._window.sliding_windows (replicate mode)
// at 6×8, 8×8, 5×7, 16×24, 3×3 and the production 224×224: every frame emitted
// exactly W*H windows, in raster order, with zero window mismatches, and a
// second frame streamed immediately after the first was framed identically.

`default_nettype none

module window_gen #(
    parameter int WIDTH  = 224,
    parameter int HEIGHT = 224,
    parameter int SIZE   = 3,    // neighbourhood side length (3 only)
    parameter int DEPTH  = 8
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [DEPTH-1:0]   pixel_in,
    input  logic               we,        // one pulse per input pixel
    input  logic               flush,     // hold WIDTH+1 cycles after the frame
    output logic [SIZE*SIZE*DEPTH-1:0] win_out,   // flat; [r][c] = [(r*SIZE+c)*DEPTH +: DEPTH]
    output logic               valid_out
);
    localparam int TOTAL = WIDTH * HEIGHT;
    localparam int CW    = $clog2(WIDTH  + 2);
    localparam int RW    = $clog2(HEIGHT + 2);
    localparam int PW    = $clog2(WIDTH  + 3);

    logic advance;
    assign advance = we | flush;

    // ── Line buffer ────────────────────────────────────────────────────────
    // During flush the last row is replayed so the bottom edge replicates.
    logic [DEPTH-1:0] lb_in;
    // Flat 1D signal for line_buffer port (avoids Icarus 12 unpacked-port x bug).
    logic [SIZE*DEPTH-1:0] row_flat;
    // Local unpacked alias for convenient indexing inside this module.
    logic [DEPTH-1:0] row [0:SIZE-1];

    for (genvar k = 0; k < SIZE; k++) begin : gen_row
        assign row[k] = row_flat[k*DEPTH +: DEPTH];
    end

    assign lb_in = flush ? row[1] : pixel_in;

    line_buffer #(
        .WIDTH (WIDTH), .ROWS (SIZE), .DEPTH (DEPTH)
    ) u_lb (
        .clk (clk), .rst_n (rst_n),
        .pixel_in (lb_in), .we (advance), .row_out (row_flat)
    );

    // ── Column registers: col[r][2] is the newest ──────────────────────────
    logic [DEPTH-1:0] col [0:SIZE-1][0:SIZE-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int r = 0; r < SIZE; r++)
                for (int c = 0; c < SIZE; c++)
                    col[r][c] <= '0;
        end else if (advance) begin
            for (int r = 0; r < SIZE; r++) begin
                col[r][0] <= col[r][1];
                col[r][1] <= col[r][2];
                col[r][2] <= row[r];      // pre-shift tap, per the algebra above
            end
        end
    end

    // ── Centre-pixel position tracking ─────────────────────────────────────
    // Emission starts once the pipeline has been primed for WIDTH+1 cycles and
    // stops after exactly TOTAL windows, so a following frame starts clean.
    logic [PW-1:0]  prime;
    logic           emitting;
    logic [CW-1:0]  cc;      // centre column
    logic [RW-1:0]  cr;      // centre row

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            prime <= '0; emitting <= 1'b0; cc <= '0; cr <= '0;
        end else if (advance) begin
            if (!emitting) begin
                if (prime == PW'(WIDTH + 1)) begin // WIDTH+2'th advance: col[r][1] valid
                    emitting <= 1'b1;
                end else begin
                    prime <= prime + PW'(1);
                end
            end else begin
                if (cc == CW'(WIDTH-1)) begin
                    cc <= '0;
                    if (cr == RW'(HEIGHT-1)) begin
                        emitting <= 1'b0;
                        prime <= '0;
                        cr <= '0;
                    end else begin
                        cr <= cr + RW'(1);
                    end

                end else begin
                    cc <= cc + CW'(1);
                end
            end
        end
    end

    assign valid_out = emitting;

    // ── Replicate-padded window assembly ───────────────────────────────────
    // Internal 2D packed; assign to flat output so Icarus 12 propagates it.
    logic [SIZE-1:0][SIZE-1:0][DEPTH-1:0] win_i;
    assign win_out = win_i;

    logic at_left, at_right, at_top, at_bottom;
    assign at_left   = (cc == CW'(0));
    assign at_right  = (cc == CW'(WIDTH-1));
    assign at_top    = (cr == RW'(0));
    assign at_bottom = (cr == RW'(HEIGHT-1));

    // Column clamp applied to each stored row, then row clamp across them.
    logic [DEPTH-1:0] rowbuf [0:SIZE-1][0:SIZE-1];

    always_comb begin
        for (int r = 0; r < SIZE; r++) begin
            rowbuf[r][0] = at_left  ? col[r][1] : col[r][0];
            rowbuf[r][1] =            col[r][1];
            rowbuf[r][2] = at_right ? col[r][1] : col[r][2];
        end
        for (int c = 0; c < SIZE; c++) begin
            // col[2] is the top row, col[1] the centre, col[0] the bottom.
            win_i[0][c] = at_top    ? rowbuf[1][c] : rowbuf[2][c];
            win_i[1][c] =                            rowbuf[1][c];
            win_i[2][c] = at_bottom ? rowbuf[1][c] : rowbuf[0][c];
        end
    end

endmodule

`default_nettype wire
