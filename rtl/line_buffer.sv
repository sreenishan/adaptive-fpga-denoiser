// line_buffer.sv — Phase 15
//
// Stores the last ROWS-1 pixel rows so that window_gen can present a
// ROWS×COLS neighbourhood for every output pixel.
//
// Interface
//   clk        — rising-edge clock
//   rst_n      — active-low synchronous reset
//   pixel_in   — 8-bit greyscale pixel, valid when we=1
//   we         — write enable (one pulse per input pixel)
//   row_out[k] — current row k delayed pixel (k=0 is the newest)
//
// The module instantiates (ROWS-1) shift-register stages, each WIDTH
// pixels long.  On every clock where we=1 a pixel advances through
// all stages.  After WIDTH×(ROWS-1) write pulses the last stage
// holds the pixel that arrived (ROWS-1) rows ago.

`default_nettype none

module line_buffer #(
    parameter int WIDTH  = 224,   // image width in pixels
    parameter int ROWS   = 3,     // neighbourhood height (must be ≥ 2)
    parameter int DEPTH  = 8      // pixel bit-width
) (
    input  logic               clk,
    input  logic               rst_n,   // unused: see the note on the always_ff
    input  logic [DEPTH-1:0]   pixel_in,
    input  logic               we,
    output logic [ROWS*DEPTH-1:0] row_out   // flat; row_out[k*DEPTH +: DEPTH] = row k
);
    // row_out[0*DEPTH +: DEPTH] is the live input; row_out[k*DEPTH +: DEPTH] is k rows delayed.
    // We implement ROWS-1 independent shift registers of depth WIDTH.

    logic [DEPTH-1:0] sr [0:ROWS-2][0:WIDTH-1];

    // Initialise all delay cells to zero. On a real FPGA, register and SRL
    // primitives power on to zero (guaranteed by Xilinx and Intel), so this
    // `initial` block matches hardware behaviour exactly. In simulation,
    // leaving the cells uninitialised produces x values that propagate through
    // window_gen's column registers and corrupt the first frame: the priming
    // period guarantees col[r][2] is written before it is read, but the older
    // stages (col[r][1], col[r][0]) still reflect the stale delay-line state
    // if it starts as x. The previous comment that no un-written cell is ever
    // read was wrong.
    initial begin
        for (int s = 0; s < ROWS-1; s++)
            for (int i = 0; i < WIDTH; i++)
                sr[s][i] = '0;
    end

    always_ff @(posedge clk) begin
        if (we) begin
            // Stage 0: shift pixel_in into the first delay line.
            sr[0][0] <= pixel_in;
            for (int i = 1; i < WIDTH; i++)
                sr[0][i] <= sr[0][i-1];

            // Stages 1..ROWS-2: cascade.
            for (int s = 1; s < ROWS-1; s++) begin
                sr[s][0] <= sr[s-1][WIDTH-1];
                for (int i = 1; i < WIDTH; i++)
                    sr[s][i] <= sr[s][i-1];
            end
        end
    end

    // Newest row is always the raw input (row 0).
    // Older rows tap the end of each delay stage.
    assign row_out[0*DEPTH +: DEPTH] = pixel_in;
    for (genvar k = 1; k < ROWS; k++) begin : gen_row_out
        assign row_out[k*DEPTH +: DEPTH] = sr[k-1][WIDTH-1];
    end

endmodule

`default_nettype wire
