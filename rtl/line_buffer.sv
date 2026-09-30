// line_buffer.sv — Phase 15
//
// Stores the last ROWS-1 pixel rows so that window_gen can present a
// ROWS×COLS neighbourhood for every output pixel.
//
// Interface
//   clk        — rising-edge clock
//   rst_n      — active-low synchronous reset (resets write pointer only)
//   pixel_in   — 8-bit greyscale pixel, valid when we=1
//   we         — write enable (one pulse per input pixel)
//   row_out[k] — row k delayed by k*WIDTH write pulses; k=0 is the live input
//
// Implementation: each delay stage is a WIDTH-deep circular buffer backed by a
// synchronous dual-port memory so yosys synth_ecp5 can infer DP16KD BRAM.
// (The earlier shift-register version had an `initial` block that zeroed all
// cells — that is the main obstacle to BRAM inference in yosys, hence the
// rewrite.)
//
// Timing: the read address presented at cycle N is wp_next = (wp+1)%WIDTH,
// i.e. the address we will overwrite NEXT cycle.  That slot was last written
// WIDTH cycles ago, so the registered BRAM output at cycle N+1 holds the value
// from WIDTH advances back — identical to the former sr[s][WIDTH-1] tap.
// When window_gen's always_ff fires at posedge N it reads s_out (which is still
// the value registered at posedge N-1, before this cycle's update), giving the
// same pixel as the shift register would have.
//
// Cascade: stage s (s≥2) takes row_out[(s-1)*DEPTH +: DEPTH] as its write
// data.  That is s_out of stage s-1, read at posedge N before the current
// non-blocking update — exactly the datum that was WIDTH cycles old for that
// stage, so the total delay is correct.
//
// Initialization: no `initial` block.  ECP5 BRAMs power on to zero.
// In simulation, reads during the priming period (first WIDTH*(ROWS-1) we
// pulses) return X, but window_gen's valid_out is low then so they never
// reach the output.

`default_nettype none

module line_buffer #(
    parameter int WIDTH  = 224,   // image width in pixels
    parameter int ROWS   = 3,     // neighbourhood height (must be ≥ 2)
    parameter int DEPTH  = 8      // pixel bit-width
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [DEPTH-1:0]   pixel_in,
    input  logic               we,
    output logic [ROWS*DEPTH-1:0] row_out   // flat; row_out[k*DEPTH +: DEPTH] = row k
);
    localparam int AW = $clog2(WIDTH);

    // Row 0 is always the live input (combinatorial, zero latency).
    assign row_out[0*DEPTH +: DEPTH] = pixel_in;

    // Write pointer shared by all stages — they all advance on every we pulse.
    logic [AW-1:0] wp;
    logic [AW-1:0] wp_next;

    always_comb wp_next = (wp == AW'(WIDTH-1)) ? '0 : AW'(wp + 1);

    always_ff @(posedge clk) begin
        if (!rst_n) wp <= '0;
        else if (we) wp <= wp_next;
    end

    // One BRAM stage per delayed row.
    //
    //   Write port: address wp, data = row_out from the previous row
    //               (row_out[0] = pixel_in for s=1; row_out[s-1] for s>1).
    //   Read port:  address wp_next — the slot about to be overwritten, which
    //               holds the value stored WIDTH advances ago.
    //
    // Both ports are gated by we (maps to clock-enable pins CEA/CEB on DP16KD).
    // Separate always_ff blocks are required for yosys memory_bram inference.
    for (genvar s = 1; s < ROWS; s++) begin : g_stage
        logic [DEPTH-1:0] mem   [0:WIDTH-1];
        logic [DEPTH-1:0] s_out;

        always_ff @(posedge clk)
            if (we) mem[wp] <= row_out[(s-1)*DEPTH +: DEPTH];

        always_ff @(posedge clk)
            if (we) s_out <= mem[wp_next];

        assign row_out[s*DEPTH +: DEPTH] = s_out;
    end

endmodule

`default_nettype wire
