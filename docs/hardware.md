# Hardware

Status: phase 1. **No board is selected, no synthesis has been run, and no
hardware has been programmed.** Every figure below is `TBD` and stays that way
until a tool report or a measurement exists.

## Target device

| Item | Value |
|---|---|
| Vendor | Lattice |
| Device | LFE5U-25F-6BG381C (ECP5-25k, speed grade 6, CABGA381) |
| Toolchain | yosys 0.57 + nextpnr-ecp5 0.9-2 (oss-cad-suite) |
| Constraint file | `fpga/constraints/ecp5_25k.lpf` |

`configs/hardware.yaml` records the vendor and device. The core RTL is written to
be board-independent; anything board-specific belongs under `fpga/`. No board
has been programmed — these are place-and-route results from
`scripts/place_and_route.py`, not measurements on silicon.

## Pixel stream interface

| Signal | Direction | Width | Meaning |
|---|---|---|---|
These are the ports `rtl/fpga_denoiser_top.sv` actually has; the table used to
list a planned interface (`in_valid`, `frame_start`, `line_start`) that the
module never grew.

| `clk` | in | 1 | pixel clock |
| `rst_n` | in | 1 | synchronous reset, active LOW |
| `s_valid` | in | 1 | `s_pixel` is valid this cycle |
| `s_pixel` | in | 8 | unsigned 0-255, raster order |
| `s_flush` | in | 1 | hold for `FLUSH_CYCLES` = IMG_WIDTH+2+PIPE_STAGES cycles after a frame to drain |
| `s_ready` | out | 1 | sink can accept (always high; no back-pressure) |
| `m_valid` | out | 1 | `m_pixel` is valid this cycle |
| `m_pixel` | out | 8 | unsigned 0-255 |
| `m_ready` | in | 1 | accepted only to detect a dropped pixel, see `m_overflow` |
| `m_overflow` | out | 1 | sticky: a pixel was produced while `m_ready` was low |
| `filter_sel` | in | 2 | `00` bypass, `01` median, `10` Gaussian, `11` Wiener |
| `noise_var` | in | 16 | Wiener noise power in squared grey levels, written per frame |

`filter_sel` and `noise_var` are combinational control: they apply to the window
in the generator that cycle, which trails the input pixel by IMG_WIDTH+1. Change
them between frames.

**Latency and flush.** The Wiener divider is pipelined `PIPE_STAGES` = 10 deep
(8 restoring steps + 1 pre-stage register + 1 output-accumulator register),
so the first `m_valid` appears `LATENCY` = IMG_WIDTH+3+PIPE_STAGES advances
after the first pixel, and the drain is `FLUSH_CYCLES` = IMG_WIDTH+2+PIPE_STAGES.
Both are localparams in `rtl/fpga_denoiser_top.sv`; read them rather than
hardcoding, because a caller still flushing for IMG_WIDTH+2 comes up
PIPE_STAGES pixels short. **Throughput is unchanged at one pixel per cycle** —
the divider is pipelined, not iterative.

Inside the top, the window generator is flushed for only IMG_WIDTH+2 of those
cycles. It counts its own frame and re-primes afterwards, so the extra
PIPE_STAGES advances the controller needs would otherwise prime it partway into
a frame that has not started and emit spurious valid pixels at the head of the
next one — which is exactly what `tb_fpga_denoiser_top` caught: the first frame
came out right and every later one was over-long.

`backpressure` is `false` in the configuration: the first implementation is a
valid-only stream. `in_ready` / `out_ready` are added later, and the config flag
is what says which of the two is built.

## Fixed-point formats

| Quantity | Format |
|---|---|
| Pixel | unsigned 8-bit, 0-255 |
| Gaussian accumulator | unsigned 13-bit (max 255 x 16 = 4080) |
| Wiener window sum | unsigned 12-bit (max 9 x 255 = 2295) |
| Wiener sum of squares | unsigned 20-bit (max 9 x 255^2 = 585225) |
| Wiener 81*variance | unsigned 24-bit, exact (9*SUM(x^2) - (SUM x)^2) |
| Wiener gain | Q8 unsigned, clamped to [0, 255] |
| Wiener accumulator | signed 24-bit |
| `noise_var` | unsigned 16-bit (255^2 = 65025 is the largest meaningful value) |

## Generic synthesis (vendor-neutral, yosys)

`python scripts/synthesize_rtl.py` maps the design to generic 6-input LUTs and
flip-flops with yosys 0.69. **This is not a fitter result for any part** — no
timing, no vendor primitives, no placement — so the board tables below stay
`TBD`. It answers two questions a simulator cannot: does the RTL synthesise at
all, and where does the logic go.

| Module | LUTs | FFs |
|---|---:|---:|
| `wiener_filter` | 3,287 | 636 |
| `line_buffer` | 0 | 3,584 |
| `median_filter` | 408 | 0 |
| `window_gen` | 112 | 97 |
| `gaussian_filter` | 112 | 0 |
| `filter_controller` | 17 | 119 |
| `fpga_denoiser_top` | 15 | 9 |
| **Total** | **3,951** | **4,445** |

Two things to know before choosing a part:

**The Wiener gain divider has been replaced, and it was worth 1,252 LUTs.**
It was a 32/24-bit division by a variable, evaluated for every pixel, and at
4,211 LUTs it was 87% of the design. It computed 32 quotient bits where the
result is clamped to 8, so it is now eight restoring steps producing those
eight bits directly: `wiener_filter` 4,211 -> 2,959 LUTs, design total
4,860 -> 3,608 (-26%). That is a cost reduction, not an approximation — the two
compute the same number, proved over 302,091 (num, den) pairs in Python and
1,352,104 side-by-side RTL vectors, so the 1 grey level budget is untouched.

**Those eight steps are now pipelined, and it was worth 2.3x the clock.** They
were eight 25-bit compare-subtracts in series, behind the window sums and two
multipliers and in front of another multiply and the final divide — one pixel's
entire computation in a single combinational cone. Each step now ends in a
register, so eight pixels are in flight at once and **throughput is unchanged**;
the cost is eight cycles of latency and 652 flip-flops, and the streaming
protocol changed with it (see *Latency and flush* above).

Measured by placing and routing the design on an ECP5-25k, the only flow here
that reports timing:

```
nextpnr-ecp5 --25k --package CABGA381 --lpf-allow-unconstrained --freq 100
```

| | Fmax | TRELLIS_FF | MULT18X18D |
|---|---:|---:|---:|
| Combinational divider | 7.97 MHz | 3,691 | 15 |
| Pipelined, 8 stages | 18.15 MHz | 4,343 | 15 |
| Pipelined + reciprocal multiply | **30.23 MHz** | 4,445 | 17 |

The third row adds a pre-stage register (breaks the 25 ns window-sum → step-0
input path) and an output-accumulator register plus reciprocal multiply (breaks
the 49 ns `c_r[8]` → multiply → carry chain output path). Both are exact — the
reciprocal multiply `floor(acc / 2304) = (acc × 233017) >> 29` is verified
exhaustively over every `acc` in 0..1,173,897 — so the 1 grey level budget is
unchanged. Total latency is now 10 cycles (8 restoring + 2 extra registers).
**This is an ECP5 number** — it is a like-for-like comparison on one part, not a
claim about any target hardware.

The current critical path (33.08 ns) is no longer in the Wiener filter: it is
now the filter_controller combinational mux (gaussian/median/bypass output)
passing through a CCU2C carry cell before registering into `comb_d[1]`.

**The reciprocal multiply uses 2 extra MULT18X18D blocks** (15 → 17 = 60% of
28) and costs **2 extra flip-flops** in Wiener. The generic LUT count is higher
(3,951 vs 3,159) because a 43×18 multiply is LUT logic without DSPs; on ECP5
nextpnr maps it to two MULT18X18D, which is why the ECP5 LUT count dropped.

**The line buffer is 3,584 flip-flops here, and should not be on a real part.**
It is written as two 224-deep shift registers. In this generic flow yosys maps
them to discrete flops; Xilinx and Intel can map the same pattern to dedicated
shift-register primitives (SRL32 / ALTSHIFT_TAPS) or block RAM, which would be
far smaller. Whether they actually do is the first thing to check in a vendor
run — the `initial` block that zeroes the cells may prevent it.

The other filters are cheap: Gaussian is 110 LUTs (adds and shifts only) and
the median network 415.

## Resource utilisation

From `scripts/place_and_route.py` on ECP5-25k (LFE5U-25F-6BG381C), yosys 0.57 +
nextpnr-ecp5 0.9-2, seed 1.  No board has been programmed.

| Resource | Used | Available | Utilisation |
|---|---:|---:|---:|
| LUT4 | 1,835 | 24,288 | 7% |
| TRELLIS_FF | 4,445 | 24,288 | 18% |
| DP16KD (BRAM18) | 0 | 56 | 0% |
| MULT18X18D (DSP) | 17 | 28 | 60% |

**The line buffer is still discrete flip-flops (3,584 of the 4,445 FFs).**
It is written as two 224-deep shift registers; nextpnr did not infer block RAM
for them, exactly as the generic-synthesis note warned.  Replacing the
shift registers with an explicit BRAM instantiation would free ~3,500 FFs and
move the line buffer to BRAM (2 × 18 kb blocks) — a straightforward
improvement for a future commit.

**The 17 MULT18X18D blocks are 60% of the available DSPs.**  15 come from the
Wiener filter's mean and variance arithmetic; 2 are the reciprocal-multiply
for `acc / 2304`.  That is acceptable on a 25k part but would be the first
thing to check on a smaller device.

## Timing

From nextpnr-ecp5 0.9-2, `--freq 100 --lpf-allow-unconstrained`, seed 1.

| Item | Value |
|---|---|
| Clock target | 100 MHz |
| Achieved Fmax | 30.23 MHz |
| Timing closure | FAIL |
| Critical path | filter_controller mux → CCU2C carry cell → `comb_d[1]` (~33 ns) |

The design does not close timing at 100 MHz.  The critical path has moved out
of the Wiener filter: it is now the combinational bypass/median/Gaussian result
mux in filter_controller passing through a CCU2C carry adder before registering
into `comb_d[1]`.  The `--lpf-allow-unconstrained` flag means the clock enters
through a general I/O cell; a dedicated clock pin (LOCATE COMP "clk" SITE "...")
would reduce I/O overhead but would not fix the combinational depth.

**Next steps to improve Fmax** (in order of likely impact):
1. Register the filter_controller output mux into an intermediate stage to break
   the 33 ns CCU2C carry-chain path.
2. Use a dedicated clock pin via a LOCATE constraint in `ecp5_25k.lpf`.
3. Target a 45k or 85k ECP5 with a higher speed grade.

## Power

No power measurement is available without a programmed board.  nextpnr-ecp5
does not produce a power report; ecppack produces a bitstream only.

| Item | Value | Source |
|---|---|---|
| Static | TBD | — |
| Dynamic | TBD | — |
| Total | TBD | — |

Estimated and measured power are labelled separately and never mixed.
