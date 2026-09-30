# Hardware

Status: phase 1. **No board is selected, no synthesis has been run, and no
hardware has been programmed.** Every figure below is `TBD` and stays that way
until a tool report or a measurement exists.

## Target device

| Item | Value |
|---|---|
| Vendor | not selected |
| Device | not selected |
| Toolchain | not selected |
| Tool version | TBD |

`configs/hardware.yaml` holds `null` for all four. The core RTL is written to be
board-independent; anything board-specific belongs under `fpga/`.

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

**Latency and flush.** The Wiener divider is pipelined `PIPE_STAGES` = 8 deep,
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
| `wiener_filter` | 2,489 | 556 |
| `line_buffer` | 0 | 3,584 |
| `median_filter` | 415 | 0 |
| `window_gen` | 113 | 97 |
| `gaussian_filter` | 110 | 0 |
| `filter_controller` | 17 | 97 |
| `fpga_denoiser_top` | 15 | 9 |
| **Total** | **3,159** | **4,343** |

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
| Pipelined, 8 stages | **18.15 MHz** | 4,343 | 15 |

Generic LUTs fell too (3,608 -> 3,159), because registering the chain lets yosys
share logic it previously had to flatten. **This is an ECP5 number and the board
tables below are still `TBD`** — it is a like-for-like comparison of two versions
of this design on one part, not a claim about any target hardware.

`wiener_filter` is still 79% of the logic, so it remains the place to look.

**The remaining constant divide is deliberate.** `acc / 2304` could be the exact
reciprocal multiply `(acc * 233017) >> 29` — verified for every `acc` in the
reachable range 0..1,173,897 — but that measured **129 LUTs worse** here,
because a 24x18 multiply becomes LUT logic when the flow has no DSP blocks. On
a part with DSPs it should win. The constant and its derivation are in the
module comment; re-measure with the vendor tool rather than assuming either
way.

**The line buffer is 3,584 flip-flops here, and should not be on a real part.**
It is written as two 224-deep shift registers. In this generic flow yosys maps
them to discrete flops; Xilinx and Intel can map the same pattern to dedicated
shift-register primitives (SRL32 / ALTSHIFT_TAPS) or block RAM, which would be
far smaller. Whether they actually do is the first thing to check in a vendor
run — the `initial` block that zeroes the cells may prevent it.

The other filters are cheap: Gaussian is 110 LUTs (adds and shifts only) and
the median network 415.

## Resource utilisation

Board-specific, from a vendor fitter. Nothing has been fitted.

| Resource | Used | Available | Utilisation |
|---|---:|---:|---:|
| LUT | TBD | TBD | TBD |
| FF | TBD | TBD | TBD |
| BRAM | TBD | TBD | TBD |
| DSP | TBD | TBD | TBD |
| I/O | TBD | TBD | TBD |

## Timing

| Item | Value |
|---|---|
| Clock constraint | TBD |
| Achieved clock | TBD |
| WNS | TBD |
| TNS | TBD |

## Power

| Item | Value | Source |
|---|---|---|
| Static | TBD | - |
| Dynamic | TBD | - |
| Total | TBD | - |

Estimated and measured power are labelled separately and never mixed.
