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

**Latency and flush.** The filter_controller pipeline is `PIPE_STAGES` = 13 deep
(1 win-input + 1 gaussian-row + 1 comb-pre + 10 delay-chain/Wiener stages), so
the first `m_valid` appears `LATENCY` = IMG_WIDTH+3+PIPE_STAGES advances after
the first pixel, and the drain is `FLUSH_CYCLES` = IMG_WIDTH+2+PIPE_STAGES.
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
| `wiener_filter` | 2,932 | 636 |
| `line_buffer` | 1,447 | 3,608 |
| `median_filter` | 408 | 0 |
| `window_gen` | 112 | 97 |
| `gaussian_filter` | 112 | 0 |
| `filter_controller` | 17 | 119 |
| `fpga_denoiser_top` | 15 | 9 |
| **Total** | **5,043** | **4,469** |

Note: `line_buffer` maps to LUT RAM in generic synthesis (no DP16KD available).
On ECP5 (`synth_ecp5`) yosys infers two DP16KD blocks and the line buffer
contributes negligible LUTs and FFs — see the Resource utilisation table.

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

| | Fmax | TRELLIS_FF | DP16KD | MULT18X18D |
|---|---:|---:|---:|---:|
| Combinational divider | 7.97 MHz | 3,691 | 0 | 15 |
| Pipelined, 8 stages | 18.15 MHz | 4,343 | 0 | 15 |
| Pipelined + reciprocal multiply | 30.23 MHz | 4,445 | 0 | 17 |
| + BRAM line buffer | 30.91 MHz | 869 | 2 | 17 |
| + filter_controller pre-register | 29.87 MHz | 888 | 2 | 17 |
| + filter_controller win-input register | 36.51 MHz | 963 | 2 | 17 |
| + gaussian accumulator pipeline | **36.65 MHz** | **1,015** | **2** | 17 |

The third row adds a pre-stage register (breaks the 25 ns window-sum → step-0
input path) and an output-accumulator register plus reciprocal multiply (breaks
the 49 ns `c_r[8]` → multiply → carry chain output path). Both are exact — the
reciprocal multiply `floor(acc / 2304) = (acc × 233017) >> 29` is verified
exhaustively over every `acc` in 0..1,173,897 — so the 1 grey level budget is
unchanged. Wiener total latency is now 10 cycles (8 restoring + 2 extra registers).

The fifth row adds a comb pre-register (`comb_r`) in filter_controller. The old
critical path (gaussian carry chain → CCU2C → comb_d[1], 33 ns) was cut, but
the path shifted to window_gen FF → gaussian accumulator → comb_r — same
combinational depth. Fmax essentially unchanged (29.87 vs 30.91 MHz, −3%).

The sixth row adds a win-input register (`win_r`) in filter_controller.
Registering `win_flat` (and `filter_sel`/`valid_in`) before the filter cores
cuts the window_gen counter propagation (~3.5 ns of routing and LUT logic) from
the critical path. The new critical path starts from `win_r` Q directly into the
gaussian/median accumulator, without the window_gen overhead. Fmax: 29.87 →
**36.51 MHz** (+22%). PIPE_STAGES = 12 throughout the design. TRELLIS_FF: 888 → 963 (+75).

The seventh row pipelines the gaussian accumulator: `gaussian_filter` gains a
clock port and a 1-cycle register splitting the 9-input adder tree into row sums
(comb) → register → final sum (comb). filter_controller adds Stage G registers
(median_r, centre_r, sel_wr2, vld_wr2) and a second wiener output register
(wiener_r2). Fmax: 36.51 → **36.65 MHz** (+0.4% — routing noise). The gaussian
adder tree was not the critical path: nextpnr reports it as `win_r` Q →
`median_px` CCU2C carry chain → `comb_r`. filter_controller total latency is now
14 cycles (1 win-input + 1 gaussian-row + 1 comb-pre + 10 delay-chain stages + 1
output register); PIPE_STAGES = 13 throughout the design. TRELLIS_FF: 963 → 1,015
(+52 for r0_q/r1_q/r2_q + median_r/centre_r + sel_wr2/vld_wr2 + wiener_r2).
LUT4: 1,853 → 1,816 (−37, minor reduction from improved synthesis of split tree).

**This is an ECP5 number** — it is a like-for-like comparison on one part, not a
claim about any target hardware.

The current critical path (~27 ns) is win_r Q → gaussian/median accumulator
carry chain → comb_px mux → comb_r setup.

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
| LUT4 | 1,816 | 24,288 | 7% |
| TRELLIS_FF | 1,015 | 24,288 | 4% |
| DP16KD (BRAM18) | 2 | 56 | 3% |
| MULT18X18D (DSP) | 17 | 28 | 60% |

**The line buffer is now in block RAM.**  The two 224-deep shift registers are
rewritten as circular buffers (synchronous dual-port memory); yosys `synth_ecp5`
infers two DP16KD blocks.  TRELLIS_FF dropped from 4,445 to 869 (−3,576 FFs,
−80%).  Only 3 of the 56 BRAM18 blocks are used.

**The 17 MULT18X18D blocks are 60% of the available DSPs.**  15 come from the
Wiener filter's mean and variance arithmetic; 2 are the reciprocal-multiply
for `acc / 2304`.  That is acceptable on a 25k part but would be the first
thing to check on a smaller device.

## Timing

From nextpnr-ecp5 0.9-2, `--freq 100 --lpf-allow-unconstrained`, seed 1.

| Item | Value |
|---|---|
| Clock target | 100 MHz |
| Achieved Fmax | 36.65 MHz |
| Timing closure | FAIL |
| Critical path | `win_r` Q → `median_px` CCU2C carry chain → `comb_r` setup (~27 ns) |

The design does not close timing at 100 MHz.  The `--lpf-allow-unconstrained`
flag means the clock enters through a general I/O cell; a dedicated clock pin
(LOCATE COMP "clk" SITE "...") would reduce I/O overhead.

The critical path is inside filter_controller: `win_r` Q feeds into the
`median_px` CCU2C carry chain (the comparator network), ending at `comb_r`.
Pipelining the gaussian accumulator (+0.4%) confirmed this; gaussian was not
the bottleneck.  To improve Fmax further, the median comparator chain must be
broken.

**Next steps to improve Fmax** (in order of likely impact):
1. Pipeline the median comparator network — register between comparison stages.
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
