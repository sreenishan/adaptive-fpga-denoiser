# Hardware

Status: place-and-route complete. **No board has been programmed** — all figures
are from `scripts/place_and_route.py` (yosys synth_ecp5 + nextpnr-ecp5 + ecppack),
not measurements on silicon. Fmax: **115.09 MHz** on ECP5-25k at 100 MHz target (PASS).
Results in `results/rtl/pnr_ecp5.json`.

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
| `filter_sel` | in | 3 | `000` bypass, `001` median, `010` Gaussian, `011` Wiener, `100` adaptive_median |
| `noise_var` | in | 16 | Wiener noise power in squared grey levels, written per frame |

`filter_sel` and `noise_var` are combinational control: they apply to the window
in the generator that cycle, which trails the input pixel by IMG_WIDTH+1. Change
them between frames.

**Latency and flush.** The filter_controller pipeline is `PIPE_STAGES` = 18 deep
(1 win-input + 1 G1 + 1 G2 + 1 G3 + 1 G4 + 1 G5 + 1 comb-pre + 11 delay-chain/Wiener stages), so
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
| + gaussian accumulator pipeline | 36.65 MHz | 1,015 | 2 | 17 |
| + median comparator pipeline | 49.15 MHz | 1,079 | 2 | 17 |
| + Wiener variance stage pipeline | 49.36 MHz | 1,181 | 2 | 17 |
| + median stage-2 pipeline | 56.82 MHz | 1,224 | 2 | 17 |
| + Wiener gain-multiply pipeline | 72.03 MHz | 1,249 | 2 | 17 |
| + Wiener s2 row-partial pipeline | 88.45 MHz | 1,349 | 2 | 17 |
| + Wiener gain-product pipeline | 88.58 MHz | 1,375 | 2 | 17 |
| + Wiener reciprocal-product pipeline | 93.76 MHz | 1,399 | 2 | 17 |
| + Wiener den_v pipeline | 94.30 MHz | 1,478 | 2 | 17 |
| + median stage-1 column-sort pipeline | 102.16 MHz | 1,550 | 2 | 17 |
| + median stage-2b-a pipeline | 102.18 MHz | 1,574 | 2 | 17 |
| + Wiener squaring stage pipeline | 115.86 MHz | 1,783 | 2 | 17 |
| + median stage-2a pipeline | 104.37 MHz | 1,839 | 2 | 17 |
| + adaptive_median filter (Phase B, 3-bit filter_sel) | 76.79 MHz | — | 2 | 17 |
| + adaptive_median min/max tree pipeline | 94.93 MHz | — | 2 | 17 |
| + Stage G6 pre-mux register | 97.24 MHz | 2,033 | 2 | 17 |
| + median/adaptive_median stage-2b-b split | **115.09 MHz** | **2,049** | **2** | **17** |

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
14 cycles; PIPE_STAGES = 13. TRELLIS_FF: 963 → 1,015 (+52); LUT4: 1,853 → 1,816.

The eighth row pipelines the median comparator network: `median_filter` gains
clock/rst_n/en ports and a 1-cycle pipeline register splitting the 19-comparator
network at the column-sort boundary (steps 1-9 sort columns → register → steps
10-19 find median). Stage 1 is 3 comparators deep; stage 2 is 5 deep. The
external `median_r` register in filter_controller's Stage G is absorbed into the
module — net pipeline depth is **unchanged** (PIPE_STAGES = 13, total latency =
14). Fmax: 36.65 → **49.15 MHz** (+34%). The median CCU2C carry chain is gone
from the critical path; nextpnr now reports `u_wiener.s_pre` Q → `den_v` →
`MULT18X18D` (~20 ns) — the Wiener variance computation. TRELLIS_FF: 1,015 →
1,079 (+64 = 9×8-bit median column-sort register − 8-bit median_r removed).

The ninth row pipelines the Wiener variance computation: a new variance-stage
register inside `wiener_filter` captures `p1_r = 9*s2_pre` (shift+add carry
chain, ~6 ns from `s2_pre` Q) and `p2_r = s_pre*s_pre` (MULT18X18D, ~4 ns from
`s_pre` Q) before the subtraction.  After the register, `v81 = p1_r - p2_r` is
a fast 24-bit subtraction (~5 ns) and the remaining path to `rem_r[1]` is
short.  Wiener total latency: 10 → 11 cycles (STAGES+3); filter_controller
`DIV_STAGES` 10 → 11; `PIPE_STAGES` 13 → 14; total pipeline latency 14 → 15
cycles.  Fmax: 49.15 → 49.36 MHz (+0.4% — routing noise; two paths are now
neck-and-neck at ~20 ns).  The Wiener variance path is no longer critical;
nextpnr now reports `u_median.q[4]` Q → median stage-2 carry chain → `comb_r`
(~20 ns).  TRELLIS_FF: 1,079 → 1,181 (+102 for p1_r + p2_r + nv81_r + s_r0 +
c_r0 registers and one extra delay-chain stage); TRELLIS_COMB: 1,816 → 1,963.

The tenth row pipelines the median stage-2 network: `median_filter`'s five-
comparator stage 2 (steps 10-19) is split at step 16 into two three-comparator
halves — stage 2a (steps 10-16, 3 deep) → register r[0..8] → stage 2b
(steps 17-19, 3 deep).  Arithmetic is unchanged; the split is verified
exact over all pseudorandom and corner-case windows in the unit bench.
`median_filter` latency: 1 → 2 cycles.  `filter_controller` adds Stage G2 —
a one-cycle register delaying `gaussian_px` and `centre_r` so all three comb
paths (median, gaussian, bypass) align at cycle 3 from `win_flat`.
`PIPE_STAGES`: 14 → 15; total latency: 15 → 16 cycles; `wiener_r3` added to
keep the Wiener path aligned.  Fmax: 49.36 → 56.82 MHz (+15%).  The median
stage-2 carry chain is gone from the critical path; nextpnr now reports
`u_ctrl.u_wiener.c_r[8]` Q → MULT18X18D (gain multiply) → second MULT →
carry chain (~17.6 ns).  TRELLIS_FF: 1,181 → 1,224 (+43 = 9×8-bit stage-2a
register + Stage G2 signals); TRELLIS_COMB: 1,963 → 1,943 (−20, simpler
stage-2b).

The eleventh row pipelines the Wiener gain-multiply path: a gain-multiply
stage register inside `wiener_filter` captures `ct_r = centre_term` (=
`c_r[STAGES] × 9 − s_r[STAGES]`, first MULT18X18D, ~8 ns from `c_r[STAGES]`
Q) alongside `gain_r` and `sext_r` before the `gain × centre_term` multiply
(second MULT18X18D).  The two-MULT cascade (`c_r[8]` Q → first MULT → second
MULT → `acc_r` setup, ~17.6 ns) is broken into two ~8 ns hops.  Not an
approximation — the arithmetic is identical; only when it runs changes.
Wiener total latency: STAGES+3 → STAGES+4 = 12 cycles.  `filter_controller`
drops `wiener_r3` (only 2 alignment registers needed now, since `wiener_px`
arrives one cycle later); `PIPE_STAGES` and total latency are **unchanged** at
15/16.  Fmax: 56.82 → 72.03 MHz (+27%).  The Wiener gain-multiply path is
gone; nextpnr now reports `u_ctrl.win_r` Q → `s2` squaring accumulation
(MULT18X18D + carry chain) → `s2_pre` register setup (~13.97 ns) — the nine-
pixel sum-of-squares in the pre-stage.  TRELLIS_FF: 1,224 → 1,249 (+25 = ct_r
+ gain_r + sext_r registers); TRELLIS_COMB: 1,943 → 1,945 (+2, routing).

The twelfth row pipelines the Wiener s2 accumulation: a row-partial-sum stage
register inside `wiener_filter` captures three row partial sums `rs_r[r]` and
`rs2_r[r]` (= `Σ wp[r*3+c]` and `Σ wp[r*3+c]²` for c=0..2) before the final
three-row accumulation.  The critical path was win_r Q → nine squaring MULTs
→ 9-input carry-chain adder tree → `s2_pre` setup (~13.97 ns).  After the
register, each row uses one 8×8 MULT18X18D + a 3-input adder (~7 ns); the
pre-stage sees only a fast 3-input adder from registered row partials (~3 ns).
Not an approximation — arithmetic is identical.  Wiener total latency:
STAGES+4 → STAGES+5 = 13 cycles.  `filter_controller` drops `wiener_r2`
(only 1 alignment register needed now); `PIPE_STAGES` and total latency
**unchanged** at 15/16.  Fmax: 72.03 → 88.45 MHz (+23%).  New critical
path: `gain_r` Q → `gain_r × ct_r` (MULT18X18D) → carry chain (acc
accumulation) → `acc_r` setup (~11.63 ns).  TRELLIS_FF: 1,249 → 1,349 (+100
= rs_r[0..2] + rs2_r[0..2] + c_rs + nv_rs); TRELLIS_COMB: 1,945 → 1,871
(−74, smaller adder tree in pre-stage).

The thirteenth row pipelines the Wiener output accumulator: a gain-product
stage register inside `wiener_filter` captures `prod_r = gain_r × ct_r`
(MULT18X18D, ~4 ns from `gain_r` Q) and `sext_r2 = sext_r` before the
three-input `acc = sext_r2<<8 + prod_r + 1152` addition.  The old path was
`gain_r` Q → MULT → carry chain (three-input addition) → `acc_r` setup
(~11.63 ns); after the register, `acc` is a fast 3-input carry-chain addition
and the path to `acc_r` is short (~4 ns).  Not an approximation — arithmetic
is identical.  Wiener total latency: STAGES+5 → STAGES+6 = 14 cycles.
`filter_controller` drops `wiener_r` entirely — `wiener_px` now arrives at
cycle 15 = `comb_d[11]`, so no alignment register is needed; `PIPE_STAGES` and
total latency unchanged at 15/16.  Fmax: 88.45 → 88.58 MHz (+0.1% — the new
critical path, `acc_r` Q → `recip_prod` MULT18X18D → carry chain →
`pixel_out` setup (~11.29 ns), was already adjacent to the old one).
TRELLIS_FF: 1,349 → 1,375 (+26 = prod_r + sext_r2); TRELLIS_COMB: 1,871 →
1,865 (−6, simpler acc).

The fourteenth row pipelines the Wiener reciprocal multiply: a reciprocal-
product stage register inside `wiener_filter` captures `recip_r = acc_r ×
RECIP` (the 43-bit intermediate, two MULT18X18D, ~4 ns from `acc_r` Q) before
the right-shift and 8-bit clamp (`quot = recip_r >> RECIP_SHR`).  The old
path was `acc_r` Q → `recip_prod` MULT18X18D → carry chain → `pixel_out`
setup (~11.29 ns); after the register, only a shift and comparison remain
(~3 ns).  Not an approximation — arithmetic is identical.  Wiener total
latency: STAGES+6 → STAGES+7 = 15 cycles.  `filter_controller`: `DIV_STAGES`
11 → 12, `wiener_px` now arrives at cycle 16 = `comb_d[12]` (no alignment
register needed); `PIPE_STAGES` 15 → 16; total latency 16 → 17 cycles.
Fmax: 88.58 → **93.76 MHz** (+5.8%).  New critical path: `p1_r` Q →
`v81 = p1_r − p2_r` (subtraction) → `den_v` (max comparison, 24-bit carry
chain) → `nxt_0` LUT → `rem_r[1]` setup (~10.41 ns) — the variance stage
into restoring-divider step 0.  TRELLIS_FF: 1,375 → 1,399 (+24 = recip_r);
TRELLIS_COMB: 1,865 → 1,863 (−2).

The fifteenth row pipelines the Wiener den_v computation: a den_v stage
register inside `wiener_filter` captures `num_v_r = max(0, v81 − nv81_r)` and
`den_v_r = max(v81, nv81_r)` (the max comparison, 24-bit carry chain) before
restoring-divider step 0.  The old path was `p1_r` Q → `v81` subtraction →
`den_v` carry chain → `nxt_0` → `rem_r[1]` setup (~10.41 ns); after the
register, step 0 sees only its 25-bit shift/compare/subtract (~5 ns).  Also
registers `s_r0a` and `c_r0a` to carry alongside.  Not an approximation —
arithmetic is identical.  Wiener total latency: STAGES+7 → STAGES+8 = 16
cycles.  `filter_controller`: `DIV_STAGES` 12 → 13, `wiener_px` at cycle 17 =
`comb_d[13]`; `PIPE_STAGES` 16 → 17; total latency 17 → 18 cycles.  Fmax:
93.76 → **94.30 MHz** (+0.6% — the new critical path, `win_r` Q → median
stage-1 column-sort carry chain → `q[]` setup (~10.25 ns), is a different
bottleneck in the median filter).  TRELLIS_FF: 1,399 → 1,478 (+79 = num_v_r +
den_v_r + s_r0a + c_r0a + 1 extra delay-chain stage); TRELLIS_COMB: 1,863 →
1,889 (+26, extra carry chain depth for den_v comparison).

The sixteenth row pipelines the median stage-1 column-sort: `median_filter`'s
3-layer column sort (steps 1-9) is split after layer 1 by inserting `s1_r[0..8]`
between stage 1a (layer 1, 1 comparator deep: CS(1,2)/CS(4,5)/CS(7,8)) and
stage 1b (layers 2-3, 2 comparators deep).  The old path was `win_r` Q →
3 column-sort layers → `q[]` setup (~10.25 ns); after the register, the worst
sub-path is `s1_r` Q → 2 layers → `q[]` setup (~7 ns).  `median_filter` latency:
2 → 3 cycles.  `filter_controller` adds Stage G3 (1 extra delay cycle for
gaussian/bypass before the `comb_px` mux); `DIV_STAGES` 13 → 12 (chain shortened
by the extra fixed stage); `PIPE_STAGES` and total latency are **unchanged** at
17/18 cycles.  Fmax: 94.30 → **102.16 MHz** (+8.3%).  New critical path:
`u_ctrl.u_median.r[4]` Q → stage-2b (steps 17-19, 3 comparators deep) →
`u_ctrl.comb_r` setup (~10.45 ns).  TRELLIS_FF: 1,478 → 1,550 (+72 = s1_r[0..8]
nine 8-bit registers); TRELLIS_COMB: 1,889 → 1,890 (+1, routing).

The eighteenth row pipelines the Wiener squaring stage: a squaring-stage register
inside `wiener_filter` captures all nine individual pixel squares
`sq_r[i] = wp[i]^2` (MULT18X18D, ~4 ns from `win_r` Q) alongside the three row
pixel sums `rsum_r[r]` before the row-partial accumulation.  The old path was
`win_r` Q → MULT18X18D → 3-input carry chain → `rs2_r` setup (~10.23 ns);
after the register the worst sub-path is `sq_r` Q → 3-input adder → `rs2_r`
setup (~3 ns).  Not an approximation — arithmetic is identical.  Wiener total
latency: STAGES+8 → STAGES+9 = 17 cycles.  `filter_controller`: `DIV_STAGES`
11 → 12, `wiener_px` now arrives at cycle 18 = `comb_d[12]`; `PIPE_STAGES`
17 → 18; total latency 18 → 19 cycles.  Fmax: 102.18 → **115.86 MHz** (+13.4%).
New critical path: `u_ctrl.u_median.q[4]` Q → stage-2a (steps 10-16, 3
comparators deep) → `u_ctrl.u_median.r[4]` setup (~9.40 ns) — the median
stage-2a inside `median_filter`.  TRELLIS_FF: 1,574 → 1,783 (+209 = sq_r[0..8]
nine 20-bit squares + rsum_r[0..2] three 12-bit sums); TRELLIS_COMB: 1,887 → 1,887.

The nineteenth row pipelines the median stage-2a comparator network:
`median_filter`'s three-layer stage-2a (steps 10-16) is split after layer 1 by
inserting `v1_r[0..8]` between stage 2a-a (steps 10-12: CS(0,3)/CS(5,8)/CS(4,7),
1 comparator deep from `q[]`) and stage 2a-b (steps 13-16: three comparators
then one, 2 layers deep from `v1_r[]`).  The old path was `q[]` Q →
3-layer stage-2a → `r[]` setup (~9.40 ns); after the register, the worst
sub-path is `v1_r` Q → 2 layers → `r[]` setup (~6 ns).  Not an approximation —
arithmetic is identical, verified exact over all unit-bench cases.
`median_filter` latency: 4 → 5 cycles.  `filter_controller` adds Stage G5
(1 extra delay cycle for gaussian/bypass before the `comb_px` mux);
`DIV_STAGES` 12 → 11 (chain shortened by the extra fixed stage); `PIPE_STAGES`
and total latency are **unchanged** at 18/19 cycles.  Fmax: 115.86 → **104.37 MHz**
(−10% — the median stage-2a path was cleared but the Wiener reciprocal-product
path `acc_r` Q → MULT18X18D → carry chain (~9.58 ns) became critical, and
routing congestion from the added logic slightly degraded placement).
New critical path: `u_ctrl.u_wiener.acc_r[15]` Q → `recip_prod` MULT18X18D →
carry chain (~9.58 ns) — the Wiener reciprocal-product accumulation.
TRELLIS_FF: 1,783 → 1,839 (+56 = v1_r[0..8] nine 8-bit registers + G5 signals);
TRELLIS_COMB: 1,887 → 1,864 (−23, simpler 2-layer stage 2a-b).

The seventeenth row pipelines the median stage-2b comparator network:
`median_filter`'s three-comparator stage-2b (steps 17-19) is split after step 17
by inserting `w_r[0..8]` between stage 2b-a (step 17: CS(4,2), 1 comparator deep
from `r[]`) and stage 2b-b (steps 18-19: CS(6,4) then CS(4,2), 2 comparators
deep from `w_r[]`).  The old path was `r[]` Q → 3 comparators → `comb_r` setup
(~10.45 ns); after the register, the worst sub-path is `w_r` Q → 2 comparators
→ `comb_r` setup (~7 ns).  Not an approximation — arithmetic is identical.
`median_filter` latency: 3 → 4 cycles.  `filter_controller` adds Stage G4
(1 extra delay cycle for gaussian/bypass); `DIV_STAGES` 12 → 11; `PIPE_STAGES`
and total latency are **unchanged** at 17/18 cycles.  Fmax: 102.16 → **102.18 MHz**
(+0.02% — the stage-2b path was broken, but the Wiener row-partial-sum path at
~10.23 ns was just behind it and is now critical).  New critical path:
`u_ctrl.win_r` Q → `u_wiener.rs2[1]` MULT18X18D → CCU2C carry chain →
`u_wiener.rs2_r[1]` setup (~10.23 ns) — the row-partial sum-of-squares accumulator.
TRELLIS_FF: 1,550 → 1,574 (+24 = w_r + gaussian_r4 + centre_r4 + sel_wr5 +
vld_wr5 — placement-merged so fewer FFs than the raw bit count); TRELLIS_COMB:
1,890 → 1,887 (−3, simpler 2-comparator stage 2b-b).

The eighteenth row adds the adaptive median filter (Phase B): a fifth filter
(`adaptive_median_filter`) with a 3-bit `filter_sel` bus.  The min/max tree
inside `adaptive_median_filter` was not pipelined and introduced a ~13 ns path,
regressing Fmax to 76.79 MHz.  After pipelining the min/max tree, Fmax
recovered to 94.93 MHz.  A `Stage G6` pre-mux register in `filter_controller`
(registering `gaussian_r5`/`centre_r5` before the `comb_px` mux) broke the
gaussian carry-chain path, reaching 97.24 MHz (FAIL).  `PIPE_STAGES` 17 → 18;
total latency 18 → 19 cycles.

The nineteenth row pipelines the stage-2b-b comparator pair in both
`median_filter` and `adaptive_median_filter`: step 18 (CS(6,4), 1 comparator)
→ `w2_r` register → step 19 (CS(4,2), 1 comparator).  The old path was
`w_r` Q → 2 comparators → `comb_r` setup (~10.28 ns); after the register the
worst sub-path is `w2_r` Q → 1 comparator → `comb_r` (~5 ns).  Not an
approximation — arithmetic is identical.  `median_filter` and
`adaptive_median_filter` latency: 5 → 6 cycles.  The external `median_pre_r`
and `adaptive_median_pre_r` Stage G6 pre-registers in `filter_controller` are
removed; the net pipeline count is **unchanged** at `PIPE_STAGES` = 18, total
latency = 19 cycles.  Fmax: 97.24 → **115.09 MHz** (+18%).  New critical path:
`u_ctrl.u_wiener.acc_r[15]` Q → `recip_prod` MULT18X18D → carry chain
(~8.69 ns) — the Wiener reciprocal-product accumulation.  TRELLIS_FF: 2,033 →
2,049 (+16 = w2_r in both filters); TRELLIS_COMB: — → 2,227.

**This is an ECP5 number** — it is a like-for-like comparison on one part, not a
claim about any target hardware.

The current critical path (~8.69 ns) is `u_ctrl.u_wiener.acc_r[15]` Q →
`recip_prod` MULT18X18D → carry chain — the Wiener reciprocal-product
accumulation inside `wiener_filter`.  The design now closes timing at 100 MHz
with 15% margin.

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
| TRELLIS_COMB (LUT4) | 2,227 | 24,288 | 9% |
| TRELLIS_FF | 2,049 | 24,288 | 8% |
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
| Achieved Fmax | 115.09 MHz |
| Timing closure | PASS at 100 MHz |
| Critical path | `acc_r[15]` Q → `recip_prod` MULT18X18D → carry chain (~8.69 ns) |

The design closes timing at 100 MHz with 15% margin.  The
`--lpf-allow-unconstrained` flag means the clock enters through a general I/O
cell; a dedicated clock pin (LOCATE COMP "clk" SITE "...") would reduce I/O
overhead.

The critical path is in the Wiener filter: `u_ctrl.u_wiener.acc_r[15]` Q →
the reciprocal-product MULT18X18D → the three-input carry-chain accumulation
(~8.69 ns total).  To improve Fmax further, pipeline the
`acc_r` Q → MULT18X18D → carry chain → `recip_r` setup path inside
`wiener_filter`.

## CNN Inference IP Core (hls4ml → Vivado HLS, Zynq-7000)

`scripts/hls4ml_convert.py` converts `models/checkpoints/best_model.pt` (5-class
CNN, 88.55% val accuracy) to a Vivado HLS C++ project under
`fpga/hls/noise_classifier_hls/`.

| Item | Value |
|---|---|
| Target device | xc7z020clg400-1 (Zynq-7020, speed grade 1) |
| Clock period | 10.0 ns (100 MHz) |
| Precision | `ap_fixed<16,6>` |
| Backend | hls4ml 1.3.0 → Vivado HLS |
| `AdaptiveAvgPool2d` | replaced with `AvgPool2d(28)` — identical for fixed 224×224 input |

### C simulation (2026-10-08)

Compiled via MSYS2 + mingw-w64 g++ 16.1.0 (`build_lib.sh`, `io_stream` +
`reuse_factor=64`).  Loaded via ctypes.

| Metric | Value |
|---|---|
| Prediction agreement vs FP32 | 90.0% (18/20 random 224×224 inputs) |
| Max logit \|diff\| | 1.35 |
| Mean logit \|diff\| | 0.96 |

The 2/20 mismatches are on borderline samples where the logit gap is smaller
than the `ap_fixed<16,6>` quantisation error — expected behaviour.

### HLS synthesis resource figures

**TBD** — requires `vitis_hls` or `vivado_hls` ≥ 2020.1, which is not
installed on this machine.  To obtain real numbers:

```bash
# Inside a Vivado HLS ≥ 2020.1 shell:
cd fpga/hls/noise_classifier_hls
vitis_hls -f build_prj.tcl csim=0 synth=1 cosim=0 export=0
# Report is written to myproject_prj/solution1/syn/report/myproject_csynth.rpt
```

When synthesis completes, read the resource table from `*_csynth.rpt` and
update the table below (do not fill it with estimates — rule 1 of CLAUDE.md).

| Resource | Used | Available | Utilisation |
|---|---|---|---|
| LUT | TBD | 53,200 | — |
| FF | TBD | 106,400 | — |
| DSP48E | TBD | 220 | — |
| BRAM_18K | TBD | 140 | — |

## Power (ECP5)

No power measurement is available without a programmed board.  nextpnr-ecp5
does not produce a power report; ecppack produces a bitstream only.

| Item | Value | Source |
|---|---|---|
| Static | TBD | — |
| Dynamic | TBD | — |
| Total | TBD | — |

Estimated and measured power are labelled separately and never mixed.

---

## ASIC flow — sky130hd (SkyWater 130 nm)

`fpga/asic/` holds a vendor-neutral RTL-to-GDSII flow using
yosys + OpenROAD + KLayout targeting the `sky130_fd_sc_hd` standard-cell
library.  The flow is staged into four scripts:

| Script | Stage | Output |
|---|---|---|
| `synth_sky130.tcl` | 16 — Synthesis | `*_sky130.v`, `synth_stats.txt` |
| `floorplan.tcl` | 17 — Floorplan | `*_floorplan.def`, `*_floorplan.odb` |
| `place_cts.tcl` | 18 — Placement + CTS | `*_cts.def`, `*_cts.odb` |
| `route_signoff.tcl` | 19-21 — Route, Signoff, GDS | `*_routed.def`, timing/area/power reports, `.gds` |

Run locally (requires Docker):
```bash
bash fpga/asic/run_flow.sh
```

Or trigger the `ASIC Flow` GitHub Actions workflow (fires automatically on
push to `master` when RTL or flow scripts change).  After the run completes,
execute `python scripts/parse_asic_reports.py` to extract the table below
from the result files.

### Synthesis (yosys + sky130hd)

| Metric | Value |
|---|---|
| Standard cells | TBD |
| Chip area (yosys estimate) | TBD µm² |

### Floorplan

| Metric | Value |
|---|---|
| Die area | 250 × 250 µm = 62,500 µm² |
| Core area | 246 × 246 µm = 60,516 µm² |
| Target utilisation | 40 % |

### Placement + CTS (Stage 18)

| Metric | Value |
|---|---|
| Core area used | TBD µm² |
| Actual utilisation | TBD % |
| CTS buffer | `sky130_fd_sc_hd__buf_{4,8,12}` |

### Routing + Signoff (Stages 19–20)

| Metric | Value |
|---|---|
| WNS (worst negative slack) | TBD ns |
| TNS (total negative slack) | TBD ns |
| Timing closure at 100 MHz | TBD |
| Internal power | TBD mW |
| Switching power (0.2 activity) | TBD mW |
| Leakage power | TBD mW |
| Total power | TBD mW |

Power figures use a static switching-activity estimate (20 % toggle, 50 %
duty).  A VCD from RTL simulation would give accurate dynamic power.

### DRC (Stage 20)

| Tool | Ruleset | Violations |
|---|---|---|
| KLayout | sky130A.drc | TBD |
| OpenROAD check_drc | geometric | TBD |

### GDSII (Stage 21)

| Item | Value |
|---|---|
| File | `fpga/asic/results/fpga_denoiser_top.gds` |
| Size | TBD |
| Routing layers | met1–met5 |

**No ASIC has been fabricated.** All figures above are place-and-route
estimates from the open-source tool chain.  `null` / TBD means the CI flow
has not yet run for this commit; run `python scripts/parse_asic_reports.py`
after the workflow completes to fill in measured values.
