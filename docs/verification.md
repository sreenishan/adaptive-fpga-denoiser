# Verification

Status: the RTL has been **simulated** in Icarus Verilog 12.0 and compared
pixel by pixel against the Python golden filters (2026-09-11). It has **not**
been synthesised, timed, or run on a board; see "What this does not show".

## Method

The Python implementation is the golden reference. For each filter:

```text
test image --> Python reference --> expected pixels
           --> RTL simulation   --> actual pixels
```

compared pixel by pixel. The report records:

- maximum absolute error
- mean absolute error
- number of mismatched pixels
- percentage mismatch

Written by `scripts/simulate_rtl.py` to `results/rtl/cosim_<W>x<H>.json`, with
the stimulus, golden and RTL pixel streams as hex under `simulation/input`,
`simulation/expected` and `simulation/output`. (No PNG diff images are produced
yet.)

## Tolerances

From `configs/hardware.yaml` (`simulation.max_abs_error`):

| Filter | Max abs error | Why |
|---|---|---|
| Median | 0 | Selection network; exact in both implementations |
| Gaussian | 0 | Fixed integer kernel and a stated rounding rule |
| Wiener | 1 | Fixed-point reciprocal; the approximation is documented with the module |

A tolerance is a decision, not a fudge factor. Raising one requires saying here
what changed and why.

## Test vectors

Both sides must use edge replication at the borders. Cases to cover: constant
image, gradient, random, all-zero, all-255, single-pixel spike, and the four
image corners.

## How the RTL is checked

Three layers, weakest to strongest. Only the third meets the method above.

1. **`tests/rtl/test_rtl_golden.py`** — a statement-for-statement Python
   transcription of the `.sv` files (`rtl_model.py`) against the golden
   filters. Runs everywhere, but cannot see a syntax, elaboration or protocol
   error, because the SystemVerilog never executes.
2. **`rtl/tb/*.sv` unit benches** (`bash rtl/tb/run_tb.sh`) — the real RTL in
   Icarus, against small references written inside each bench. 6/6 pass.
   Three of them had never compiled: they drove a `win` array port the filters
   no longer have (the RTL was flattened to `win_flat`). Fixed in the benches;
   the RTL was not changed. Each filter bench was mutation-checked — an
   off-by-one in the median network, a rounding constant 8 -> 7 in the
   Gaussian, a +2 offset in the Wiener — and each mutant failed.
3. **Golden co-simulation** (`python scripts/simulate_rtl.py`) — full frames
   streamed through the complete `fpga_denoiser_top` (window generator, filter
   controller, all four filters) via `rtl/tb/tb_golden_image.sv`, every output
   pixel judged by the Python golden filters. The bench checks only the stream
   protocol (output count, `m_overflow`, `s_ready`); a planted Gaussian rounding
   bug dropped the run to 18/24 frames with a nonzero exit.
   `tests/rtl/test_rtl_cosim.py` runs the small-frame version in the normal
   suite and **skips, with a reason, when Icarus is absent** — it never passes
   without simulating.

## Results

Golden co-simulation, 224x224 (the configured stream geometry), 12 images:
the clean synthetic source, salt-and-pepper at 0.02 / 0.05 / 0.10, Gaussian at
sigma 0.03 / 0.06 / 0.10, speckle at 0.03 / 0.06 / 0.10, and a random image of
only 0 and 255 pixels. One image per filter was also streamed with random
input stalls. Wiener runs at the noise power the host measured for each frame
and wrote to the `noise_var` port — 42 to 14435 squared grey levels across
these images — the same integer the golden filter was given.

| Filter | Frames | Pixels | Max abs error | Mean abs error | Mismatched pixels | Status |
|---|---|---|---|---|---|---|
| Bypass | 12 | 602,112 | 0 | 0 | 0 (0%) | pass |
| Median | 12 | 602,112 | 0 | 0 | 0 (0%) | pass, bit-exact |
| Gaussian | 12 | 602,112 | 0 | 0 | 0 (0%) | pass, bit-exact |
| Wiener | 12 | 602,112 | 1 | 0.0321 | 19,336 (3.21%) | pass, within 1 LSB |

48/48 frames within tolerance; 58-77 s of simulation per frame.

The edge vectors listed under *Test vectors* — all-zero, all-255, constant
mid-grey, a horizontal gradient, and single-pixel spikes at the centre and all
four corners — were run on a 31x17 frame, where the border is ~20% of the
image rather than ~2%: 44/44 frames within tolerance, every one of them with
zero mismatches.

## What this does not show

- **Timing, utilisation, power.** Nothing has been synthesised. `docs/hardware.md`
  stays `TBD` until a real toolchain run fills it.
- **Behaviour on hardware.** Simulation proves the RTL computes the golden
  result for the stimulus given. It does not prove it meets timing or survives
  a real clock, reset and I/O.
- **Adaptive median, and two-pass decisions.** Adaptive median has no RTL at
  all, and the top module runs one pass per frame; neither is simulated here.

## Resolved: the hardware Wiener is now the software Wiener

`noise_var` was a compile-time parameter fixed at 100, so the hardware ran the
fixed-variance filter while the pipeline estimated the noise power per image.
Co-simulation measured that gap at up to 6.1 dB and 109 grey levels: the two
shared a name and were not the same filter.

It is an input port now. The host estimates the noise power for the frame,
rounds it to an integer, and writes it alongside `filter_sel`; the golden filter
is given the same integer. PSNR against the clean source, 224x224:

| Case | `noise_var` sent | Old fixed 100 | Host integer | Software float |
|---|---|---|---|---|
| salt_pepper 0.02 | 379 | 22.63 | 23.55 | 23.55 |
| salt_pepper 0.05 | 862 | 18.74 | 21.00 | 21.00 |
| salt_pepper 0.10 | 1682 | 15.65 | 19.36 | 19.36 |
| gaussian 0.03 | 95 | 39.12 | 39.04 | 39.03 |
| gaussian 0.06 | 249 | 28.24 | 32.30 | 32.30 |
| gaussian 0.10 | 601 | 21.54 | 27.63 | 27.63 |
| speckle 0.03 | 516 | 22.25 | 26.61 | 26.61 |
| speckle 0.06 | 942 | 18.79 | 23.76 | 23.76 |
| speckle 0.10 | 1476 | 16.45 | 21.72 | 21.72 |

The hardware now tracks the software filter to within one grey level in every
case, gaining up to 6.09 dB over the fixed value. Gaussian sigma 0.03 is the
one exception and is 0.08 dB *worse*: the estimate there is 95, and a fixed 100
happened to suit that noise level marginally better. Matching the software
filter is the goal, not beating it — a hardware output that disagrees with the
software decision is the failure this closes.

`tb_wiener_filter` now sweeps `noise_var` over {0, 1, 25, 100, 400, 4000,
65025} in a single run, which a compile-time parameter could not do. The module
header had claimed agreement across that range without it being checked in any
recorded run.

## The divider refactor changed no pixel

`wiener_filter`'s gain divide became eight restoring steps (a 26% cut in the
design's logic; see `docs/hardware.md`). It is a cost reduction, not an
approximation, and three independent checks say so:

- the arithmetic identity over 302,091 `(num, den)` pairs — exhaustive for small
  values, random across the 24-bit range, plus the `num = den` and `den = 0`
  edges: zero mismatches;
- the old and new modules instantiated side by side in Icarus and compared for
  **exact** equality over 1,352,104 vectors — every flat window at every grey
  level, structured extremes, 150k random windows, each swept across
  `noise_var` {0, 1, 25, 100, 400, 4000, 65025, 65535}: identical everywhere;
- the full 224x224 co-simulation re-run afterwards, reproducing the Wiener row
  of the results table to the pixel — max 1, mean 0.0321, 19,336 of 602,112
  mismatched, and the same PSNR for every case.

A formal SAT equivalence proof (yosys `miter -equiv` plus `sat -prove`) was
attempted and **did not finish** — the window statistics contain several
multipliers, which SAT handles badly. It is not evidence either way, and the
claim above rests on the three checks that did complete.
