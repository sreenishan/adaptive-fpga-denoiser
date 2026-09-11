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
input stalls. Wiener at `NOISE_VAR = 100`, the RTL's compile-time value.

| Filter | Frames | Pixels | Max abs error | Mean abs error | Mismatched pixels | Status |
|---|---|---|---|---|---|---|
| Bypass | 12 | 602,112 | 0 | 0 | 0 (0%) | pass |
| Median | 12 | 602,112 | 0 | 0 | 0 (0%) | pass, bit-exact |
| Gaussian | 12 | 602,112 | 0 | 0 | 0 (0%) | pass, bit-exact |
| Wiener | 12 | 602,112 | 1 | 0.0237 | 14,289 (2.37%) | pass, within 1 LSB |

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
- **The software pipeline's Wiener.** The RTL Wiener uses a fixed
  `NOISE_VAR = 100`; the software pipeline estimates the noise power per
  image. They are different filters. At 224x224 the fixed-variance filter is
  up to 6.1 dB worse (Gaussian sigma 0.10: 21.54 dB against 27.63 dB) and
  differs by up to 109 grey levels, so a Wiener decision made in software does
  **not** describe what the FPGA would output. Only at Gaussian sigma 0.03 do
  the two agree (39.12 vs 39.03 dB). Closing this needs the noise power as a
  runtime input to the RTL, which it does not have.
- **Adaptive median, and two-pass decisions.** Adaptive median has no RTL at
  all, and the top module runs one pass per frame; neither is simulated here.
