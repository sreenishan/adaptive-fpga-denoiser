# Working notes for the adaptive FPGA denoising project

## What this is

An AI noise classifier (PC-side, Python) decides which of four filters an image
needs; an FPGA performs the filtering. The Python implementation is the
**golden reference**: RTL is verified against it, not the other way round.

## Non-negotiables

1. **Never fabricate a number.** Accuracy, MSE, PSNR, SSIM, FPS, latency, LUT /
   FF / BRAM / DSP counts, clock frequency and power come from an actual run or
   they are `TBD` / `null` / absent. A plausible-looking placeholder is
   indistinguishable from a measurement once it is in a document.
2. **Software first, hardware second.** Every RTL filter must be compared
   pixel-by-pixel against its Python reference before it counts as working.
3. **Do not silently swap an algorithm.** If the hardware Wiener filter is an
   approximation, it is documented as one and its error against the reference is
   reported. Replacing it with something cheaper and still calling it Wiener is
   the failure mode this rule exists to prevent.
4. **No board in the core RTL.** Everything in `rtl/` (the modules listed in
   `rtl/filelist.f`, plus the benches in `rtl/tb/`) is vendor-neutral. Constraints, camera and display code live under
   `fpga/`.
5. **No absolute paths.** Everything resolves against the repository root via
   `denoising.config.PROJECT_ROOT`. A test asserts the configs contain none.

## The boundary policy is a single decision, held in two files

The RTL window generator replicates edge pixels. The software reference must use
the same policy or the golden comparison compares two different algorithms and
reports the difference as an RTL bug. `configs/hardware.yaml`
(`stream.boundary_policy`) and `configs/inference.yaml`
(`filters.boundary_mode`) must agree; `test_software_and_rtl_boundary_policies_agree`
fails if they drift apart. The loader rejects any hardware policy other than
`replicate`, because that is the only one implemented.

## Configuration

Four files under `configs/`, loaded into frozen dataclasses by
`src/denoising/config.py`. Validation is deliberately strict and every error
names the dotted key that is wrong — a bad value should fail before a dataset is
generated, not halfway through a training run.

Things the loader refuses on purpose: split ratios that do not sum to 1, even
kernel sizes, a `num_classes` that disagrees with `CLASSES`, a pixel width other
than 8, a boundary policy other than `replicate`, a `bool` where a number
belongs (`True` is an `int` in Python and would otherwise pass as a width).

`null` means *nobody has measured this*, which is not the same claim as `0`.
That is why `synthesis.*` is all `null` and `wiener.noise_variance` is `null`
(= estimate it from local statistics).

## `CLASSES` order is the label order

`("clean", "salt_pepper", "gaussian", "speckle")` — a class's index in that
tuple is its integer label in the dataset, the model output layer, the confusion
matrix and the 2-bit RTL control code. Reordering it silently relabels every
saved checkpoint.

## Importing the package must not import torch

`denoising/__init__.py` imports only `config` and `logging_utils`. PyTorch is an
optional extra so the noise generators, filters and metrics stay usable on a
machine with no ML framework installed; a test asserts `torch` is absent from
`sys.modules` after import.

## Toolchain in this environment

Python 3.12.10 with numpy, opencv, scipy, scikit-image, scikit-learn, pandas,
matplotlib, pyyaml, pytest. **Icarus Verilog 12.0 is installed at
`C:/iverilog/bin` and is on PATH.** yosys 0.57 (oss-cad-suite) is at
`C:/oss-cad-suite/oss-cad-suite` — both its `bin` AND `lib` are on PATH (the
binaries fail on a missing DLL without `lib`). For `rtl/tb/run_tb.sh` export
`PATH="/c/iverilog/bin:$PATH"` first. No Verilator, Vivado or Quartus: synthesis
so far is vendor-neutral only, and nothing has been fitted to a part.

PyTorch 2.13.0+cpu is installed. On 2026-09-11 Windows Application Control
blocked its DLL (`import torch` -> "An Application Control policy has blocked
this file"); on 2026-09-12 it loaded again and the app reported "CNN loaded".
If it returns, the app degrades to manual and reports "PyTorch unavailable",
and `tests/python/test_model.py` fails at collection — run the suite with
`--ignore=tests/python/test_model.py` and say so. It is a machine policy, not a
code fault: do not "fix" it in code.

## Before saying it works

```bash
python -m pytest && python scripts/check_config.py
```

## Order of work

Follow the phase order in the development spec. Implement the smallest missing
module, test it, then move on. Do not rewrite a working module, and inspect
dependants before changing an interface.

Done: phase 1 (foundation), phase 2 (noise generators), phases 3-4 (dataset
generation, splitting, manifest).

Done additionally: phase 5 (preprocessing), phases 9-10 (filters, selector),
phase 12 (metrics), phase 11 (pipeline), and the Streamlit app (spec 39).

Done: phases 6-8, the CNN classifier. `src/denoising/model/` holds the network
(`cnn.py`), training (`train.py`, `train_cli.py`) and inference
(`inference.py`); `scripts/train.py` drives it and `tests/python/test_model.py`
covers architecture, class weighting, early stopping, checkpoint contents and
prediction. It plugs into `process_image(classifier=...)` exactly as planned —
nothing else changed and the app picked it up without edits.

`models/checkpoints/best_model.pt` is a real trained checkpoint; when PyTorch
loads (see Toolchain — currently blocked on this machine) the app reports a
loaded classifier rather than falling back to manual selection. It stays out of `requirements.txt` on purpose:
~2 GB exceeds Streamlit Community Cloud's install budget, and the app degrades
gracefully to manual classification when it is absent. That is why the deployed
build shows "Manual" where this machine shows "CNN" — not a bug.

Done: severity estimation (design stage 4) and severity-driven filter
selection, adaptive median, single-frame camera input, and the seven-stage flow
panel on the processing page. See the section on severity below.

Done: RTL simulation (2026-09-11). All six `rtl/tb` unit benches pass in
Icarus, each filter bench mutation-checked; three had never compiled because
they drove a `win` port the flattened filters no longer have. The golden
co-simulation (`scripts/simulate_rtl.py`) streams full 224x224 frames through
`fpga_denoiser_top` and judges every pixel by the Python filters: 48/48 frames,
median and Gaussian bit-exact on 602,112 pixels each, Wiener max 1 grey level.
Results and method are in `docs/verification.md`; the small-frame version runs
in pytest as `tests/rtl/test_rtl_cosim.py` and skips (never passes) without
Icarus. Say "simulated against the golden model", never "verified on hardware".

**The Wiener noise power is a runtime port, and must stay one.** It was a
compile-time parameter fixed at 100 while the pipeline estimated the variance
per image — co-simulation put that gap at up to 6.1 dB and 109 grey levels, so
the two were different filters sharing a name. The host now estimates the noise
power, rounds it to an integer and writes `noise_var` alongside `filter_sel`
(`host_noise_var()` in `scripts/simulate_rtl.py` is the reference), and the
hardware tracks the software filter to within one grey level. Both are
combinational control: change them between frames, not mid-frame. Turning
`noise_var` back into a parameter would silently reopen a 6 dB gap.

Done: vendor-neutral synthesis (2026-09-12). yosys 0.57 from oss-cad-suite
(`C:/oss-cad-suite/oss-cad-suite`, add its `bin` AND `lib` to PATH or the
binaries fail on a missing DLL) synthesises the design via
`scripts/synthesize_rtl.py`: 3,951 LUTs and 4,445 FFs generic after the
pipeline work below, table in `docs/hardware.md`. It found a latch that
simulation cannot: `median_filter` swapped through a module-level temporary
assigned only inside `if` branches, so a latch was inferred for it and yosys
refused the design. The comparator stages are concatenated swaps now, with no
temporary — keep them that way.

Done: the Wiener gain divider. It was a variable 32/24-bit divide per pixel
and 87% of the logic; it is eight restoring steps now, which produce the same
eight clamped bits. `wiener_filter` 4,211 -> 2,959 LUTs, total 4,860 -> 3,608.
**Not an approximation** — identical over 302,091 arithmetic pairs and
1,352,104 side-by-side RTL vectors — so the 1 grey level budget still describes
the output rounding and nothing was re-characterised. Keep it that way: any
future replacement that is genuinely approximate has to be re-characterised
against the golden model and the budget re-justified.

Done: Wiener Fmax improvement (2026-09-30). Three pipeline stages added to break
the critical paths on ECP5-25k: a pre-stage register after the window sums
(broke the 25 ns input path), an output-accumulator register plus reciprocal
multiply for `acc / 2304` (broke the 49 ns carry-chain output path), and a
variance-stage register capturing `p1_r = 9*s2_pre` and `p2_r = s_pre*s_pre`
before the subtraction (broke the ~20 ns path from `s_pre/s2_pre` through
`den_v` to `rem_r[1]`). All are exact. Wiener total latency is now 11 cycles
(STAGES+3 = 8+3). ECP5 Fmax at that stage: 20.42 → 30.23 → 49.36 MHz.
MULT18X18D: 15 → 17 (2 DSPs for the reciprocal multiply; variance stage uses
the same DSPs already present).

`acc / 2304` is now the reciprocal multiply on ECP5, where DSP blocks make it
faster. The constant `RECIP = 233017 = floor(2^29 / 2304)` and shift `RECIP_SHR
= 29` are in the module comment. Do not revert to integer division — it would
reopen the 49 ns critical path.

The variance-stage register captures `p1_r = 9*s2_pre` and `p2_r = s_pre*s_pre`
before the `v81 = p1_r - p2_r` subtraction. The key insight: both products are
~5-8 ns from the pre-stage registers, but the full chain through the subtraction
and `den_v` comparison was ~20 ns. After the register, `v81 = p1_r - p2_r` is a
single fast subtraction. Do not collapse these back into one cycle — it would
reopen the 20 ns critical path.

Done: BRAM line buffer (2026-09-30). `line_buffer.sv` rewrote the two 224-deep
shift registers as circular buffers (synchronous dual-port memory pattern).
yosys `synth_ecp5` now infers two DP16KD blocks. TRELLIS_FF: 4,445 → **869**
(−3,576 = −80%); DP16KD: 0 → 2; Fmax: 30.23 → 30.91 MHz (routing noise, same
critical path). The previous `initial` block that zeroed the shift-register
cells is gone — it was the main blocker for BRAM inference. On ECP5, BRAMs
power on to zero; in simulation, X reads during the priming period are masked
by window_gen's valid_out being low.

The timing contract with window_gen is unchanged: the registered BRAM output
at cycle N holds the pixel from WIDTH cycles ago, which is what window_gen's
`always_ff` reads at the same posedge — same as the old shift-register tap.
Do not add back an `initial` block; it will re-break BRAM inference.

Done: filter_controller pre-register + win-input register (2026-09-30).

First: a `comb_r`/`sel_r`/`vld_r` pre-register after the combinational
`comb_px` mux. This broke the old 33 ns path (gaussian carry chain → comb_d[1])
but the critical path shifted to window_gen FF → gaussian accumulator → comb_r
(same combinational depth), so Fmax was essentially unchanged: 30.91 → 29.87 MHz.

Second: a `win_r`/`sel_wr`/`vld_wr` input register in filter_controller,
registering win_flat (and filter_sel/valid_in) before the filter cores. This
cuts the ~3.5 ns window_gen counter propagation from the critical path.
Fmax: 29.87 → **36.51 MHz** (+22%). `PIPE_STAGES` = 12.

Done: gaussian accumulator pipeline (2026-09-30). `gaussian_filter` gains
`clk`/`rst_n`/`en` ports and a 1-cycle register splitting the 9-input adder
tree into row sums (comb) → register → final sum (comb). Arithmetic is
bit-identical (addition is associative within range); max_abs_error = 0
unchanged. filter_controller adds Stage G registers (median_r, centre_r,
sel_wr2, vld_wr2) and wiener_r2 to keep all paths aligned at 14 cycles.
Fmax: 36.51 → **36.65 MHz** (+0.4% — routing noise). The gaussian adder tree
was not the critical path; nextpnr reports win_r Q → median_px CCU2C carry chain
→ comb_r (~27 ns). LUT4: 1,853 → 1,816; TRELLIS_FF: 963 → 1,015 (+52).
`PIPE_STAGES` = 13 throughout the design. All 5 unit benches and 6 slow RTL
tests pass.

Done: median comparator pipeline (2026-09-30). `median_filter` gains
clk/rst_n/en ports and a 1-cycle pipeline register splitting the 19-comparator
network at the column-sort boundary: steps 1-9 (sort columns, 3 deep) →
register → steps 10-19 (select median, 5 deep). The external `median_r`
register in filter_controller's Stage G is absorbed here — net pipeline depth
unchanged (PIPE_STAGES = 13, total latency = 14). Fmax: 36.65 → **49.15 MHz**
(+34%). The median CCU2C carry chain is gone; nextpnr reports the critical path
as `u_wiener.s_pre` Q → `den_v` → MULT18X18D input (~20 ns). TRELLIS_FF:
1,015 → 1,079 (+64). All 5 unit benches and 6 slow RTL tests pass.

Done: Wiener variance stage pipeline (2026-09-30). A variance-stage register
inside `wiener_filter` captures `p1_r = 9*s2_pre` (shift+add, ~6 ns) and
`p2_r = s_pre*s_pre` (MULT18X18D, ~4 ns) before the `v81 = p1_r - p2_r`
subtraction. `nv81_r = 81*nv_pre` and the s/centre pixel are also latched here.
After the register, `v81` is a fast subtraction and the path to `rem_r[1]` is
short. Wiener latency: 10 → 11 cycles (STAGES+3); DIV_STAGES: 10 → 11;
PIPE_STAGES: 13 → 14; total latency: 14 → 15 cycles.
Fmax: 49.15 → 49.36 MHz (+0.4% — both Wiener variance and median stage-2
paths were ~20 ns; pipelining the Wiener path transferred the critical path to
median stage-2). TRELLIS_FF: 1,079 → 1,181 (+102); TRELLIS_COMB: 1,816 → 1,963.
All 5 unit benches pass.

Done: median stage-2 pipeline (2026-09-30). `median_filter`'s five-comparator
stage 2 (steps 10-19) split at step 16 into two three-comparator sub-stages:
stage 2a (steps 10-16, 3 deep) → register r[0..8] → stage 2b (steps 17-19,
3 deep). Split verified exact over all unit-bench cases. `median_filter` latency:
1 → 2 cycles. `filter_controller` adds Stage G2 (new `gaussian_r2`, `centre_r2`,
`sel_wr3`, `vld_wr3`) to delay gaussian/bypass by one cycle so all comb paths
align at cycle 3 from `win_flat`. `PIPE_STAGES`: 14 → 15; total latency: 15 → 16
cycles; `wiener_r3` added. Fmax: 49.36 → 56.82 MHz (+15%). The median
carry chain is gone from the critical path; nextpnr reports
`u_ctrl.u_wiener.c_r[8]` Q → MULT18X18D → second MULT → carry chain (~17.6 ns).
TRELLIS_FF: 1,181 → 1,224 (+43); TRELLIS_COMB: 1,963 → 1,943 (−20).
All 5 unit benches pass.

Done: Wiener gain-multiply pipeline (2026-10-01). A gain-multiply stage register
inside `wiener_filter` captures `ct_r = centre_term` (first MULT18X18D output,
~8 ns from `c_r[STAGES]` Q) alongside `gain_r` and `sext_r` before the
`gain × ct_r` multiply (second MULT18X18D). Breaks the two-MULT cascade
(`c_r[8]` Q → first MULT → second MULT → `acc_r` setup, ~17.6 ns) into two
~8 ns hops. Not an approximation — identical arithmetic, only timing changes.
Wiener total latency: STAGES+3 → STAGES+4 = 12 cycles. `filter_controller`
drops `wiener_r3` (wiener_px now at cycle 13 → only 2 alignment regs needed).
`PIPE_STAGES` and total latency **unchanged** at 15/16. Fmax: 56.82 →
72.03 MHz (+27%). All 5 unit benches pass.

Done: Wiener s2 row-partial-sum pipeline (2026-10-01). A row-partial-sum stage
register inside `wiener_filter` captures `rs_r[r]` and `rs2_r[r]` (= three-
pixel row sums and row sums-of-squares) before the final three-row accumulation.
The old path: win_r Q → nine squaring MULTs → 9-input carry-chain adder tree
→ `s2_pre` setup (~13.97 ns). After the register: one MULT + 3-input adder per
row (~7 ns); pre-stage sees a fast 3-input adder from registered row partials
(~3 ns). Not an approximation. Wiener total latency: STAGES+4 → STAGES+5 = 13
cycles. `filter_controller` drops `wiener_r2` (wiener_px now at cycle 14 → only
1 alignment reg needed). `PIPE_STAGES` and total latency **unchanged** at 15/16.
Fmax: 72.03 → **88.45 MHz** (+23%). New critical path: `gain_r` Q →
`gain_r × ct_r` (MULT18X18D) → carry chain (acc = sext_r<<8 + product + 1152)
→ `acc_r` setup (~11.63 ns). TRELLIS_FF: 1,249 → 1,349 (+100);
TRELLIS_COMB: 1,945 → 1,871 (−74). All 5 unit benches pass.

The current critical path (~11.63 ns) is `u_ctrl.u_wiener.gain_r` Q →
`gain_r × ct_r` (MULT18X18D) → acc carry chain → `acc_r` setup. To improve
Fmax further, pipeline the output accumulator — register the multiply product
before adding `sext_r<<8 + 1152`.

`configs/hardware.yaml` names the ECP5-25k device. **No board has been
programmed** — all figures in `docs/hardware.md` are place-and-route results
from `scripts/place_and_route.py`, not measurements on silicon. Power is TBD.

## The filters are a contract with the hardware

All three read the same replicated-edge 3x3 window (`filters/_window.py`), which
is the golden model for `rtl/window_gen.sv`. Median and Gaussian are
exact integer arithmetic and must match the RTL **bit for bit**; Wiener needs a
division and is allowed one grey level.

The Gaussian filter has TWO kernels and they are different filters. The default
is the integer binomial `[1 2 1; 2 4 2; 1 2 1]/16` the RTL implements, rounded
**half up** as `(acc + 8) >> 4` — one adder in hardware, written down here so
the two languages cannot round apart. It corresponds to sigma ~0.85, NOT the
`sigma: 1.0` in the config, which is why `filters.gaussian.integer_kernel` is an
explicit switch rather than a sigma silently selecting a kernel nobody built.

Note this is a different rounding rule from the noise generators' `numpy.rint`
(half to even). That is deliberate and not a drift: the noise path converts a
float to a pixel, where half-to-even is right; the Gaussian path is integer
throughout, where the tie is exact and hardware rounds half up with one adder.

## Severity picks the strength; measurement picked the policy

`src/denoising/severity.py` reads how strong the noise is **from the noisy image
alone** — impulse fraction for salt-and-pepper, Immerkaer sigma for Gaussian,
sigma/mean for speckle. The cut points in `inference.yaml` are midpoints between
the dataset's three generation levels and classify 360/360 held-out images per
class; that is a claim about the synthetic generator, and the UI calls a real
camera's level an estimate.

`SEVERITY_POLICY` in `filters/selector.py` maps (class, level) to a filter and a
pass count. **Every entry is a measurement** — best mean PSNR over 40 sources,
figures recorded beside the table. Do not edit an entry without re-measuring;
a 0.1 dB margin is a tie, ties go to fewer passes, then to the class default.
Speckle ties go to Wiener on purpose: one Gaussian pass matches two Wiener
passes within 0.08 dB and would halve hardware passes, but that is a design
change to make deliberately, not through a tie.

Severity off, or severity `None`, reproduces the old class mapping exactly. The
low-confidence fallback ignores severity: an untrusted class cannot pick a
strength. `clean` has no severity — no noise has no level.

**Adaptive median has no RTL and must stay out of `FILTERS`.** `FILTERS` is the
hardware's 2-bit control-code space; a filter in it is assumed to run on the
FPGA. It lives in `SOFTWARE_FILTERS`, its `control_code` is `None` (never a
borrowed 2'b01, which would make the FPGA run a plain median), and
`FilterDecision.hardware` reports `software`. Two passes of an RTL filter report
`rtl_multipass`: the top module streams one pass per frame, so a second needs a
cascaded core that is not built. The UI renders all three honestly; keep it so.

The camera tab is single-frame `st.camera_input`. Continuous video would need
`streamlit-webrtc`, a new dependency that works poorly on Streamlit Cloud — ask
before adding it.

## The pipeline refuses to invent two things

**A manual class choice has no confidence** — `confidence` is `None`, never 1.0.
**Metrics need a clean reference** — no original, no MSE/PSNR/SSIM, and
`metrics_note` says why. Comparing an output to its own noisy input measures how
much the filter changed and says nothing about quality.

Both rules live in `pipeline/adaptive_pipeline.py` so no UI can get them wrong,
and `app/streamlit_app.py` contains no image processing of its own for the same
reason.

## Dataset invariants

**The split is assigned per source, never per sample.** Every noisy version of
one clean image lands in the same split; otherwise the test set holds near-copies
of training images and the accuracy measures memorisation. `assign_splits` sorts
the ids before shuffling, so the result depends on the set of sources and the
seed and not on the order the filesystem yielded them. Adding a source
reshuffles everything — a permutation is not stable under insertion — which is
why a dataset is regenerated whole and `generate_dataset` refuses to merge into
an existing one.

**A sample's seed is `blake2b(master_seed, source_id, noise_type, level)`, not a
counter.** Inserting a source must not renumber every sample after it, silently
changing images already on disk. `test_sample_seed_is_stable` pins the
derivation with a measured constant: change the recipe and every seed in every
existing manifest stops describing the image beside it.

**The manifest and the per-sample JSON sidecars are written from one record in
one pass.** They hold the same facts, which is two copies of the truth — the
spec requires both — so they are produced together and a test asserts they
agree. Never give them independent sources.

**The classes are imbalanced by construction**: one clean sample per source
against one per configured intensity for each noisy class, so with three
intensities it is 1:3:3:3. Phase 7 must apply class weights or balanced
sampling; do not "fix" it by writing duplicate clean copies, which adds rows
without adding information.

## Noise generator conventions

`sigma` and `variance` are in normalised [0, 1] units, `amount` is a fraction of
pixels, and the speckle argument is a **variance** — a test asserts the
empirical spread is `sqrt(variance)` so nobody can quietly pass a standard
deviation. Clipping happens before the cast back to uint8, because uint8 wraps
silently and an unclipped bright pixel comes out dark, which would make every
class look like impulse noise. Rounding is `numpy.rint` (half to even)
everywhere a float becomes a pixel; the filters must use the same rule or the
RTL comparison inherits an off-by-one.

A seed reproduces an image exactly **for a given NumPy version**. NumPy does not
promise stream stability for `Generator` across feature releases, so the
manifest stores the seed and the parameters, and a regenerated dataset is a new
dataset rather than a bit-identical copy.
