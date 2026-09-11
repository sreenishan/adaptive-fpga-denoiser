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
`C:/iverilog/bin`, but NOT on PATH** — `scripts/simulate_rtl.py` and
`tests/rtl/test_rtl_cosim.py` find it there; for `rtl/tb/run_tb.sh` export
`PATH="/c/iverilog/bin:$PATH"` first. No Verilator, Vivado or Quartus: nothing has
been synthesised.

PyTorch 2.13.0+cpu is installed, but as of 2026-09-11 **Windows Application
Control blocks its DLL** (`import torch` -> "An Application Control policy has
blocked this file"). While that holds, the app runs in manual mode and reports
"PyTorch unavailable", and `tests/python/test_model.py` fails at collection —
run the suite with `--ignore=tests/python/test_model.py` and say so. It is a
machine policy, not a code fault: do not "fix" it in code.

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

**Open design issue — the hardware Wiener is not the software Wiener.** The RTL
fixes `NOISE_VAR = 100` at compile time; the pipeline estimates noise power per
image (`noise_variance: null`). Co-simulation matches the RTL to the golden
filter *at the same fixed value*, but that filter is up to 6.1 dB worse than the
software one and differs by up to 109 grey levels. So `SEVERITY_POLICY`'s Wiener
entries — measured with the estimated variance — describe the software pipeline,
not what the FPGA would produce. Fixing it needs the noise power as a runtime
RTL input; until then, do not present a software Wiener result as the FPGA's.

Next up: synthesis. `configs/hardware.yaml` names no vendor or device and
**no board has been programmed**, so every figure in `docs/hardware.md` is
`TBD`. Those tables get filled from a real toolchain run or not at all —
timing, utilisation and power are measurements, and a plausible number in that
table would be indistinguishable from a measured one.

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
