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
4. **No board in the core RTL.** `rtl/common`, `rtl/filters`, `rtl/control` and
   `rtl/top` are vendor-neutral. Constraints, camera and display code live under
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
matplotlib, pyyaml, pytest. **No PyTorch, no Icarus Verilog, no Verilator, no
Vivado, no Quartus.** Phases needing them are not implemented; do not claim a
simulation ran.

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

**PyTorch is installed** (2.13.0+cpu) and `models/checkpoints/best_model.pt` is
a real trained checkpoint, so the app reports a loaded classifier rather than
falling back to manual selection. It stays out of `requirements.txt` on purpose:
~2 GB exceeds Streamlit Community Cloud's install budget, and the app degrades
gracefully to manual classification when it is absent. That is why the deployed
build shows "Manual" where this machine shows "CNN" — not a bug.

Next up: synthesis. Every module in `rtl/` is written and its testbench passes
against the Python golden model, but `configs/hardware.yaml` names no vendor or
device and **no board has been programmed**, so every figure in
`docs/hardware.md` is `TBD`. Those tables get filled from a real toolchain run
or not at all — timing, utilisation and power are measurements, and a plausible
number in that table would be indistinguishable from a measured one.

## The filters are a contract with the hardware

All three read the same replicated-edge 3x3 window (`filters/_window.py`), which
is the golden model for `rtl/common/window_3x3.sv`. Median and Gaussian are
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
