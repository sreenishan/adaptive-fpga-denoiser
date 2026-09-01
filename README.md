# Adaptive FPGA-Based Image Noise Detection and Reduction System

An AI classifier decides *what kind of noise* an image carries; an FPGA performs
the matching filtering operation. The software reference comes first and is the
golden model the RTL is verified against.

```text
Camera / Image Input
        v
Pre-processing
        v
AI Noise Detection          (clean | salt_pepper | gaussian | speckle)
        v
Automatic Filter Selection
        v
FPGA-Accelerated Filtering  (bypass | median | gaussian | wiener)
        v
Denoised Output
        v
Quality & Performance Evaluation
```

| Noise class | Filter |
|---|---|
| `clean` | bypass |
| `salt_pepper` | median 3x3 |
| `gaussian` | Gaussian 3x3 |
| `speckle` | adaptive / Wiener 3x3 |

## Status

Phases 1 to 5 and 9 to 12 are complete, and a Streamlit demo runs on top of them and tested. Everything below them is not yet
implemented — this README lists only commands that actually run today.

| Phase | Module | State |
|---|---|---|
| 1 | repository, configs, config loader, logging, tests | **done** |
| 2 | noise generators | **done** |
| 3-4 | dataset generation and splitting | **done** |
| 5 | preprocessing | **done** |
| 6-8 | CNN classifier, training, evaluation | not started (needs PyTorch) |
| 9-10 | software filters, filter selector | **done** |
| 11-12 | adaptive pipeline, MSE/PSNR/SSIM | **done** |
| 13-14 | denoising evaluation, baseline comparison | not started |
| 39 | Streamlit demo application | **done** |
| 15-22 | RTL: window, filters, controller, top | not started |
| 23-25 | RTL simulation and golden-reference verification | not started |
| 26-27 | synthesis and hardware integration | not started |

**No accuracy, PSNR, SSIM, FPS, latency, resource or power figure appears
anywhere in this repository yet, because none has been measured.** Unmeasured
values are `null` in the configs and `TBD` in the documents.

## Architecture

```text
src/denoising/        Python package (software reference and golden model)
  config.py           typed loading and validation of configs/*.yaml
  logging_utils.py    logging setup
  cli.py              configuration check entry point
  noise/              salt-pepper, Gaussian and speckle generators
  dataset/            sources, generation, splitting, manifest
  model/              phase 6-8 CNN classifier
  filters/            phase 9-10 reference filters and selector
  pipeline/           phase 11 end-to-end pipeline
  metrics/            phase 12 MSE / PSNR / SSIM
  visualization/      report plots

rtl/                  board-independent SystemVerilog core
  common/             line buffer, 3x3 window generator
  filters/            median, Gaussian, Wiener
  control/            filter controller (2-bit noise class in)
  top/                adaptive denoiser top
  tb/                 testbenches

fpga/                 board-specific constraints and platform code only
simulation/           RTL test vectors: input, expected, output
results/              generated reports (classifier, denoising, rtl, hardware)
docs/                 architecture, algorithm, verification, hardware, results
```

The split is deliberate: nothing under `rtl/common`, `rtl/filters`,
`rtl/control` or `rtl/top` may name a vendor primitive or a board. Anything
board-specific lives under `fpga/`.

## Requirements

- Python 3.10+ (developed on 3.12)
- For the RTL phases: Icarus Verilog or Verilator
- For synthesis: AMD/Xilinx Vivado or Intel Quartus

Present in this development environment: Python 3.12.10, numpy, opencv-python,
scipy, scikit-image, scikit-learn, pandas, matplotlib, pyyaml, pytest.
**Not** present: PyTorch, Icarus Verilog, Verilator, Vivado, Quartus. The
phases that need them are not implemented yet.

## Installation

```bash
pip install -r requirements.txt
```

Or, for an editable install that also provides the console script:

```bash
pip install -e ".[dev]"
```

PyTorch is an optional extra (`pip install -e ".[train]"`) and is only needed
from phase 6 onward. Importing `denoising` never imports it.

## Configuration

All tunable values live in `configs/`; none are hard-coded inside an algorithm.

| File | Holds |
|---|---|
| `configs/dataset.yaml` | image size, split ratios and seed, noise intensity ranges, data paths |
| `configs/training.yaml` | model shape, optimiser, epochs, early stopping, checkpoint paths |
| `configs/inference.yaml` | model path, confidence threshold and fallback, filter parameters, evaluation |
| `configs/hardware.yaml` | pixel stream parameters, simulator, RTL comparison tolerance, synthesis target |

Relative paths are resolved against the repository root, so no configuration
file contains an absolute path.

Validate them:

```bash
python scripts/check_config.py
```

This loads all four files, prints what was parsed, and exits non-zero on the
first invalid value. It is worth running before any long phase.

## Running the tests

```bash
python -m pytest
```

224 tests, all passing as of this commit. `pytest` adds `src/` to the path
itself, so the suite runs in a fresh checkout without installing the package.

## Running the application

```bash
streamlit run app/streamlit_app.py
```

Opens at <http://localhost:8501>. Pick a sample image or upload one, add
synthetic noise, choose the noise class, and the mapped filter is applied and
measured.

The UI holds no image processing of its own — every pixel on screen comes from
`denoising.pipeline.process_image`. Three things it will not do:

- **Invent a confidence.** The classifier does not exist yet, so the class is
  chosen by a person and the page shows `n/a`, not a percentage beside a guess.
- **Invent quality metrics.** MSE, PSNR and SSIM need a clean original. Add
  noise to a sample and the numbers are real; upload an already-noisy photo and
  the page says it cannot measure quality rather than comparing the output to
  its own input.
- **Claim hardware.** Timings are Python on this CPU. No FPGA has run.

## Generating a dataset

```bash
python scripts/generate_dataset.py --synthetic 20 --dry-run
```

Drops clean images into `data/raw/` and run it without `--synthetic` to use
them. Each source produces one `clean` sample plus one per configured intensity
of each noise model, written as PNG with a JSON sidecar, plus
`data/generated/manifest.csv`.

Two behaviours are deliberate:

- **An empty `data/raw/` is an error, not a cue to invent data.** Synthetic
  sources happen only when `--synthetic N` asks for them, and they are marked
  `synthetic` in every manifest row — a classifier trained on patterns has an
  accuracy on patterns.
- **An existing dataset is never merged into.** `--overwrite` replaces it;
  without the flag a populated output directory stops the run.

The split is assigned per *source*, so every noisy version of one image lands in
the same split. The classes are imbalanced by design (one clean sample per
source against one per intensity for each noisy class); training weights them.

## Not yet available

These commands are named in the development spec and will exist in later
phases. They are listed here so nobody looks for them:
`train.py`, `evaluate_classifier.py`, `run_inference.py`, `run_pipeline.py`,
`evaluate_denoising.py`, `generate_report.py`, `make rtl-test`, `make synth`.

## Results

None yet. `results/` is empty by design; every file under it is generated by a
script from a real run, never written by hand. See `docs/results.md`.

## Limitations

- Grayscale, 8-bit only. Colour is not in the initial scope.
- The RTL boundary policy is edge replication, and the software reference must
  use the same policy or the golden-reference comparison is meaningless. The
  config loader rejects any other value in `hardware.yaml` and the test suite
  asserts that the two files agree.
- Fixed 3x3 kernels. The Gaussian kernel is the integer `[1 2 1; 2 4 2; 1 2 1]/16`
  so it can be bit-exact in hardware.
- Stage A integration only is planned first: AI on the PC, the 2-bit noise class
  supplied to the FPGA as an external control signal.

## Documentation

- `docs/architecture.md` — block diagram, data flow, software/hardware partition
- `docs/algorithm.md` — noise models, CNN, filters, selection, metrics
- `docs/verification.md` — golden reference methodology and tolerances
- `docs/hardware.md` — device, clock, interfaces, utilisation, timing, power
- `docs/results.md` — classifier, denoising, baseline and hardware results

## Future work

Phases 3 onward, in the order given in the development spec and in the table
above. The next module is the dataset generator under
`src/denoising/dataset/`: clean sources plus the configured intensities, a
split that keeps every noisy version of one source image in one split, and the
manifest at `data/generated/manifest.csv`.

## License

MIT. See `LICENSE`.
