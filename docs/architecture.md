# Architecture

Status: phase 1. The software/hardware partition and the directory layout are
fixed; the modules below marked *planned* do not exist yet.

## Block diagram

```text
                      +---------------------------+
  image or camera --> |  preprocessing (Python)   |
                      |  resize, grayscale, float |
                      +------------+--------------+
                                   |
                      +------------v--------------+
                      |  CNN noise classifier     |   5 classes
                      |  (Python, phase 6)        |   + confidence
                      +------------+--------------+
                                   |
                      +------------v--------------+
                      |  filter selector          |   one mapping,
                      |  (Python, phase 10)       |   one file
                      +------------+--------------+
                                   |  3-bit class code
              PC boundary  - - - - + - - - - - - - - - - - - - -
                                   |
                      +------------v--------------+
                      |  filter_controller.sv     |
                      +------------+--------------+
                                   |
   pixel stream --> line_buffer -> window_3x3 -> +-> median_filter_3x3 ----------+
                                                 +-> gaussian_filter_3x3 --------+--> MUX --> out
                                                 +-> wiener_filter_3x3 ----------+
                                                 +-> adaptive_median_filter_3x3 -+
                                                 +-> bypass ---------------------+
```

## Software / hardware partition

| Stage | Runs where | Why |
|---|---|---|
| Preprocessing for the CNN | PC | Model input geometry, not pixel-rate work |
| Noise classification | PC (stage A, done) | Decouples AI development from RTL development |
| Noise classification on FPGA | HLS project generated (stage C) | Full on-chip pipeline; synthesis requires Vivado HLS |
| Class to filter mapping | PC, one function | Centralised so it cannot drift |
| Per-pixel filtering | FPGA | The only stage that is pixel-rate |
| Quality metrics | PC | Offline evaluation |

Staged integration (spec section 33): stage A is AI on the PC with the class
sent to the FPGA as a 3-bit code.  Stage B (embedded CPU on-board) is future
work.  Stage C (CNN inference on the FPGA via HLS) has been started:
``scripts/hls4ml_convert.py`` generates the HLS C++ project under
``fpga/hls/noise_classifier_hls/`` using hls4ml 1.3.0, targeting an
xc7z020clg400-1 (Zynq-7000) at 100 MHz.  Vivado HLS synthesis is needed
to obtain real resource figures; see ``fpga/hls/hls4ml_report.txt``.

## Data flow

| Item | Representation |
|---|---|
| Source image | uint8 grayscale |
| CNN input | float32, normalised, `image.width` x `image.height` |
| Pixel stream | 8-bit unsigned, one pixel per clock, valid-only initially |
| Noise class | 3 bits: `000` bypass, `001` median, `010` Gaussian, `011` Wiener, `100` adaptive_median |
| Filter output | 8-bit unsigned |

The CNN preprocessing path and the FPGA pixel path are kept separate on purpose:
the first resizes and normalises, the second must not.

## Module map

| Path | Role | State |
|---|---|---|
| `src/denoising/config.py` | typed config loading and validation | done |
| `src/denoising/logging_utils.py` | logging setup | done |
| `src/denoising/cli.py` | `check_config` entry point | done |
| `src/denoising/noise/` | salt-pepper, Gaussian, speckle, Rician generators | done |
| `src/denoising/dataset/` | sources, generation, split, manifest | done |
| `src/denoising/dataset/loader.py` | training-time loading | done |
| `src/denoising/model/` | CNN, train, evaluate, inference | done |
| `src/denoising/severity.py` | noise severity estimation (stage 4) | done |
| `src/denoising/filters/` | median, Gaussian, Wiener, selector | done |
| `src/denoising/pipeline/` | `process_image` end to end | done |
| `src/denoising/metrics/` | MSE, PSNR, SSIM | done |
| `src/denoising/preprocessing.py` | CNN input path | done |
| `app/streamlit_app.py` | demonstration UI (spec 39) | done |
| `rtl/line_buffer.sv` | line buffering | simulated, matches golden |
| `rtl/window_gen.sv` | 3x3 neighbourhood, edge replication | simulated, matches golden |
| `rtl/median_filter.sv`, `gaussian_filter.sv`, `wiener_filter.sv`, `adaptive_median_filter.sv` | four filters | simulated, matches golden (see verification.md) |
| `rtl/filter_controller.sv` | 3-bit code to filter select | simulated, matches golden |
| `rtl/fpga_denoiser_top.sv` | window, filters, MUX, stream protocol | simulated at 224x224; synthesised at 115 MHz on ECP5-25k |
