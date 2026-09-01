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
                      |  CNN noise classifier     |   4 classes
                      |  (Python, phase 6)        |   + confidence
                      +------------+--------------+
                                   |
                      +------------v--------------+
                      |  filter selector          |   one mapping,
                      |  (Python, phase 10)       |   one file
                      +------------+--------------+
                                   |  2-bit class code
              PC boundary  - - - - + - - - - - - - - - - - - - -
                                   |
                      +------------v--------------+
                      |  filter_controller.sv     |
                      +------------+--------------+
                                   |
   pixel stream --> line_buffer -> window_3x3 -> +-> median_filter_3x3 --+
                                                 +-> gaussian_filter_3x3 +--> MUX --> out
                                                 +-> wiener_filter_3x3 --+
                                                 +-> bypass -------------+
```

## Software / hardware partition

| Stage | Runs where | Why |
|---|---|---|
| Preprocessing for the CNN | PC | Model input geometry, not pixel-rate work |
| Noise classification | PC (stage A) | Decouples AI development from RTL development |
| Class to filter mapping | PC, one function | Centralised so it cannot drift |
| Per-pixel filtering | FPGA | The only stage that is pixel-rate |
| Quality metrics | PC | Offline evaluation |

Staged integration (spec section 33): stage A is AI on the PC with the class
sent to the FPGA as a 2-bit code. Stages B (embedded CPU) and C (inference on
the FPGA) come later and are not attempted before stage A works.

## Data flow

| Item | Representation |
|---|---|
| Source image | uint8 grayscale |
| CNN input | float32, normalised, `image.width` x `image.height` |
| Pixel stream | 8-bit unsigned, one pixel per clock, valid-only initially |
| Noise class | 2 bits: `00` bypass, `01` median, `10` Gaussian, `11` Wiener |
| Filter output | 8-bit unsigned |

The CNN preprocessing path and the FPGA pixel path are kept separate on purpose:
the first resizes and normalises, the second must not.

## Module map

| Path | Role | State |
|---|---|---|
| `src/denoising/config.py` | typed config loading and validation | done |
| `src/denoising/logging_utils.py` | logging setup | done |
| `src/denoising/cli.py` | `check_config` entry point | done |
| `src/denoising/noise/` | salt-pepper, Gaussian, speckle generators | done |
| `src/denoising/dataset/` | sources, generation, split, manifest | done |
| `src/denoising/dataset/loader.py` | training-time loading | planned |
| `src/denoising/model/` | CNN, train, evaluate, inference | planned |
| `src/denoising/filters/` | median, Gaussian, Wiener, selector | done |
| `src/denoising/pipeline/` | `process_image` end to end | done |
| `src/denoising/metrics/` | MSE, PSNR, SSIM | done |
| `src/denoising/preprocessing.py` | CNN input path | done |
| `app/streamlit_app.py` | demonstration UI (spec 39) | done |
| `rtl/common/line_buffer.sv` | 3-line buffering | planned |
| `rtl/common/window_3x3.sv` | 3x3 neighbourhood, edge replication | planned |
| `rtl/filters/*.sv` | the three filters | planned |
| `rtl/control/filter_controller.sv` | 2-bit class to filter select | planned |
| `rtl/top/adaptive_denoiser_top.sv` | window, filters, MUX | planned |
