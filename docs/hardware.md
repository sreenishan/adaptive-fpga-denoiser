# Hardware

Status: phase 1. **No board is selected, no synthesis has been run, and no
hardware has been programmed.** Every figure below is `TBD` and stays that way
until a tool report or a measurement exists.

## Target device

| Item | Value |
|---|---|
| Vendor | not selected |
| Device | not selected |
| Toolchain | not selected |
| Tool version | TBD |

`configs/hardware.yaml` holds `null` for all four. The core RTL is written to be
board-independent; anything board-specific belongs under `fpga/`.

## Pixel stream interface

| Signal | Direction | Width | Meaning |
|---|---|---|---|
| `clk` | in | 1 | pixel clock |
| `rst` | in | 1 | synchronous reset, active high |
| `in_valid` | in | 1 | `in_pixel` is valid this cycle |
| `in_pixel` | in | 8 | unsigned 0-255 |
| `frame_start` | in | 1 | first pixel of a frame |
| `line_start` | in | 1 | first pixel of a line |
| `out_valid` | out | 1 | `out_pixel` is valid this cycle |
| `out_pixel` | out | 8 | unsigned 0-255 |
| `noise_class` | in | 2 | `00` bypass, `01` median, `10` Gaussian, `11` Wiener |

`backpressure` is `false` in the configuration: the first implementation is a
valid-only stream. `in_ready` / `out_ready` are added later, and the config flag
is what says which of the two is built.

## Fixed-point formats

| Quantity | Format |
|---|---|
| Pixel | unsigned 8-bit, 0-255 |
| Gaussian accumulator | unsigned 13-bit (max 255 x 16 = 4080) |
| Wiener intermediates | TBD when the module is written |

## Resource utilisation

| Resource | Used | Available | Utilisation |
|---|---:|---:|---:|
| LUT | TBD | TBD | TBD |
| FF | TBD | TBD | TBD |
| BRAM | TBD | TBD | TBD |
| DSP | TBD | TBD | TBD |
| I/O | TBD | TBD | TBD |

## Timing

| Item | Value |
|---|---|
| Clock constraint | TBD |
| Achieved clock | TBD |
| WNS | TBD |
| TNS | TBD |

## Power

| Item | Value | Source |
|---|---|---|
| Static | TBD | - |
| Dynamic | TBD | - |
| Total | TBD | - |

Estimated and measured power are labelled separately and never mixed.
