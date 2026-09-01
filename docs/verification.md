# Verification

Status: phase 1. No RTL exists yet, so no comparison has been run. Nothing in
this document is a result.

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

Written to `results/rtl/verification.json`, alongside `reference.png`,
`rtl_output.png` and `diff.png`.

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

## Results

| Filter | Max abs error | Mean abs error | Mismatched pixels | Status |
|---|---|---|---|---|
| Median | TBD | TBD | TBD | not run |
| Gaussian | TBD | TBD | TBD | not run |
| Wiener | TBD | TBD | TBD | not run |
