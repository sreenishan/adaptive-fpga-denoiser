# Algorithms

Status: phases 1-2. Parameters are configured in `configs/`; the sections
below marked with a later phase are not implemented yet.

## Noise models (phase 2, implemented)

`src/denoising/noise/`. Every generator takes a 2-D uint8 image and a `seed`,
returns a new array of the same shape and dtype, and never modifies its input.

| Model | Form | Parameters |
|---|---|---|
| Salt and pepper | replace a fraction of pixels with 0 or 255 | `amount`, `salt_vs_pepper` |
| Gaussian | `out = clip(in + N(mean, sigma^2))` | `mean`, `sigma` |
| Speckle | `out = clip(in + in * N(0, variance))` | `variance` |

`sigma` and `variance` are in normalised [0, 1] image units, not 0-255 counts:
`sigma = 0.08` is 8% of full scale, about 20 grey levels. The speckle argument
is a **variance**, not a standard deviation; a test asserts the empirical spread
matches `sqrt(variance)`, because confusing the two changes the intensity by a
square root and looks like a mislabelled dataset rather than a bug.

Three decisions worth knowing:

- **Salt and pepper corrupts an exact count**, `round(amount * pixels)` chosen
  without replacement, not a per-pixel coin flip. A fixed count means the label
  "5% salt and pepper" describes that sample, not the distribution it came from.
- **Clipping happens before the conversion back to uint8.** uint8 arithmetic
  wraps silently, so an unclipped bright pixel would come out dark — the noise
  would look like impulse noise in every class. Rounding is `numpy.rint`
  (half to even), the single rounding rule used wherever a float becomes a pixel.
- **Speckle leaves black pixels exactly black**, since the perturbation scales
  with the pixel. That is the property that distinguishes it from additive
  noise, and a test guards it: if it ever fails, the model has become additive.

Intensities come from the low/medium/high lists in `configs/dataset.yaml`; a
single fixed noise level would teach the classifier to detect loudness rather
than kind, and a test asserts every configured level actually changes pixels.

**Reproducibility caveat.** A seed reproduces an image exactly for a given NumPy
version. NumPy does not guarantee that `Generator`'s distribution streams stay
identical across feature releases, so the manifest records the seed *and* the
parameters, and a regenerated dataset should be treated as a new dataset rather
than assumed bit-identical to one built on another machine.

## Dataset construction (phases 3-4, implemented)

`src/denoising/dataset/`. One clean source produces one `clean` sample plus one
per configured intensity of each noise model — with three intensities, ten
samples per source.

| Concern | Decision |
|---|---|
| Split | assigned per source, 70/15/15, largest remainder so counts sum exactly |
| Leakage | every sample of one source shares its split |
| Seed | `blake2b(master_seed, source_id, noise_type, level)`, not a counter |
| Order | sources sorted before shuffling, so filesystem order cannot decide a split |
| Resize | before noise, never after — resampling low-passes the noise |
| Format | PNG, lossless; JPEG artefacts would be one more thing to learn |
| Metadata | JSON sidecar per sample plus `manifest.csv`, written in one pass |

Manifest columns are fixed by the spec: `path`, `split`, `label`, `source_id`,
`noise_type`, `noise_parameter`, `seed`. `label` is the index in `CLASSES`,
`noise_parameter` is the primary intensity (`amount`, `sigma` or `variance`) and
is empty for the clean class; the full parameter set is in the sidecar.

**The classes are imbalanced by construction** — 1:3:3:3 with three intensities,
since a clean source has only one clean version. Training must weight the
classes rather than duplicate clean rows, which would add rows without adding
information.

**Sources may be synthetic.** With no images in `data/raw/`, generation fails
with an error naming both options rather than quietly inventing data;
`--synthetic N` builds deterministic gradient/texture/shape patterns, and every
row derived from them says `synthetic`. Accuracy measured on synthetic sources
is accuracy on synthetic sources.

## Classifier (phase 6)

Four classes, in label order: `clean`, `salt_pepper`, `gaussian`, `speckle`.
Lightweight conv/ReLU/pool stack ending in a 4-way output, kept small enough to
remain a candidate for FPGA inference later.

Confidence is reported alongside the prediction. Below `confidence.threshold`
the pipeline uses `confidence.fallback` and says so; a low-confidence prediction
is never hidden.

## Filters (phase 9)

All three operate on a 3x3 neighbourhood with edge replication at the borders,
matching the RTL window generator.

**Median.** Output is the median of the nine window pixels. Exact in both
implementations, so RTL comparison is bit-exact.

**Gaussian.** Fixed integer kernel

```text
1 2 1
2 4 2   / 16
1 2 1
```

chosen so the hardware needs shifts and adds only. Maximum accumulator value is
255 x 16 = 4080, so 13 bits are required. Bit-exactness depends on both sides
rounding identically; the rounding rule is stated with the implementation.

**Wiener / adaptive.** Local-statistics form over the same window:

```text
out = mean + max(var - noise_var, 0) / max(var, noise_var) * (pixel - mean)
```

where `mean` and `var` are the local window statistics and `noise_var` is either
configured or estimated. The denominator is clamped away from zero, so a flat
window returns the local mean rather than dividing by nothing. The hardware
version will approximate the division; whichever approximation is chosen gets
documented here and its error against this reference reported in
`docs/verification.md`. It stays a Wiener filter.

## Filter selection (phase 10)

```text
clean       -> bypass
salt_pepper -> median
gaussian    -> gaussian
speckle     -> wiener
```

One mapping in one module, used by the Python pipeline and by the 2-bit code
sent to `filter_controller.sv`. Duplicating it is how the two ends come to
disagree.

## Metrics (phase 12)

```text
MSE  = mean((reference - output)^2)
PSNR = 10 * log10(MAX^2 / MSE)          MAX = 255 for uint8
SSIM = scikit-image structural_similarity
```

Zero MSE means identical images; PSNR is reported as infinite rather than as a
large finite number that looks measured.
