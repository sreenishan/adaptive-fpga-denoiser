"""Reproducible noise generators (spec section 7).

Three models, one convention: every generator takes a 2-D uint8 grayscale image
and a ``seed``, returns a new array of the same shape and dtype, and never
modifies its input.

```python
from denoising.noise import add_gaussian_noise

noisy = add_gaussian_noise(clean, mean=0.0, sigma=0.08, seed=12345)
```

Intensities (``sigma``, ``variance``) are in normalised [0, 1] image units, so
they mean the same thing whatever the pixel depth; ``amount`` is a fraction of
pixels. The values used to build the dataset live in ``configs/dataset.yaml``
and are never hard-coded here.

Passing ``seed=None`` draws fresh entropy and is therefore *not* reproducible;
pass an int, or thread one :class:`numpy.random.Generator` through a whole
dataset build.
"""

from __future__ import annotations

from .gaussian import add_gaussian_noise
from .salt_pepper import add_salt_pepper_noise
from .speckle import add_speckle_noise

__all__ = [
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "add_speckle_noise",
]
