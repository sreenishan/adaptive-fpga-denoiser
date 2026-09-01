"""Preprocessing for the classifier (spec section 10).

```text
load -> grayscale -> resize -> float32 -> normalise -> model input
```

**This is the CNN's input path, not the FPGA's.** The two are deliberately
separate: this one resizes and normalises to whatever the model was trained on,
while the pixel stream that reaches the hardware must arrive as unmodified 8-bit
samples. Filtering a resized, normalised copy and calling the result denoised
would return an image that is not the one anybody submitted.

So the pipeline keeps both: the uint8 image is what gets filtered, and a
preprocessed copy is what gets classified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import ImageConfig
from .dataset.sources import read_gray_image, resize_to
from .noise._common import PIXEL_MAX, GrayImage, validate_image

__all__ = [
    "load_image",
    "to_grayscale",
    "preprocess",
    "to_model_input",
]


def load_image(path: Path | str) -> GrayImage:
    """Load an image from disk as 2-D uint8 grayscale.

    Args:
        path: Image file. Any format OpenCV can decode.

    Returns:
        A 2-D uint8 array at the file's own resolution — resizing is a separate
        step so a caller that wants the original size can have it.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    image = read_gray_image(path)
    if image is None:
        raise ValueError(f"{path} could not be decoded as an image")
    return image


def to_grayscale(image: npt.NDArray[np.uint8]) -> GrayImage:
    """Reduce a colour image to grayscale, or pass a grayscale one through.

    Uses the ITU-R BT.601 luma weights (0.299 R, 0.587 G, 0.114 B), which is
    what OpenCV's ``COLOR_BGR2GRAY`` applies. Channel order is assumed **RGB**
    here, since that is what an image arriving from a UI upload carries; the
    loader above returns grayscale already and never reaches this function.

    Raises:
        ValueError: if the array is not 2-D, or 3-D with 3 or 4 channels.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
        luma = image[:, :, :3].astype(np.float64) @ weights
        return np.rint(np.clip(luma, 0, PIXEL_MAX)).astype(np.uint8)
    raise ValueError(
        f"expected a 2-D grayscale or 3-channel colour image, got shape {image.shape}"
    )


def preprocess(image: GrayImage, config: ImageConfig) -> GrayImage:
    """Bring an image to the configured geometry, still as uint8.

    The last point at which the data is still exactly what the filters will
    see. :func:`to_model_input` takes it the rest of the way.

    Args:
        image: 2-D uint8 grayscale image.
        config: Target geometry.

    Returns:
        A uint8 array of shape ``(config.height, config.width)``.
    """
    image = validate_image(image)
    return resize_to(image, config.width, config.height)


def to_model_input(
    image: GrayImage, config: ImageConfig, *, add_batch: bool = True
) -> npt.NDArray[np.float32]:
    """Convert a uint8 image to a normalised float32 tensor.

    Normalisation is a plain divide by 255 into [0, 1] — not a mean/std
    standardisation, because that would need statistics from the training set
    and none have been computed. When a model is trained with different
    normalisation, this function changes and the checkpoint metadata records it.

    Args:
        image: 2-D uint8 grayscale image, any size.
        config: Target geometry.
        add_batch: Prepend a batch axis, giving ``(1, 1, H, W)`` — the
            channels-first layout PyTorch expects. Without it, ``(1, H, W)``.

    Returns:
        A float32 array in [0, 1].
    """
    resized = preprocess(image, config)
    values = resized.astype(np.float32) / np.float32(PIXEL_MAX)
    values = values[np.newaxis, ...]
    if add_batch:
        values = values[np.newaxis, ...]
    return values
