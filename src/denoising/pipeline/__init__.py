"""The end-to-end adaptive pipeline (spec section 16).

```python
from denoising.pipeline import process_image

result = process_image(image, config, noise_class="salt_pepper")
```

The classifier is injected, so this works with a hand-picked class today and
with a trained model later without the callers changing.
"""

from __future__ import annotations

from .adaptive_pipeline import NoiseClassifier, PipelineResult, process_image

__all__ = ["NoiseClassifier", "PipelineResult", "process_image"]
