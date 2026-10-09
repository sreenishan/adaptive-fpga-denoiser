"""Dataset generation, splitting and the manifest (spec sections 8 and 9).

```bash
python scripts/generate_dataset.py --synthetic 20
```

One clean source produces one ``clean`` sample plus one per configured
intensity of each noise model. The split is assigned per source, so every noisy
version of one image lands in the same split — anything else puts near-copies
of training images into the test set.
"""

from __future__ import annotations

from .generate import (
    MANIFEST_COLUMNS,
    SPLITS,
    GenerationSummary,
    PlannedSample,
    assign_splits,
    class_counts,
    generate_dataset,
    plan_dataset,
    render_sample,
    sample_seed,
)
from .loader import load_manifest, sample_class_weights, split_counts
from .sources import MODALITIES, SourceImage, load_sources, synthetic_sources

__all__ = [
    "MANIFEST_COLUMNS",
    "MODALITIES",
    "SPLITS",
    "GenerationSummary",
    "PlannedSample",
    "SourceImage",
    "assign_splits",
    "class_counts",
    "generate_dataset",
    "load_manifest",
    "load_sources",
    "plan_dataset",
    "render_sample",
    "sample_class_weights",
    "sample_seed",
    "split_counts",
    "synthetic_sources",
]
