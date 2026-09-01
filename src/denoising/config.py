"""Typed configuration loading and validation.

Every tunable number in this project lives in ``configs/*.yaml``; nothing is
hard-coded inside an algorithm. This module turns those files into frozen
dataclasses and refuses malformed values loudly at load time, so a bad split
ratio or an even kernel size fails before a dataset is generated rather than
halfway through a training run.

Relative paths in the YAML are resolved against :data:`PROJECT_ROOT`, so no
absolute path is ever written into a configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

__all__ = [
    "CLASSES",
    "BOUNDARY_MODES",
    "FALLBACK_FILTERS",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "ConfigError",
    "ImageConfig",
    "SplitConfig",
    "SaltPepperNoiseConfig",
    "GaussianNoiseConfig",
    "SpeckleNoiseConfig",
    "NoiseConfig",
    "DatasetPaths",
    "DatasetConfig",
    "ModelConfig",
    "EarlyStoppingConfig",
    "TrainingConfig",
    "MedianFilterConfig",
    "GaussianFilterConfig",
    "WienerFilterConfig",
    "FiltersConfig",
    "ConfidenceConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "StreamConfig",
    "SimulationConfig",
    "SynthesisConfig",
    "HardwareConfig",
    "load_yaml",
    "load_dataset_config",
    "load_training_config",
    "load_inference_config",
    "load_hardware_config",
]

#: The four classes the system distinguishes, in label-index order. A class's
#: index here is its integer label everywhere else in the project.
CLASSES: Final[tuple[str, ...]] = ("clean", "salt_pepper", "gaussian", "speckle")

#: Boundary handling supported by the software reference filters. The mode used
#: for RTL comparison must match ``hardware.yaml``'s ``boundary_policy``.
BOUNDARY_MODES: Final[tuple[str, ...]] = ("replicate", "zero", "reflect")

#: What a below-threshold prediction may fall back to.
FALLBACK_FILTERS: Final[tuple[str, ...]] = ("bypass", "median", "gaussian", "wiener")

_OPTIMIZERS: Final[tuple[str, ...]] = ("adam", "adamw", "sgd")
_SCHEDULERS: Final[tuple[Any, ...]] = (None, "cosine", "step", "plateau")
_DEVICES: Final[tuple[str, ...]] = ("auto", "cpu", "cuda")
_SIMULATORS: Final[tuple[str, ...]] = ("verilator", "iverilog")
_VENDORS: Final[tuple[Any, ...]] = (None, "xilinx", "intel")

#: Repository root: ``<root>/src/denoising/config.py`` -> ``<root>``.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is missing a key or holds a bad value."""


# --------------------------------------------------------------------------- #
# Primitive accessors. Each carries the dotted path of the offending key so the
# error names the line to fix.
# --------------------------------------------------------------------------- #


def _dotted(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _get(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping):
        raise ConfigError(
            f"{where or 'config'} must be a mapping, got {type(mapping).__name__}"
        )
    if key not in mapping:
        raise ConfigError(f"missing required key '{_dotted(where, key)}'")
    return mapping[key]


def _section(mapping: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = _get(mapping, key, where)
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"'{_dotted(where, key)}' must be a mapping, got {type(value).__name__}"
        )
    return value


def _int(
    mapping: Mapping[str, Any], key: str, where: str, *, minimum: int | None = None
) -> int:
    value = _get(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{_dotted(where, key)}' must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{_dotted(where, key)}' must be >= {minimum}, got {value}")
    return value


def _float(
    mapping: Mapping[str, Any],
    key: str,
    where: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
) -> float:
    value = _get(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{_dotted(where, key)}' must be a number, got {value!r}")
    value = float(value)
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        bound = ">" if exclusive_min else ">="
        raise ConfigError(f"'{_dotted(where, key)}' must be {bound} {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"'{_dotted(where, key)}' must be <= {maximum}, got {value}")
    return value


def _bool(mapping: Mapping[str, Any], key: str, where: str) -> bool:
    value = _get(mapping, key, where)
    if not isinstance(value, bool):
        raise ConfigError(f"'{_dotted(where, key)}' must be true or false, got {value!r}")
    return value


def _choice(
    mapping: Mapping[str, Any],
    key: str,
    where: str,
    allowed: Sequence[Any],
    *,
    lower: bool = True,
) -> Any:
    value = _get(mapping, key, where)
    if lower and isinstance(value, str):
        value = value.lower()
    if value not in allowed:
        names = ", ".join("null" if a is None else str(a) for a in allowed)
        raise ConfigError(f"'{_dotted(where, key)}' must be one of [{names}], got {value!r}")
    return value


def _float_list(
    mapping: Mapping[str, Any],
    key: str,
    where: str,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    value = _get(mapping, key, where)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"'{_dotted(where, key)}' must be a list of numbers, got {value!r}")
    if not value:
        raise ConfigError(f"'{_dotted(where, key)}' must not be empty")
    out: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"'{_dotted(where, key)}[{index}]' must be a number, got {item!r}")
        item = float(item)
        if not minimum < item <= maximum:
            raise ConfigError(
                f"'{_dotted(where, key)}[{index}]' must be in ({minimum}, {maximum}], got {item}"
            )
        out.append(item)
    return tuple(out)


def _path(mapping: Mapping[str, Any], key: str, where: str, *, root: Path) -> Path:
    value = _get(mapping, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"'{_dotted(where, key)}' must be a non-empty path string, got {value!r}"
        )
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _kernel_size(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = _int(mapping, key, where, minimum=3)
    if value % 2 == 0:
        raise ConfigError(f"'{_dotted(where, key)}' must be odd, got {value}")
    return value


def _optional_float(mapping: Mapping[str, Any], key: str, where: str) -> float | None:
    if mapping.get(key) is None:
        return None
    return _float(mapping, key, where, minimum=0.0)


def _optional_str(mapping: Mapping[str, Any], key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"'{_dotted(where, key)}' must be null or a non-empty string, got {value!r}"
        )
    return value


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageConfig:
    """Model input geometry. Distinct from the RTL frame size in hardware.yaml."""

    width: int
    height: int
    grayscale: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "image") -> "ImageConfig":
        return cls(
            width=_int(data, "width", where, minimum=1),
            height=_int(data, "height", where, minimum=1),
            grayscale=_bool(data, "grayscale", where),
        )


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test proportions and the seed that makes them repeatable."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    seed: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "split") -> "SplitConfig":
        split = cls(
            train_ratio=_float(
                data, "train_ratio", where, minimum=0.0, maximum=1.0, exclusive_min=True
            ),
            validation_ratio=_float(
                data, "validation_ratio", where, minimum=0.0, maximum=1.0, exclusive_min=True
            ),
            test_ratio=_float(
                data, "test_ratio", where, minimum=0.0, maximum=1.0, exclusive_min=True
            ),
            seed=_int(data, "seed", where, minimum=0),
        )
        total = split.train_ratio + split.validation_ratio + split.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"'{where}' ratios must sum to 1.0, got {total:.6f}")
        return split


@dataclass(frozen=True)
class SaltPepperNoiseConfig:
    """Salt-and-pepper intensities. ``amounts`` is the fraction of pixels hit."""

    amounts: tuple[float, ...]
    salt_vs_pepper: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> "SaltPepperNoiseConfig":
        return cls(
            amounts=_float_list(data, "amounts", where, minimum=0.0, maximum=1.0),
            salt_vs_pepper=_float(data, "salt_vs_pepper", where, minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Additive Gaussian intensities, in normalised [0, 1] image units."""

    mean: float
    sigmas: tuple[float, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> "GaussianNoiseConfig":
        return cls(
            mean=_float(data, "mean", where),
            sigmas=_float_list(data, "sigmas", where, minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True)
class SpeckleNoiseConfig:
    """Multiplicative speckle intensities, in normalised [0, 1] image units."""

    variances: tuple[float, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> "SpeckleNoiseConfig":
        return cls(variances=_float_list(data, "variances", where, minimum=0.0, maximum=1.0))


@dataclass(frozen=True)
class NoiseConfig:
    """Intensity ranges for the three noise models."""

    salt_pepper: SaltPepperNoiseConfig
    gaussian: GaussianNoiseConfig
    speckle: SpeckleNoiseConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "noise") -> "NoiseConfig":
        return cls(
            salt_pepper=SaltPepperNoiseConfig.from_mapping(
                _section(data, "salt_pepper", where), f"{where}.salt_pepper"
            ),
            gaussian=GaussianNoiseConfig.from_mapping(
                _section(data, "gaussian", where), f"{where}.gaussian"
            ),
            speckle=SpeckleNoiseConfig.from_mapping(
                _section(data, "speckle", where), f"{where}.speckle"
            ),
        )


@dataclass(frozen=True)
class DatasetPaths:
    """Where clean sources are read from and generated samples are written."""

    raw_dir: Path
    generated_dir: Path
    manifest: Path

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], where: str, *, root: Path
    ) -> "DatasetPaths":
        return cls(
            raw_dir=_path(data, "raw_dir", where, root=root),
            generated_dir=_path(data, "generated_dir", where, root=root),
            manifest=_path(data, "manifest", where, root=root),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """Contents of ``configs/dataset.yaml``."""

    image: ImageConfig
    split: SplitConfig
    noise: NoiseConfig
    paths: DatasetPaths

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, root: Path = PROJECT_ROOT
    ) -> "DatasetConfig":
        return cls(
            image=ImageConfig.from_mapping(_section(data, "image", "")),
            split=SplitConfig.from_mapping(_section(data, "split", "")),
            noise=NoiseConfig.from_mapping(_section(data, "noise", "")),
            paths=DatasetPaths.from_mapping(_section(data, "paths", ""), "paths", root=root),
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Classifier shape. ``num_classes`` must agree with :data:`CLASSES`."""

    num_classes: int
    input_channels: int
    base_channels: int
    dropout: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "model") -> "ModelConfig":
        num_classes = _int(data, "num_classes", where, minimum=2)
        if num_classes != len(CLASSES):
            raise ConfigError(
                f"'{where}.num_classes' must be {len(CLASSES)} to match CLASSES "
                f"{CLASSES}, got {num_classes}"
            )
        return cls(
            num_classes=num_classes,
            input_channels=_int(data, "input_channels", where, minimum=1),
            base_channels=_int(data, "base_channels", where, minimum=1),
            dropout=_float(data, "dropout", where, minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Validation-loss patience settings."""

    enabled: bool
    patience: int
    min_delta: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str) -> "EarlyStoppingConfig":
        return cls(
            enabled=_bool(data, "enabled", where),
            patience=_int(data, "patience", where, minimum=1),
            min_delta=_float(data, "min_delta", where, minimum=0.0),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Contents of ``configs/training.yaml``."""

    model: ModelConfig
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str | None
    seed: int
    device: str
    num_workers: int
    resume: bool
    checkpoint_dir: Path
    metadata_dir: Path
    early_stopping: EarlyStoppingConfig

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, root: Path = PROJECT_ROOT
    ) -> "TrainingConfig":
        where = "training"
        training = _section(data, "training", "")
        scheduler = training.get("scheduler")
        if isinstance(scheduler, str):
            scheduler = scheduler.lower()
        if scheduler not in _SCHEDULERS:
            names = ", ".join("null" if s is None else str(s) for s in _SCHEDULERS)
            raise ConfigError(f"'{where}.scheduler' must be one of [{names}], got {scheduler!r}")
        return cls(
            model=ModelConfig.from_mapping(_section(data, "model", "")),
            batch_size=_int(training, "batch_size", where, minimum=1),
            epochs=_int(training, "epochs", where, minimum=1),
            learning_rate=_float(training, "learning_rate", where, minimum=0.0, exclusive_min=True),
            weight_decay=_float(training, "weight_decay", where, minimum=0.0),
            optimizer=_choice(training, "optimizer", where, _OPTIMIZERS),
            scheduler=scheduler,
            seed=_int(training, "seed", where, minimum=0),
            device=_choice(training, "device", where, _DEVICES),
            num_workers=_int(training, "num_workers", where, minimum=0),
            resume=_bool(training, "resume", where),
            checkpoint_dir=_path(training, "checkpoint_dir", where, root=root),
            metadata_dir=_path(training, "metadata_dir", where, root=root),
            early_stopping=EarlyStoppingConfig.from_mapping(
                _section(training, "early_stopping", where), f"{where}.early_stopping"
            ),
        )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MedianFilterConfig:
    """Median filter parameters."""

    kernel_size: int


@dataclass(frozen=True)
class GaussianFilterConfig:
    """Gaussian filter parameters.

    ``integer_kernel`` selects the fixed binomial kernel the RTL implements,
    which is exact and ignores ``sigma``. The two are different filters and the
    choice is explicit for that reason.
    """

    kernel_size: int
    sigma: float
    integer_kernel: bool


@dataclass(frozen=True)
class WienerFilterConfig:
    """Wiener filter parameters.

    ``noise_variance`` is ``None`` when the variance is to be estimated from the
    image's local statistics rather than asserted as a value nobody measured.
    """

    kernel_size: int
    noise_variance: float | None


@dataclass(frozen=True)
class FiltersConfig:
    """Software reference filter parameters and the shared boundary policy."""

    boundary_mode: str
    median: MedianFilterConfig
    gaussian: GaussianFilterConfig
    wiener: WienerFilterConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "filters") -> "FiltersConfig":
        median = _section(data, "median", where)
        gaussian = _section(data, "gaussian", where)
        wiener = _section(data, "wiener", where)
        return cls(
            boundary_mode=_choice(data, "boundary_mode", where, BOUNDARY_MODES),
            median=MedianFilterConfig(
                kernel_size=_kernel_size(median, "kernel_size", f"{where}.median")
            ),
            gaussian=GaussianFilterConfig(
                kernel_size=_kernel_size(gaussian, "kernel_size", f"{where}.gaussian"),
                sigma=_float(
                    gaussian, "sigma", f"{where}.gaussian", minimum=0.0, exclusive_min=True
                ),
                integer_kernel=_bool(gaussian, "integer_kernel", f"{where}.gaussian"),
            ),
            wiener=WienerFilterConfig(
                kernel_size=_kernel_size(wiener, "kernel_size", f"{where}.wiener"),
                noise_variance=_optional_float(wiener, "noise_variance", f"{where}.wiener"),
            ),
        )


@dataclass(frozen=True)
class ConfidenceConfig:
    """Threshold below which a prediction is reported but not acted on."""

    threshold: float
    fallback: str

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], where: str = "confidence"
    ) -> "ConfidenceConfig":
        return cls(
            threshold=_float(data, "threshold", where, minimum=0.0, maximum=1.0),
            fallback=_choice(data, "fallback", where, FALLBACK_FILTERS),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Image-quality evaluation settings."""

    max_pixel_value: int
    results_dir: Path

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        where: str = "evaluation",
        *,
        root: Path = PROJECT_ROOT,
    ) -> "EvaluationConfig":
        return cls(
            max_pixel_value=_int(data, "max_pixel_value", where, minimum=1),
            results_dir=_path(data, "results_dir", where, root=root),
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Contents of ``configs/inference.yaml``."""

    model_path: Path
    device: str
    image: ImageConfig
    confidence: ConfidenceConfig
    filters: FiltersConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, root: Path = PROJECT_ROOT
    ) -> "InferenceConfig":
        return cls(
            model_path=_path(data, "model_path", "", root=root),
            device=_choice(data, "device", "", _DEVICES),
            image=ImageConfig.from_mapping(_section(data, "image", "")),
            confidence=ConfidenceConfig.from_mapping(_section(data, "confidence", "")),
            filters=FiltersConfig.from_mapping(_section(data, "filters", "")),
            evaluation=EvaluationConfig.from_mapping(_section(data, "evaluation", ""), root=root),
        )


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StreamConfig:
    """Board-independent pixel stream parameters."""

    pixel_width: int
    image_width: int
    image_height: int
    boundary_policy: str
    backpressure: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = "stream") -> "StreamConfig":
        pixel_width = _int(data, "pixel_width", where, minimum=1)
        if pixel_width != 8:
            raise ConfigError(
                f"'{where}.pixel_width' must be 8: only 8-bit grayscale is "
                f"implemented, got {pixel_width}"
            )
        return cls(
            pixel_width=pixel_width,
            image_width=_int(data, "image_width", where, minimum=3),
            image_height=_int(data, "image_height", where, minimum=3),
            # The window generator replicates edge pixels; nothing else is built.
            boundary_policy=_choice(data, "boundary_policy", where, ("replicate",)),
            backpressure=_bool(data, "backpressure", where),
        )


@dataclass(frozen=True)
class SimulationConfig:
    """RTL simulation directories and the per-filter comparison tolerance.

    ``max_abs_error`` is the largest absolute pixel difference tolerated against
    the Python golden reference; 0 means bit-exact.
    """

    simulator: str
    input_dir: Path
    expected_dir: Path
    output_dir: Path
    max_abs_error: Mapping[str, int]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        where: str = "simulation",
        *,
        root: Path = PROJECT_ROOT,
    ) -> "SimulationConfig":
        tolerances = _section(data, "max_abs_error", where)
        parsed = {
            name: _int(tolerances, name, f"{where}.max_abs_error", minimum=0)
            for name in ("median", "gaussian", "wiener")
        }
        return cls(
            simulator=_choice(data, "simulator", where, _SIMULATORS),
            input_dir=_path(data, "input_dir", where, root=root),
            expected_dir=_path(data, "expected_dir", where, root=root),
            output_dir=_path(data, "output_dir", where, root=root),
            max_abs_error=parsed,
        )


@dataclass(frozen=True)
class SynthesisConfig:
    """Synthesis target.

    Every field is ``None`` until a real tool run fills it in. These are never
    populated by hand.
    """

    vendor: str | None
    device: str | None
    clock_mhz: float | None
    tool_version: str | None

    @property
    def configured(self) -> bool:
        """True when a board has actually been selected."""
        return self.vendor is not None and self.device is not None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], where: str = "synthesis"
    ) -> "SynthesisConfig":
        vendor = data.get("vendor")
        if isinstance(vendor, str):
            vendor = vendor.lower()
        if vendor not in _VENDORS:
            names = ", ".join("null" if v is None else str(v) for v in _VENDORS)
            raise ConfigError(f"'{where}.vendor' must be one of [{names}], got {vendor!r}")
        clock_mhz = data.get("clock_mhz")
        if clock_mhz is not None:
            clock_mhz = _float(data, "clock_mhz", where, minimum=0.0, exclusive_min=True)
        return cls(
            vendor=vendor,
            device=_optional_str(data, "device", where),
            clock_mhz=clock_mhz,
            tool_version=_optional_str(data, "tool_version", where),
        )


@dataclass(frozen=True)
class HardwareConfig:
    """Contents of ``configs/hardware.yaml``."""

    stream: StreamConfig
    simulation: SimulationConfig
    synthesis: SynthesisConfig

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, root: Path = PROJECT_ROOT
    ) -> "HardwareConfig":
        return cls(
            stream=StreamConfig.from_mapping(_section(data, "stream", "")),
            simulation=SimulationConfig.from_mapping(_section(data, "simulation", ""), root=root),
            synthesis=SynthesisConfig.from_mapping(_section(data, "synthesis", "")),
        )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_yaml(path: Path | str) -> Mapping[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Raises:
        ConfigError: if the file is missing, unparseable, empty, or does not
            hold a mapping at the top level.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message varies by parser
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_dataset_config(
    path: Path | str | None = None, *, root: Path = PROJECT_ROOT
) -> DatasetConfig:
    """Load ``configs/dataset.yaml``, or *path* if given."""
    return DatasetConfig.from_mapping(load_yaml(path or CONFIG_DIR / "dataset.yaml"), root=root)


def load_training_config(
    path: Path | str | None = None, *, root: Path = PROJECT_ROOT
) -> TrainingConfig:
    """Load ``configs/training.yaml``, or *path* if given."""
    return TrainingConfig.from_mapping(load_yaml(path or CONFIG_DIR / "training.yaml"), root=root)


def load_inference_config(
    path: Path | str | None = None, *, root: Path = PROJECT_ROOT
) -> InferenceConfig:
    """Load ``configs/inference.yaml``, or *path* if given."""
    return InferenceConfig.from_mapping(
        load_yaml(path or CONFIG_DIR / "inference.yaml"), root=root
    )


def load_hardware_config(
    path: Path | str | None = None, *, root: Path = PROJECT_ROOT
) -> HardwareConfig:
    """Load ``configs/hardware.yaml``, or *path* if given."""
    return HardwareConfig.from_mapping(load_yaml(path or CONFIG_DIR / "hardware.yaml"), root=root)
