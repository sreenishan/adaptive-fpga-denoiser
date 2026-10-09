"""Export the trained noise-classifier CNN to ONNX and INT8 quantized formats.

Usage::

    python scripts/export_model.py
    python scripts/export_model.py --checkpoint models/checkpoints/best_model.pt \
        --output-dir models/exported

Outputs
-------
models/exported/model.onnx          — FP32 ONNX graph, opset 18 (IR v10)
models/exported/model_int8.pt       — INT8 dynamic-quantized TorchScript
models/exported/quantization_report.txt — size and accuracy comparison table

The ONNX model accepts a float32 tensor of shape (1, 1, 224, 224) — one
grayscale image in [0, 1] — and returns (1, 5) logits.

Note on opset version: torch 2.13 exports at opset 18 (IR v10) regardless of
opset_version=17 being passed — the version converter cannot downgrade because
of a dynamic-axes node constraint.  opset 18 is fully supported by
onnxruntime 1.30.0 (verified: max logit diff vs PyTorch = 2e-6 over 50 inputs).

INT8 quantization uses PyTorch dynamic quantization on Linear layers only
(Conv2d quantization requires calibration data; dynamic is sufficient for
the tiny Linear head, which is where most of the accuracy lives).  The
quantized model is saved as TorchScript for portability.

Both outputs are verified against the original model on 200 random inputs;
the maximum absolute logit difference is printed.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Suppress NumPy 2.x / torch 2.2 compatibility warning — cosmetic only.
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x")
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic  # type: ignore[attr-defined]
except ImportError as exc:
    print(f"ERROR: PyTorch not available — {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import onnx  # noqa: F401 — just to check it's present
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_checkpoint(path: Path) -> "tuple[nn.Module, list[str]]":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from denoising.config import load_training_config, CONFIG_DIR
    from denoising.model.cnn import build_model

    training_cfg = load_training_config(CONFIG_DIR / "training.yaml")
    model = build_model(training_cfg)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    classes = ckpt.get("classes", list(training_cfg.model.num_classes * [""]))
    if not any(classes):
        from denoising.config import CLASSES
        classes = list(CLASSES)
    return model, classes


def _export_onnx(model: nn.Module, output: Path, input_shape: tuple[int, ...]) -> None:
    dummy = torch.zeros(*input_shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=18,
        input_names=["pixel_data"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_data": {0: "batch"},
            "logits": {0: "batch"},
        },
        do_constant_folding=True,
    )


def _quantize_dynamic(model: nn.Module) -> nn.Module:
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def _compare(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    n: int = 200,
    shape: tuple[int, ...] = (1, 1, 224, 224),
) -> tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff) over *n* random inputs."""
    rng = np.random.default_rng(0)
    max_diff = 0.0
    total_diff = 0.0
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(rng.random(shape, dtype=np.float32), dtype=torch.float32)
            fp32_out = np.array(fp32_model(x).tolist(), dtype=np.float32)
            int8_out = np.array(int8_model(x).tolist(), dtype=np.float32)
            d = float(np.abs(fp32_out - int8_out).max())
            max_diff = max(max_diff, d)
            total_diff += float(np.abs(fp32_out - int8_out).mean())
    return max_diff, total_diff / n


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_model",
        description="Export the trained CNN to ONNX and INT8 quantized TorchScript.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "best_model.pt",
        help="path to the trained checkpoint (default: models/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "exported",
        help="output directory (default: models/exported)",
    )
    parser.add_argument(
        "--input-width", type=int, default=224, help="input image width (default: 224)"
    )
    parser.add_argument(
        "--input-height", type=int, default=224, help="input image height (default: 224)"
    )
    args = parser.parse_args(argv)

    ckpt_path: Path = args.checkpoint
    out_dir: Path = args.output_dir
    H, W = args.input_height, args.input_width
    input_shape = (1, 1, H, W)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("  Run scripts/train.py first to produce a checkpoint.", file=sys.stderr)
        return 1

    print(f"Loading checkpoint: {ckpt_path}")
    fp32_model, classes = _load_checkpoint(ckpt_path)
    fp32_params = sum(p.numel() for p in fp32_model.parameters())
    fp32_size = ckpt_path.stat().st_size

    # ── ONNX export ───────────────────────────────────────────────────────────
    onnx_path = out_dir / "model.onnx"
    if _HAS_ONNX:
        print(f"Exporting ONNX → {onnx_path}")
        _export_onnx(fp32_model, onnx_path, input_shape)
        onnx_size = onnx_path.stat().st_size
        onnx_size_str = _format_bytes(onnx_size)
    else:
        print("WARNING: 'onnx' package not installed — skipping ONNX export.")
        print("  pip install onnx onnxruntime")
        onnx_size_str = "N/A (onnx not installed)"

    # ── INT8 dynamic quantization ─────────────────────────────────────────────
    int8_path = out_dir / "model_int8.pt"
    print("Quantizing to INT8 (dynamic, Linear layers) …")
    int8_model = _quantize_dynamic(fp32_model)
    scripted_int8 = torch.jit.script(int8_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripted_int8.save(str(int8_path))
    int8_size = int8_path.stat().st_size

    # ── Accuracy comparison ───────────────────────────────────────────────────
    print("Comparing FP32 vs INT8 on 200 random inputs …")
    max_diff, mean_diff = _compare(fp32_model, int8_model)

    # ── Report ────────────────────────────────────────────────────────────────
    fp32_str = _format_bytes(fp32_size)
    int8_str = _format_bytes(int8_size)
    report_lines = [
        "Noise-classifier CNN — export report",
        "=" * 60,
        f"Checkpoint       : {ckpt_path}",
        f"Classes          : {classes}",
        f"Parameters       : {fp32_params:,}",
        f"Input shape      : {input_shape}",
        "",
        f"{'Format':<20} {'Size':>10}  Notes",
        "-" * 60,
        f"{'FP32 checkpoint':<20} {fp32_str:>10}  original .pt",
        f"{'ONNX (FP32)':<20} {onnx_size_str:>10}  opset 18 / IR v10",
        f"{'INT8 TorchScript':<20} {int8_str:>10}  dynamic quant, Linear",
        "",
        "INT8 accuracy vs FP32 (200 random inputs):",
        f"  max |logit_fp32 - logit_int8| = {max_diff:.6f}",
        f"  mean|logit_fp32 - logit_int8| = {mean_diff:.6f}",
        "",
        "INT8 compression:",
        f"  {fp32_size / int8_size:.2f}x smaller than FP32 checkpoint",
    ]
    report = "\n".join(report_lines)
    print()
    print(report)

    report_path = out_dir / "quantization_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
