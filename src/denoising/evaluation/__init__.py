"""Denoising evaluation — baseline comparison and per-class metrics."""

from .denoising_eval import ClassMetrics, EvalReport, StrategyResult, evaluate

__all__ = ["ClassMetrics", "EvalReport", "StrategyResult", "evaluate"]
