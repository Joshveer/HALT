"""
Evaluation metrics and pipeline for HALT.

Provides comprehensive evaluation of hallucination detection and mitigation
performance across multiple metrics.
"""

from .metrics import EvaluationMetrics
from .evaluator import HaltEvaluator

__all__ = ["EvaluationMetrics", "HaltEvaluator"]
