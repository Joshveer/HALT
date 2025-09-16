"""
Unified detection logic for HALT.

Combines token-level uncertainty and reasoning-level disagreement
into a unified hallucination detector.
"""

from .unified_detector import UnifiedDetector
from .threshold_optimizer import ThresholdOptimizer

__all__ = ["UnifiedDetector", "ThresholdOptimizer"]
