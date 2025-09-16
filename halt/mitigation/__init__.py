"""
Mitigation strategies for hallucination correction.

Includes self-consistency methods and lightweight RAG-based
ground truth checking for hallucination mitigation.
"""

from .self_consistency import SelfConsistencyMitigator
from .rag_checker import LightweightRAGChecker

__all__ = ["SelfConsistencyMitigator", "LightweightRAGChecker"]
