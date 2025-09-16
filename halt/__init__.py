"""
HALT: Hallucination Analysis with Logits & Traces

A comprehensive pipeline for detecting and mitigating hallucinations in LLMs.
"""

__version__ = "0.1.0"
__author__ = "HALT Team"

from .pipeline import HaltPipeline

__all__ = ["HaltPipeline"]
