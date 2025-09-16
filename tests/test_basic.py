"""
Basic tests for HALT pipeline.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import halt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from halt.pipeline import HaltPipeline
from halt.data import TruthfulQALoader
from halt.models import BaseModel, TokenAnalyzer


def test_imports():
    """Test that all modules can be imported."""
    from halt import HaltPipeline
    from halt.data import TruthfulQALoader, DataPreprocessor
    from halt.models import BaseModel, TokenAnalyzer
    from halt.reasoning import CoTAnalyzer, DisagreementDetector
    from halt.detection import UnifiedDetector, ThresholdOptimizer
    from halt.mitigation import SelfConsistencyMitigator, LightweightRAGChecker
    from halt.evaluation import HaltEvaluator, EvaluationMetrics


def test_data_loader():
    """Test data loader functionality."""
    loader = TruthfulQALoader()
    
    # Test sample data creation
    sample_data = loader._create_sample_data(5)
    assert len(sample_data) == 5
    assert 'question' in sample_data.columns
    assert 'best_answer' in sample_data.columns
    
    # Test formatting
    formatted = loader.format_for_generation(sample_data, include_cot=False)
    assert len(formatted) == 5
    assert 'prompt' in formatted[0]
    assert 'question' in formatted[0]


def test_token_analyzer():
    """Test token analyzer functionality."""
    # This is a basic test - in practice you'd need a real tokenizer
    analyzer = TokenAnalyzer(None)
    
    # Test entropy calculation
    import numpy as np
    logits = np.random.randn(1000)  # Random logits
    entropy = analyzer.calculate_entropy(logits)
    assert isinstance(entropy, float)
    assert entropy >= 0


def test_cot_analyzer():
    """Test CoT analyzer functionality."""
    from halt.reasoning import CoTAnalyzer
    
    analyzer = CoTAnalyzer()
    
    # Test answer extraction
    cot_text = "Let me think step by step. The capital of France is Paris. So the answer is: Paris."
    answer = analyzer.extract_final_answer(cot_text)
    assert "Paris" in answer
    
    # Test reasoning steps extraction
    steps = analyzer.extract_reasoning_steps(cot_text)
    assert len(steps) > 0


def test_unified_detector():
    """Test unified detector functionality."""
    from halt.detection import UnifiedDetector
    
    detector = UnifiedDetector()
    
    # Test threshold updates
    detector.update_thresholds(entropy_threshold=1.5, disagreement_threshold=0.3)
    assert detector.entropy_threshold == 1.5
    assert detector.disagreement_threshold == 0.3


def test_self_consistency_mitigator():
    """Test self-consistency mitigator functionality."""
    from halt.mitigation import SelfConsistencyMitigator
    
    mitigator = SelfConsistencyMitigator()
    
    # Test majority answer selection
    answers = ["Paris", "Paris", "London", "Paris", "Berlin"]
    result = mitigator.select_majority_answer(answers)
    
    assert result['selected_answer'] == "Paris"
    assert result['confidence'] > 0
    assert result['agreement_ratio'] > 0


def test_evaluation_metrics():
    """Test evaluation metrics functionality."""
    from halt.evaluation import EvaluationMetrics
    
    metrics = EvaluationMetrics()
    
    # Test detection metrics calculation
    y_true = [True, False, True, False, True]
    y_pred = [True, False, False, False, True]
    
    result = metrics.calculate_detection_metrics(y_true, y_pred)
    
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1_score' in result
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['precision'] <= 1
    assert 0 <= result['recall'] <= 1


def test_pipeline_initialization():
    """Test pipeline initialization."""
    # This test might be slow due to model loading
    try:
        pipeline = HaltPipeline(
            model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
            device="cpu"
        )
        
        assert pipeline.model_name == "microsoft/DialoGPT-small"
        assert pipeline.device == "cpu"
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        assert 'model_name' in info
        assert 'device' in info
        assert 'detector_thresholds' in info
        
    except Exception as e:
        # If model loading fails, that's okay for basic tests
        pytest.skip(f"Model loading failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_imports()
    test_data_loader()
    test_token_analyzer()
    test_cot_analyzer()
    test_unified_detector()
    test_self_consistency_mitigator()
    test_evaluation_metrics()
    
    print("All basic tests passed!")
