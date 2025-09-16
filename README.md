# HALT: Hallucination Analysis with Logits & Traces

Pipeline that analyzes tokenization uncertaintiy (input) and CoT disagreement (reasoning) to detect hallucinations in LLMs.

## Overview

HALT combines two complementary approaches to detect hallucinations:
1. **Token-level uncertainty**: Analyzing logit entropy and tokenization fragility
2. **Reasoning-level disagreement**: Measuring consistency across multiple chain-of-thought reasoning paths

The system also includes lightweight RAG-based ground truth checking for mitigation.

## Features

- **Token Entropy Analysis**: Measures uncertainty at the token level using logit probabilities
- **Chain-of-Thought Disagreement**: Generates multiple reasoning traces and measures convergence
- **Lightweight RAG Checker**: Uses Wikipedia retrieval for ground truth approximation
- **Self-Consistency Mitigation**: Selects majority answers from multiple generations
- **Comprehensive Evaluation**: Tests on TruthfulQA with multiple metrics
- **Threshold Optimization**: Automatically optimizes detection thresholds

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/halt.git
cd halt

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from halt import HaltPipeline

# Initialize the pipeline
pipeline = HaltPipeline(
    model_name="microsoft/DialoGPT-medium",  # or any HuggingFace model
    device="cpu"  # or "cuda" for GPU
)

# Analyze a single example
result = pipeline.analyze_single_example(
    "What is the capital of France?",
    num_cot_samples=5
)

print(f"Answer: {result['original_answer']}")
print(f"Is Hallucination: {result['is_hallucination']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Full Evaluation

```python
# Run comprehensive evaluation
results = pipeline.run_evaluation(
    subset_size=200,
    num_cot_samples=5,
    entropy_threshold=2.0,
    disagreement_threshold=0.4
)

print(f"Detection F1: {results['detection']['metrics']['f1_score']:.3f}")
print(f"Mitigation Improvement: {results['mitigation']['metrics']['accuracy_improvement']:.3f}")
print(f"Overall Score: {results['system']['overall_score']:.3f}")
```

### Detection Only

```python
# Run detection evaluation
detection_results = pipeline.run_detection_only(
    subset_size=100,
    num_cot_samples=5,
    entropy_threshold=2.0,
    disagreement_threshold=0.4
)

print(f"Detection F1: {detection_results['metrics']['f1_score']:.3f}")
print(f"Detection Precision: {detection_results['metrics']['precision']:.3f}")
print(f"Detection Recall: {detection_results['metrics']['recall']:.3f}")
```

### Mitigation Only

```python
# Run mitigation evaluation
mitigation_results = pipeline.run_mitigation_only(
    subset_size=100,
    num_samples=5
)

print(f"Original Accuracy: {mitigation_results['metrics']['original_accuracy']:.3f}")
print(f"Mitigated Accuracy: {mitigation_results['metrics']['mitigated_accuracy']:.3f}")
print(f"Accuracy Improvement: {mitigation_results['metrics']['accuracy_improvement']:.3f}")
```

## Examples

The `examples/` directory contains several example scripts:

- `basic_usage.py`: Simple example of using HALT
- `full_evaluation.py`: Comprehensive evaluation example
- `custom_model.py`: Using different models and custom parameters

Run examples with:

```bash
python examples/basic_usage.py
python examples/full_evaluation.py
python examples/custom_model.py
```

## Project Structure

```
halt/
├── data/                    # Dataset loading and preprocessing
│   ├── truthfulqa_loader.py
│   └── preprocessing.py
├── models/                  # Model integration and token analysis
│   ├── base_model.py
│   └── token_analyzer.py
├── reasoning/               # Chain-of-thought analysis
│   ├── cot_analyzer.py
│   └── disagreement_detector.py
├── detection/               # Unified detection logic
│   ├── unified_detector.py
│   └── threshold_optimizer.py
├── mitigation/              # Self-consistency and RAG
│   ├── self_consistency.py
│   └── rag_checker.py
├── evaluation/              # Metrics and evaluation
│   ├── metrics.py
│   └── evaluator.py
├── pipeline.py              # Main pipeline orchestration
└── __init__.py
```

## Configuration

### Model Selection

HALT supports any HuggingFace model with logit access. Popular choices include:

- `microsoft/DialoGPT-medium` (recommended for testing)
- `microsoft/DialoGPT-small` (faster, less accurate)
- `gpt2` (smaller, faster)
- `gpt2-medium` (larger, more accurate)

### Detection Thresholds

- `entropy_threshold`: Threshold for token entropy (default: 2.0)
- `disagreement_threshold`: Threshold for CoT disagreement (default: 0.4)
- `uncertainty_threshold`: Threshold for overall uncertainty (default: 0.6)

### Mitigation Parameters

- `num_samples`: Number of samples for self-consistency (default: 5)
- `similarity_threshold`: RAG similarity threshold (default: 0.3)
- `min_agreement`: Minimum agreement for majority selection (default: 0.4)

## Evaluation Metrics

### Detection Metrics
- **Accuracy**: Overall detection accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

### Mitigation Metrics
- **Accuracy Improvement**: Increase in accuracy after mitigation
- **Hallucination Reduction**: Number of hallucinations reduced
- **Mitigation Effectiveness**: Percentage of incorrect answers that were corrected

### System Metrics
- **Overall Score**: Weighted combination of detection and mitigation scores
- **Performance Level**: Categorical assessment (excellent/good/fair/poor)

## Advanced Usage

### Custom Thresholds

```python
# Update detection thresholds
pipeline.unified_detector.update_thresholds(
    entropy_threshold=1.5,
    disagreement_threshold=0.3,
    uncertainty_threshold=0.5
)
```

### Threshold Optimization

```python
# Optimize thresholds based on validation data
optimization_results = pipeline.optimize_thresholds(
    subset_size=100,
    target_precision=0.8
)

print(f"Optimal Entropy Threshold: {optimization_results['optimal_entropy_threshold']:.3f}")
print(f"Optimal Disagreement Threshold: {optimization_results['optimal_disagreement_threshold']:.3f}")
```

### Custom RAG Configuration

```python
from halt.mitigation import LightweightRAGChecker

# Custom RAG checker
custom_rag = LightweightRAGChecker(
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.2,
    max_retrieval_attempts=5
)
```

## Performance Considerations

- **GPU Usage**: Set `device="cuda"` for faster processing
- **Batch Size**: Adjust `subset_size` based on available memory
- **CoT Samples**: More samples improve detection but increase computation time
- **RAG Retrieval**: Wikipedia access may be rate-limited
