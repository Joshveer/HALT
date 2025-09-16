#!/usr/bin/env python3
"""
Demo script for HALT project.

This script demonstrates the project structure and functionality
without requiring all dependencies to be installed.
"""

import os
import sys
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_file_structure():
    """Print the project file structure."""
    print_section("PROJECT STRUCTURE")
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(Path(directory).iterdir())
        for i, item in enumerate(items):
            if item.name.startswith('.'):
                continue
                
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(".")
    print(f"\nTotal files: {sum(1 for _ in Path('.').rglob('*.py'))} Python files")

def print_module_overview():
    """Print overview of each module."""
    print_section("MODULE OVERVIEW")
    
    modules = {
        "Data Module": {
            "files": ["halt/data/truthfulqa_loader.py", "halt/data/preprocessing.py"],
            "description": "Dataset loading and preprocessing for TruthfulQA",
            "key_features": ["TruthfulQA dataset integration", "Text preprocessing", "Answer evaluation"]
        },
        "Models Module": {
            "files": ["halt/models/base_model.py", "halt/models/token_analyzer.py"],
            "description": "Model integration and token-level analysis",
            "key_features": ["HuggingFace model wrapper", "Token entropy analysis", "Tokenization fragility detection"]
        },
        "Reasoning Module": {
            "files": ["halt/reasoning/cot_analyzer.py", "halt/reasoning/disagreement_detector.py"],
            "description": "Chain-of-thought reasoning analysis",
            "key_features": ["CoT trace analysis", "Answer extraction", "Disagreement detection"]
        },
        "Detection Module": {
            "files": ["halt/detection/unified_detector.py", "halt/detection/threshold_optimizer.py"],
            "description": "Unified hallucination detection",
            "key_features": ["Token + CoT signal combination", "Threshold optimization", "Confidence scoring"]
        },
        "Mitigation Module": {
            "files": ["halt/mitigation/self_consistency.py", "halt/mitigation/rag_checker.py"],
            "description": "Hallucination mitigation strategies",
            "key_features": ["Self-consistency voting", "Lightweight RAG checking", "Answer correction"]
        },
        "Evaluation Module": {
            "files": ["halt/evaluation/metrics.py", "halt/evaluation/evaluator.py"],
            "description": "Comprehensive evaluation framework",
            "key_features": ["Detection metrics", "Mitigation metrics", "Visualization"]
        },
        "Pipeline Module": {
            "files": ["halt/pipeline.py"],
            "description": "Main orchestration pipeline",
            "key_features": ["End-to-end workflow", "Configuration management", "Result aggregation"]
        }
    }
    
    for module_name, info in modules.items():
        print(f"\n{module_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Files: {', '.join(info['files'])}")
        print(f"  Key Features:")
        for feature in info['key_features']:
            print(f"    • {feature}")

def print_usage_examples():
    """Print usage examples."""
    print_section("USAGE EXAMPLES")
    
    print("1. Basic Usage:")
    print("""
from halt import HaltPipeline

# Initialize pipeline
pipeline = HaltPipeline(
    model_name="microsoft/DialoGPT-medium",
    device="cpu"
)

# Analyze single example
result = pipeline.analyze_single_example(
    "What is the capital of France?",
    num_cot_samples=5
)

print(f"Answer: {result['original_answer']}")
print(f"Is Hallucination: {result['is_hallucination']}")
print(f"Confidence: {result['confidence']:.3f}")
""")
    
    print("2. Full Evaluation:")
    print("""
# Run comprehensive evaluation
results = pipeline.run_evaluation(
    subset_size=200,
    num_cot_samples=5,
    entropy_threshold=2.0,
    disagreement_threshold=0.4
)

print(f"Detection F1: {results['detection']['metrics']['f1_score']:.3f}")
print(f"Mitigation Improvement: {results['mitigation']['metrics']['accuracy_improvement']:.3f}")
""")
    
    print("3. Command Line Interface:")
    print("""
# Run evaluation from command line
halt-eval --model microsoft/DialoGPT-medium --subset-size 100
halt-eval --detection-only --entropy-threshold 2.0
halt-eval --mitigation-only --num-samples 5
""")

def print_installation_guide():
    """Print installation guide."""
    print_section("INSTALLATION GUIDE")
    
    print("1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Install Package:")
    print("   pip install -e .")
    
    print("\n3. Run Examples:")
    print("   python examples/basic_usage.py")
    print("   python examples/full_evaluation.py")
    print("   python examples/custom_model.py")
    
    print("\n4. Run Tests:")
    print("   python tests/test_basic.py")

def print_key_features():
    """Print key features of the system."""
    print_section("KEY FEATURES")
    
    features = [
        "🔍 Token-level uncertainty analysis using logit entropy",
        "🧠 Chain-of-thought disagreement detection",
        "🔄 Self-consistency mitigation with majority voting",
        "📚 Lightweight RAG-based ground truth checking",
        "📊 Comprehensive evaluation metrics and visualization",
        "⚙️ Automatic threshold optimization",
        "🎯 Support for multiple HuggingFace models",
        "📈 Performance analysis and reporting",
        "🖥️ Command-line interface for easy usage",
        "📝 Extensive documentation and examples"
    ]
    
    for feature in features:
        print(f"  {feature}")

def print_architecture_diagram():
    """Print a simple architecture diagram."""
    print_section("SYSTEM ARCHITECTURE")
    
    diagram = """
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Input Data    │───▶│  Token Analysis │───▶│  Uncertainty    │
    │  (Questions)    │    │   (Entropy)     │    │   Detection     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                           │
    ┌─────────────────┐    ┌─────────────────┐             ▼
    │  CoT Generation │───▶│  Disagreement   │    ┌─────────────────┐
    │  (Multiple)     │    │   Analysis      │───▶│  Unified        │
    └─────────────────┘    └─────────────────┘    │  Detection      │
                                                  └─────────────────┘
                                                           │
    ┌─────────────────┐    ┌─────────────────┐             ▼
    │ Self-Consistency│───▶│   RAG Checker   │    ┌─────────────────┐
    │   Mitigation    │    │  (Wikipedia)    │───▶│   Mitigation    │
    └─────────────────┘    └─────────────────┘    │   & Correction  │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   Evaluation    │
                                                  │   & Metrics     │
                                                  └─────────────────┘
    """
    
    print(diagram)

def main():
    """Main demo function."""
    print("HALT: Hallucination Analysis with Logits & Traces")
    print("A comprehensive pipeline for detecting and mitigating hallucinations in LLMs")
    
    print_file_structure()
    print_module_overview()
    print_key_features()
    print_architecture_diagram()
    print_usage_examples()
    print_installation_guide()
    
    print_section("NEXT STEPS")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install package: pip install -e .")
    print("3. Run examples: python examples/basic_usage.py")
    print("4. Read documentation: README.md")
    print("5. Explore code: halt/ directory")
    
    print(f"\n{'='*60}")
    print(" Demo completed! HALT is ready for hallucination detection.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
