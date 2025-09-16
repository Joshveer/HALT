"""
Basic usage example for HALT pipeline.

This example demonstrates how to use HALT for hallucination detection
and mitigation on a simple question-answering task.
"""

from halt import HaltPipeline


def main():
    """Run basic HALT example."""
    print("HALT Basic Usage Example")
    print("=" * 40)
    
    # Initialize the pipeline
    print("Initializing HALT pipeline...")
    pipeline = HaltPipeline(
        model_name="microsoft/DialoGPT-medium",  # You can change this to any HuggingFace model
        device="cpu"  # Use "cuda" if you have a GPU
    )
    
    # Analyze a single example
    print("\nAnalyzing single example...")
    question = "What is the capital of France?"
    
    result = pipeline.analyze_single_example(question, num_cot_samples=3)
    
    print(f"Question: {result['question']}")
    print(f"Original Answer: {result['original_answer']}")
    print(f"Mitigated Answer: {result['mitigated_answer']}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Run detection-only evaluation
    print("\nRunning detection evaluation...")
    detection_results = pipeline.run_detection_only(
        subset_size=50,  # Small subset for quick demo
        num_cot_samples=3,
        entropy_threshold=2.0,
        disagreement_threshold=0.4
    )
    
    print(f"Detection F1 Score: {detection_results['metrics']['f1_score']:.3f}")
    print(f"Detection Precision: {detection_results['metrics']['precision']:.3f}")
    print(f"Detection Recall: {detection_results['metrics']['recall']:.3f}")
    
    # Run mitigation-only evaluation
    print("\nRunning mitigation evaluation...")
    mitigation_results = pipeline.run_mitigation_only(
        subset_size=50,
        num_samples=3
    )
    
    print(f"Original Accuracy: {mitigation_results['metrics']['original_accuracy']:.3f}")
    print(f"Mitigated Accuracy: {mitigation_results['metrics']['mitigated_accuracy']:.3f}")
    print(f"Accuracy Improvement: {mitigation_results['metrics']['accuracy_improvement']:.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
