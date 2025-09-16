"""
Full evaluation example for HALT pipeline.

This example demonstrates how to run a comprehensive evaluation
of the HALT system on the TruthfulQA dataset.
"""

from halt import HaltPipeline
import matplotlib.pyplot as plt


def main():
    """Run full HALT evaluation."""
    print("HALT Full Evaluation Example")
    print("=" * 40)
    
    # Initialize the pipeline
    print("Initializing HALT pipeline...")
    pipeline = HaltPipeline(
        model_name="microsoft/DialoGPT-medium",
        device="cpu"  # Change to "cuda" if you have a GPU
    )
    
    # Run full evaluation
    print("\nRunning full evaluation...")
    evaluation_results = pipeline.run_evaluation(
        subset_size=100,  # Adjust based on your computational resources
        num_cot_samples=5,
        num_mitigation_samples=5,
        entropy_threshold=2.0,
        disagreement_threshold=0.4,
        show_progress=True
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Detection F1: {evaluation_results['detection']['metrics']['f1_score']:.3f}")
    print(f"Detection Precision: {evaluation_results['detection']['metrics']['precision']:.3f}")
    print(f"Detection Recall: {evaluation_results['detection']['metrics']['recall']:.3f}")
    print(f"Mitigation Improvement: {evaluation_results['mitigation']['metrics']['accuracy_improvement']:.3f}")
    print(f"Overall Score: {evaluation_results['system']['overall_score']:.3f}")
    print(f"Performance Level: {evaluation_results['system']['performance_level']}")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    try:
        pipeline.evaluator.plot_results(save_dir="./plots")
        print("Plots saved to ./plots/")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Optimize thresholds
    print("\nOptimizing detection thresholds...")
    optimization_results = pipeline.optimize_thresholds(
        subset_size=50,
        target_precision=0.8
    )
    
    print(f"Optimal Entropy Threshold: {optimization_results['optimal_entropy_threshold']:.3f}")
    print(f"Optimal Disagreement Threshold: {optimization_results['optimal_disagreement_threshold']:.3f}")
    print(f"Optimized F1 Score: {optimization_results['f1_score']:.3f}")
    
    print("\nFull evaluation completed successfully!")


if __name__ == "__main__":
    main()
