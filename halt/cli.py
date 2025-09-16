"""Brief description."""

import argparse
import sys
from .pipeline import HaltPipeline


def main():
    """Brief description."""
    parser = argparse.ArgumentParser(
        description="HALT: Hallucination Analysis with Logits & Traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Brief description."""
    )
    
    parser.add_argument(
        "--model", 
        default="microsoft/DialoGPT-medium",
        help="HuggingFace model name (default: microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run on (default: auto)"
    )
    
    parser.add_argument(
        "--subset-size",
        type=int,
        default=200,
        help="Number of examples to evaluate (default: 200)"
    )
    parser.add_argument(
        "--num-cot-samples",
        type=int,
        default=5,
        help="Number of CoT samples for detection (default: 5)"
    )
    parser.add_argument(
        "--num-mitigation-samples",
        type=int,
        default=5,
        help="Number of samples for mitigation (default: 5)"
    )
    
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=2.0,
        help="Token entropy threshold (default: 2.0)"
    )
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        default=0.4,
        help="CoT disagreement threshold (default: 0.4)"
    )
    
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Run detection evaluation only"
    )
    parser.add_argument(
        "--mitigation-only",
        action="store_true",
        help="Run mitigation evaluation only"
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize detection thresholds"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    try:
        if args.verbose:
            print(f"Initializing HALT pipeline with model: {args.model}")
            print(f"Using device: {device}")
        
        pipeline = HaltPipeline(
            model_name=args.model,
            device=device
        )
        
        if args.detection_only:
            if args.verbose:
                print("Running detection-only evaluation...")
            
            results = pipeline.run_detection_only(
                subset_size=args.subset_size,
                num_cot_samples=args.num_cot_samples,
                entropy_threshold=args.entropy_threshold,
                disagreement_threshold=args.disagreement_threshold
            )
            
            print(f"Detection F1: {results['metrics']['f1_score']:.3f}")
            print(f"Detection Precision: {results['metrics']['precision']:.3f}")
            print(f"Detection Recall: {results['metrics']['recall']:.3f}")
            
        elif args.mitigation_only:
            if args.verbose:
                print("Running mitigation-only evaluation...")
            
            results = pipeline.run_mitigation_only(
                subset_size=args.subset_size,
                num_samples=args.num_mitigation_samples
            )
            
            print(f"Original Accuracy: {results['metrics']['original_accuracy']:.3f}")
            print(f"Mitigated Accuracy: {results['metrics']['mitigated_accuracy']:.3f}")
            print(f"Accuracy Improvement: {results['metrics']['accuracy_improvement']:.3f}")
            
        elif args.optimize_thresholds:
            if args.verbose:
                print("Optimizing detection thresholds...")
            
            results = pipeline.optimize_thresholds(
                subset_size=args.subset_size,
                target_precision=0.8
            )
            
            print(f"Optimal Entropy Threshold: {results['optimal_entropy_threshold']:.3f}")
            print(f"Optimal Disagreement Threshold: {results['optimal_disagreement_threshold']:.3f}")
            print(f"Optimized F1 Score: {results['f1_score']:.3f}")
            
        else:
            if args.verbose:
                print("Running full evaluation...")
            
            results = pipeline.run_evaluation(
                subset_size=args.subset_size,
                num_cot_samples=args.num_cot_samples,
                num_mitigation_samples=args.num_mitigation_samples,
                entropy_threshold=args.entropy_threshold,
                disagreement_threshold=args.disagreement_threshold,
                show_progress=args.verbose
            )
            
            print(f"Detection F1: {results['detection']['metrics']['f1_score']:.3f}")
            print(f"Mitigation Improvement: {results['mitigation']['metrics']['accuracy_improvement']:.3f}")
            print(f"Overall Score: {results['system']['overall_score']:.3f}")
            print(f"Performance Level: {results['system']['performance_level']}")
        
        if not args.no_plots and not args.optimize_thresholds:
            if args.verbose:
                print("Generating plots...")
            
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            pipeline.evaluator.plot_results(save_dir=args.output_dir)
            
            if args.verbose:
                print(f"Plots saved to {args.output_dir}/")
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
