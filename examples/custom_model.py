"""
Custom model example for HALT pipeline.

This example demonstrates how to use HALT with different models
and customize the detection and mitigation parameters.
"""

from halt import HaltPipeline
from halt.models import BaseModel, TokenAnalyzer
from halt.detection import UnifiedDetector
from halt.mitigation import SelfConsistencyMitigator, LightweightRAGChecker
from halt.evaluation import HaltEvaluator


def main():
    """Run HALT with custom model and parameters."""
    print("HALT Custom Model Example")
    print("=" * 40)
    
    # List of available models (you can try different ones)
    models_to_try = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium", 
        "gpt2",
        "gpt2-medium"
    ]
    
    for model_name in models_to_try:
        print(f"\nTesting with model: {model_name}")
        print("-" * 30)
        
        try:
            # Initialize pipeline with custom model
            pipeline = HaltPipeline(
                model_name=model_name,
                device="cpu"
            )
            
            # Test single example
            question = "What is the largest planet in our solar system?"
            result = pipeline.analyze_single_example(question, num_cot_samples=3)
            
            print(f"Question: {result['question']}")
            print(f"Answer: {result['original_answer']}")
            print(f"Is Hallucination: {result['is_hallucination']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            # Test with different thresholds
            print(f"\nTesting different thresholds...")
            
            # Low threshold (more sensitive)
            pipeline.unified_detector.update_thresholds(
                entropy_threshold=1.5,
                disagreement_threshold=0.3
            )
            
            result_low = pipeline.analyze_single_example(question, num_cot_samples=3)
            print(f"Low threshold - Is Hallucination: {result_low['is_hallucination']}, Confidence: {result_low['confidence']:.3f}")
            
            # High threshold (less sensitive)
            pipeline.unified_detector.update_thresholds(
                entropy_threshold=2.5,
                disagreement_threshold=0.6
            )
            
            result_high = pipeline.analyze_single_example(question, num_cot_samples=3)
            print(f"High threshold - Is Hallucination: {result_high['is_hallucination']}, Confidence: {result_high['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue
    
    # Test with custom RAG parameters
    print(f"\nTesting custom RAG parameters...")
    print("-" * 30)
    
    try:
        # Initialize with custom RAG checker
        base_model = BaseModel("microsoft/DialoGPT-medium", device="cpu")
        token_analyzer = TokenAnalyzer(base_model.tokenizer)
        unified_detector = UnifiedDetector(tokenizer=base_model.tokenizer)
        self_consistency_mitigator = SelfConsistencyMitigator()
        
        # Custom RAG checker with different parameters
        custom_rag_checker = LightweightRAGChecker(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.2  # Lower threshold for more lenient matching
        )
        
        evaluator = HaltEvaluator(
            model=base_model,
            token_analyzer=token_analyzer,
            unified_detector=unified_detector,
            self_consistency_mitigator=self_consistency_mitigator,
            rag_checker=custom_rag_checker
        )
        
        # Test RAG mitigation
        question = "Who wrote Romeo and Juliet?"
        original_answer = "William Shakespeare wrote Romeo and Juliet."
        
        rag_result = custom_rag_checker.apply_rag_mitigation(question, original_answer)
        
        print(f"Question: {question}")
        print(f"Original Answer: {rag_result['original_answer']}")
        print(f"Mitigated Answer: {rag_result['mitigated_answer']}")
        print(f"Mitigation Applied: {rag_result['mitigation_applied']}")
        print(f"Is Supported by Knowledge: {rag_result['is_supported']}")
        print(f"Confidence: {rag_result['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error with custom RAG: {e}")
    
    print("\nCustom model example completed!")


if __name__ == "__main__":
    main()
