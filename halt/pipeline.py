"""Main HALT pipeline for hallucination detection and mitigation."""

import os
import time
from typing import List, Dict, Any, Optional

from .data import TruthfulQALoader, DataPreprocessor
from .models import BaseModel, TokenAnalyzer
from .reasoning import CoTAnalyzer, DisagreementDetector
from .detection import UnifiedDetector, ThresholdOptimizer
from .mitigation import SelfConsistencyMitigator, LightweightRAGChecker
from .evaluation import HaltEvaluator, EvaluationMetrics


class HaltPipeline:
    """Main pipeline for HALT hallucination detection and mitigation."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """Initialize the HALT pipeline."""
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "halt")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_components()
        
        print(f"HALT Pipeline initialized with model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        print("Initializing HALT components...")
        
        self.data_loader = TruthfulQALoader(cache_dir=self.cache_dir)
        self.data_preprocessor = DataPreprocessor()
        
        self.model = BaseModel(self.model_name, device=self.device)
        self.token_analyzer = TokenAnalyzer(self.model.tokenizer)
        
        self.cot_analyzer = CoTAnalyzer()
        self.disagreement_detector = DisagreementDetector()
        
        self.unified_detector = UnifiedDetector(tokenizer=self.model.tokenizer)
        self.threshold_optimizer = ThresholdOptimizer()
        
        self.self_consistency_mitigator = SelfConsistencyMitigator()
        self.rag_checker = LightweightRAGChecker()
        
        self.evaluator = HaltEvaluator(
            model=self.model,
            token_analyzer=self.token_analyzer,
            unified_detector=self.unified_detector,
            self_consistency_mitigator=self.self_consistency_mitigator,
            rag_checker=self.rag_checker
        )
        
        print("All components initialized successfully!")
    
    def load_data(self, 
                 dataset_name: str = "truthful_qa",
                 subset_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load and preprocess dataset."""
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "truthful_qa":
            df = self.data_loader.load_dataset(subset_size)
            formatted_data = self.data_loader.format_for_generation(df, include_cot=False)
            preprocessed_data = self.data_preprocessor.batch_preprocess(formatted_data)
            
            print(f"Loaded {len(preprocessed_data)} examples from TruthfulQA")
            return preprocessed_data
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def detect_hallucinations(self, 
                            examples: List[Dict[str, Any]], 
                            num_cot_samples: int = 5,
                            entropy_threshold: float = 2.0,
                            disagreement_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Detect hallucinations in generated text."""
        print(f"Detecting hallucinations in {len(examples)} examples...")
        
        self.unified_detector.update_thresholds(
            entropy_threshold=entropy_threshold,
            disagreement_threshold=disagreement_threshold
        )
        
        detection_results = []
        
        for i, example in enumerate(examples):
            try:
                prompt = example['original_question']
                generation_result = self.model.generate(prompt, return_logits=True)
                generated_text = generation_result['generated_text']
                
                cot_prompt = f"Question: {example['original_question']}\n\nLet's think step by step to answer this question accurately.\n\nAnswer:"
                cot_generations = self.model.generate_multiple(cot_prompt, num_cot_samples)
                cot_texts = [gen['generated_text'] for gen in cot_generations]
                
                token_prob_result = self.model.get_token_probabilities(prompt)
                if 'tokens' in token_prob_result:
                    token_analysis = self.token_analyzer.analyze_token_uncertainty(token_prob_result['tokens'])
                else:
                    token_analysis = {'error': 'Token analysis failed'}
                
                detection_result = self.unified_detector.detect_unified(
                    token_analysis=token_analysis,
                    cot_texts=cot_texts
                )
                
                result = {
                    'example_id': i,
                    'question': example['original_question'],
                    'generated_answer': generated_text,
                    'detection_result': detection_result,
                    'token_analysis': token_analysis,
                    'cot_texts': cot_texts,
                    'is_hallucination': detection_result.get('unified_detection', {}).get('is_hallucination', False),
                    'confidence': detection_result.get('unified_detection', {}).get('confidence', 0.0)
                }
                
                detection_results.append(result)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                detection_results.append({
                    'example_id': i,
                    'error': str(e),
                    'is_hallucination': False,
                    'confidence': 0.0
                })
        
        print(f"Detection completed. Found {sum(r.get('is_hallucination', False) for r in detection_results)} potential hallucinations.")
        return detection_results
    
    def mitigate_hallucinations(self, 
                              examples: List[Dict[str, Any]], 
                              num_samples: int = 5) -> List[Dict[str, Any]]:
        """Mitigate hallucinations using self-consistency and RAG."""
        print(f"Mitigating hallucinations in {len(examples)} examples...")
        
        mitigation_results = []
        
        for i, example in enumerate(examples):
            try:
                prompt = example['original_question']
                original_generation = self.model.generate(prompt)
                original_answer = original_generation['generated_text']
                
                generations = self.model.generate_multiple(prompt, num_samples)
                
                self_consistency_result = self.self_consistency_mitigator.apply_self_consistency(
                    generations, method='majority'
                )
                self_consistency_answer = self_consistency_result['mitigated_answer']
                
                rag_result = self.rag_checker.apply_rag_mitigation(
                    example['original_question'], self_consistency_answer
                )
                final_mitigated_answer = rag_result['mitigated_answer']
                
                result = {
                    'example_id': i,
                    'question': example['original_question'],
                    'original_answer': original_answer,
                    'mitigated_answer': final_mitigated_answer,
                    'self_consistency_result': self_consistency_result,
                    'rag_result': rag_result,
                    'mitigation_applied': rag_result['mitigation_applied']
                }
                
                mitigation_results.append(result)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                mitigation_results.append({
                    'example_id': i,
                    'error': str(e),
                    'original_answer': '',
                    'mitigated_answer': '',
                    'mitigation_applied': False
                })
        
        print(f"Mitigation completed. Applied mitigation to {sum(r.get('mitigation_applied', False) for r in mitigation_results)} examples.")
        return mitigation_results
    
    def run_evaluation(self, 
                      subset_size: int = 200,
                      num_cot_samples: int = 5,
                      num_mitigation_samples: int = 5,
                      entropy_threshold: float = 2.0,
                      disagreement_threshold: float = 0.4,
                      show_progress: bool = True) -> Dict[str, Any]:
        """Run comprehensive evaluation of the HALT system."""
        print("=" * 60)
        print("HALT EVALUATION PIPELINE")
        print("=" * 60)
        
        print("\n1. Loading dataset...")
        test_data = self.load_data(subset_size=subset_size)
        
        self.unified_detector.update_thresholds(
            entropy_threshold=entropy_threshold,
            disagreement_threshold=disagreement_threshold
        )
        
        print("\n2. Running evaluation...")
        evaluation_results = self.evaluator.evaluate_full_system(
            test_data=test_data,
            num_cot_samples=num_cot_samples,
            num_mitigation_samples=num_mitigation_samples,
            show_progress=show_progress
        )
        
        print("\n3. Evaluation Results:")
        print(evaluation_results['report'])
        
        results_file = os.path.join(self.cache_dir, f"evaluation_results_{int(time.time())}.json")
        self.evaluator.save_results(results_file)
        
        print(f"\nResults saved to: {results_file}")
        print("=" * 60)
        
        return evaluation_results
    
    def run_detection_only(self, 
                          subset_size: int = 200,
                          num_cot_samples: int = 5,
                          entropy_threshold: float = 2.0,
                          disagreement_threshold: float = 0.4) -> Dict[str, Any]:
        """Run detection evaluation only."""
        print("Running detection-only evaluation...")
        
        test_data = self.load_data(subset_size=subset_size)
        
        self.unified_detector.update_thresholds(
            entropy_threshold=entropy_threshold,
            disagreement_threshold=disagreement_threshold
        )
        
        detection_results = self.evaluator.evaluate_detection(
            test_data=test_data,
            num_cot_samples=num_cot_samples,
            show_progress=True
        )
        
        metrics = detection_results['metrics']
        print(f"\nDetection Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        
        return detection_results
    
    def run_mitigation_only(self, 
                           subset_size: int = 200,
                           num_samples: int = 5) -> Dict[str, Any]:
        """Run mitigation evaluation only."""
        print("Running mitigation-only evaluation...")
        
        test_data = self.load_data(subset_size=subset_size)
        
        mitigation_results = self.evaluator.evaluate_mitigation(
            test_data=test_data,
            num_samples=num_samples,
            show_progress=True
        )
        
        metrics = mitigation_results['metrics']
        print(f"\nMitigation Results:")
        print(f"Original Accuracy: {metrics['original_accuracy']:.3f}")
        print(f"Mitigated Accuracy: {metrics['mitigated_accuracy']:.3f}")
        print(f"Accuracy Improvement: {metrics['accuracy_improvement']:.3f}")
        print(f"Hallucination Reduction: {metrics['hallucination_reduction']} cases")
        
        return mitigation_results
    
    def optimize_thresholds(self, 
                          subset_size: int = 100,
                          target_precision: float = 0.8) -> Dict[str, Any]:
        """Optimize detection thresholds using validation data."""
        print("Optimizing detection thresholds...")
        
        test_data = self.load_data(subset_size=subset_size)
        
        detection_results = self.run_detection_only(
            subset_size=subset_size,
            num_cot_samples=3
        )
        
        details = detection_results['details']
        entropy_scores = []
        disagreement_scores = []
        ground_truth = []
        
        for detail in details:
            if 'error' not in detail:
                token_analysis = detail.get('token_analysis', {})
                entropy_scores.append(token_analysis.get('avg_entropy', 0))
                
                cot_analysis = detail.get('cot_analysis', {})
                disagreement_scores.append(cot_analysis.get('disagreement_score', 0))
                
                ground_truth.append(detail.get('is_hallucination', False))
        
        optimization_result = self.threshold_optimizer.optimize_combined_thresholds(
            entropy_scores=entropy_scores,
            disagreement_scores=disagreement_scores,
            ground_truth=ground_truth,
            target_precision=target_precision
        )
        
        print(f"Optimization Results:")
        print(f"Optimal Entropy Threshold: {optimization_result['optimal_entropy_threshold']:.3f}")
        print(f"Optimal Disagreement Threshold: {optimization_result['optimal_disagreement_threshold']:.3f}")
        print(f"Precision: {optimization_result['precision']:.3f}")
        print(f"Recall: {optimization_result['recall']:.3f}")
        print(f"F1 Score: {optimization_result['f1_score']:.3f}")
        
        return optimization_result
    
    def analyze_single_example(self, 
                             question: str, 
                             num_cot_samples: int = 5) -> Dict[str, Any]:
        """Analyze a single example for hallucination detection and mitigation."""
        print(f"Analyzing single example: {question}")
        
        prompt = f"Question: {question}\n\nAnswer:"
        
        generation_result = self.model.generate(prompt, return_logits=True)
        generated_text = generation_result['generated_text']
        
        cot_prompt = f"Question: {question}\n\nLet's think step by step to answer this question accurately.\n\nAnswer:"
        cot_generations = self.model.generate_multiple(cot_prompt, num_cot_samples)
        cot_texts = [gen['generated_text'] for gen in cot_generations]
        
        token_prob_result = self.model.get_token_probabilities(prompt)
        if 'tokens' in token_prob_result:
            token_analysis = self.token_analyzer.analyze_token_uncertainty(token_prob_result['tokens'])
        else:
            token_analysis = {'error': 'Token analysis failed'}
        
        detection_result = self.unified_detector.detect_unified(
            token_analysis=token_analysis,
            cot_texts=cot_texts
        )
        
        mitigation_result = self.self_consistency_mitigator.apply_self_consistency(
            cot_generations, method='cot_based'
        )
        
        rag_result = self.rag_checker.apply_rag_mitigation(
            question, mitigation_result['mitigated_answer']
        )
        
        return {
            'question': question,
            'original_answer': generated_text,
            'mitigated_answer': rag_result['mitigated_answer'],
            'detection_result': detection_result,
            'token_analysis': token_analysis,
            'cot_analysis': self.cot_analyzer.analyze_cot_traces(cot_texts),
            'mitigation_result': mitigation_result,
            'rag_result': rag_result,
            'is_hallucination': detection_result.get('unified_detection', {}).get('is_hallucination', False),
            'confidence': detection_result.get('unified_detection', {}).get('confidence', 0.0)
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'cache_dir': self.cache_dir,
            'detector_thresholds': {
                'entropy_threshold': self.unified_detector.entropy_threshold,
                'disagreement_threshold': self.unified_detector.disagreement_threshold,
                'uncertainty_threshold': self.unified_detector.uncertainty_threshold
            },
            'rag_config': {
                'model_name': self.rag_checker.model_name,
                'similarity_threshold': self.rag_checker.similarity_threshold
            }
        }