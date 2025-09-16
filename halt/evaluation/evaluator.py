"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
from .metrics import EvaluationMetrics
from ..data.truthfulqa_loader import TruthfulQALoader
from ..models.base_model import BaseModel
from ..models.token_analyzer import TokenAnalyzer
from ..reasoning.disagreement_detector import DisagreementDetector
from ..detection.unified_detector import UnifiedDetector
from ..mitigation.self_consistency import SelfConsistencyMitigator
from ..mitigation.rag_checker import LightweightRAGChecker


class HaltEvaluator:
    """Brief description."""
    
    def __init__(self, 
                 model: BaseModel,
                 token_analyzer: TokenAnalyzer,
                 unified_detector: UnifiedDetector,
                 self_consistency_mitigator: SelfConsistencyMitigator,
                 rag_checker: LightweightRAGChecker):
        """Brief description."""
        self.model = model
        self.token_analyzer = token_analyzer
        self.unified_detector = unified_detector
        self.self_consistency_mitigator = self_consistency_mitigator
        self.rag_checker = rag_checker
        self.metrics = EvaluationMetrics()
        
        self.evaluation_results = []
        self.detection_results = []
        self.mitigation_results = []
    
    def evaluate_detection(self, 
                          test_data: List[Dict[str, Any]], 
                          num_cot_samples: int = 5,
                          show_progress: bool = True) -> Dict[str, Any]:
        """Brief description."""
        print("Evaluating hallucination detection...")
        
        ground_truth = []
        predictions = []
        confidence_scores = []
        detection_details = []
        
        iterator = tqdm(test_data, desc="Detection Evaluation") if show_progress else test_data
        
        for i, example in enumerate(iterator):
            try:
                prompt = example['prompt']
                generation_result = self.model.generate(prompt, return_logits=True)
                generated_text = generation_result['generated_text']
                
                evaluation = self.model.tokenizer._tokenizer.evaluate_answer(
                    generated_text, 
                    example['correct_answers'], 
                    example['incorrect_answers']
                )
                
                is_hallucination = evaluation['is_hallucination']
                ground_truth.append(is_hallucination)
                
                cot_prompt = f"Question: {example['question']}\n\nLet's think step by step to answer this question accurately.\n\nAnswer:"
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
                
                unified_detection = detection_result.get('unified_detection', {})
                predicted_hallucination = unified_detection.get('is_hallucination', False)
                confidence = unified_detection.get('confidence', 0.0)
                
                predictions.append(predicted_hallucination)
                confidence_scores.append(confidence)
                
                detection_details.append({
                    'example_id': i,
                    'question': example['question'],
                    'generated_answer': generated_text,
                    'is_hallucination': is_hallucination,
                    'predicted_hallucination': predicted_hallucination,
                    'confidence': confidence,
                    'token_analysis': token_analysis,
                    'cot_analysis': detection_result.get('cot_detection', {}),
                    'detection_reason': unified_detection.get('reason', 'unknown')
                })
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                ground_truth.append(False)
                predictions.append(False)
                confidence_scores.append(0.0)
                detection_details.append({
                    'example_id': i,
                    'error': str(e),
                    'is_hallucination': False,
                    'predicted_hallucination': False,
                    'confidence': 0.0
                })
        
        detection_metrics = self.metrics.calculate_detection_metrics(
            ground_truth, predictions, confidence_scores
        )
        
        self.detection_results = detection_details
        
        return {
            'metrics': detection_metrics,
            'details': detection_details,
            'ground_truth': ground_truth,
            'predictions': predictions,
            'confidence_scores': confidence_scores
        }
    
    def evaluate_mitigation(self, 
                           test_data: List[Dict[str, Any]], 
                           num_samples: int = 5,
                           show_progress: bool = True) -> Dict[str, Any]:
        """Brief description."""
        print("Evaluating hallucination mitigation...")
        
        original_answers = []
        mitigated_answers = []
        ground_truth_answers = []
        original_correct = []
        mitigated_correct = []
        mitigation_details = []
        
        iterator = tqdm(test_data, desc="Mitigation Evaluation") if show_progress else test_data
        
        for i, example in enumerate(iterator):
            try:
                prompt = example['prompt']
                original_generation = self.model.generate(prompt)
                original_answer = original_generation['generated_text']
                original_answers.append(original_answer)
                
                original_evaluation = self.model.tokenizer._tokenizer.evaluate_answer(
                    original_answer, 
                    example['correct_answers'], 
                    example['incorrect_answers']
                )
                original_correct.append(original_evaluation['is_correct'])
                
                generations = self.model.generate_multiple(prompt, num_samples)
                
                self_consistency_result = self.self_consistency_mitigator.apply_self_consistency(
                    generations, method='majority'
                )
                self_consistency_answer = self_consistency_result['mitigated_answer']
                
                rag_result = self.rag_checker.apply_rag_mitigation(
                    example['question'], self_consistency_answer
                )
                final_mitigated_answer = rag_result['mitigated_answer']
                
                mitigated_answers.append(final_mitigated_answer)
                
                mitigated_evaluation = self.model.tokenizer._tokenizer.evaluate_answer(
                    final_mitigated_answer, 
                    example['correct_answers'], 
                    example['incorrect_answers']
                )
                mitigated_correct.append(mitigated_evaluation['is_correct'])
                
                ground_truth_answers.append(example['correct_answers'][0] if example['correct_answers'] else '')
                
                mitigation_details.append({
                    'example_id': i,
                    'question': example['question'],
                    'original_answer': original_answer,
                    'mitigated_answer': final_mitigated_answer,
                    'ground_truth': example['correct_answers'][0] if example['correct_answers'] else '',
                    'original_correct': original_evaluation['is_correct'],
                    'mitigated_correct': mitigated_evaluation['is_correct'],
                    'self_consistency_result': self_consistency_result,
                    'rag_result': rag_result,
                    'improvement': mitigated_evaluation['is_correct'] and not original_evaluation['is_correct']
                })
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                original_answers.append('')
                mitigated_answers.append('')
                ground_truth_answers.append('')
                original_correct.append(False)
                mitigated_correct.append(False)
                mitigation_details.append({
                    'example_id': i,
                    'error': str(e),
                    'original_correct': False,
                    'mitigated_correct': False
                })
        
        mitigation_metrics = self.metrics.calculate_mitigation_metrics(
            original_answers, mitigated_answers, ground_truth_answers,
            original_correct, mitigated_correct
        )
        
        self.mitigation_results = mitigation_details
        
        return {
            'metrics': mitigation_metrics,
            'details': mitigation_details,
            'original_answers': original_answers,
            'mitigated_answers': mitigated_answers,
            'ground_truth_answers': ground_truth_answers,
            'original_correct': original_correct,
            'mitigated_correct': mitigated_correct
        }
    
    def evaluate_full_system(self, 
                           test_data: List[Dict[str, Any]], 
                           num_cot_samples: int = 5,
                           num_mitigation_samples: int = 5,
                           show_progress: bool = True) -> Dict[str, Any]:
        """Brief description."""
        print("Evaluating full HALT system...")
        
        detection_results = self.evaluate_detection(
            test_data, num_cot_samples, show_progress
        )
        
        mitigation_results = self.evaluate_mitigation(
            test_data, num_mitigation_samples, show_progress
        )
        
        system_metrics = self.metrics.calculate_system_metrics(
            detection_results['metrics'],
            mitigation_results['metrics'],
            detection_results['ground_truth']
        )
        
        report = self.metrics.generate_evaluation_report(
            detection_results['metrics'],
            mitigation_results['metrics'],
            system_metrics
        )
        
        self.evaluation_results = {
            'detection': detection_results,
            'mitigation': mitigation_results,
            'system': system_metrics,
            'report': report
        }
        
        return self.evaluation_results
    
    def evaluate_per_category(self, 
                            test_data: List[Dict[str, Any]], 
                            categories: List[str],
                            num_cot_samples: int = 5) -> Dict[str, Dict[str, Any]]:
        """Brief description."""
        print("Evaluating per-category performance...")
        
        category_results = {}
        
        for category in categories:
            print(f"Evaluating category: {category}")
            
            category_data = [ex for ex in test_data if ex.get('category') == category]
            
            if not category_data:
                category_results[category] = {'error': 'no_data_for_category'}
                continue
            
            category_eval = self.evaluate_full_system(
                category_data, num_cot_samples, show_progress=False
            )
            
            category_results[category] = category_eval
        
        return category_results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Brief description."""
        if not self.evaluation_results:
            return {'error': 'No evaluation results available'}
        
        detection = self.evaluation_results['detection']
        mitigation = self.evaluation_results['mitigation']
        system = self.evaluation_results['system']
        
        return {
            'detection_f1': detection['metrics']['f1_score'],
            'detection_precision': detection['metrics']['precision'],
            'detection_recall': detection['metrics']['recall'],
            'mitigation_improvement': mitigation['metrics']['accuracy_improvement'],
            'hallucination_reduction': mitigation['metrics']['hallucination_reduction'],
            'overall_score': system['overall_score'],
            'performance_level': system['performance_level'],
            'total_samples': len(detection['ground_truth'])
        }
    
    def save_results(self, filepath: str) -> None:
        """Brief description."""
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def plot_results(self, save_dir: Optional[str] = None) -> None:
        """Brief description."""
        if not self.evaluation_results:
            print("No evaluation results available for plotting")
            return
        
        detection = self.evaluation_results['detection']
        mitigation = self.evaluation_results['mitigation']
        
        self.metrics.plot_confusion_matrix(
            detection['ground_truth'], 
            detection['predictions'],
            title="Hallucination Detection Confusion Matrix",
            save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
        )
        
        if detection['confidence_scores']:
            self.metrics.plot_precision_recall_curve(
                detection['ground_truth'],
                detection['confidence_scores'],
                title="Precision-Recall Curve",
                save_path=f"{save_dir}/precision_recall_curve.png" if save_dir else None
            )
        
        self.metrics.plot_mitigation_improvement(
            mitigation['metrics']['original_accuracy'],
            mitigation['metrics']['mitigated_accuracy'],
            title="Mitigation Improvement",
            save_path=f"{save_dir}/mitigation_improvement.png" if save_dir else None
        )
