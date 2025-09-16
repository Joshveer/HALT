"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from .cot_analyzer import CoTAnalyzer


class DisagreementDetector:
    """Brief description."""
    
    def __init__()(self):
        """Brief description."""
        """Brief description."""
        self.cot_analyzer = CoTAnalyzer()
    
    def detect_answer_disagreement(self, answers: List[str], threshold: float = 0.3) -> Dict[str, Any]:
        """Brief description."""
        if len(answers) < 2:
            return {
                'is_disagreeing': False,
                'disagreement_score': 0.0,
                'confidence': 0.0,
                'reason': 'insufficient_samples'
            }
        
        analysis = self.cot_analyzer.analyze_answer_disagreement(answers)
        
        disagreement_score = analysis.get('disagreement_score', 0.0)
        is_disagreeing = disagreement_score > threshold
        
        confidence = min(1.0, disagreement_score * 2)  # Scale to 0-1
        
        reason = "agreement" if not is_disagreeing else "high_disagreement"
        if disagreement_score > 0.8:
            reason = "very_high_disagreement"
        elif disagreement_score > 0.5:
            reason = "moderate_disagreement"
        
        return {
            'is_disagreeing': is_disagreeing,
            'disagreement_score': disagreement_score,
            'confidence': confidence,
            'reason': reason,
            'analysis': analysis
        }
    
    def detect_reasoning_divergence(self, cot_texts: List[str], threshold: float = 0.5) -> Dict[str, Any]:
        """Brief description."""
        if len(cot_texts) < 2:
            return {
                'is_divergent': False,
                'divergence_score': 0.0,
                'confidence': 0.0,
                'reason': 'insufficient_samples'
            }
        
        analysis = self.cot_analyzer.analyze_cot_traces(cot_texts)
        divergence_analysis = analysis.get('divergence_analysis', {})
        
        divergence_score = divergence_analysis.get('divergence_score', 0.0)
        is_divergent = divergence_score > threshold
        
        confidence = min(1.0, divergence_score * 1.5)  # Scale to 0-1
        
        reason = "convergent" if not is_divergent else "high_divergence"
        if divergence_score > 0.8:
            reason = "very_high_divergence"
        elif divergence_score > 0.6:
            reason = "moderate_divergence"
        
        return {
            'is_divergent': is_divergent,
            'divergence_score': divergence_score,
            'confidence': confidence,
            'reason': reason,
            'analysis': divergence_analysis
        }
    
    def detect_overall_disagreement(self, cot_texts: List[str], 
                                  answer_threshold: float = 0.3,
                                  reasoning_threshold: float = 0.5) -> Dict[str, Any]:
        """Brief description."""
        if len(cot_texts) < 2:
            return {
                'is_disagreeing': False,
                'overall_score': 0.0,
                'confidence': 0.0,
                'reason': 'insufficient_samples',
                'components': {}
            }
        
        final_answers = [self.cot_analyzer.extract_final_answer(cot) for cot in cot_texts]
        
        answer_detection = self.detect_answer_disagreement(final_answers, answer_threshold)
        
        reasoning_detection = self.detect_reasoning_divergence(cot_texts, reasoning_threshold)
        
        answer_score = answer_detection.get('disagreement_score', 0.0)
        reasoning_score = reasoning_detection.get('divergence_score', 0.0)
        
        overall_score = 0.7 * answer_score + 0.3 * reasoning_score
        
        is_disagreeing = (answer_detection.get('is_disagreeing', False) or 
                         reasoning_detection.get('is_divergent', False))
        
        answer_confidence = answer_detection.get('confidence', 0.0)
        reasoning_confidence = reasoning_detection.get('confidence', 0.0)
        overall_confidence = 0.7 * answer_confidence + 0.3 * reasoning_confidence
        
        if answer_detection.get('is_disagreeing', False) and reasoning_detection.get('is_divergent', False):
            reason = "both_answer_and_reasoning_disagreement"
        elif answer_detection.get('is_disagreeing', False):
            reason = "answer_disagreement"
        elif reasoning_detection.get('is_divergent', False):
            reason = "reasoning_divergence"
        else:
            reason = "agreement"
        
        return {
            'is_disagreeing': is_disagreeing,
            'overall_score': overall_score,
            'confidence': overall_confidence,
            'reason': reason,
            'components': {
                'answer_detection': answer_detection,
                'reasoning_detection': reasoning_detection
            }
        }
    
    def get_disagreement_summary(self, cot_texts: List[str]) -> Dict[str, Any]:
        """Brief description."""
        overall_analysis = self.detect_overall_disagreement(cot_texts)
        
        cot_analysis = self.cot_analyzer.analyze_cot_traces(cot_texts)
        
        final_answers = cot_analysis.get('final_answers', [])
        answer_analysis = cot_analysis.get('answer_analysis', {})
        divergence_analysis = cot_analysis.get('divergence_analysis', {})
        
        num_traces = len(cot_texts)
        num_unique_answers = answer_analysis.get('num_unique_answers', 1)
        answer_diversity = num_unique_answers / num_traces if num_traces > 0 else 0
        
        overall_score = overall_analysis.get('overall_score', 0.0)
        if overall_score > 0.8:
            severity = "high"
        elif overall_score > 0.5:
            severity = "medium"
        elif overall_score > 0.2:
            severity = "low"
        else:
            severity = "minimal"
        
        return {
            'overall_analysis': overall_analysis,
            'cot_analysis': cot_analysis,
            'final_answers': final_answers,
            'answer_analysis': answer_analysis,
            'divergence_analysis': divergence_analysis,
            'metrics': {
                'num_traces': num_traces,
                'num_unique_answers': num_unique_answers,
                'answer_diversity': answer_diversity,
                'overall_score': overall_score,
                'severity': severity
            }
        }
