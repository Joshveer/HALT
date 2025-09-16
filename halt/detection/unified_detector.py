"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..models.token_analyzer import TokenAnalyzer
from ..reasoning.disagreement_detector import DisagreementDetector


class UnifiedDetector:
    """Brief description."""
    
    def __init__(self, 
                 entropy_threshold: float = 2.0,
                 disagreement_threshold: float = 0.4,
                 uncertainty_threshold: float = 0.6,
                 tokenizer=None):
        """Brief description."""
        self.entropy_threshold = entropy_threshold
        self.disagreement_threshold = disagreement_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        self.token_analyzer = TokenAnalyzer(tokenizer) if tokenizer else None
        self.disagreement_detector = DisagreementDetector()
    
    def detect_from_token_analysis(self, token_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Brief description."""
        if 'error' in token_analysis:
            return {
                'is_hallucination': False,
                'confidence': 0.0,
                'reason': 'token_analysis_error',
                'signals': {}
            }
        
        avg_entropy = token_analysis.get('avg_entropy', 0)
        confidence_score = token_analysis.get('confidence_score', 1.0)
        uncertainty_score = token_analysis.get('patterns', {}).get('uncertainty_score', 0)
        
        high_entropy = avg_entropy > self.entropy_threshold
        low_confidence = confidence_score < 0.3
        high_uncertainty = uncertainty_score > self.uncertainty_threshold
        
        token_signals = {
            'high_entropy': high_entropy,
            'low_confidence': low_confidence,
            'high_uncertainty': high_uncertainty,
            'avg_entropy': avg_entropy,
            'confidence_score': confidence_score,
            'uncertainty_score': uncertainty_score
        }
        
        is_hallucination = high_entropy or low_confidence or high_uncertainty
        
        signal_strength = 0
        if high_entropy:
            signal_strength += (avg_entropy - self.entropy_threshold) / self.entropy_threshold
        if low_confidence:
            signal_strength += (0.3 - confidence_score) / 0.3
        if high_uncertainty:
            signal_strength += (uncertainty_score - self.uncertainty_threshold) / self.uncertainty_threshold
        
        confidence = min(1.0, signal_strength / 3.0)  # Normalize to 0-1
        
        reasons = []
        if high_entropy:
            reasons.append("high_token_entropy")
        if low_confidence:
            reasons.append("low_confidence")
        if high_uncertainty:
            reasons.append("high_uncertainty")
        
        reason = "agreement" if not reasons else "_".join(reasons)
        
        return {
            'is_hallucination': is_hallucination,
            'confidence': confidence,
            'reason': reason,
            'signals': token_signals
        }
    
    def detect_from_cot_disagreement(self, cot_texts: List[str]) -> Dict[str, Any]:
        """Brief description."""
        if len(cot_texts) < 2:
            return {
                'is_hallucination': False,
                'confidence': 0.0,
                'reason': 'insufficient_cot_samples',
                'signals': {}
            }
        
        disagreement_analysis = self.disagreement_detector.detect_overall_disagreement(
            cot_texts, 
            answer_threshold=self.disagreement_threshold,
            reasoning_threshold=self.disagreement_threshold
        )
        
        is_disagreeing = disagreement_analysis.get('is_disagreeing', False)
        overall_score = disagreement_analysis.get('overall_score', 0)
        confidence = disagreement_analysis.get('confidence', 0)
        reason = disagreement_analysis.get('reason', 'agreement')
        
        is_hallucination = is_disagreeing and overall_score > self.disagreement_threshold
        
        cot_signals = {
            'is_disagreeing': is_disagreeing,
            'disagreement_score': overall_score,
            'confidence': confidence,
            'reason': reason
        }
        
        return {
            'is_hallucination': is_hallucination,
            'confidence': confidence,
            'reason': reason,
            'signals': cot_signals
        }
    
    def detect_unified(self, 
                      token_analysis: Optional[Dict[str, Any]] = None,
                      cot_texts: Optional[List[str]] = None,
                      text: Optional[str] = None) -> Dict[str, Any]:
        """Brief description."""
        detection_results = {
            'token_detection': None,
            'cot_detection': None,
            'unified_detection': None
        }
        
        if token_analysis is not None:
            token_detection = self.detect_from_token_analysis(token_analysis)
            detection_results['token_detection'] = token_detection
        elif text is not None and self.token_analyzer is not None:
            token_detection = {
                'is_hallucination': False,
                'confidence': 0.0,
                'reason': 'token_analysis_not_available',
                'signals': {}
            }
            detection_results['token_detection'] = token_detection
        
        if cot_texts is not None:
            cot_detection = self.detect_from_cot_disagreement(cot_texts)
            detection_results['cot_detection'] = cot_detection
        
        unified_result = self._combine_detections(detection_results)
        detection_results['unified_detection'] = unified_result
        
        return detection_results
    
    def _combine_detections(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Brief description."""
        token_detection = detection_results.get('token_detection', {})
        cot_detection = detection_results.get('cot_detection', {})
        
        token_hallucination = token_detection.get('is_hallucination', False)
        token_confidence = token_detection.get('confidence', 0.0)
        
        cot_hallucination = cot_detection.get('is_hallucination', False)
        cot_confidence = cot_detection.get('confidence', 0.0)
        
        is_hallucination = token_hallucination or cot_hallucination
        
        if token_hallucination and cot_hallucination:
            combined_confidence = (token_confidence + cot_confidence) / 2
            reason = "both_token_and_cot_signals"
        elif token_hallucination or cot_hallucination:
            combined_confidence = max(token_confidence, cot_confidence)
            if token_hallucination:
                reason = "token_signal_only"
            else:
                reason = "cot_signal_only"
        else:
            combined_confidence = 0.0
            reason = "no_signals"
        
        if token_hallucination and cot_hallucination:
            combined_confidence *= 1.2  # Boost confidence when signals agree
        elif token_hallucination or cot_hallucination:
            combined_confidence *= 0.8  # Reduce confidence for single signal
        
        combined_confidence = min(1.0, combined_confidence)
        
        return {
            'is_hallucination': is_hallucination,
            'confidence': combined_confidence,
            'reason': reason,
            'token_signal': token_hallucination,
            'cot_signal': cot_hallucination,
            'token_confidence': token_confidence,
            'cot_confidence': cot_confidence
        }
    
    def update_thresholds(self, 
                         entropy_threshold: Optional[float] = None,
                         disagreement_threshold: Optional[float] = None,
                         uncertainty_threshold: Optional[float] = None):
        """Brief description."""
        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
        if disagreement_threshold is not None:
            self.disagreement_threshold = disagreement_threshold
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
    
    def get_detection_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Brief description."""
        unified = detection_results.get('unified_detection', {})
        token = detection_results.get('token_detection', {})
        cot = detection_results.get('cot_detection', {})
        
        return {
            'final_decision': unified.get('is_hallucination', False),
            'confidence': unified.get('confidence', 0.0),
            'reason': unified.get('reason', 'unknown'),
            'token_contributed': token.get('is_hallucination', False),
            'cot_contributed': cot.get('is_hallucination', False),
            'token_confidence': token.get('confidence', 0.0),
            'cot_confidence': cot.get('confidence', 0.0),
            'thresholds': {
                'entropy_threshold': self.entropy_threshold,
                'disagreement_threshold': self.disagreement_threshold,
                'uncertainty_threshold': self.uncertainty_threshold
            }
        }
