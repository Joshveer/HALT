"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from ..reasoning.cot_analyzer import CoTAnalyzer


class SelfConsistencyMitigator:
    """Brief description."""
    
    def __init__()(self):
        """Brief description."""
        """Brief description."""
        self.cot_analyzer = CoTAnalyzer()
    
    def select_majority_answer(self, answers: List[str], 
                              min_agreement: float = 0.4) -> Dict[str, Any]:
        """Brief description."""
        if not answers:
            return {
                'selected_answer': '',
                'confidence': 0.0,
                'agreement_ratio': 0.0,
                'method': 'no_answers'
            }
        
        cleaned_answers = [ans.strip().lower() for ans in answers if ans.strip()]
        
        if not cleaned_answers:
            return {
                'selected_answer': '',
                'confidence': 0.0,
                'agreement_ratio': 0.0,
                'method': 'no_valid_answers'
            }
        
        answer_counts = Counter(cleaned_answers)
        total_answers = len(cleaned_answers)
        
        most_common = answer_counts.most_common(1)[0]
        most_common_answer = most_common[0]
        most_common_count = most_common[1]
        
        agreement_ratio = most_common_count / total_answers
        
        if agreement_ratio >= min_agreement:
            confidence = agreement_ratio
            method = 'majority_consensus'
        else:
            confidence = agreement_ratio * 0.7  # Reduce confidence for weak consensus
            method = 'weak_consensus'
        
        original_answer = None
        for ans in answers:
            if ans.strip().lower() == most_common_answer:
                original_answer = ans.strip()
                break
        
        if original_answer is None:
            original_answer = most_common_answer
        
        return {
            'selected_answer': original_answer,
            'confidence': confidence,
            'agreement_ratio': agreement_ratio,
            'method': method,
            'answer_distribution': dict(answer_counts),
            'total_answers': total_answers,
            'unique_answers': len(answer_counts)
        }
    
    def select_best_cot_answer(self, cot_texts: List[str], 
                              min_agreement: float = 0.4) -> Dict[str, Any]:
        """Brief description."""
        if not cot_texts:
            return {
                'selected_answer': '',
                'confidence': 0.0,
                'agreement_ratio': 0.0,
                'method': 'no_cot_texts'
            }
        
        final_answers = [self.cot_analyzer.extract_final_answer(cot) for cot in cot_texts]
        
        majority_result = self.select_majority_answer(final_answers, min_agreement)
        
        cot_analysis = self.cot_analyzer.analyze_cot_traces(cot_texts)
        reasoning_quality = self._assess_reasoning_quality(cot_texts)
        
        base_confidence = majority_result['confidence']
        reasoning_penalty = max(0, 1.0 - reasoning_quality)
        adjusted_confidence = base_confidence * (1.0 - reasoning_penalty * 0.3)
        
        return {
            'selected_answer': majority_result['selected_answer'],
            'confidence': adjusted_confidence,
            'agreement_ratio': majority_result['agreement_ratio'],
            'method': majority_result['method'],
            'reasoning_quality': reasoning_quality,
            'cot_analysis': cot_analysis,
            'answer_distribution': majority_result['answer_distribution'],
            'total_answers': majority_result['total_answers'],
            'unique_answers': majority_result['unique_answers']
        }
    
    def _assess_reasoning_quality(self, cot_texts: List[str]) -> float:
        """Brief description."""
        if not cot_texts:
            return 0.0
        
        quality_scores = []
        
        for cot_text in cot_texts:
            score = 0.0
            
            reasoning_indicators = [
                'because', 'since', 'therefore', 'thus', 'hence',
                'first', 'second', 'third', 'next', 'then',
                'let me', 'i need to', 'i should', 'i will'
            ]
            
            cot_lower = cot_text.lower()
            indicator_count = sum(1 for indicator in reasoning_indicators 
                                if indicator in cot_lower)
            score += min(0.3, indicator_count * 0.05)  # Max 0.3 for indicators
            
            step_patterns = [
                r'step \d+', r'first', r'second', r'third', r'next', r'then'
            ]
            step_count = sum(1 for pattern in step_patterns 
                           if re.search(pattern, cot_lower))
            score += min(0.2, step_count * 0.05)  # Max 0.2 for structure
            
            connectors = ['because', 'since', 'therefore', 'thus', 'hence', 'so']
            connector_count = sum(1 for connector in connectors 
                                if connector in cot_lower)
            score += min(0.2, connector_count * 0.05)  # Max 0.2 for connectors
            
            question_indicators = ['question', 'asked', 'wants to know', 'looking for']
            question_count = sum(1 for indicator in question_indicators 
                               if indicator in cot_lower)
            score += min(0.1, question_count * 0.05)  # Max 0.1 for question analysis
            
            conclusion_indicators = ['answer', 'conclusion', 'therefore', 'so the answer']
            conclusion_count = sum(1 for indicator in conclusion_indicators 
                                 if indicator in cot_lower)
            score += min(0.2, conclusion_count * 0.05)  # Max 0.2 for conclusion
            
            quality_scores.append(min(1.0, score))
        
        return np.mean(quality_scores)
    
    def apply_self_consistency(self, 
                             generations: List[Dict[str, Any]], 
                             method: str = 'majority',
                             min_agreement: float = 0.4) -> Dict[str, Any]:
        """Brief description."""
        if not generations:
            return {
                'mitigated_answer': '',
                'confidence': 0.0,
                'method': 'no_generations',
                'original_answers': []
            }
        
        if method == 'cot_based':
            cot_texts = [gen.get('generated_text', '') for gen in generations]
            result = self.select_best_cot_answer(cot_texts, min_agreement)
        else:
            answers = [gen.get('generated_text', '') for gen in generations]
            result = self.select_majority_answer(answers, min_agreement)
        
        original_answers = [gen.get('generated_text', '') for gen in generations]
        answer_diversity = len(set(ans.lower().strip() for ans in original_answers 
                                 if ans.strip())) / len(original_answers)
        
        return {
            'mitigated_answer': result['selected_answer'],
            'confidence': result['confidence'],
            'agreement_ratio': result['agreement_ratio'],
            'method': result['method'],
            'original_answers': original_answers,
            'answer_diversity': answer_diversity,
            'improvement_metrics': {
                'consensus_achieved': result['agreement_ratio'] >= min_agreement,
                'confidence_level': result['confidence'],
                'answer_consistency': 1.0 - answer_diversity
            }
        }
    
    def compare_with_original(self, 
                            original_answer: str,
                            mitigated_answer: str,
                            ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Brief description."""
        original_lower = original_answer.lower().strip()
        mitigated_lower = mitigated_answer.lower().strip()
        
        original_words = set(original_lower.split())
        mitigated_words = set(mitigated_lower.split())
        
        if original_words or mitigated_words:
            lexical_similarity = len(original_words.intersection(mitigated_words)) / \
                               len(original_words.union(mitigated_words))
        else:
            lexical_similarity = 1.0 if original_lower == mitigated_lower else 0.0
        
        answers_changed = original_lower != mitigated_lower
        
        ground_truth_comparison = None
        if ground_truth:
            ground_truth_lower = ground_truth.lower().strip()
            
            original_gt_sim = len(original_words.intersection(set(ground_truth_lower.split()))) / \
                             len(original_words.union(set(ground_truth_lower.split()))) if original_words or ground_truth_lower else 0.0
            
            mitigated_gt_sim = len(mitigated_words.intersection(set(ground_truth_lower.split()))) / \
                              len(mitigated_words.union(set(ground_truth_lower.split()))) if mitigated_words or ground_truth_lower else 0.0
            
            ground_truth_comparison = {
                'original_gt_similarity': original_gt_sim,
                'mitigated_gt_similarity': mitigated_gt_sim,
                'improved_gt_similarity': mitigated_gt_sim > original_gt_sim,
                'similarity_improvement': mitigated_gt_sim - original_gt_sim
            }
        
        return {
            'original_answer': original_answer,
            'mitigated_answer': mitigated_answer,
            'lexical_similarity': lexical_similarity,
            'answers_changed': answers_changed,
            'ground_truth_comparison': ground_truth_comparison,
            'mitigation_applied': answers_changed
        }
