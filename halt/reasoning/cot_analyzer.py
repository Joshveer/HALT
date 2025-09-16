"""Chain-of-thought reasoning analysis for hallucination detection."""

import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class CoTAnalyzer:
    """Analyzes chain-of-thought reasoning patterns and disagreement."""
    
    def __init__(self):
        """Initialize the CoT analyzer."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def extract_final_answer(self, cot_text: str) -> str:
        """Extract the final answer from a chain-of-thought response."""
        patterns = [
            r'So the answer is:?\s*(.+?)(?:\.|$)',
            r'Therefore,?\s*(.+?)(?:\.|$)',
            r'Thus,?\s*(.+?)(?:\.|$)',
            r'Hence,?\s*(.+?)(?:\.|$)',
            r'In conclusion,?\s*(.+?)(?:\.|$)',
            r'The answer is:?\s*(.+?)(?:\.|$)',
            r'Answer:?\s*(.+?)(?:\.|$)',
            r'Final answer:?\s*(.+?)(?:\.|$)',
            r'Therefore the answer is:?\s*(.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'^[:\-\s]+', '', answer)
                answer = re.sub(r'[:\-\s]+$', '', answer)
                if answer and len(answer) > 1:
                    return answer
        
        sentences = sent_tokenize(cot_text)
        if sentences:
            last_sentence = sentences[-1].strip()
            last_sentence = re.sub(r'^(So|Therefore|Thus|Hence|In conclusion|The answer|Answer|Final answer)', 
                                 '', last_sentence, flags=re.IGNORECASE).strip()
            if last_sentence and len(last_sentence) > 1:
                return last_sentence
        
        return cot_text[-50:].strip()
    
    def extract_reasoning_steps(self, cot_text: str) -> List[str]:
        """Extract individual reasoning steps from chain-of-thought text."""
        step_patterns = [
            r'(?:Step \d+:|First|Second|Third|Fourth|Fifth|Next|Then|Also|Additionally|Furthermore|Moreover)',
            r'(?:Let me|Let\'s|I need to|I should|I will|I can)',
            r'(?:To answer|To solve|To find|To determine)',
            r'(?:Looking at|Considering|Given that|Since|Because)',
            r'(?:This means|This suggests|This indicates|This shows)',
            r'(?:Therefore|Thus|Hence|So|In conclusion)',
        ]
        
        steps = []
        current_text = cot_text
        
        for pattern in step_patterns:
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    if i < len(matches) - 1:
                        end = matches[i + 1].start()
                        step_text = current_text[start:end].strip()
                    else:
                        step_text = current_text[start:].strip()
                    
                    if step_text and len(step_text) > 10:
                        steps.append(step_text)
                break
        
        if not steps:
            sentences = sent_tokenize(cot_text)
            steps = [s.strip() for s in sentences if s.strip()]
        
        cleaned_steps = []
        for step in steps:
            step = step.strip()
            if step and len(step) > 5:
                cleaned_steps.append(step)
        
        return cleaned_steps
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using word overlap."""
        words1 = set(word.lower() for word in word_tokenize(text1) 
                    if word.lower() not in self.stop_words and word.isalpha())
        words2 = set(word.lower() for word in word_tokenize(text2) 
                    if word.lower() not in self.stop_words and word.isalpha())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using difflib."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def analyze_reasoning_divergence(self, reasoning_steps_list: List[List[str]]) -> Dict[str, Any]:
        """Analyze divergence in reasoning steps across multiple CoT traces."""
        if len(reasoning_steps_list) < 2:
            return {'error': 'Need at least 2 reasoning traces for divergence analysis'}
        
        similarities = []
        for i in range(len(reasoning_steps_list)):
            for j in range(i + 1, len(reasoning_steps_list)):
                steps1 = reasoning_steps_list[i]
                steps2 = reasoning_steps_list[j]
                
                step_similarities = []
                max_steps = max(len(steps1), len(steps2))
                
                for k in range(max_steps):
                    step1 = steps1[k] if k < len(steps1) else ""
                    step2 = steps2[k] if k < len(steps2) else ""
                    
                    if step1 and step2:
                        semantic_sim = self.calculate_semantic_similarity(step1, step2)
                        sequence_sim = self.calculate_sequence_similarity(step1, step2)
                        combined_sim = (semantic_sim + sequence_sim) / 2
                        step_similarities.append(combined_sim)
                
                avg_similarity = np.mean(step_similarities) if step_similarities else 0.0
                similarities.append(avg_similarity)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        similarity_std = np.std(similarities)
        
        divergence_score = 1.0 - avg_similarity
        
        return {
            'avg_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'similarity_std': similarity_std,
            'divergence_score': divergence_score,
            'high_divergence': divergence_score > 0.5,
            'num_traces': len(reasoning_steps_list)
        }
    
    def analyze_cot_traces(self, cot_texts: List[str]) -> Dict[str, Any]:
        """Analyze multiple chain-of-thought traces for disagreement and divergence."""
        if len(cot_texts) < 2:
            return {'error': 'Need at least 2 CoT traces for analysis'}
        
        final_answers = [self.extract_final_answer(cot) for cot in cot_texts]
        reasoning_steps_list = [self.extract_reasoning_steps(cot) for cot in cot_texts]
        
        answer_analysis = self.analyze_answer_disagreement(final_answers)
        divergence_analysis = self.analyze_reasoning_divergence(reasoning_steps_list)
        
        answer_disagreement = answer_analysis.get('disagreement_score', 0)
        reasoning_divergence = divergence_analysis.get('divergence_score', 0)
        
        overall_disagreement = (answer_disagreement + reasoning_divergence) / 2
        
        return {
            'final_answers': final_answers,
            'reasoning_steps': reasoning_steps_list,
            'answer_analysis': answer_analysis,
            'divergence_analysis': divergence_analysis,
            'overall_disagreement': overall_disagreement,
            'is_disagreeing': overall_disagreement > 0.4,
            'num_traces': len(cot_texts)
        }
    
    def analyze_answer_disagreement(self, answers: List[str]) -> Dict[str, Any]:
        """Analyze disagreement between final answers."""
        if len(answers) < 2:
            return {'error': 'Need at least 2 answers for disagreement analysis'}
        
        cleaned_answers = [ans.strip().lower() for ans in answers if ans.strip()]
        
        if len(cleaned_answers) < 2:
            return {'error': 'Need at least 2 non-empty answers'}
        
        answer_counts = Counter(cleaned_answers)
        total_answers = len(cleaned_answers)
        
        most_common_count = answer_counts.most_common(1)[0][1]
        agreement_ratio = most_common_count / total_answers
        disagreement_score = 1.0 - agreement_ratio
        
        similarities = []
        for i in range(len(cleaned_answers)):
            for j in range(i + 1, len(cleaned_answers)):
                sim = self.calculate_semantic_similarity(cleaned_answers[i], cleaned_answers[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        is_disagreeing = disagreement_score > 0.3 or avg_similarity < 0.5
        
        return {
            'answers': cleaned_answers,
            'answer_counts': dict(answer_counts),
            'agreement_ratio': agreement_ratio,
            'disagreement_score': disagreement_score,
            'avg_similarity': avg_similarity,
            'is_disagreeing': is_disagreeing,
            'num_answers': total_answers
        }