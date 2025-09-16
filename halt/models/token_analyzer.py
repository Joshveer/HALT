"""Token-level uncertainty analysis for hallucination detection."""

import numpy as np
import torch
from typing import List, Dict, Any
import re


class TokenAnalyzer:
    """Analyzes token-level uncertainty and tokenization patterns."""
    
    def __init__(self, tokenizer):
        """Initialize the token analyzer."""
        self.tokenizer = tokenizer
    
    def calculate_entropy(self, logits: np.ndarray) -> float:
        """Calculate Shannon entropy from logits."""
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def calculate_perplexity(self, logits: np.ndarray) -> float:
        """Calculate perplexity from logits."""
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        perplexity = np.exp(entropy)
        return float(perplexity)
    
    def analyze_token_uncertainty(self, token_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze uncertainty metrics for a sequence of tokens."""
        if not token_data:
            return {'error': 'No token data provided'}
        
        logits_list = [token['logits'] for token in token_data]
        probabilities = [token['probability'] for token in token_data]
        
        token_entropies = []
        token_perplexities = []
        
        for logits in logits_list:
            entropy = self.calculate_entropy(logits)
            perplexity = self.calculate_perplexity(logits)
            token_entropies.append(entropy)
            token_perplexities.append(perplexity)
        
        avg_entropy = np.mean(token_entropies)
        max_entropy = np.max(token_entropies)
        min_entropy = np.min(token_entropies)
        entropy_std = np.std(token_entropies)
        
        avg_perplexity = np.mean(token_perplexities)
        max_perplexity = np.max(token_perplexities)
        
        avg_probability = np.mean(probabilities)
        min_probability = np.min(probabilities)
        probability_std = np.std(probabilities)
        
        confidence_score = 1.0 / (1.0 + avg_entropy)
        
        return {
            'token_entropies': token_entropies,
            'token_perplexities': token_perplexities,
            'avg_entropy': avg_entropy,
            'max_entropy': max_entropy,
            'min_entropy': min_entropy,
            'entropy_std': entropy_std,
            'avg_perplexity': avg_perplexity,
            'max_perplexity': max_perplexity,
            'avg_probability': avg_probability,
            'min_probability': min_probability,
            'probability_std': probability_std,
            'confidence_score': confidence_score,
            'num_tokens': len(token_data)
        }
    
    def analyze_tokenization_fragility(self, text: str) -> Dict[str, Any]:
        """Analyze tokenization patterns that might indicate fragility."""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        subword_count = 0
        rare_token_count = 0
        unknown_token_count = 0
        
        subword_pattern = re.compile(r'^(##|▁)')
        
        for token in tokens:
            if subword_pattern.match(token):
                subword_count += 1
            
            if len(token) > 10 or re.search(r'[^\w\s]', token):
                rare_token_count += 1
        
        unknown_token_id = self.tokenizer.unk_token_id
        if unknown_token_id is not None:
            unknown_token_count = sum(1 for token_id in token_ids if token_id == unknown_token_id)
        
        total_tokens = len(tokens)
        subword_ratio = subword_count / total_tokens if total_tokens > 0 else 0
        rare_token_ratio = rare_token_count / total_tokens if total_tokens > 0 else 0
        unknown_token_ratio = unknown_token_count / total_tokens if total_tokens > 0 else 0
        
        avg_token_length = np.mean([len(token) for token in tokens]) if tokens else 0
        
        fragility_score = (subword_ratio * 0.4 + 
                          rare_token_ratio * 0.3 + 
                          unknown_token_ratio * 0.3)
        
        return {
            'total_tokens': total_tokens,
            'subword_count': subword_count,
            'subword_ratio': subword_ratio,
            'rare_token_count': rare_token_count,
            'rare_token_ratio': rare_token_ratio,
            'unknown_token_count': unknown_token_count,
            'unknown_token_ratio': unknown_token_ratio,
            'avg_token_length': avg_token_length,
            'fragility_score': fragility_score,
            'tokens': tokens
        }
    
    def detect_uncertainty_patterns(self, token_analysis: Dict[str, Any], 
                                  tokenization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns that might indicate hallucinations."""
        patterns = {}
        
        avg_entropy = token_analysis.get('avg_entropy', 0)
        entropy_std = token_analysis.get('entropy_std', 0)
        
        patterns['high_avg_entropy'] = avg_entropy > 2.0
        patterns['high_entropy_variance'] = entropy_std > 1.0
        patterns['very_high_entropy'] = avg_entropy > 3.0
        
        confidence_score = token_analysis.get('confidence_score', 1.0)
        patterns['low_confidence'] = confidence_score < 0.3
        
        fragility_score = tokenization_analysis.get('fragility_score', 0)
        subword_ratio = tokenization_analysis.get('subword_ratio', 0)
        rare_token_ratio = tokenization_analysis.get('rare_token_ratio', 0)
        
        patterns['high_fragility'] = fragility_score > 0.5
        patterns['high_subword_ratio'] = subword_ratio > 0.3
        patterns['high_rare_token_ratio'] = rare_token_ratio > 0.2
        
        uncertainty_factors = [
            avg_entropy / 3.0,
            (1.0 - confidence_score),
            fragility_score,
        ]
        
        uncertainty_score = np.mean(uncertainty_factors)
        patterns['uncertainty_score'] = uncertainty_score
        patterns['high_uncertainty'] = uncertainty_score > 0.6
        
        return patterns
    
    def get_uncertainty_summary(self, text: str, token_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a comprehensive uncertainty summary for generated text."""
        token_analysis = self.analyze_token_uncertainty(token_data)
        tokenization_analysis = self.analyze_tokenization_fragility(text)
        patterns = self.detect_uncertainty_patterns(token_analysis, tokenization_analysis)
        
        return {
            'text': text,
            'token_analysis': token_analysis,
            'tokenization_analysis': tokenization_analysis,
            'patterns': patterns,
            'overall_uncertainty': patterns.get('uncertainty_score', 0),
            'is_uncertain': patterns.get('high_uncertainty', False)
        }