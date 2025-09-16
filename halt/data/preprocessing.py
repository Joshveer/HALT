"""Data preprocessing utilities for HALT."""

import re
import string
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreprocessor:
    """Utilities for preprocessing text data."""
    
    def __init__(self):
        """Initialize the preprocessor and download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        keywords = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2
            and token not in string.punctuation
        ]
        
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_keywords[:max_keywords]]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple lexical similarity between two texts."""
        tokens1 = set(self.extract_keywords(text1))
        tokens2 = set(self.extract_keywords(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def preprocess_qa_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Preprocess a question-answer pair."""
        cleaned_question = self.clean_text(question)
        cleaned_answer = self.clean_text(answer)
        
        question_keywords = self.extract_keywords(cleaned_question)
        answer_keywords = self.extract_keywords(cleaned_answer)
        
        return {
            'original_question': question,
            'original_answer': answer,
            'cleaned_question': cleaned_question,
            'cleaned_answer': cleaned_answer,
            'question_keywords': question_keywords,
            'answer_keywords': answer_keywords,
            'answer_length': len(cleaned_answer.split()),
            'question_length': len(cleaned_question.split())
        }
    
    def batch_preprocess(self, data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Preprocess a batch of question-answer pairs."""
        preprocessed = []
        
        for item in data:
            if 'question' in item and 'answer' in item:
                preprocessed.append(self.preprocess_qa_pair(item['question'], item['answer']))
            else:
                question = item.get('question', item.get('prompt', ''))
                answer = item.get('answer', item.get('generated_answer', ''))
                preprocessed.append(self.preprocess_qa_pair(question, answer))
        
        return preprocessed