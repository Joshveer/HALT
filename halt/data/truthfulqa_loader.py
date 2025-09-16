"""TruthfulQA dataset loader and preprocessing."""

import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd


class TruthfulQALoader:
    """Loader for TruthfulQA dataset with preprocessing capabilities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the TruthfulQA loader."""
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "halt")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_dataset(self, subset_size: Optional[int] = None) -> pd.DataFrame:
        """Load TruthfulQA dataset from HuggingFace datasets."""
        try:
            dataset = load_dataset("truthful_qa", "multiple_choice", cache_dir=self.cache_dir)
            train_data = dataset["validation"]
            
            data = []
            for item in train_data:
                data.append({
                    'question': item['question'],
                    'best_answer': item['best_answer'],
                    'correct_answers': item['correct_answers'],
                    'incorrect_answers': item['incorrect_answers'],
                    'category': item.get('category', 'unknown')
                })
            
            df = pd.DataFrame(data)
            
            if subset_size is not None:
                df = df.sample(n=min(subset_size, len(df)), random_state=42)
                df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading TruthfulQA from HuggingFace: {e}")
            print("Falling back to local sample data...")
            return self._create_sample_data(subset_size or 50)
    
    def _create_sample_data(self, size: int) -> pd.DataFrame:
        """Create sample data for testing when TruthfulQA is not available."""
        sample_questions = [
            {
                'question': 'What is the capital of France?',
                'best_answer': 'Paris',
                'correct_answers': ['Paris'],
                'incorrect_answers': ['London', 'Berlin', 'Madrid'],
                'category': 'geography'
            },
            {
                'question': 'Who wrote "Romeo and Juliet"?',
                'best_answer': 'William Shakespeare',
                'correct_answers': ['William Shakespeare', 'Shakespeare'],
                'incorrect_answers': ['Charles Dickens', 'Mark Twain', 'Jane Austen'],
                'category': 'literature'
            },
            {
                'question': 'What is the largest planet in our solar system?',
                'best_answer': 'Jupiter',
                'correct_answers': ['Jupiter'],
                'incorrect_answers': ['Saturn', 'Earth', 'Neptune'],
                'category': 'science'
            },
            {
                'question': 'In what year did World War II end?',
                'best_answer': '1945',
                'correct_answers': ['1945'],
                'incorrect_answers': ['1944', '1946', '1943'],
                'category': 'history'
            },
            {
                'question': 'What is the chemical symbol for gold?',
                'best_answer': 'Au',
                'correct_answers': ['Au'],
                'incorrect_answers': ['Go', 'Gd', 'Ag'],
                'category': 'science'
            }
        ]
        
        repeated_data = []
        for i in range(size):
            base_question = sample_questions[i % len(sample_questions)]
            question = base_question['question']
            if i >= len(sample_questions):
                question = f"Question {i+1}: {question}"
            
            repeated_data.append({
                'question': question,
                'best_answer': base_question['best_answer'],
                'correct_answers': base_question['correct_answers'],
                'incorrect_answers': base_question['incorrect_answers'],
                'category': base_question['category']
            })
        
        return pd.DataFrame(repeated_data)
    
    def format_for_generation(self, df: pd.DataFrame, include_cot: bool = False) -> List[Dict[str, str]]:
        """Format dataset for model generation."""
        formatted_data = []
        
        for _, row in df.iterrows():
            if include_cot:
                prompt = f"Question: {row['question']}\n\nLet's think step by step to answer this question accurately.\n\nAnswer:"
            else:
                prompt = f"Question: {row['question']}\n\nAnswer:"
            
            formatted_data.append({
                'prompt': prompt,
                'question': row['question'],
                'best_answer': row['best_answer'],
                'correct_answers': row['correct_answers'],
                'incorrect_answers': row['incorrect_answers'],
                'category': row['category']
            })
        
        return formatted_data
    
    def evaluate_answer(self, generated_answer: str, correct_answers: List[str], 
                       incorrect_answers: List[str]) -> Dict[str, Any]:
        """Evaluate a generated answer against ground truth."""
        generated_lower = generated_answer.lower().strip()
        correct_lower = [ans.lower().strip() for ans in correct_answers]
        incorrect_lower = [ans.lower().strip() for ans in incorrect_answers]
        
        is_correct = any(correct in generated_lower for correct in correct_lower)
        is_incorrect = any(incorrect in generated_lower for incorrect in incorrect_lower)
        is_hallucination = not is_correct and (is_incorrect or len(generated_answer.strip()) > 0)
        
        return {
            'is_correct': is_correct,
            'is_incorrect': is_incorrect,
            'is_hallucination': is_hallucination,
            'generated_answer': generated_answer,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers
        }