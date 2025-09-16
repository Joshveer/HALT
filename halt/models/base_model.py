"""Base model integration with Hugging Face transformers."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
warnings.filterwarnings("ignore")


class BaseModel:
    """Base model wrapper for Hugging Face transformers with logit access."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512):
        """Initialize the base model."""
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        if self.device != 'cuda' or not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generation_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                temperature: float = 0.7, return_logits: bool = False) -> Dict[str, Any]:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        config = self.generation_config
        config.max_new_tokens = max_new_tokens
        config.temperature = temperature
        
        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=config,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                logits = []
                for score in outputs.scores:
                    logits.append(score[0].cpu().numpy())
                
                return {
                    'generated_text': generated_text,
                    'logits': logits,
                    'generated_ids': generated_ids.cpu().numpy(),
                    'input_length': inputs['input_ids'].shape[1]
                }
            else:
                outputs = self.model.generate(**inputs, generation_config=config)
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return {
                    'generated_text': generated_text,
                    'generated_ids': generated_ids.cpu().numpy(),
                    'input_length': inputs['input_ids'].shape[1]
                }
    
    def generate_multiple(self, prompt: str, num_samples: int = 5, 
                         max_new_tokens: int = 100, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """Generate multiple samples from the same prompt."""
        results = []
        
        for i in range(num_samples):
            sample_temp = temperature + np.random.normal(0, 0.1)
            sample_temp = max(0.1, min(2.0, sample_temp))
            
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=sample_temp,
                return_logits=False
            )
            result['sample_id'] = i
            results.append(result)
        
        return results
    
    def get_token_probabilities(self, prompt: str, max_new_tokens: int = 100) -> Dict[str, Any]:
        """Get token probabilities for generated text."""
        result = self.generate(prompt, max_new_tokens, return_logits=True)
        
        if 'logits' not in result:
            return {'error': 'Logits not available'}
        
        logits = result['logits']
        generated_ids = result['generated_ids']
        
        probabilities = []
        tokens = []
        
        for i, (logit, token_id) in enumerate(zip(logits, generated_ids)):
            probs = torch.softmax(torch.tensor(logit), dim=-1).numpy()
            token_prob = probs[token_id]
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            probabilities.append(token_prob)
            tokens.append({
                'id': int(token_id),
                'text': token_text,
                'probability': float(token_prob),
                'logits': logit.tolist()
            })
        
        return {
            'tokens': tokens,
            'probabilities': probabilities,
            'generated_text': result['generated_text']
        }
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)