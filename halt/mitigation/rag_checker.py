"""Brief description."""

import re
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import wikipedia
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import warnings
warnings.filterwarnings("ignore")


class LightweightRAGChecker:
    """Brief description."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_retrieval_attempts: int = 3,
                 similarity_threshold: float = 0.3):
        """Brief description."""
        self.model_name = model_name
        self.max_retrieval_attempts = max_retrieval_attempts
        self.similarity_threshold = similarity_threshold
        
        print(f"Loading sentence transformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        wikipedia.set_lang("en")
        
        self.retrieval_cache = {}
    
    def extract_search_terms(self, question: str, answer: str) -> List[str]:
        """Brief description."""
        combined_text = f"{question} {answer}"
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
        
        search_terms = list(set(word for word in words if word not in stop_words))
        
        search_terms.sort(key=len, reverse=True)
        return search_terms[:5]  # Top 5 terms
    
    def retrieve_wikipedia_content(self, search_terms: List[str]) -> Dict[str, Any]:
        """Brief description."""
        cache_key = "_".join(sorted(search_terms))
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        retrieved_content = []
        successful_searches = 0
        
        for term in search_terms:
            if successful_searches >= 3:  # Limit to 3 successful retrievals
                break
                
            try:
                search_results = wikipedia.search(term, results=2)
                
                for page_title in search_results:
                    try:
                        page = wikipedia.page(page_title)
                        content = page.content
                        
                        paragraphs = content.split('\n\n')[:3]
                        relevant_content = '\n'.join(paragraphs)
                        
                        retrieved_content.append({
                            'title': page_title,
                            'content': relevant_content,
                            'url': page.url,
                            'search_term': term
                        })
                        
                        successful_searches += 1
                        break  # Move to next search term
                        
                    except wikipedia.exceptions.DisambiguationError as e:
                        try:
                            page = wikipedia.page(e.options[0])
                            content = page.content
                            paragraphs = content.split('\n\n')[:3]
                            relevant_content = '\n'.join(paragraphs)
                            
                            retrieved_content.append({
                                'title': e.options[0],
                                'content': relevant_content,
                                'url': page.url,
                                'search_term': term
                            })
                            
                            successful_searches += 1
                            break
                        except:
                            continue
                            
                    except Exception as e:
                        continue
                
                time.sleep(0.5)
                
            except Exception as e:
                continue
        
        result = {
            'content': retrieved_content,
            'num_sources': len(retrieved_content),
            'search_terms': search_terms,
            'successful_searches': successful_searches
        }
        
        self.retrieval_cache[cache_key] = result
        return result
    
    def calculate_similarity(self, answer: str, retrieved_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Brief description."""
        if not retrieved_content:
            return {
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'similarity_scores': [],
                'best_match': None,
                'is_supported': False
            }
        
        answer_embedding = self.encoder.encode([answer])
        
        similarities = []
        best_match = None
        max_similarity = 0.0
        
        for content_item in retrieved_content:
            content_embedding = self.encoder.encode([content_item['content']])
            
            similarity = cosine_similarity(answer_embedding, content_embedding)[0][0]
            similarities.append(similarity)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = content_item
        
        avg_similarity = np.mean(similarities)
        is_supported = max_similarity > self.similarity_threshold
        
        return {
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'similarity_scores': similarities,
            'best_match': best_match,
            'is_supported': is_supported,
            'threshold': self.similarity_threshold
        }
    
    def check_answer_against_knowledge(self, 
                                     question: str, 
                                     answer: str) -> Dict[str, Any]:
        """Brief description."""
        search_terms = self.extract_search_terms(question, answer)
        
        retrieval_result = self.retrieve_wikipedia_content(search_terms)
        
        similarity_result = self.calculate_similarity(answer, retrieval_result['content'])
        
        is_supported = similarity_result['is_supported']
        confidence = similarity_result['max_similarity']
        
        if is_supported:
            reason = "supported_by_knowledge"
        elif retrieval_result['num_sources'] == 0:
            reason = "no_knowledge_retrieved"
        elif similarity_result['max_similarity'] < 0.1:
            reason = "very_low_similarity"
        else:
            reason = "below_threshold"
        
        return {
            'is_supported': is_supported,
            'confidence': confidence,
            'reason': reason,
            'search_terms': search_terms,
            'retrieval_result': retrieval_result,
            'similarity_result': similarity_result,
            'threshold': self.similarity_threshold
        }
    
    def suggest_corrections(self, 
                          question: str, 
                          answer: str, 
                          knowledge_check: Dict[str, Any]) -> Dict[str, Any]:
        """Brief description."""
        if knowledge_check['is_supported']:
            return {
                'needs_correction': False,
                'confidence': knowledge_check['confidence'],
                'suggested_answer': answer,
                'reason': 'answer_supported_by_knowledge'
            }
        
        best_match = knowledge_check['similarity_result'].get('best_match')
        
        if best_match is None:
            return {
                'needs_correction': True,
                'confidence': 0.0,
                'suggested_answer': answer,
                'reason': 'no_knowledge_available',
                'correction_available': False
            }
        
        content = best_match['content']
        
        sentences = content.split('.')
        potential_answers = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 200:  # Reasonable length
                question_words = set(question.lower().split())
                sentence_words = set(sentence.lower().split())
                overlap = len(question_words.intersection(sentence_words))
                
                if overlap > 0:
                    potential_answers.append({
                        'text': sentence,
                        'overlap': overlap,
                        'length': len(sentence)
                    })
        
        potential_answers.sort(key=lambda x: (x['overlap'], x['length']), reverse=True)
        
        if potential_answers:
            suggested_answer = potential_answers[0]['text']
            correction_confidence = min(0.8, knowledge_check['confidence'] + 0.2)
            
            return {
                'needs_correction': True,
                'confidence': correction_confidence,
                'suggested_answer': suggested_answer,
                'reason': 'correction_from_knowledge',
                'correction_available': True,
                'potential_answers': [pa['text'] for pa in potential_answers[:3]]
            }
        else:
            return {
                'needs_correction': True,
                'confidence': 0.0,
                'suggested_answer': answer,
                'reason': 'no_suitable_correction_found',
                'correction_available': False
            }
    
    def apply_rag_mitigation(self, 
                           question: str, 
                           answer: str) -> Dict[str, Any]:
        """Brief description."""
        knowledge_check = self.check_answer_against_knowledge(question, answer)
        
        correction_suggestions = self.suggest_corrections(question, answer, knowledge_check)
        
        if correction_suggestions['needs_correction'] and correction_suggestions['correction_available']:
            final_answer = correction_suggestions['suggested_answer']
            mitigation_applied = True
        else:
            final_answer = answer
            mitigation_applied = False
        
        return {
            'original_answer': answer,
            'mitigated_answer': final_answer,
            'mitigation_applied': mitigation_applied,
            'knowledge_check': knowledge_check,
            'correction_suggestions': correction_suggestions,
            'confidence': knowledge_check['confidence'],
            'is_supported': knowledge_check['is_supported']
        }
