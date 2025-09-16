"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class ThresholdOptimizer:
    """Brief description."""
    
    def __init__()(self):
        """Brief description."""
        """Brief description."""
        pass
    
    def optimize_entropy_threshold(self, 
                                  entropy_scores: List[float], 
                                  ground_truth: List[bool],
                                  target_precision: float = 0.8) -> Dict[str, Any]:
        """Brief description."""
        entropy_scores = np.array(entropy_scores)
        ground_truth = np.array(ground_truth)
        
        precision, recall, thresholds = precision_recall_curve(ground_truth, entropy_scores)
        
        valid_indices = precision >= target_precision
        if not np.any(valid_indices):
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        else:
            valid_recalls = recall[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            best_idx = np.argmax(valid_recalls)
            best_idx = np.where(thresholds == valid_thresholds[best_idx])[0][0]
        
        optimal_threshold = thresholds[best_idx]
        optimal_precision = precision[best_idx]
        optimal_recall = recall[best_idx]
        optimal_f1 = 2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall + 1e-10)
        
        auc_score = auc(recall, precision)
        
        return {
            'optimal_threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': optimal_f1,
            'auc': auc_score,
            'target_precision': target_precision,
            'achieved_target': optimal_precision >= target_precision
        }
    
    def optimize_disagreement_threshold(self, 
                                       disagreement_scores: List[float], 
                                       ground_truth: List[bool],
                                       target_precision: float = 0.8) -> Dict[str, Any]:
        """Brief description."""
        disagreement_scores = np.array(disagreement_scores)
        ground_truth = np.array(ground_truth)
        
        precision, recall, thresholds = precision_recall_curve(ground_truth, disagreement_scores)
        
        valid_indices = precision >= target_precision
        if not np.any(valid_indices):
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
        else:
            valid_recalls = recall[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            best_idx = np.argmax(valid_recalls)
            best_idx = np.where(thresholds == valid_thresholds[best_idx])[0][0]
        
        optimal_threshold = thresholds[best_idx]
        optimal_precision = precision[best_idx]
        optimal_recall = recall[best_idx]
        optimal_f1 = 2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall + 1e-10)
        
        auc_score = auc(recall, precision)
        
        return {
            'optimal_threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': optimal_f1,
            'auc': auc_score,
            'target_precision': target_precision,
            'achieved_target': optimal_precision >= target_precision
        }
    
    def optimize_combined_thresholds(self, 
                                   entropy_scores: List[float],
                                   disagreement_scores: List[float],
                                   ground_truth: List[bool],
                                   target_precision: float = 0.8) -> Dict[str, Any]:
        """Brief description."""
        entropy_scores = np.array(entropy_scores)
        disagreement_scores = np.array(disagreement_scores)
        ground_truth = np.array(ground_truth)
        
        entropy_thresholds = np.linspace(0.5, 4.0, 20)
        disagreement_thresholds = np.linspace(0.1, 0.8, 20)
        
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        best_entropy_thresh = 2.0
        best_disagreement_thresh = 0.4
        
        results = []
        
        for entropy_thresh in entropy_thresholds:
            for disagreement_thresh in disagreement_thresholds:
                entropy_detections = entropy_scores > entropy_thresh
                disagreement_detections = disagreement_scores > disagreement_thresh
                
                combined_detections = entropy_detections | disagreement_detections
                
                precision = precision_score(ground_truth, combined_detections, zero_division=0)
                recall = recall_score(ground_truth, combined_detections, zero_division=0)
                f1 = f1_score(ground_truth, combined_detections, zero_division=0)
                
                results.append({
                    'entropy_threshold': entropy_thresh,
                    'disagreement_threshold': disagreement_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
                if precision >= target_precision and f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    best_entropy_thresh = entropy_thresh
                    best_disagreement_thresh = disagreement_thresh
        
        return {
            'optimal_entropy_threshold': best_entropy_thresh,
            'optimal_disagreement_threshold': best_disagreement_thresh,
            'precision': best_precision,
            'recall': best_recall,
            'f1_score': best_f1,
            'target_precision': target_precision,
            'achieved_target': best_precision >= target_precision,
            'all_results': results
        }
    
    def plot_precision_recall_curve(self, 
                                   scores: List[float], 
                                   ground_truth: List[bool],
                                   title: str = "Precision-Recall Curve",
                                   save_path: Optional[str] = None) -> None:
        """Brief description."""
        precision, recall, thresholds = precision_recall_curve(ground_truth, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        f1_scores = np.linspace(0.1, 0.9, 9)
        for f1 in f1_scores:
            if f1 < 1:
                x = np.linspace(0.01, 1, 100)
                y = f1 * x / (2 * x - f1)
                y = y[(y > 0) & (y < 1)]
                x = x[(y > 0) & (y < 1)]
                if len(x) > 0:
                    plt.plot(x, y, '--', alpha=0.3, color='gray')
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_thresholds(self, 
                           entropy_scores: List[float],
                           disagreement_scores: List[float],
                           ground_truth: List[bool],
                           entropy_threshold: float,
                           disagreement_threshold: float) -> Dict[str, Any]:
        """Brief description."""
        entropy_scores = np.array(entropy_scores)
        disagreement_scores = np.array(disagreement_scores)
        ground_truth = np.array(ground_truth)
        
        entropy_detections = entropy_scores > entropy_threshold
        disagreement_detections = disagreement_scores > disagreement_threshold
        
        combined_detections = entropy_detections | disagreement_detections
        
        precision = precision_score(ground_truth, combined_detections, zero_division=0)
        recall = recall_score(ground_truth, combined_detections, zero_division=0)
        f1 = f1_score(ground_truth, combined_detections, zero_division=0)
        
        tp = np.sum((combined_detections == True) & (ground_truth == True))
        fp = np.sum((combined_detections == True) & (ground_truth == False))
        fn = np.sum((combined_detections == False) & (ground_truth == True))
        tn = np.sum((combined_detections == False) & (ground_truth == False))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_negative': int(tn)
            },
            'entropy_threshold': entropy_threshold,
            'disagreement_threshold': disagreement_threshold,
            'entropy_detections': int(np.sum(entropy_detections)),
            'disagreement_detections': int(np.sum(disagreement_detections)),
            'combined_detections': int(np.sum(combined_detections))
        }
