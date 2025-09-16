"""Brief description."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationMetrics:
    """Brief description."""
    
    def __init__()(self):
        """Brief description."""
        """Brief description."""
        pass
    
    def calculate_detection_metrics(self, 
                                  y_true: List[bool], 
                                  y_pred: List[bool],
                                  y_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Brief description."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        auc = None
        if y_scores is not None:
            try:
                auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'auc': auc,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(np.sum(~y_true))
        }
    
    def calculate_mitigation_metrics(self, 
                                   original_answers: List[str],
                                   mitigated_answers: List[str],
                                   ground_truth: List[str],
                                   original_correct: List[bool],
                                   mitigated_correct: List[bool]) -> Dict[str, Any]:
        """Brief description."""
        original_correct = np.array(original_correct)
        mitigated_correct = np.array(mitigated_correct)
        
        original_accuracy = np.mean(original_correct)
        mitigated_accuracy = np.mean(mitigated_correct)
        accuracy_improvement = mitigated_accuracy - original_accuracy
        
        original_hallucinations = np.sum(~original_correct)
        mitigated_hallucinations = np.sum(~mitigated_correct)
        hallucination_reduction = original_hallucinations - mitigated_hallucinations
        hallucination_reduction_rate = hallucination_reduction / len(original_correct)
        
        helped_cases = np.sum((~original_correct) & mitigated_correct)
        hurt_cases = np.sum(original_correct & (~mitigated_correct))
        unchanged_cases = np.sum(original_correct == mitigated_correct)
        
        total_incorrect_original = np.sum(~original_correct)
        mitigation_effectiveness = helped_cases / total_incorrect_original if total_incorrect_original > 0 else 0
        
        answer_changes = []
        for orig, mit in zip(original_answers, mitigated_answers):
            answer_changes.append(orig.lower().strip() != mit.lower().strip())
        
        change_rate = np.mean(answer_changes)
        
        return {
            'original_accuracy': original_accuracy,
            'mitigated_accuracy': mitigated_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'original_hallucinations': int(original_hallucinations),
            'mitigated_hallucinations': int(mitigated_hallucinations),
            'hallucination_reduction': int(hallucination_reduction),
            'hallucination_reduction_rate': hallucination_reduction_rate,
            'helped_cases': int(helped_cases),
            'hurt_cases': int(hurt_cases),
            'unchanged_cases': int(unchanged_cases),
            'mitigation_effectiveness': mitigation_effectiveness,
            'answer_change_rate': change_rate,
            'total_samples': len(original_correct)
        }
    
    def calculate_system_metrics(self, 
                               detection_results: Dict[str, Any],
                               mitigation_results: Dict[str, Any],
                               ground_truth: List[bool]) -> Dict[str, Any]:
        """Brief description."""
        detection_f1 = detection_results.get('f1_score', 0)
        detection_precision = detection_results.get('precision', 0)
        detection_recall = detection_results.get('recall', 0)
        
        accuracy_improvement = mitigation_results.get('accuracy_improvement', 0)
        hallucination_reduction_rate = mitigation_results.get('hallucination_reduction_rate', 0)
        mitigation_effectiveness = mitigation_results.get('mitigation_effectiveness', 0)
        
        detection_score = (detection_f1 + detection_precision + detection_recall) / 3
        mitigation_score = (accuracy_improvement + hallucination_reduction_rate + mitigation_effectiveness) / 3
        
        detection_score = min(1.0, max(0.0, detection_score))
        mitigation_score = min(1.0, max(0.0, mitigation_score))
        
        overall_score = 0.6 * detection_score + 0.4 * mitigation_score
        
        if overall_score >= 0.8:
            performance_level = "excellent"
        elif overall_score >= 0.6:
            performance_level = "good"
        elif overall_score >= 0.4:
            performance_level = "fair"
        else:
            performance_level = "poor"
        
        return {
            'overall_score': overall_score,
            'performance_level': performance_level,
            'detection_score': detection_score,
            'mitigation_score': mitigation_score,
            'detection_metrics': {
                'f1_score': detection_f1,
                'precision': detection_precision,
                'recall': detection_recall
            },
            'mitigation_metrics': {
                'accuracy_improvement': accuracy_improvement,
                'hallucination_reduction_rate': hallucination_reduction_rate,
                'mitigation_effectiveness': mitigation_effectiveness
            }
        }
    
    def calculate_per_category_metrics(self, 
                                     results: List[Dict[str, Any]], 
                                     categories: List[str]) -> Dict[str, Dict[str, Any]]:
        """Brief description."""
        category_metrics = {}
        
        for category in categories:
            category_results = [r for r in results if r.get('category') == category]
            
            if not category_results:
                category_metrics[category] = {'error': 'no_data_for_category'}
                continue
            
            y_true = [r.get('is_hallucination', False) for r in category_results]
            y_pred = [r.get('detected_hallucination', False) for r in category_results]
            
            metrics = self.calculate_detection_metrics(y_true, y_pred)
            category_metrics[category] = metrics
        
        return category_metrics
    
    def plot_confusion_matrix(self, 
                            y_true: List[bool], 
                            y_pred: List[bool],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """Brief description."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Hallucination', 'Hallucination'],
                   yticklabels=['Not Hallucination', 'Hallucination'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, 
                                  y_true: List[bool], 
                                  y_scores: List[float],
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> None:
        """Brief description."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_mitigation_improvement(self, 
                                  original_accuracy: float,
                                  mitigated_accuracy: float,
                                  categories: Optional[List[str]] = None,
                                  category_improvements: Optional[Dict[str, float]] = None,
                                  title: str = "Mitigation Improvement",
                                  save_path: Optional[str] = None) -> None:
        """Brief description."""
        if categories and category_improvements:
            improvements = [category_improvements.get(cat, 0) for cat in categories]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(categories, improvements, color=['red' if x < 0 else 'green' for x in improvements])
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Category')
            plt.ylabel('Accuracy Improvement')
            plt.title(title)
            plt.xticks(rotation=45)
            
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{improvement:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        else:
            plt.figure(figsize=(8, 6))
            categories = ['Original', 'Mitigated']
            accuracies = [original_accuracy, mitigated_accuracy]
            
            bars = plt.bar(categories, accuracies, color=['red', 'green'])
            plt.ylabel('Accuracy')
            plt.title(title)
            plt.ylim([0, 1])
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, 
                                 detection_metrics: Dict[str, Any],
                                 mitigation_metrics: Dict[str, Any],
                                 system_metrics: Dict[str, Any]) -> str:
        """Brief description."""
        report = []
        report.append("=" * 60)
        report.append("HALT EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("DETECTION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Accuracy: {detection_metrics['accuracy']:.3f}")
        report.append(f"Precision: {detection_metrics['precision']:.3f}")
        report.append(f"Recall: {detection_metrics['recall']:.3f}")
        report.append(f"F1 Score: {detection_metrics['f1_score']:.3f}")
        report.append(f"Balanced Accuracy: {detection_metrics['balanced_accuracy']:.3f}")
        if detection_metrics['auc'] is not None:
            report.append(f"AUC: {detection_metrics['auc']:.3f}")
        report.append("")
        
        report.append("MITIGATION PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Original Accuracy: {mitigation_metrics['original_accuracy']:.3f}")
        report.append(f"Mitigated Accuracy: {mitigation_metrics['mitigated_accuracy']:.3f}")
        report.append(f"Accuracy Improvement: {mitigation_metrics['accuracy_improvement']:.3f}")
        report.append(f"Hallucination Reduction: {mitigation_metrics['hallucination_reduction']} cases")
        report.append(f"Hallucination Reduction Rate: {mitigation_metrics['hallucination_reduction_rate']:.3f}")
        report.append(f"Mitigation Effectiveness: {mitigation_metrics['mitigation_effectiveness']:.3f}")
        report.append("")
        
        report.append("SYSTEM PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Overall Score: {system_metrics['overall_score']:.3f}")
        report.append(f"Performance Level: {system_metrics['performance_level'].upper()}")
        report.append(f"Detection Score: {system_metrics['detection_score']:.3f}")
        report.append(f"Mitigation Score: {system_metrics['mitigation_score']:.3f}")
        report.append("")
        
        cm = detection_metrics['confusion_matrix']
        report.append("CONFUSION MATRIX")
        report.append("-" * 30)
        report.append(f"True Negatives: {cm['true_negative']}")
        report.append(f"False Positives: {cm['false_positive']}")
        report.append(f"False Negatives: {cm['false_negative']}")
        report.append(f"True Positives: {cm['true_positive']}")
        report.append("")
        
        report.append("SUMMARY")
        report.append("-" * 30)
        if system_metrics['performance_level'] == 'excellent':
            report.append("✓ System performs excellently with high detection and mitigation effectiveness.")
        elif system_metrics['performance_level'] == 'good':
            report.append("✓ System performs well with good detection and mitigation capabilities.")
        elif system_metrics['performance_level'] == 'fair':
            report.append("⚠ System performs fairly with room for improvement in detection or mitigation.")
        else:
            report.append("✗ System performance is poor and requires significant improvements.")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
