import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    brier_score_loss, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    
    # Performance metrics
    auc: float
    auprc: float
    f1_score: float
    brier_score: float
    log_loss: float
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    specificity: float
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    
    # Calibration metrics
    calibration_slope: Optional[float] = None
    calibration_intercept: Optional[float] = None
    
    # Additional info
    threshold: float = 0.5
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'auc': self.auc,
            'auprc': self.auprc,
            'f1_score': self.f1_score,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'specificity': self.specificity,
            'calibration_slope': self.calibration_slope,
            'calibration_intercept': self.calibration_intercept,
            'threshold': self.threshold,
            'n_samples': self.n_samples
        }
    
    def __str__(self) -> str:
        return (
            f"ModelMetrics(\n"
            f"  AUC: {self.auc:.4f}\n"
            f"  AUPRC: {self.auprc:.4f}\n"
            f"  F1: {self.f1_score:.4f}\n"
            f"  Brier Score: {self.brier_score:.4f}\n"
            f"  Log Loss: {self.log_loss:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  Specificity: {self.specificity:.4f}\n"
            f"  Threshold: {self.threshold:.4f}\n"
            f"  N Samples: {self.n_samples}\n"
            f")"
        )


class ModelEvaluator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_proba: np.ndarray, 
                       y_pred: Optional[np.ndarray] = None,
                       threshold: float = 0.5) -> ModelMetrics:
        # Validate inputs
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        
        if len(y_true) != len(y_proba):
            raise ValueError("y_true and y_proba must have the same length")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")
        
        if not np.all((y_proba >= 0) & (y_proba <= 1)):
            raise ValueError("y_proba must be between 0 and 1")
        
        if y_pred is None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = np.asarray(y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.5 if len(np.unique(y_true)) == 1 else np.nan
        
        try:
            auprc = average_precision_score(y_true, y_proba)
        except ValueError:
            auprc = np.mean(y_true) if len(np.unique(y_true)) == 1 else np.nan
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        brier = brier_score_loss(y_true, y_proba)
        
        try:
            logloss = log_loss(y_true, y_proba, labels=[0, 1])
        except ValueError:
            logloss = np.nan
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            if len(np.unique(y_true)) == 1:
                if y_true[0] == 1:  # All true positives
                    tp = len(y_true)
                    tn = fp = fn = 0
                else:  # All true negatives
                    tn = len(y_true)
                    tp = fp = fn = 0
            else:
                # Fallback
                tp = fp = fn = tn = 0
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        cal_slope, cal_intercept = self._compute_calibration_metrics(y_true, y_proba)
        
        return ModelMetrics(
            auc=auc,
            auprc=auprc,
            f1_score=f1,
            brier_score=brier,
            log_loss=logloss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            confusion_matrix=cm,
            calibration_slope=cal_slope,
            calibration_intercept=cal_intercept,
            threshold=threshold,
            n_samples=len(y_true)
        )
    
    def _compute_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Convert probabilities to logits
            epsilon = 1e-15
            y_proba_clipped = np.clip(y_proba, epsilon, 1 - epsilon)
            logits = np.log(y_proba_clipped / (1 - y_proba_clipped))
            
            # Fit logistic regression: y_true ~ logits
            cal_model = LogisticRegression(fit_intercept=True)
            cal_model.fit(logits.reshape(-1, 1), y_true)
            
            slope = cal_model.coef_[0][0]
            intercept = cal_model.intercept_[0]
            
            return slope, intercept
            
        except Exception:
            return None, None
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      title: str = "ROC Curve", figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve", 
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
        baseline = np.mean(y_true)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUPRC = {auprc:.3f})')
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Baseline (Prevalence = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              n_bins: int = 10, title: str = "Calibration Curve",
                              figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', linewidth=2,
               label='Model Calibration')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = False, title: str = "Confusion Matrix",
                             figsize: Tuple[int, int] = (6, 5)) -> plt.Figure:
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray,
                               thresholds: Optional[np.ndarray] = None,
                               title: str = "Threshold Analysis",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'specificity': []
        }
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                if len(np.unique(y_pred)) == 1:
                    if y_pred[0] == 0:  # All predicted as negative
                        tp, fp, fn, tn = 0, 0, np.sum(y_true), np.sum(1 - y_true)
                    else:  # All predicted as positive
                        tp, fp, fn, tn = np.sum(y_true), np.sum(1 - y_true), 0, 0
                else:
                    continue
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)
            metrics['accuracy'].append(accuracy)
            metrics['specificity'].append(specificity)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        axes[0, 0].plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
        axes[0, 0].plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision and Recall')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot F1 score
        axes[0, 1].plot(thresholds, metrics['f1_score'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot accuracy and specificity
        axes[1, 0].plot(thresholds, metrics['accuracy'], label='Accuracy', linewidth=2)
        axes[1, 0].plot(thresholds, metrics['specificity'], label='Specificity', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Accuracy and Specificity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot all metrics together
        for metric_name, values in metrics.items():
            axes[1, 1].plot(thresholds, values, label=metric_name.replace('_', ' ').title(), linewidth=2)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('All Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_models(self, model_results: Dict[str, Dict[str, np.ndarray]],
                      metrics_to_plot: List[str] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        if metrics_to_plot is None:
            metrics_to_plot = ['auc', 'auprc', 'f1_score', 'accuracy', 'brier_score']
        
        # Compute metrics for all models
        model_metrics = {}
        for model_name, results in model_results.items():
            metrics = self.compute_metrics(results['y_true'], results['y_proba'])
            model_metrics[model_name] = metrics.to_dict()
        
        # Create comparison plots
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
                
            model_names = list(model_metrics.keys())
            metric_values = [model_metrics[name][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, metric_values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            if len(max(model_names, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_proba: np.ndarray,
                                 model_name: str = "Model", 
                                 save_plots: bool = False,
                                 output_dir: str = ".") -> Dict[str, Any]:
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_proba)
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Generate plots
        plots = {}
        plots['roc_curve'] = self.plot_roc_curve(y_true, y_proba, 
                                                title=f"{model_name} - ROC Curve")
        plots['pr_curve'] = self.plot_precision_recall_curve(y_true, y_proba,
                                                           title=f"{model_name} - PR Curve")
        plots['calibration'] = self.plot_calibration_curve(y_true, y_proba,
                                                         title=f"{model_name} - Calibration")
        plots['confusion_matrix'] = self.plot_confusion_matrix(y_true, y_pred,
                                                             title=f"{model_name} - Confusion Matrix")
        plots['threshold_analysis'] = self.plot_threshold_analysis(y_true, y_proba,
                                                                 title=f"{model_name} - Threshold Analysis")
        
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            for plot_name, fig in plots.items():
                fig.savefig(os.path.join(output_dir, f"{model_name}_{plot_name}.png"), 
                           dpi=300, bbox_inches='tight')
        
        return {
            'metrics': metrics,
            'plots': plots,
            'summary': {
                'model_name': model_name,
                'n_samples': len(y_true),
                'positive_rate': np.mean(y_true),
                'key_metrics': {
                    'AUC': metrics.auc,
                    'AUPRC': metrics.auprc,
                    'F1': metrics.f1_score,
                    'Accuracy': metrics.accuracy,
                    'Brier Score': metrics.brier_score
                }
            }
        }