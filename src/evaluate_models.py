"""
Advanced Model Evaluation Module
Generates detailed metrics and visualizations
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Get absolute paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, 'models')
data_dir = os.path.join(project_root, 'data')

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)

class ModelEvaluator:
    def __init__(self, model_path, vectorizer_path, test_data_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.test_data = pd.read_csv(test_data_path)
        
    def prepare_test_data(self):
        """Prepare test features"""
        X_test = self.vectorizer.transform(self.test_data['processed_text'])
        y_test = self.test_data['label']
        return X_test, y_test
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate all classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr.tolist()
            metrics['tpr'] = tpr.tolist()
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_proba, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(models_dir, f'{model_name}_roc_curve.png'))
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(models_dir, f'{model_name}_pr_curve.png'))
        plt.close()
    
    def generate_report(self, model_name):
        """Generate comprehensive evaluation report"""
        X_test, y_true = self.prepare_test_data()
        y_pred = self.model.predict(X_test)
        
        # Get probabilities if model supports it
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities using sigmoid
            y_proba = 1 / (1 + np.exp(-self.model.decision_function(X_test)))
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Create visualizations
        if y_proba is not None:
            self.plot_roc_curve(y_true, y_proba, model_name)
            self.plot_precision_recall_curve(y_true, y_proba, model_name)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['FAKE', 'REAL'], 
                    yticklabels=['FAKE', 'REAL'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(models_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # Save metrics to JSON
        with open(os.path.join(models_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

# Compare multiple models
def compare_all_models():
    """Compare performance of all trained models"""
    models = ['logistic_regression', 'svm', 'random_forest', 'naive_bayes']
    results = {}
    
    for model_name in models:
        try:
            evaluator = ModelEvaluator(
                model_path=os.path.join(models_dir, f'{model_name}.pkl'),
                vectorizer_path=os.path.join(models_dir, 'tfidf_vectorizer.pkl'),
                test_data_path=os.path.join(data_dir, 'processed_news.csv')
            )
            metrics = evaluator.generate_report(model_name)
            results[model_name] = metrics
            print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv(os.path.join(models_dir, 'model_comparison.csv'))
    print("\nComparison saved to model_comparison.csv")
    
    return comparison_df

if __name__ == "__main__":
    comparison = compare_all_models()
    print("\nModel Performance Summary:")
    print(comparison[['accuracy', 'precision', 'recall', 'f1_score']])