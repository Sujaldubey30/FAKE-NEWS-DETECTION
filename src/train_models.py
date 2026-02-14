"""
Model Training Module
Trains multiple ML models for fake news detection
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.models = {}
        self.results = {}
        
    def prepare_features(self):
        """
        Split data and create TF-IDF features
        """
        # Split data
        X = self.df['processed_text']
        y = self.df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Training set: {X_train_tfidf.shape}")
        print(f"Test set: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model
        """
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        return model
    
    def train_svm(self, X_train, y_train):
        """
        Train Linear SVM model
        """
        print("Training Linear SVM...")
        model = LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42,
            dual=False  # For large datasets
        )
        model.fit(X_train, y_train)
        self.models['SVM'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model
        """
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        return model
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Multinomial Naive Bayes
        """
        print("Training Naive Bayes...")
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train, y_train)
        self.models['Naive Bayes'] = model
        return model
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print('='*50)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.results[name] = accuracy
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['FAKE', 'REAL'], 
                        yticklabels=['FAKE', 'REAL'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            # Get absolute path to models directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_dir = os.path.join(project_root, 'models')
            os.makedirs(models_dir, exist_ok=True)   # create folder if it doesn't exist

            plt.savefig(os.path.join(models_dir, f'{name.replace(" ", "_")}_cm.png'))
            plt.close()
    
    def save_models(self):
     script_dir = os.path.dirname(os.path.abspath(__file__))
     project_root = os.path.dirname(script_dir)
     models_dir = os.path.join(project_root, 'models')
     os.makedirs(models_dir, exist_ok=True)

    # Save vectorizer
     joblib.dump(self.vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
     print("\nVectorizer saved!")

    # Save models
     for name, model in self.models.items():
        filename = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, filename)
        print(f"{name} saved to {filename}")
    
    def plot_accuracy_comparison(self):
     script_dir = os.path.dirname(os.path.abspath(__file__))
     project_root = os.path.dirname(script_dir)
     models_dir = os.path.join(project_root, 'models')
     script_dir = os.path.dirname(os.path.abspath(__file__))
     project_root = os.path.dirname(script_dir)
     models_dir = os.path.join(project_root, 'models')
     os.makedirs(models_dir, exist_ok=True)
     plt.savefig(os.path.join(models_dir, 'accuracy_comparison.png')) 
# Main training pipeline
def main():
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    # Get the script's directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed_news.csv')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed_news.csv')
    trainer = ModelTrainer(data_path)
    
    # Prepare features
    X_train, X_test, y_train, y_test = trainer.prepare_features()
    
    # Train all models
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_svm(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_naive_bayes(X_train, y_train)
    
    # Evaluate models
    trainer.evaluate_models(X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    # Plot comparison
    trainer.plot_accuracy_comparison()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*60)
    for model, accuracy in trainer.results.items():
        print(f"{model:20s}: {accuracy:.4f}")
    
    # Best model
    best_model = max(trainer.results, key=trainer.results.get)
    print(f"\n🏆 Best Model: {best_model} with accuracy {trainer.results[best_model]:.4f}")

if __name__ == "__main__":
    main()