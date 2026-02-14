"""
Data Preprocessing Module for Fake News Detection
Handles loading, cleaning, and preparing the dataset
"""
import os
print("Current working directory:", os.getcwd())
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(script_dir)
# Build correct paths to the CSV files
true_path = os.path.join(project_root, 'data', 'True.csv')
fake_path = os.path.join(project_root, 'data', 'Fake.csv')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def load_data(self, true_path, fake_path):
        """
        Load and combine real and fake news datasets
        """
        # Load datasets
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        
        # Add labels
        true_df['label'] = 1  # 1 for REAL news
        fake_df['label'] = 0  # 0 for FAKE news
        
        # Combine datasets
        df = pd.concat([true_df, fake_df], ignore_index=True)
        
        print(f"Dataset loaded: {len(df)} articles")
        print(f"Real news: {len(true_df)}, Fake news: {len(fake_df)}")
        
        return df
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """
        Advanced preprocessing: tokenization, stopword removal, lemmatization
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_dataset(self, true_path, fake_path, sample_size=None):
        """
        Complete data preparation pipeline
        """
        # Load data
        df = self.load_data(true_path, fake_path)
        
        # Combine title and text for better context
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Apply preprocessing
        print("Preprocessing text...")
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Sample if needed (for balanced training)
        if sample_size:
            real_sample = df[df['label'] == 1].sample(n=min(sample_size, len(df[df['label'] == 1])))
            fake_sample = df[df['label'] == 0].sample(n=min(sample_size, len(df[df['label'] == 0])))
            df = pd.concat([real_sample, fake_sample], ignore_index=True)
        
        print(f"Prepared dataset: {len(df)} articles")
        print(f"Classes: Real={sum(df['label']==1)}, Fake={sum(df['label']==0)}")
        
        return df[['processed_text', 'label']]

# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.prepare_dataset(
    true_path=true_path,
    fake_path=fake_path,
    sample_size=5000
)
    processed_path = os.path.join(project_root, 'data', 'processed_news.csv') 
    df.to_csv(processed_path, index=False)
    print("Preprocessed data saved!")