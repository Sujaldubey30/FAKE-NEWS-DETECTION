"""
Utility functions for Fake News Detection
"""
import os
import re
import pickle
import json
import hashlib
from datetime import datetime

def get_model_info(model_path):
    """Get metadata about trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        info = {
            'model_type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A',
            'file_size': f"{round(os.path.getsize(model_path)/1024/1024, 2)} MB",
            'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        }
        return info
    except Exception as e:
        return {'error': str(e)}

def text_hash(text):
    """Create hash of text for caching/deduplication"""
    return hashlib.md5(text.encode()).hexdigest()

def extract_key_phrases(text, top_n=5):
    """Extract key phrases from text (simple version)"""
    from collections import Counter
    
    # Simple noun phrase extraction
    words = text.lower().split()
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

def format_prediction_result(label, confidence, model_name):
    """Format prediction result for API response"""
    return {
        'prediction': 'REAL' if label == 1 else 'FAKE',
        'confidence': float(confidence),
        'model_used': model_name,
        'timestamp': datetime.now().isoformat(),
        'is_reliable': confidence > 0.7  # Simple reliability flag
    }

def validate_input_text(text):
    """Validate user input text"""
    if not text or not isinstance(text, str):
        return False, "Invalid input: Text must be a non-empty string"
    
    if len(text.strip()) < 20:
        return False, "Text too short: Please enter at least 20 characters"
    
    if len(text) > 10000:
        return False, "Text too long: Maximum 10,000 characters allowed"
    
    return True, "Valid"