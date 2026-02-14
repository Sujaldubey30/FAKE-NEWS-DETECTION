"""
Flask Web Application for Fake News Detection
"""

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import format_prediction_result, validate_input_text

app = Flask(__name__)

# Load models and vectorizer
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
print(f"Loading models from {MODELS_DIR}")

try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    
    # Load all available models
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'SVM': 'svm.pkl',
        'Random Forest': 'random_forest.pkl',
        'Naive Bayes': 'naive_bayes.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
            print(f"✅ Loaded {name}")
        else:
            print(f"⚠️ {filename} not found")
    
    if not models:
        print("❌ No models found! Please train models first.")
        
except Exception as e:
    print(f"❌ Error loading models: {e}")
    models = {}
    vectorizer = None

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', models=list(models.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input text
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
            model_name = data.get('model', 'Logistic Regression')
        else:
            text = request.form.get('news_text', '')
            model_name = request.form.get('model', 'Logistic Regression')
        
        # Validate input
        is_valid, message = validate_input_text(text)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if models are loaded
        if not models or vectorizer is None:
            return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
        
        # Transform text
        text_tfidf = vectorizer.transform([text])
        
        # Get prediction from selected model
        if model_name not in models:
            model_name = list(models.keys())[0]  # Default to first model
        
        model = models[model_name]
        
        # Get prediction and confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_tfidf)[0]
            confidence = max(proba)
            prediction = model.predict(text_tfidf)[0]
        else:
            prediction = model.predict(text_tfidf)[0]
            # For models without predict_proba, use decision function
            if hasattr(model, 'decision_function'):
                confidence = 1 / (1 + np.exp(-model.decision_function(text_tfidf)[0]))
                confidence = float(np.abs(confidence))
            else:
                confidence = 0.85  # Default confidence
        
        # Format result
        result = format_prediction_result(prediction, confidence, model_name)
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('index.html', 
                                 prediction=result['prediction'],
                                 confidence=f"{result['confidence']:.2%}",
                                 model_used=result['model_used'],
                                 models=list(models.keys()))
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        else:
            return render_template('index.html', error=error_msg, models=list(models.keys()))

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Predict using all models for comparison"""
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('news_text', '')
        
        # Validate input
        is_valid, message = validate_input_text(text)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        if not models or vectorizer is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Transform text
        text_tfidf = vectorizer.transform([text])
        
        # Get predictions from all models
        results = {}
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_tfidf)[0]
                prediction = model.predict(text_tfidf)[0]
                confidence = max(proba)
            else:
                prediction = model.predict(text_tfidf)[0]
                confidence = 0.85
            
            results[name] = {
                'prediction': 'REAL' if prediction == 1 else 'FAKE',
                'confidence': float(confidence)
            }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    info = {}
    for name, model in models.items():
        info[name] = {
            'type': type(model).__name__,
            'parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
        }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)