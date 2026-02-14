# Fake News Detection

Lightweight Python project for detecting fake news using classical ML models and a small Flask web UI.

## Project overview

This repository contains scripts to preprocess news data, train and evaluate several models, and a simple Flask app to demo predictions.

## Repo structure

- `app/` — Flask app and static files
- `data/` — CSV datasets (Fake.csv, True.csv, processed_news.csv)
- `models/` — saved model artifacts and metrics
- `src/` — preprocessing, training, evaluation, and utility scripts
- `requirements.txt` — Python dependencies

## Setup

1. Create a Python 3.8+ virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Preprocess data:

```bash
python src/data_preprocessing.py
```

- Train models:

```bash
python src/train_models.py
```

- Evaluate models:

```bash
python src/evaluate_models.py
```

- Run the Flask app (development):

```bash
python app/app.py
# then open http://127.0.0.1:5000
```

## Data

Place raw datasets in `data/`. This project includes small CSVs; remove or replace them if you prefer not to track data in Git.

## Files of interest

- `src/data_preprocessing.py` — cleaning and feature extraction
- `src/train_models.py` — training pipeline for classifiers
- `src/evaluate_models.py` — compute metrics and comparisons
- `app/app.py` — Flask frontend for demo

## Notes

- This repository is intended for educational/demo purposes. For production, add proper input validation, model serialization, and secure secret management.

## License

Add a LICENSE file or include licensing terms as needed.
