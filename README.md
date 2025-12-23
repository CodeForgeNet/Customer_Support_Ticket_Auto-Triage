# Customer Support Ticket Auto-Triage

## Overview
A machine learning project that automates the classification and routing of customer support tickets using Natural Language Processing (NLP). The system predicts categories like **Bug Report**, **Feature Request**, **Technical Issue**, **Billing Inquiry**, and **Account Management** based on ticket text.

## Features
- **Synthetic Data Generation**: Creates realistic support tickets.
- **Text Processing Pipeline**: Cleans and vectorizes text using TF-IDF.
- **High-Performance Model**: Optimized Logistic Regression model with <1ms inference latency.
- **REST API**: Flask-based API for real-time predictions.
- **Comprehensive Testing**: Unit and integration tests included.

## Project Structure
```
├── data/               # Raw and processed datasets
├── docs/               # Documentation and reports
├── models/             # Trained models and artifacts (joblib)
├── notebooks/          # Exploratory Data Analysis (EDA)
├── reports/            # Generated metrics and figures
├── src/
│   ├── api/            # Flask application
│   ├── data/           # Data generation and preprocessing scripts
│   ├── features/       # Feature engineering scripts
│   └── models/         # Training and evaluation scripts
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## How to Run

### 1. Setup Environment
Clone the repository and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Pipeline
Generate data, preprocess, and build features:
```bash
# Generate synthetic dataset
python src/data/make_dataset.py

# Clean and split data
python src/data/preprocess.py

# Vectorize text and encode labels
python src/features/build_features.py
```

### 3. Model Training
Train and optimize the model:
```bash
# Train baseline models
python src/models/train_models.py

# Tune hyperparameters
python src/models/optimize_model.py

# Evaluate performance
python src/models/evaluate_model.py
```

### 4. Run API
Start the Flask API server:
```bash
python src/api/app.py
```
The API will start on `http://localhost:5001`.

### 5. Test API
You can test the API using the provided script or `curl`:
```bash
bash src/api/test_api.sh
```

## Testing
Run the full test suite using pytest:
```bash
pytest tests/
```
