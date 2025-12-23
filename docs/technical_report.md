# Customer Support Ticket Auto-Triage - Technical Report

## 1. Project Overview

### Objective
The goal of this project was to build a machine learning system capable of automatically classifying customer support tickets into five distinct categories: Bug Report, Feature Request, Technical Issue, Billing Inquiry, and Account Management. This automation aims to reduce manual triage time and improve response efficiency.

### Scope
The system includes:
- A synthetic dataset generator to simulate real-world ticket data.
- A data processing pipeline for cleaning and vectorizing text.
- A machine learning model (Logistic Regression) optimized for high accuracy.
- A REST API developed with Flask for real-time inference.
- Comprehensive testing and documentation.

## 2. Methodology

### Data Generation
Since real-world support data is sensitive and not readily available, we developed a `make_dataset.py` script.
- **Volume**: 5,000 tickets.
- **Attributes**: `ticket_id`, `subject`, `description`, `category`, `priority`, `timestamp`.
- **Logic**: Used templates with random variations to ensure diversity in text length and vocabulary.

### Data Preprocessing
The cleaning pipeline (`preprocess.py`) performs:
1.  **Lowercasing**: To ensure uniformity.
2.  **Noise Removal**: Stripping special characters while retaining alphanumeric content and basic punctuation.
3.  **Whitespace structure**: reducing multiple spaces to single spaces.
4.  **Splitting**: 80/20 Stratified split to maintain class distribution in training and test sets.

### Feature Engineering
We utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** for text vectorization:
- **N-grams**: Unigrams and Bigrams range `(1, 2)` to capture phrases like "login failed".
- **Max Features**: Capped at 5,000 to prevent overfitting and reduce dimensionality.
- **Stop words**: Standard English stop words removed.

## 3. Model Selection & Optimization

We experimented with three distinct algorithms:
1.  **Logistic Regression**: Selected as the baseline.
2.  **Linear SVM**: Known for high performance on high-dimensional sparse text data.
3.  **Random Forest**: A robust ensemble method.

### Results
All models achieved near-perfect performance on the synthetic dataset due to distinct vocabulary separation in the generation templates.
- **Selected Model**: Logistic Regression.
- **Reasoning**: It offered the best balance of speed (training and inference) and model size while maintaining 100% accuracy.

### Optimization
We performed Grid Search Cross-Validation (`optimize_model.py`) to fine-tune hyperparameters.
- **Best Params**: `{'C': 0.01, 'solver': 'lbfgs'}`.
- **Result**: The tuned model maintained perfect scores with improved regularization.

## 4. Evaluation Results

### Metrics
- **Accuracy**: 100%
- **F1-Score (Weighted)**: 1.00
- **Precision & Recall**: 1.00 across all classes.

### Latency
Performance is critical for an API. We measured inference latency over 1,000 requests:
- **Average Latency**: ~0.07 ms (sub-millisecond).
- **95th Percentile**: ~0.10 ms.
This confirms the system is highly performant and suitable for high-throughput environments.

## 5. System Architecture

1.  **Input**: JSON payload containing `subject` and `description`.
2.  **API Layer**: Flask application (`app.py`) receives the request.
3.  **Preprocessing**: Applies `clean_text` function.
4.  **Vectorization**: Loads pre-fitted `TfidfVectorizer` to transform text.
5.  **Inference**: Loads trained `LogisticRegression` model to predict class index.
6.  **Decoding**: Maps index to label using `LabelEncoder`.
7.  **Output**: JSON response with `predicted_category` and `confidence`.

## 6. Future Improvements

-   **Real Data**: Retrain on actual human-written tickets to handle more noise and ambiguity.
-   **Deep Learning**: Implement BERT or DistilBERT for better semantic understanding if the dataset becomes more complex.
-   **Feedback Loop**: Implement a mechanism to correct misclassifications and retrain the model periodically.
