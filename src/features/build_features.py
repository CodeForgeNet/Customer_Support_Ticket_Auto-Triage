import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

def build_features():
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    models_dir = "models"
    processed_dir = os.path.join("data", "processed")
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed data not found. Run preprocess.py first.")
    
    print("Loading processed data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Fill NAs just in case (though we cleaned it)
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    # Initialize Vectorizer
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit on train, transform on train and test
    X_train = tfidf.fit_transform(train_df['text'])
    X_test = tfidf.transform(test_df['text'])
    
    # Encode Target Labels
    print("Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['category'])
    y_test = le.transform(test_df['category'])
    
    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    
    # Save vectorizer and label encoder
    print("Saving models...")
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
    
    # Save features
    print("Saving features...")
    # Use scipy.sparse.save_npz for sparse matrices to save space
    sparse.save_npz(os.path.join(processed_dir, "X_train.npz"), X_train)
    sparse.save_npz(os.path.join(processed_dir, "X_test.npz"), X_test)
    
    # Save labels as numpy arrays
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    print(f"Features saved to {processed_dir}")
    print(f"Models saved to {models_dir}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Classes: {le.classes_}")

if __name__ == "__main__":
    build_features()
