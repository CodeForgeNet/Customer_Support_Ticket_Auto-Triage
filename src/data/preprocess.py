import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Applies basic text cleaning:
    - Lowercase
    - Remove special characters (keep alphanumeric and spaces)
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars (keep only letters, numbers, and basic punctuation)
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    input_path = os.path.join("data", "raw", "tickets.csv")
    output_train_path = os.path.join("data", "processed", "train.csv")
    output_test_path = os.path.join("data", "processed", "test.csv")
    
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}. Please run make_dataset.py first.")
        
    df = pd.read_csv(input_path)
    
    # Basic cleaning
    print("Cleaning text fields...")
    df['cleaned_subject'] = df['subject'].apply(clean_text)
    df['cleaned_description'] = df['description'].apply(clean_text)
    
    # Combine subject and description for model input
    df['text'] = df['cleaned_subject'] + " " + df['cleaned_description']
    
    # Split data (Stratified to maintain class balance)
    print("Splitting data into train/test sets...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['category'], 
        random_state=42
    )
    
    # Save processed data
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    print(f"Saved training set: {len(train_df)} rows to {output_train_path}")
    print(f"Saved test set: {len(test_df)} rows to {output_test_path}")
    
    # Verify split distribution
    print("\nTraining Class Distribution:")
    print(train_df['category'].value_counts(normalize=True))
    
    print("\nTest Class Distribution:")
    print(test_df['category'].value_counts(normalize=True))

if __name__ == "__main__":
    preprocess_data()
