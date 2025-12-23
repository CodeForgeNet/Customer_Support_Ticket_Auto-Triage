import numpy as np
import os
import joblib
import json
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

def train_models():
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    

    print("Loading data...")
    X_train = sparse.load_npz(os.path.join(processed_dir, "X_train.npz"))
    X_test = sparse.load_npz(os.path.join(processed_dir, "X_test.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # Load label encoder to map back to class names
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    target_names = [str(cls) for cls in le.classes_]
    

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(random_state=42, dual='auto'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_f1 = 0.0
    results = {}
    
    print("\nTraining and Evaluating Models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            "accuracy": accuracy,
            "f1_score": f1
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    print("-" * 60)
    print(f"Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    

    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    with open(os.path.join(models_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nClassification Report (Best Model):")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=target_names))

if __name__ == "__main__":
    train_models()
