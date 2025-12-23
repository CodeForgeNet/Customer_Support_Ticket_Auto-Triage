import numpy as np
import os
import joblib
import json
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

def optimize_model():
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    
    # Load data
    print("Loading data...")
    X_train = sparse.load_npz(os.path.join(processed_dir, "X_train.npz"))
    X_test = sparse.load_npz(os.path.join(processed_dir, "X_test.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # Load label encoder
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    target_names = [str(cls) for cls in le.classes_]
    
    # Use Logistic Regression as it was likely the best or tied for best (fastest)
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Define Parameter Grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
    
    print(f"Starting Grid Search with {base_model.__class__.__name__}...")
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on Test Set
    print("\nEvaluating Best Model on Test Set...")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save tuned model
    tuned_model_path = os.path.join(models_dir, "best_model_tuned.joblib")
    joblib.dump(best_model, tuned_model_path)
    print(f"Saved tuned model to {tuned_model_path}")

if __name__ == "__main__":
    optimize_model()
