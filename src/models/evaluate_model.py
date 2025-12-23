import numpy as np
import os
import joblib
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def evaluate_model():
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X_test = sparse.load_npz(os.path.join(processed_dir, "X_test.npz"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    # Load model and label encoder
    print("Loading model...")
    model = joblib.load(os.path.join(models_dir, "best_model_tuned.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    target_names = [str(cls) for cls in le.classes_]
    
    # Measure Latency
    print("Measuring latency...")
    latencies = []
    # Warmup
    for _ in range(10):
        model.predict(X_test[0])
        
    # Test
    for i in range(1000):
        idx = i % X_test.shape[0]
        start_time = time.time()
        model.predict(X_test[idx])
        latencies.append((time.time() - start_time) * 1000) # ms
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"95th Percentile Latency: {p95_latency:.4f} ms")
    
    # Predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "average_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency
    }
    
    # Save Metrics
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    
    print(f"Evaluation complete. Reports saved to {reports_dir}")

if __name__ == "__main__":
    evaluate_model()
