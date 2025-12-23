import os
import sys
import joblib
import re
from flask import Flask, request, jsonify

# Ensure we can import from src if needed, though we will likely just duplicate clean_text for simplicity regarding paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__)

model = None
vectorizer = None
label_encoder = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts():
    global model, vectorizer, label_encoder
    models_dir = "models"
    
    # Check if we are running from src/api or root
    if not os.path.exists(models_dir):
        # Try going up two levels if running from src/api
        models_dir = os.path.join("..", "..", "models")
        if not os.path.exists(models_dir):
             models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

    print(f"Loading artifacts from {models_dir}...")
    try:
        model = joblib.load(os.path.join(models_dir, "best_model_tuned.joblib"))
        vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        sys.exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "ticket-triage-api"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    subject = data.get("subject", "")
    description = data.get("description", "")
    ticket_id = data.get("ticket_id", "unknown")
    
    if not subject and not description:
        return jsonify({"error": "Missing subject or description"}), 400
    
    # Preprocess
    full_text = clean_text(subject) + " " + clean_text(description)
    
    # Vectorize
    vectorized_text = vectorizer.transform([full_text])
    
    # Predict
    prediction_idx = model.predict(vectorized_text)[0]
    predicted_category = label_encoder.inverse_transform([prediction_idx])[0]
    
    # Confidence (if supported by model)
    confidence = "N/A"
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized_text)
        confidence = float(probs[0][prediction_idx])
    
    response = {
        "ticket_id": ticket_id,
        "predicted_category": predicted_category,
        "confidence": confidence
    }
    
    return jsonify(response), 200

if __name__ == '__main__':
    load_artifacts()
    app.run(host='0.0.0.0', port=5001)
