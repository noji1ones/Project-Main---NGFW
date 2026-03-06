from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import joblib

# Define the Envelope
class TrafficData(BaseModel):
    features: list

app = FastAPI(title="NGFW Inference Engine")

# ill load artifacts via the Docker volume mount
scaler = joblib.load('/app/model/scaler.pkl')
top_features_idx = np.load('/app/model/top_features_idx.npy')
ort_session = ort.InferenceSession('/app/model/ngfw_ae.onnx')

@app.post("/predict")
def predict(data: TrafficData):
    # Convert incoming list into a 2D numpy array
    raw_features = np.array(data.features).reshape(1, -1)
    
    # 1. Ensure exactly 77 features for the scaler
    if raw_features.shape[1] > 77:
        raw_features = raw_features[:, :77]
    elif raw_features.shape[1] < 77:
        padded = np.zeros((1, 77))
        padded[0, :raw_features.shape[1]] = raw_features
        raw_features = padded

    # 2. Scale the full dataset FIRST 
    scaled_full_features = scaler.transform(raw_features)
    
    # 3. Extract ONLY the 40 critical features using the Random Forest index
    final_features = scaled_full_features[:, top_features_idx].astype(np.float32)
    
    # 4. High-Speed C++ ONNX Inference
    inputs = {ort_session.get_inputs()[0].name: final_features}
    reconstruction = ort_session.run(None, inputs)[0]
    
    # 5. Calculate Mean Squared Error (Anomaly Score)
    mse = np.mean(np.square(final_features - reconstruction))
    is_threat = bool(mse > 0.02) 
    
    return {"threat_detected": is_threat, "malicious_score": float(mse)}