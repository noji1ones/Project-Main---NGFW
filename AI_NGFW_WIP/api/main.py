from fastapi import FastAPI
from pydantic import BaseModel  
import tensorflow as tf
import numpy as np
import joblib

#Define the Envelope
class TrafficData(BaseModel):
    features: list

app = FastAPI()

#Load model + scaler
model = tf.keras.models.load_model('/app/model/ngfw_model.h5')
scaler = joblib.load('/app/model/scaler.pkl')

@app.post("/predict")
def predict(data: TrafficData):
    #Extract the list from the envelop
    packet_data = data.features
    
    #This should fix missing zeros pls work
    expected_features = scaler.n_features_in_
    
    if len(packet_data) < expected_features:
        packet_data = packet_data + [0] * (expected_features - len(packet_data))
    elif len(packet_data) > expected_features:
        packet_data = packet_data[:expected_features]
  

    # Predict data
    scaled_data = scaler.transform([packet_data])
    prediction = model.predict(scaled_data)
    
    return {"malicious_score": float(prediction[0][0])}