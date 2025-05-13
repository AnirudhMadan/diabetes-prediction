import os
from tensorflow.keras.models import load_model
import joblib

MODEL_PATH = 'saved_models/best_dl_model.keras'
SCALER_PATH = 'saved_models/scaler.pkl'

def load_dl_model_and_scaler(verbose=False):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}.")
    
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if verbose:
        print(f"Loaded model from {MODEL_PATH}")
        print(f"Loaded scaler from {SCALER_PATH}")
    return model, scaler
