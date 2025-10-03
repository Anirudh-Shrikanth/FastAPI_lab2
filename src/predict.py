import joblib
import os

# Always resolve relative to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "iris_model.pkl")

# Global model object
model = None

def load_model():
    """
    Load the Iris model from disk and store in global `model`.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    #model = joblib.load("../model/iris_model.pkl")
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    return y_pred