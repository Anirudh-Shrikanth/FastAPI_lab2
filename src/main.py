from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", status_code=status.HTTP_200_OK)
async def get_info():
    """Basic information about the API."""
    return {
        "app": "Iris Prediction API",
        "version": "1.0.0",
        "author": "Anirudh",
        "endpoints": ["/", "/predict", "/info", "/features", "/reload-model"]
    }

@app.get("/features", status_code=status.HTTP_200_OK)
async def get_features():
    """Return expected feature names and descriptions."""
    return {
        "features": {
            "sepal_length": "Length of the sepal in cm",
            "sepal_width": "Width of the sepal in cm",
            "petal_length": "Length of the petal in cm",
            "petal_width": "Width of the petal in cm"
        }
    }

@app.post("/reload-model", status_code=status.HTTP_200_OK)
async def reload_model():
    """
    Reload the model from disk (in case it was retrained).
    """
    try:
        # Lazy import to avoid circular import
        import predict
        predict.model = None  # reset
        predict.load_model()  # reload
        return {"detail": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")