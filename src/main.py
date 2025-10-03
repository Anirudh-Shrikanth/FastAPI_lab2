from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from predict import predict_data, load_model

app = FastAPI()

# Pydantic model for input
class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine: WineData):
    try:
        features = [[
            wine.alcohol,
            wine.malic_acid,
            wine.ash,
            wine.alcalinity_of_ash,
            wine.magnesium,
            wine.total_phenols,
            wine.flavanoids,
            wine.nonflavanoid_phenols,
            wine.proanthocyanins,
            wine.color_intensity,
            wine.hue,
            wine.od280_od315_of_diluted_wines,
            wine.proline
        ]]
        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", status_code=status.HTTP_200_OK)
async def get_info():
    return {
        "app": "Wine Classification API",
        "version": "1.0.0",
        "author": "Anirudh",
        "endpoints": ["/", "/predict", "/info", "/features", "/reload-model"]
    }

@app.get("/features", status_code=status.HTTP_200_OK)
async def get_features():
    return {
        "features": [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
        ]
    }

@app.post("/reload-model", status_code=status.HTTP_200_OK)
async def reload_model():
    try:
        load_model()
        return {"detail": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
