from cProfile import label

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Breast Cancer Classification API")

# ── Load scaler ──────────────────────────────────────────────
with open('scaler_weights.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Load model ───────────────────────────────────────────────
model = load_model('model_weights.keras')


# ── Input schema (30 features) ───────────────────────────────
class TumorFeatures(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Welcome to the Breast Cancer Classification API!"}


@app.post("/predict")
def predict(features: TumorFeatures):
    try:
        input_data = np.array([[
            features.mean_radius, features.mean_texture,
            features.mean_perimeter, features.mean_area,
            features.mean_smoothness, features.mean_compactness,
            features.mean_concavity, features.mean_concave_points,
            features.mean_symmetry, features.mean_fractal_dimension,
            features.radius_se, features.texture_se,
            features.perimeter_se, features.area_se,
            features.smoothness_se, features.compactness_se,
            features.concavity_se, features.concave_points_se,
            features.symmetry_se, features.fractal_dimension_se,
            features.worst_radius, features.worst_texture,
            features.worst_perimeter, features.worst_area,
            features.worst_smoothness, features.worst_compactness,
            features.worst_concavity, features.worst_concave_points,
            features.worst_symmetry, features.worst_fractal_dimension
        ]])

        # Scale
        scaled = scaler.transform(input_data)

        # Predict
        probability = float(model.predict(scaled)[0][0])
        label = 1 if probability >= 0.5 else 0

# بما إن 0 = Malignant في sklearn
        diagnosis = "Benign" if label == 1 else "Malignant"
        probability_malignant = round(1 - probability, 4)  # ← هذا السطر

        return {
        "status": "success",
        "diagnosis": diagnosis,
        "label": label,
        "probability_malignant": probability_malignant
}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
