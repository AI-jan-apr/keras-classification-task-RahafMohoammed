# Breast Cancer Classification API

A deep learning model built with Keras to classify tumors as Benign or Malignant, deployed using FastAPI.

---

## Project Structure

```
├── deploy.py
├── model_weights.keras
├── scaler_weights.pkl
└── keras-classification-task.ipynb
```

---

## How to Run

```bash
uvicorn deploy:app --reload
```

Then open: http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/predict` | Classify tumor as Benign or Malignant |

---

## Example Request

```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

## Example Response

```json
{
  "status": "success",
  "diagnosis": "Malignant",
  "label": 0,
  "probability_malignant": 1
}
```

---

## Dataset

Breast Cancer Wisconsin Diagnostic dataset — 569 samples, 30 features.

| Class | Count |
|-------|-------|
| Benign | 357 |
| Malignant | 212 |

## Model

- Framework: TensorFlow / Keras
- Type: Sequential Neural Network with Dropout
- Loss: Binary Crossentropy
- Optimizer: Adam
- Scaler: MinMaxScaler
- Callback: EarlyStopping
- Accuracy: 99%
