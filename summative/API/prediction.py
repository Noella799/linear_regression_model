"""
PCIAT Early-Warning API
Predicts a child's Parent-Child Internet Addiction Test (PCIAT) total score (0-100)
from 10 everyday measures, so counsellors and parents can offer support early.

Run locally:   uvicorn prediction:app --reload
Swagger UI:    http://127.0.0.1:8000/docs
"""
import copy
import io
import os
import threading
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"

FEATURES = ["age", "sex", "internet_hours_per_day", "height_in", "weight_lb",
            "sleep_disturbance_t", "cgas_score", "systolic_bp", "heart_rate", "enroll_season"]
TARGET = "pciat_total"
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
RETRAIN_EPOCHS = 10  # passes over the new rows when continuing SGD training

# The model is loaded once from the repository at startup and kept in memory.
# /retrain swaps it for the updated model; the lock stops a prediction from
# reading the model while it is being replaced.
model = joblib.load(MODEL_PATH)
model_lock = threading.Lock()


# ---------------------------------------------------------------- request / response schemas
class ChildInput(BaseModel):
    """One child's measures. Every field is required, typed and range-checked."""
    model_config = ConfigDict(
        strict=True,        # "12" (a string) is rejected for a number field instead of being converted
        extra="forbid",     # unknown fields are rejected
        json_schema_extra={"example": {
            "age": 12, "sex": 0, "internet_hours_per_day": 3, "height_in": 60.0,
            "weight_lb": 100.0, "sleep_disturbance_t": 65, "cgas_score": 60,
            "systolic_bp": 115.0, "heart_rate": 80.0, "enroll_season": "Fall"}},
    )

    age: int = Field(..., ge=5, le=22, description="Age in years (5-22)")
    sex: int = Field(..., ge=0, le=1, description="0 = male, 1 = female")
    internet_hours_per_day: int = Field(..., ge=0, le=3,
                                        description="Daily computer/internet use: 0 = <1h, 1 = ~1h, 2 = ~2h, 3 = 3h or more")
    height_in: float = Field(..., ge=30, le=85, description="Height in inches (30-85)")
    weight_lb: float = Field(..., ge=20, le=350, description="Weight in pounds (20-350)")
    sleep_disturbance_t: int = Field(..., ge=38, le=100, description="Sleep Disturbance Scale T-score (38-100)")
    cgas_score: int = Field(..., ge=1, le=100, description="Children's Global Assessment Scale, general functioning (1-100)")
    systolic_bp: float = Field(..., ge=60, le=200, description="Systolic blood pressure in mmHg (60-200)")
    heart_rate: float = Field(..., ge=40, le=180, description="Resting heart rate in bpm (40-180)")
    enroll_season: Literal["Spring", "Summer", "Fall", "Winter"] = Field(..., description="Season the child enrolled")


class PredictionOut(BaseModel):
    predicted_pciat_total: float
    risk_level: str
    message: str


class RetrainOut(BaseModel):
    status: str
    model_type: str
    method: str
    rows_used: int
    mse_on_new_data_before: float
    mse_on_new_data_after: float


# ---------------------------------------------------------------- app + CORS
app = FastAPI(
    title="PCIAT Early-Warning API",
    description="Predicts a child's problematic-internet-use score (PCIAT total, 0-100). "
                "A score of 40 or more suggests the child may need support.",
    version="1.0.0",
)

# CORS decides which WEBSITES (browser origins) may call this API from JavaScript.
# - Allowed origins: only my own deployed Render site and my local Flutter web debug
#   server. No "*" wildcard, so an unknown website cannot use the API from a visitor's
#   browser. Extra origins can be added through the ALLOWED_ORIGINS environment variable.
# - Allowed methods: only GET (docs, health) and POST (/predict, /retrain), the only ones
#   the API uses. PUT, DELETE and PATCH are refused.
# - Allowed headers: only Content-Type (JSON or file upload) and Accept, which is all the
#   clients send. Custom headers are refused.
# - Credentials: False. The API uses no cookies or login sessions, so browsers are not
#   allowed to send them, which removes a whole class of cross-site attacks.
# The Flutter mobile app is not a browser, so CORS does not apply to it; it calls the API directly.
ALLOWED_ORIGINS = [
    "https://pciat-regression-api.onrender.com",   # deployed service (update to your Render URL)
    "http://localhost:8080",                        # Flutter web debug server
    "http://127.0.0.1:8080",
] + [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


# ---------------------------------------------------------------- helpers
def risk_level(score: float) -> tuple[str, str]:
    if score >= 50:
        return "High", "Strong signs of problematic internet use. Talk with the child kindly and refer them to a counsellor."
    if score >= 40:
        return "Moderate", "Signs of problematic internet use. A supportive check-in and screen-time plan are recommended."
    if score >= 30:
        return "Mild", "Some signs to watch. Keep an eye on sleep and screen habits."
    return "Low", "No strong signs of problematic internet use."


def predict_one(row: dict) -> float:
    """Load-free version of the notebook's predict_one: predicts one child with the in-memory model."""
    with model_lock:
        current = model
    value = float(current.predict(pd.DataFrame([row])[FEATURES])[0])
    return float(np.clip(value, 0, 100))  # the PCIAT scale runs from 0 to 100


def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="The file could not be read as a CSV.")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"CSV is missing columns: {missing}. Expected: {FEATURES + [TARGET]}")
    df = df[FEATURES + [TARGET]].copy()
    numeric = [c for c in FEATURES if c != "enroll_season"] + [TARGET]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df.loc[~df["enroll_season"].isin(SEASONS), "enroll_season"] = np.nan
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(0, 100)]
    if len(df) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 rows with a valid pciat_total (0-100) to retrain.")
    return df


# ---------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    with model_lock:
        name = type(model.named_steps["model"]).__name__
    return {"status": "ok", "model": name}


@app.post("/predict", response_model=PredictionOut)
def predict(child: ChildInput):
    score = predict_one(child.model_dump())
    level, message = risk_level(score)
    return PredictionOut(predicted_pciat_total=round(score, 1), risk_level=level, message=message)


@app.post("/retrain", response_model=RetrainOut)
async def retrain(file: UploadFile = File(..., description="CSV with the same columns as training_data.csv")):
    """Upload new data; the model is retrained immediately and /predict uses it straight away."""
    global model
    new = read_uploaded_csv(await file.read())
    X_new, y_new = new[FEATURES], new[TARGET].to_numpy()

    with model_lock:
        current = model
    mse_before = mean_squared_error(y_new, current.predict(X_new))

    if isinstance(current.named_steps["model"], SGDRegressor):
        # Best model is SGD: continue training the saved model on the new rows with partial_fit.
        # The preprocessing (imputer, scaler, encoder) stays as fitted on the original training set.
        updated = copy.deepcopy(current)     # keep the live model untouched until the swap
        sgd = updated.named_steps["model"]
        X_t = updated.named_steps["preprocess"].transform(X_new)
        rng = np.random.default_rng(42)
        for _ in range(RETRAIN_EPOCHS):
            idx = rng.permutation(len(y_new))
            sgd.partial_fit(X_t[idx], y_new[idx])
        method = f"partial_fit: {RETRAIN_EPOCHS} epochs on the new rows"
    else:
        # Any other model: refit from scratch on the original data plus the new rows.
        original = pd.read_csv(TRAINING_DATA_PATH)
        combined = pd.concat([original, new], ignore_index=True)
        updated = clone(current).fit(combined[FEATURES], combined[TARGET])
        combined.to_csv(TRAINING_DATA_PATH, index=False)
        method = f"full refit on {len(original)} original + {len(new)} new rows"

    mse_after = mean_squared_error(y_new, updated.predict(X_new))
    with model_lock:
        model = updated                      # /predict uses the new model from now on
    joblib.dump(updated, MODEL_PATH)         # kept until the server restarts (Render free plan)

    return RetrainOut(status="Model retrained and now in use", model_type=type(updated.named_steps["model"]).__name__,
                      method=method, rows_used=len(new),
                      mse_on_new_data_before=round(float(mse_before), 2),
                      mse_on_new_data_after=round(float(mse_after), 2))

