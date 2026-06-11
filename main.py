"""
main.py — FastAPI server for Telco Churn Prediction
Wraps the pipeline functions from app.py into REST endpoints.

Endpoints:
    GET  /health         → health check
    POST /predict        → predict churn for one customer
"""

import os
import warnings
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import pipeline functions from app.py
from app import (
    encode_features,
    fix_multicollinearity,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# App state (in-memory model store)
# ─────────────────────────────────────────────────────────────

MODEL_PATH = "churn_model.joblib"
app_state: dict = {
    "model":        None,
    "feature_cols": None,
    "threshold":    0.30,
    "trained":      False,
}


def load_model_from_disk():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        app_state["model"] = data["model"]
        app_state["feature_cols"] = data["feature_cols"]
        app_state["threshold"] = data.get("threshold", app_state["threshold"])
        app_state["trained"] = True
        print("[startup] Model loaded from disk.")
    else:
        print("[startup] WARNING: churn_model.joblib not found. Predictions will fail until a model is placed in the project root.")


# ─────────────────────────────────────────────────────────────
# Lifespan — load model on startup if it exists
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_disk()
    yield

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Get churn predictions for Telco customers via REST.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve CSS/JS files
app.mount("/static", StaticFiles(directory="."), name="static")

# Open index.html when user visits /
@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")

# Enable CORS so the local HTML file can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """One customer's raw features — same field names as the CSV."""
    gender:            str   = Field(..., example="Male")
    SeniorCitizen:     int   = Field(..., example=0)
    Partner:           str   = Field(..., example="Yes")
    Dependents:        str   = Field(..., example="No")
    tenure:            int   = Field(..., example=12)
    PhoneService:      str   = Field(..., example="Yes")
    MultipleLines:     str   = Field(..., example="No")
    InternetService:   str   = Field(..., example="Fiber optic")
    OnlineSecurity:    str   = Field(..., example="No")
    OnlineBackup:      str   = Field(..., example="Yes")
    DeviceProtection:  str   = Field(..., example="No")
    TechSupport:       str   = Field(..., example="No")
    StreamingTV:       str   = Field(..., example="No")
    StreamingMovies:   str   = Field(..., example="No")
    Contract:          str   = Field(..., example="Month-to-month")
    PaperlessBilling:  str   = Field(..., example="Yes")
    PaymentMethod:     str   = Field(..., example="Electronic check")
    MonthlyCharges:    float = Field(..., example=70.35)
    TotalCharges:      float = Field(..., example=844.2)


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int   # 1 = likely churn, 0 = likely stay
    risk_label:        str   # "HIGH" / "LOW"


# ─────────────────────────────────────────────────────────────
# Helper — preprocess a single customer dict → model-ready row
# ─────────────────────────────────────────────────────────────

def preprocess_single(customer: dict, feature_cols: list) -> pd.DataFrame:
    """
    Encode one customer record and align columns to match training features.
    """
    df = pd.DataFrame([customer])
    df = encode_features(df)
    df = fix_multicollinearity(df)

    # Drop Churn if present (shouldn't be, but safe)
    df = df.drop(columns=["Churn"], errors="ignore")

    # Align to training columns — fill any missing dummies with 0
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":      "ok",
        "service":     "Telco Churn API",
        "model_ready": app_state["trained"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    Requires the pre-trained model (churn_model.joblib).
    """
    if not app_state["trained"]:
        raise HTTPException(status_code=503, detail="Model not loaded or trained yet.")

    try:
        row = preprocess_single(customer.model_dump(), app_state["feature_cols"])
        proba = float(app_state["model"].predict_proba(row)[:, 1][0])
        threshold = app_state["threshold"]
        pred = int(proba >= threshold)

        return PredictResponse(
            churn_probability=round(proba, 4),
            churn_prediction=pred,
            risk_label="HIGH" if pred == 1 else "LOW",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
