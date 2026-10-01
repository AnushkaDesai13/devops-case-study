"""
model.py
Trains a RandomForestRegressor to predict delivery_time_minutes.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "delivery_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

CATEGORICAL = ["traffic_level", "weather"]
FEATURES    = ["distance_km", "traffic_level", "weather",
               "prep_time", "partner_rating", "partner_available", "premium_order"]
TARGET      = "delivery_time_minutes"


def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """Encode categorical columns. Returns (processed_df, encoders)."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col])
    return df, encoders


def train_model(df: pd.DataFrame):
    """Train RandomForestRegressor, save model + encoders, return metrics."""
    df_enc, encoders = preprocess(df, fit=True)

    X = df_enc[FEATURES]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    # Persist
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"[Model] Trained RandomForestRegressor | MAE = {mae:.2f} min")
    print(f"[Model] Saved → {MODEL_PATH}, {ENCODERS_PATH}")
    return model, encoders, mae


def load_model():
    """Load saved model and encoders."""
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


def predict_delivery_time(order: dict, model, encoders) -> float:
    """Predict delivery time for a single order dict."""
    df_order = pd.DataFrame([order])[FEATURES]
    df_enc, _ = preprocess(df_order, encoders=encoders, fit=False)
    return round(float(model.predict(df_enc)[0]), 1)
