"""
deploy.py
Covers FR12 (predict_churn scoring function) and FR13 (persisted model artifact).

Simulates how this model would be called from an API/CRM integration in production:
a raw customer record in, a churn probability + recommendation out.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd

from data_prep import load_data, clean_data, engineer_features, split_and_scale, BINARY_COLS, MULTI_CAT_COLS
from train import train_models

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), '..', 'churn_model_artifact.json')


def save_artifact(model, feature_names, scaler, threshold=0.5, path=ARTIFACT_PATH):
    """FR13: Persist trained weights + preprocessing metadata to a JSON artifact.
    This is what a serving layer (e.g. a Flask/FastAPI endpoint) would load.
    """
    artifact = {
        'weights': model.weights.tolist(),
        'bias': model.bias,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'decision_threshold': threshold,
    }
    with open(path, 'w') as f:
        json.dump(artifact, f)
    print(f"Saved model artifact -> {path}")


def _build_feature_row(customer: dict, feature_names):
    """Turn a raw customer dict (same schema as the source CSV, minus customerID/Churn)
    into the same one-hot encoded feature vector used at training time.
    """
    row = pd.DataFrame([customer])
    row['gender'] = (row['gender'] == 'Male').astype(int)
    for c in BINARY_COLS:
        row[c] = (row[c] == 'Yes').astype(int)
    row = pd.get_dummies(row, columns=MULTI_CAT_COLS, drop_first=False)

    aligned = pd.DataFrame(0, index=row.index, columns=feature_names)
    for c in row.columns:
        if c in aligned.columns:
            aligned[c] = row[c].values
    return aligned.values.astype(float)


def predict_churn(customer: dict, artifact_path=ARTIFACT_PATH):
    """FR12: Score a single raw customer record end-to-end.
    Returns {'churn_probability': float, 'recommendation': str}.
    """
    with open(artifact_path) as f:
        art = json.load(f)

    w = np.array(art['weights'])
    b = art['bias']
    mean = np.array(art['scaler_mean'])
    scale = np.array(art['scaler_scale'])
    threshold = art['decision_threshold']

    x = _build_feature_row(customer, art['feature_names'])
    x_scaled = (x - mean) / scale
    z = x_scaled @ w + b
    proba = float(1 / (1 + np.exp(-np.clip(z, -500, 500)))[0])

    recommendation = (
        'CHURN RISK — recommend retention outreach'
        if proba >= threshold else 'Likely to stay'
    )
    return {'churn_probability': round(proba, 4), 'recommendation': recommendation}


if __name__ == '__main__':
    # python src/deploy.py
    df = clean_data(load_data('data/telco.csv'))
    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)

    _, model_weighted = train_models(X_train, y_train, verbose=False)
    save_artifact(model_weighted, feature_names, scaler)

    # --- Demo: a new, high-risk-looking customer ---
    at_risk_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.0, 'TotalCharges': 190.0
    }
    print("At-risk customer  ->", predict_churn(at_risk_customer))

    # --- Demo: a loyal, low-risk-looking customer ---
    loyal_customer = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 55.0, 'TotalCharges': 3300.0
    }
    print("Loyal customer    ->", predict_churn(loyal_customer))
