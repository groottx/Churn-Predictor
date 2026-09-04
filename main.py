"""
main.py
Runs the full churn-prediction pipeline end to end:
data prep -> train (weighted + unweighted) -> evaluate (+ sklearn benchmark) -> deploy demo.

Usage:
    python main.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_prep import load_data, clean_data, eda_summary, engineer_features, split_and_scale
from train import train_models
from evaluate import evaluate, plot_confusion_matrices, plot_roc_curves, feature_importance
from deploy import save_artifact, predict_churn

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import pandas as pd


def main():
    print("=" * 60)
    print("STEP 1-3: Data loading, cleaning, feature engineering")
    print("=" * 60)
    df = clean_data(load_data('data/telco.csv'))
    print(eda_summary(df))
    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}  Features: {len(feature_names)}\n")

    print("=" * 60)
    print("STEP 4-8: Training (from-scratch, unweighted vs class-weighted)")
    print("=" * 60)
    model_unweighted, model_weighted = train_models(X_train, y_train, verbose=False)
    print(f"Unweighted final loss: {model_unweighted.loss_history[-1]:.4f}")
    print(f"Weighted final loss:   {model_weighted.loss_history[-1]:.4f}\n")

    print("=" * 60)
    print("STEP 9-11: Evaluation + scikit-learn benchmark + feature importance")
    print("=" * 60)
    results = []
    preds, probas = {}, {}

    proba_u = model_unweighted.predict_proba(X_test)
    pred_u = model_unweighted.predict(X_test)
    results.append(evaluate("Scratch LR (unweighted)", y_test, pred_u, proba_u))
    preds["Unweighted"], probas["Unweighted"] = pred_u, proba_u

    proba_w = model_weighted.predict_proba(X_test)
    pred_w = model_weighted.predict(X_test)
    results.append(evaluate("Scratch LR (class-weighted)", y_test, pred_w, proba_w))
    preds["Class-weighted"], probas["Class-weighted"] = pred_w, proba_w

    sk_model = SKLogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    sk_model.fit(X_train, y_train)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]
    sk_pred = sk_model.predict(X_test)
    results.append(evaluate("Scikit-learn LR (balanced)", y_test, sk_pred, sk_proba))
    preds["Scikit-learn"], probas["Scikit-learn"] = sk_pred, sk_proba

    results_df = pd.DataFrame(results).set_index('model').round(4)
    print(results_df, "\n")

    plot_confusion_matrices(y_test, preds)
    plot_roc_curves(y_test, probas)
    coef_df = feature_importance(model_weighted, feature_names)
    print("\nTop churn drivers:\n", coef_df.head(10), "\n")

    print("=" * 60)
    print("STEP 12-13: Deployment artifact + demo scoring")
    print("=" * 60)
    save_artifact(model_weighted, feature_names, scaler)

    at_risk_customer = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.0, 'TotalCharges': 190.0
    }
    print("At-risk customer  ->", predict_churn(at_risk_customer))

    loyal_customer = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 55.0, 'TotalCharges': 3300.0
    }
    print("Loyal customer    ->", predict_churn(loyal_customer))

    print("\nDone. See confusion_matrices.png, roc_curves.png, feature_importance.png,")
    print("and churn_model_artifact.json in the project root.")


if __name__ == '__main__':
    main()
