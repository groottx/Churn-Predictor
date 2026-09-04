"""
evaluate.py
Covers FR9 (metrics/confusion matrix/ROC), FR10 (sklearn benchmark),
FR11 (feature importance).
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from data_prep import load_data, clean_data, engineer_features, split_and_scale
from model import compute_class_weights
from train import train_models


def evaluate(name, y_true, y_pred, y_proba, verbose=True):
    """FR9: Compute the full metric suite for one model."""
    result = {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }
    if verbose:
        print(f"--- {name} ---")
        for k, v in result.items():
            if k != 'model':
                print(f"{k:10s}: {v:.4f}")
        print()
    return result


def plot_confusion_matrices(y_test, preds_by_model, out_path='confusion_matrices.png'):
    """FR9: Save confusion matrix plots for each model."""
    n = len(preds_by_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, preds_by_model.items()):
        cm = confusion_matrix(y_test, pred)
        ax.imshow(cm, cmap='Blues')
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_yticklabels(['No Churn', 'Churn'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def plot_roc_curves(y_test, probas_by_model, out_path='roc_curves.png'):
    """FR9: Save ROC curves for each model on one plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def feature_importance(model_weighted, feature_names, top_n=15, out_path='feature_importance.png'):
    """FR11: Standardized coefficients, sorted by magnitude."""
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model_weighted.weights
    }).sort_values('coefficient', key=abs, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#C44E52' if c > 0 else '#4C72B0' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'][::-1], coef_df['coefficient'][::-1], color=colors[::-1])
    ax.set_xlabel('Standardized coefficient (impact on churn log-odds)')
    ax.set_title(f'Top {top_n} Drivers of Churn')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")

    return coef_df


if __name__ == '__main__':
    # python src/evaluate.py
    df = clean_data(load_data('data/telco.csv'))
    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)

    model_unweighted, model_weighted = train_models(X_train, y_train, verbose=False)

    results = []
    preds = {}
    probas = {}

    proba_u = model_unweighted.predict_proba(X_test)
    pred_u = model_unweighted.predict(X_test)
    results.append(evaluate("Scratch LR (unweighted)", y_test, pred_u, proba_u))
    preds["Unweighted"] = pred_u
    probas["Unweighted"] = proba_u

    proba_w = model_weighted.predict_proba(X_test)
    pred_w = model_weighted.predict(X_test)
    results.append(evaluate("Scratch LR (class-weighted)", y_test, pred_w, proba_w))
    preds["Class-weighted"] = pred_w
    probas["Class-weighted"] = proba_w

    # FR10: scikit-learn benchmark
    sk_model = SKLogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    sk_model.fit(X_train, y_train)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]
    sk_pred = sk_model.predict(X_test)
    results.append(evaluate("Scikit-learn LR (balanced)", y_test, sk_pred, sk_proba))
    preds["Scikit-learn"] = sk_pred
    probas["Scikit-learn"] = sk_proba

    results_df = pd.DataFrame(results).set_index('model').round(4)
    print(results_df)

    plot_confusion_matrices(y_test, preds)
    plot_roc_curves(y_test, probas)
    coef_df = feature_importance(model_weighted, feature_names)
    print("\nTop features:\n", coef_df)
