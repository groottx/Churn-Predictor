"""
train.py
Covers FR8: train both an unweighted and a class-weighted version of the
scratch logistic regression, for comparison.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_prep import load_data, clean_data, engineer_features, split_and_scale
from model import LogisticRegressionScratch, compute_class_weights


def train_models(X_train, y_train, lr=0.1, n_iters=3000, l2=0.01, verbose=False):
    """Train both an unweighted and a class-weighted scratch model.
    Returns (model_unweighted, model_weighted).
    """
    model_unweighted = LogisticRegressionScratch(
        lr=lr, n_iters=n_iters, l2=l2, class_weight=None, verbose=verbose
    )
    model_unweighted.fit(X_train, y_train)

    w_neg, w_pos = compute_class_weights(y_train)
    model_weighted = LogisticRegressionScratch(
        lr=lr, n_iters=n_iters, l2=l2, class_weight=(w_neg, w_pos), verbose=verbose
    )
    model_weighted.fit(X_train, y_train)

    return model_unweighted, model_weighted


if __name__ == '__main__':
    # python src/train.py
    df = clean_data(load_data('data/telco.csv'))
    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)

    model_unweighted, model_weighted = train_models(X_train, y_train, verbose=True)

    print("\nUnweighted final loss:", round(model_unweighted.loss_history[-1], 4))
    print("Weighted final loss:  ", round(model_weighted.loss_history[-1], 4))
