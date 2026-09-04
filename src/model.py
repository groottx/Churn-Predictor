"""
model.py
Covers FR4 (sigmoid), FR5 (log-loss), FR6 (gradient descent), FR7 (class weighting).

A binary logistic regression classifier trained with batch gradient descent,
implemented from scratch using only NumPy.
"""

import numpy as np


class LogisticRegressionScratch:
    """Binary logistic regression trained with batch gradient descent.

    Supports L2 regularization and per-class sample weights (for imbalance handling).
    """

    def __init__(self, lr=0.1, n_iters=3000, l2=0.01, class_weight=None, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.class_weight = class_weight  # None, or (weight_for_class_0, weight_for_class_1)
        self.verbose = verbose

        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        """FR4: Sigmoid activation. Clipped to avoid overflow for large |z|."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_weights(self, y):
        """FR7: Map each sample to a weight based on its class."""
        if self.class_weight is None:
            return np.ones_like(y)
        w_neg, w_pos = self.class_weight
        return np.where(y == 1, w_pos, w_neg)

    def _log_loss(self, y, y_hat, sample_w):
        """FR5: Weighted binary cross-entropy + L2 penalty."""
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        data_loss = -np.mean(
            sample_w * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        )
        reg_loss = (self.l2 / (2 * len(y))) * np.sum(self.weights ** 2)
        return data_loss + reg_loss

    def fit(self, X, y):
        """FR6: Batch gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        sample_w = self._sample_weights(y)

        for i in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            error = (y_hat - y) * sample_w
            grad_w = (X.T @ error) / n_samples + (self.l2 / n_samples) * self.weights
            grad_b = np.sum(error) / n_samples

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            loss = self._log_loss(y, y_hat, sample_w)
            self.loss_history.append(loss)

            if self.verbose and i % 500 == 0:
                print(f"iter {i:5d}  log-loss = {loss:.4f}")

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_class_weights(y):
    """FR7: Balanced class weights, matching sklearn's class_weight='balanced' formula:
    weight_c = n_samples / (n_classes * count_c)
    """
    n = len(y)
    n_pos = y.sum()
    n_neg = n - n_pos
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return w_neg, w_pos


if __name__ == '__main__':
    # Quick manual check: python src/model.py
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_prep import load_data, clean_data, engineer_features, split_and_scale

    df = clean_data(load_data('data/telco.csv'))
    df_model, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_model, feature_names)

    w_neg, w_pos = compute_class_weights(y_train)
    print(f"class weights -> no-churn: {w_neg:.3f}, churn: {w_pos:.3f}")

    model = LogisticRegressionScratch(lr=0.1, n_iters=3000, l2=0.01,
                                       class_weight=(w_neg, w_pos), verbose=True)
    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
