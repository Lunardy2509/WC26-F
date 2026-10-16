from __future__ import annotations

import numpy as np


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    W = np.linalg.solve(A, Xs.T @ (Y - Y.mean(0)))
    b = Y.mean(0)
    return (mu, sd, W, b)


def ridge_predict(model, X):
    mu, sd, W, b = model
    return ((X - mu) / sd) @ W + b
