from __future__ import annotations

import math

import numpy as np


def icc21(test: np.ndarray, ref: np.ndarray) -> float:
    m = np.isfinite(test) & np.isfinite(ref)
    x = np.column_stack([test[m], ref[m]])
    n, k = x.shape
    if n < 2:
        return math.nan
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    msr = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    msc = n * float(np.sum((col_means - grand) ** 2)) / (k - 1)
    mse = float(np.sum((x - row_means[:, None] - col_means[None, :] + grand) ** 2)) / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom != 0 else math.nan
