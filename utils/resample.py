from __future__ import annotations

import numpy as np


def resample_matrix(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    if time.size < 2 or time[-1] == time[0]:
        return np.full((len(target), values.shape[1]), np.nan)
    u = (time - time[0]) / (time[-1] - time[0])
    u, idx = np.unique(u, return_index=True)
    if u.size < 2:
        return np.full((len(target), values.shape[1]), np.nan)
    v = values[idx]
    pos = np.clip(np.searchsorted(u, target, side="right") - 1, 0, u.size - 2)
    span = u[pos + 1] - u[pos]
    w = np.where(span > 0, (target - u[pos]) / np.where(span > 0, span, 1.0), 0.0)
    return v[pos] + (v[pos + 1] - v[pos]) * w[:, None]
