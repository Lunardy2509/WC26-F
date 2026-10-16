from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader.footprint import ACTIVE_FRACTION

N_PHASE = 101                      # pipeline_1 is already resampled to 101 frames

def remap(points: np.ndarray, source_box, target_box) -> np.ndarray:
    """Per-axis affine remap of (row, col) points between two contact boxes."""
    s_r0, s_r1, s_c0, s_c1 = source_box
    t_r0, t_r1, t_c0, t_c1 = target_box
    out = points.copy().astype(float)
    out[:, 0] = t_r0 + (points[:, 0] - s_r0) * (t_r1 - t_r0) / (s_r1 - s_r0)
    out[:, 1] = t_c0 + (points[:, 1] - s_c0) * (t_c1 - t_c0) / (s_c1 - s_c0)
    return out


def contact_box(peak_map: np.ndarray, fraction: float = ACTIVE_FRACTION):
    peak = float(peak_map.max())
    if peak <= 0:
        return None
    active = peak_map > peak * fraction
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    if rows.size < 2 or cols.size < 2:
        return None
    return float(rows.min()), float(rows.max()), float(cols.min()), float(cols.max())


def field_cop(frames: np.ndarray) -> np.ndarray:
    """Pressure-weighted centroid per frame -> (T, 2) as (row, col). Eq. (1)."""
    tot = frames.sum(axis=(1, 2))
    rows = np.arange(frames.shape[1], dtype=float)
    cols = np.arange(frames.shape[2], dtype=float)
    r = (frames.sum(axis=2) * rows).sum(axis=1)
    c = (frames.sum(axis=1) * cols).sum(axis=1)
    out = np.full((len(frames), 2), np.nan)
    ok = tot > 0
    out[ok, 0] = r[ok] / tot[ok]
    out[ok, 1] = c[ok] / tot[ok]
    return out


# ------------------------------------------------------- raw (unregistered) route
def temporal_resample(fs: np.ndarray, t: int = N_PHASE) -> np.ndarray:
    """Crop inactive head/tail, then interpolate to t frames (example-code method)."""
    act = fs.max(axis=(1, 2))
    nz = np.flatnonzero(act > 0)
    if nz.size < 2:
        return None
    fs = fs[nz[0]:nz[-1] + 1]
    src = np.linspace(0.0, 1.0, len(fs))
    dst = np.linspace(0.0, 1.0, t)
    idx = np.searchsorted(src, dst).clip(1, len(fs) - 1)
    lo, hi = idx - 1, idx
    w = ((dst - src[lo]) / (src[hi] - src[lo]))[:, None, None]
    return fs[lo] * (1 - w) + fs[hi] * w


def extract_raw_steps(trial: np.ndarray, meta: pd.DataFrame, side: str,
                      resample: bool = True):
    """Footsteps straight from the recording, with NO spatial normalisation.

    pipeline_1 is PCA-rotated, resized to a fixed 25x70 sole and registered to the
    MUN104 template. That removes precisely the between-subject shape variation
    that makes the ML axis hard, and it leaves a no-input template control scoring
    ML 0.86 - the problem is normalised away before any model sees it. Our real
    F-SCAN recordings get none of that treatment, so for a transferable result the
    public data must be handled the same way: bounding-box crop and a 180-degree
    walking-direction flip only (example-code load_data/normalize cells), plus the
    temporal resampling every trajectory comparison needs.
    """
    keep = meta[(meta.Exclude == 0) & (meta.Incomplete == 0) & (meta.Side == side)]
    for _, r in keep.iterrows():
        s, e = int(r.StartFrame), int(r.EndFrame)
        y0, y1, x0, x1 = int(r.Ymin), int(r.Ymax), int(r.Xmin), int(r.Xmax)
        fs = trial[s:e + 1, y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        if fs.size == 0 or fs.max() <= 0:
            continue
        if int(r.Orientation) == 1:            # walking direction -> upright
            fs = np.flip(fs, axis=(1, 2))
        if side == "Left":                     # canonicalise chirality
            fs = fs[:, :, ::-1]
        if resample:
            fs = temporal_resample(fs)
        else:
            # Scalar features need true stance duration, so keep the natural
            # frame count and let features_for_stances do its own resampling.
            act = fs.max(axis=(1, 2))
            nz = np.flatnonzero(act > 0)
            fs = fs[nz[0]:nz[-1] + 1] if nz.size >= 2 else None
        if fs is None or len(fs) < 3:
            continue
        yield np.ascontiguousarray(fs)
