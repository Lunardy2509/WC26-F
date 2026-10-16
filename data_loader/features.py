from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import *
from data_loader.io import parse_fscan, parse_sock


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return 0.0
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def features_for_stances(frame_df: pd.DataFrame, stances: list[tuple[int, int]], trial: Trial) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for step_idx, (start, end) in enumerate(stances, start=1):
        step = frame_df.iloc[start:end]
        time = step["time_s"].to_numpy()
        total = step["total_pressure"].to_numpy()
        peak = step["peak_pressure"].to_numpy()
        peak_x = step["peak_x"].to_numpy()
        peak_y = step["peak_y"].to_numpy()
        cop_x = step["cop_x"].to_numpy()
        cop_y = step["cop_y"].to_numpy()
        peak_frame_idx = peak_index(peak)
        duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
        if len(time) > 1:
            _trapezoid = getattr(np, "trapezoid", None) or np.trapz
            pressure_impulse = float(_trapezoid(total, time))
            target = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
            feature_cop_x = smooth_edge(resample_series(time, cop_x, target), COP_FEATURE_SMOOTH_WINDOW)
            feature_cop_y = smooth_edge(resample_series(time, cop_y, target), COP_FEATURE_SMOOTH_WINDOW)
            distances = np.sqrt(np.diff(feature_cop_x) ** 2 + np.diff(feature_cop_y) ** 2)
            path_length = float(np.nansum(distances))
            total_curve = resample_series(time, total, target)
            fwhm_percent = half_max_width_percent(total_curve)
            gradient_max, gradient_min = pressure_gradients(total_curve, duration)
            peak_pressure_value = float(np.nanmax(peak)) if np.isfinite(peak).any() else math.nan
        else:
            pressure_impulse = 0.0
            path_length = 0.0
            feature_cop_x = cop_x
            feature_cop_y = cop_y
            fwhm_percent = math.nan
            gradient_max = gradient_min = math.nan
            peak_pressure_value = math.nan

        mean_pressure = pressure_impulse / duration if duration > 0 else math.nan
        scale_free = math.isfinite(peak_pressure_value) and peak_pressure_value > 0
        rows.append(
            {
                "sex": trial.sex,
                "device": trial.device,
                "subject": trial.subject,
                "condition": trial.condition,
                "foot": trial.foot,
                "step_index": step_idx,
                "start_time_s": float(time[0]),
                "end_time_s": float(time[-1]),
                "stance_duration": duration,
                "total_pressure": float(total.sum()),
                "peak_pressure": peak_pressure_value,
                "peak_pressure_time_percent": peak_time_percent(time, peak),
                "peak_pressure_ml_coordinate": (
                    float(peak_x[peak_frame_idx]) if peak_frame_idx is not None else math.nan
                ),
                "peak_pressure_ap_coordinate": (
                    float(peak_y[peak_frame_idx]) if peak_frame_idx is not None else math.nan
                ),
                "pressure_impulse": pressure_impulse,
                "mean_pressure": mean_pressure,
                "pressure_gradient_max": gradient_max,
                "pressure_gradient_min": gradient_min,
                "pressure_fwhm_percent": fwhm_percent,
                "mean_to_peak_pressure_ratio": (
                    mean_pressure / peak_pressure_value if scale_free and math.isfinite(mean_pressure) else math.nan
                ),
                "pressure_gradient_max_norm": (
                    gradient_max / peak_pressure_value if scale_free and math.isfinite(gradient_max) else math.nan
                ),
                "pressure_gradient_min_norm": (
                    abs(gradient_min) / peak_pressure_value if scale_free and math.isfinite(gradient_min) else math.nan
                ),
                "cop_path_length": path_length,
                "ap_displacement": float(np.nanmax(feature_cop_y) - np.nanmin(feature_cop_y)),
                "ml_displacement": float(np.nanmax(feature_cop_x) - np.nanmin(feature_cop_x)),
                "cop_velocity": path_length / duration if duration > 0 else math.nan,
                "cop_area": polygon_area(feature_cop_x, feature_cop_y),
            }
        )
    return pd.DataFrame(rows)


def trajectories_for_stances(frame_df: pd.DataFrame, stances: list[tuple[int, int]], trial: Trial) -> pd.DataFrame:
    target = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
    rows: list[dict[str, float | int | str]] = []
    for step_idx, (start, end) in enumerate(stances, start=1):
        step = frame_df.iloc[start:end]
        time = step["time_s"].to_numpy(dtype=float)
        cop_x = resample_series(time, step["cop_x"].to_numpy(dtype=float), target)
        cop_y = resample_series(time, step["cop_y"].to_numpy(dtype=float), target)
        total = resample_series(time, step["total_pressure"].to_numpy(dtype=float), target)
        total_max = np.nanmax(total) if np.isfinite(total).any() else math.nan
        total_norm = total / total_max if np.isfinite(total_max) and total_max > 0 else np.full_like(total, np.nan)
        for point_idx, stance_pct in enumerate(target * 100.0):
            rows.append(
                {
                    "device": trial.device,
                    "sex": trial.sex,
                    "subject": trial.subject,
                    "condition": trial.condition,
                    "foot": trial.foot,
                    "step_index": step_idx,
                    "point_index": point_idx,
                    "stance_percent": stance_pct,
                    "cop_x": cop_x[point_idx],
                    "cop_y": cop_y[point_idx],
                    "total_pressure": total[point_idx],
                    "total_pressure_norm": total_norm[point_idx],
                }
            )
    return pd.DataFrame(rows)


def read_trial(trial: Trial) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    frame_df = parse_fscan(trial.path) if trial.device == "fscan" else parse_sock(trial.path, trial.foot, trial)
    original_frames = len(frame_df)
    frame_df = frame_df[frame_df["time_s"] >= WARMUP_SECONDS].reset_index(drop=True)
    stances, threshold = find_stances(frame_df)
    features = features_for_stances(frame_df, stances, trial)
    trajectories = trajectories_for_stances(frame_df, stances, trial)
    qc = {
        "device": trial.device,
        "sex": trial.sex,
        "subject": trial.subject,
        "condition": trial.condition,
        "foot": trial.foot,
        "file": str(trial.path),
        "frames": len(frame_df),
        "excluded_warmup_seconds": WARMUP_SECONDS,
        "original_frames": original_frames,
        "threshold": threshold,
        "stance_count": len(stances),
        "median_total_pressure": float(frame_df["total_pressure"].median()),
        "max_total_pressure": float(frame_df["total_pressure"].max()),
    }
    return features, trajectories, qc
