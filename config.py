from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PPS_DATA_DIR", ROOT / "data"))
STEPUP_DIR = Path(os.environ.get("PPS_STEPUP_DIR", ROOT / "StepUP-P150_dataset"))
OUT_DIR = Path(os.environ.get("PPS_OUT_DIR", ROOT / "results"))

# ------------------------------------------------------------------ device scaling and cleaning
SOCK_BASELINE_PERCENTILE = 10         
SOCK_NOISE_PERCENTILE = 10
SOCK_NOISE_STD_MULTIPLIER = 3          
SOCK_MIN_CLEANED_PRESSURE = 18000      
SOCK_RAW_SCALE = 10.0
FSCAN_RAW_SCALE = 1000.0              
FSCAN_MIN_CLEANED_PRESSURE = 20000.0 

FSCAN_ROWS = 60
FSCAN_COLS = 21
SOCK_ML_MIDLINE = (1.0 + FSCAN_COLS) / 2.0

# ------------------------------------------------------------------ stances and resampling
MIN_STANCE_SECONDS = 0.2
MERGE_GAP_SECONDS = 0.08
RESAMPLE_POINTS = 100 
COP_FEATURE_SMOOTH_WINDOW = 5
WARMUP_SECONDS = 15.0 

# ------------------------------------------------------------------ sock geometry
SOCK_COORDS_FULL = np.array(
    [
        (4.500, 4.500), (9.155, 5.063), (12.951, 6.584), (16.400, 8.696), (19.500, 11.086),
        (5.848, 13.323), (9.052, 15.949), (6.265, 19.482), (9.151, 19.608), (12.135, 19.811),
        (14.931, 19.761), (9.284, 22.983), (17.782, 26.599), (6.660, 30.083), (9.391, 30.143),
        (17.782, 33.272), (6.699, 33.535), (9.434, 33.606), (9.520, 39.984), (9.537, 43.643),
        (9.507, 47.161), (12.307, 50.722), (15.004, 51.055), (12.178, 54.370), (15.201, 55.037),
        (15.257, 58.500),
    ],
    dtype=float,
)
SOCK_SENSOR_INDICES = np.array([0, 8, 10, 12, 19, 20, 21, 25], dtype=int)
SOCK_SENSOR_COUNT = int(len(SOCK_SENSOR_INDICES))
MIRROR_SOCK_LEFT_FOOT = True

FSCAN_X = np.tile(np.arange(1, FSCAN_COLS + 1, dtype=float), FSCAN_ROWS)
FSCAN_Y = np.repeat(np.arange(1, FSCAN_ROWS + 1, dtype=float), FSCAN_COLS)
FSCAN_HEEL_MASK = FSCAN_Y >= 40.0


def sock_coords_for_foot(foot: str) -> np.ndarray:
    coords = SOCK_COORDS_FULL.copy()
    if MIRROR_SOCK_LEFT_FOOT and foot == "L":
        coords[:, 0] = 2.0 * SOCK_ML_MIDLINE - coords[:, 0]
    return coords[SOCK_SENSOR_INDICES]


def select_sock_sensor_values(values: np.ndarray) -> np.ndarray:
    return values[:, SOCK_SENSOR_INDICES]


def clean_sock_values(values: np.ndarray, trial: "Trial | None" = None) -> np.ndarray:
    baseline = np.percentile(values, SOCK_BASELINE_PERCENTILE, axis=0)
    low_values = np.minimum(values, baseline)
    noise_std = np.std(low_values, axis=0)
    noise_floor = np.percentile(values, SOCK_NOISE_PERCENTILE, axis=0)
    threshold = np.maximum(noise_floor, baseline + SOCK_NOISE_STD_MULTIPLIER * noise_std)
    corrected = values - baseline
    corrected[values <= threshold] = 0.0
    corrected = np.clip(corrected, 0.0, None)
    corrected[corrected < SOCK_MIN_CLEANED_PRESSURE] = 0.0
    return corrected


# ------------------------------------------------------------------ trial discovery
@dataclass(frozen=True)
class Trial:
    sex: str
    device: str
    subject: str
    condition: str
    foot: str
    path: Path


def condition_from_name(device: str, name: str) -> str:
    """'mix*' = sock and F-Scan worn at the same time; 'only*' = separate recordings."""
    lower = name.lower()
    if lower.startswith("mix"):
        return "mix"
    if lower.startswith("only"):
        return "only"
    raise ValueError(f"Cannot infer condition from {name}")


def _dataset_config() -> dict:
    path = DATA_DIR / "dataset_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


_CFG = _dataset_config()
PARTICIPANT_ALIASES: dict[str, str] = _CFG.get("participant_aliases", {})
RECORDING_SET: dict[str, int] = {k: int(v) for k, v in _CFG.get("recording_set", {}).items()}
_FILE_RE = re.compile(r"_([lr])(?:_(\d+))?$")
MIX_AS_VALIDATION_PAIR = True
MIX_VALIDATION_FEET: set[tuple[str, str]] = set()


def participant_of(subject: str) -> str:
    return PARTICIPANT_ALIASES.get(subject, subject)


def _promote_mix_pairs(trials: list[Trial]) -> list[Trial]:
    have = {(t.subject, t.device, t.foot, t.condition) for t in trials}
    promoted = {
        (t.subject, t.foot) for t in trials
        if t.condition == "mix"
        and {(t.subject, d, t.foot, "mix") for d in ("fscan", "sock")} <= have
        and not {(t.subject, d, t.foot, "only") for d in ("fscan", "sock")} <= have
    }
    MIX_VALIDATION_FEET.clear()
    MIX_VALIDATION_FEET.update(promoted)
    return [replace(t, condition="only") if t.condition == "mix" and (t.subject, t.foot) in promoted else t
            for t in trials]


def discover_trials() -> list[Trial]:
    """Walk <DATA_DIR>/<group>/{f_scan,sock}/<subject>/<file>.csv; file suffix _l/_r sets the foot."""
    trials: list[Trial] = []
    for group_dir in sorted([p for p in DATA_DIR.iterdir() if p.is_dir()]):
        sex = group_dir.name
        for device_dir, device in [(group_dir / "f_scan", "fscan"), (group_dir / "sock", "sock")]:
            if not device_dir.exists():
                continue
            for path in sorted(device_dir.glob("*/*.csv")):
                subject = path.parent.name
                condition = condition_from_name(device, path.name)
                m = _FILE_RE.search(path.stem.lower())
                if not m:
                    raise ValueError(f"Cannot infer foot from {path.name}")
                foot = m.group(1).upper()
                if m.group(2) is not None:
                    if subject not in RECORDING_SET:
                        raise ValueError(f"{path}: numbered set, but dataset_config.json names none for {subject}")
                    if int(m.group(2)) != RECORDING_SET[subject]:
                        continue
                trials.append(Trial(sex, device, subject, condition, foot, path))
    return _promote_mix_pairs(trials) if MIX_AS_VALIDATION_PAIR else trials


# ------------------------------------------------------------------ signal helpers
def peak_index(values: np.ndarray) -> int | None:
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return None
    valid_values = np.where(valid, values, -np.inf)
    idx = int(np.argmax(valid_values))
    return idx if np.isfinite(valid_values[idx]) else None


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(signal.astype(float), kernel, mode="same")


def smooth_edge(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(float)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(signal.astype(float), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def stance_threshold(total: np.ndarray) -> float:
    nonzero = total[total > 0]
    if nonzero.size == 0:
        return math.inf
    p10, p95 = np.percentile(nonzero, [10, 95])
    return float(p10 + 0.35 * (p95 - p10))


def onset_threshold(signal: np.ndarray) -> float:
    nonzero = signal[signal > 0]
    if nonzero.size == 0:
        return math.inf
    p05, p95 = np.percentile(nonzero, [5, 95])
    return float(p05 + 0.05 * (p95 - p05))


def find_stances(frame_df: pd.DataFrame) -> tuple[list[tuple[int, int]], float]:
    """Stance segments from total load, extended to heel-strike onset and toe-off. -> (segments, threshold)."""
    time = frame_df["time_s"].to_numpy()
    total = frame_df["total_pressure"].to_numpy()
    heel = frame_df["heel_pressure"].to_numpy() if "heel_pressure" in frame_df.columns else total
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.01
    smooth = moving_average(total, max(1, round(0.05 / dt)))
    heel_smooth = moving_average(heel, max(1, round(0.03 / dt)))
    heel_slope = np.diff(heel_smooth, prepend=heel_smooth[0]) if len(heel_smooth) else heel_smooth
    threshold = stance_threshold(smooth)
    low_threshold = onset_threshold(smooth)
    heel_threshold = onset_threshold(heel_smooth)
    heel_slope_threshold = onset_threshold(np.clip(heel_slope, 0.0, None))
    active = smooth > threshold

    segments: list[tuple[int, int]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(active)))

    merge_gap = max(1, round(MERGE_GAP_SECONDS / dt))
    merged: list[tuple[int, int]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    min_len = max(2, round(MIN_STANCE_SECONDS / dt))
    filtered_core = [(s, e) for s, e in merged if e - s >= min_len]
    expanded: list[tuple[int, int]] = []
    for idx, (start, end) in enumerate(filtered_core):
        left_bound = expanded[-1][1] if expanded else 0
        right_bound = filtered_core[idx + 1][0] if idx + 1 < len(filtered_core) else len(active)

        expanded_start = start
        heel_onset = None
        search_start = max(left_bound, start - max(2, round(0.20 / dt)))
        for prev in range(start - 1, search_start - 1, -1):
            if heel_slope[prev] > heel_slope_threshold and heel_smooth[prev] > 0:
                heel_onset = prev
            elif heel_onset is not None:
                break
        if heel_onset is not None:
            expanded_start = heel_onset
            while expanded_start > left_bound:
                prev = expanded_start - 1
                if heel_smooth[prev] > 0 and heel_slope[prev] >= 0:
                    expanded_start = prev
                else:
                    break
        else:
            while expanded_start > left_bound:
                prev = expanded_start - 1
                if heel_smooth[prev] > heel_threshold or smooth[prev] > low_threshold:
                    expanded_start = prev
                else:
                    break

        expanded_end = end
        while expanded_end < right_bound and smooth[expanded_end] > low_threshold:
            expanded_end += 1

        if expanded and expanded_start < expanded[-1][1]:
            expanded_start = expanded[-1][1]
        if expanded_end - expanded_start >= min_len:
            expanded.append((expanded_start, expanded_end))

    return expanded, threshold


def peak_time_percent(time: np.ndarray, values: np.ndarray) -> float:
    idx = peak_index(np.where(np.isfinite(time), values, np.nan))
    if idx is None:
        return math.nan
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    if duration <= 0:
        return 0.0
    return float((time[idx] - time[0]) / duration * 100.0)


def half_max_width_percent(curve: np.ndarray) -> float:
    """FWHM of a stance-resampled curve in percent of stance (outermost half-maximum crossings)."""
    valid = np.isfinite(curve)
    if valid.sum() < 2:
        return math.nan
    values = np.where(valid, curve, -np.inf)
    peak = float(values.max())
    if not math.isfinite(peak) or peak <= 0:
        return math.nan
    half = peak / 2.0
    above = np.flatnonzero(values >= half)
    if above.size == 0:
        return math.nan
    n = len(curve)
    step = 100.0 / (n - 1)
    first, last = int(above[0]), int(above[-1])

    def crossing(inside: int, outside: int) -> float:
        a, b = values[outside], values[inside]
        if not (math.isfinite(a) and math.isfinite(b)) or b == a:
            return float(inside)
        return outside + (half - a) / (b - a) * (inside - outside)

    left = crossing(first, first - 1) if first > 0 else float(first)
    right = crossing(last, last + 1) if last < n - 1 else float(last)
    return float((right - left) * step)


def pressure_gradients(curve: np.ndarray, duration: float) -> tuple[float, float]:
    """(max dP/dt, min dP/dt) of an evenly spaced stance-resampled curve, per second."""
    valid = np.isfinite(curve)
    if valid.sum() < 2 or not math.isfinite(duration) or duration <= 0:
        return math.nan, math.nan
    dt = duration / (len(curve) - 1)
    gradient = np.gradient(np.asarray(curve, dtype=float), dt)
    gradient = gradient[np.isfinite(gradient)]
    if gradient.size == 0:
        return math.nan, math.nan
    return float(gradient.max()), float(gradient.min())


def resample_series(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(values)
    time = time[valid]
    values = values[valid]
    if time.size < 2 or time[-1] == time[0]:
        return np.full_like(target, np.nan, dtype=float)
    normalized_time = (time - time[0]) / (time[-1] - time[0])
    unique_time, unique_idx = np.unique(normalized_time, return_index=True)
    if unique_time.size < 2:
        return np.full_like(target, np.nan, dtype=float)
    return np.interp(target, unique_time, values[unique_idx])


def resample_matrix_segment(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(time)
    time = time[valid]
    values = values[valid]
    if time.size < 2 or time[-1] == time[0]:
        return np.full((len(target), values.shape[1]), np.nan, dtype=np.float32)
    normalized_time = (time - time[0]) / (time[-1] - time[0])
    unique_time, unique_idx = np.unique(normalized_time, return_index=True)
    if unique_time.size < 2:
        return np.full((len(target), values.shape[1]), np.nan, dtype=np.float32)
    unique_values = values[unique_idx]
    out = np.empty((len(target), unique_values.shape[1]), dtype=np.float32)
    for col_idx in range(unique_values.shape[1]):
        out[:, col_idx] = np.interp(target, unique_time, unique_values[:, col_idx])
    return out

FEATURE_COLUMNS = [
    "total_pressure", "peak_pressure", "peak_pressure_time_percent", "peak_pressure_ml_coordinate",
    "peak_pressure_ap_coordinate", "pressure_impulse", "mean_pressure", "pressure_gradient_max",
    "pressure_gradient_min", "pressure_fwhm_percent", "mean_to_peak_pressure_ratio",
    "pressure_gradient_max_norm", "pressure_gradient_min_norm", "cop_path_length", "ap_displacement",
    "ml_displacement", "cop_velocity", "cop_area", "stance_duration",
]
