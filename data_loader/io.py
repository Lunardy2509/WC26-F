from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import *


def parse_fscan(path: Path) -> pd.DataFrame:
    times: list[float] = []
    totals: list[float] = []
    peaks: list[float] = []
    peak_xs: list[float] = []
    peak_ys: list[float] = []
    heels: list[float] = []
    cop_xs: list[float] = []
    cop_ys: list[float] = []
    seconds_per_frame = 0.01
    frame_re = re.compile(r"^Frame\s+\d+,\s*([^,]*)")

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        lines = iter(handle)
        for line in lines:
            if line.startswith("SECONDS_PER_FRAME"):
                try:
                    seconds_per_frame = float(line.strip().split()[-1])
                except ValueError:
                    pass
                continue

            match = frame_re.match(line)
            if not match:
                continue

            time_text = match.group(1).strip()
            time_s = float(time_text) if time_text else len(times) * seconds_per_frame
            values: list[float] = []
            for _ in range(FSCAN_ROWS):
                row = next(lines).strip().split(",")
                for cell in row[:FSCAN_COLS]:
                    if cell == "B" or cell == "":
                        values.append(0.0)
                    else:
                        values.append(float(cell) * FSCAN_RAW_SCALE)

            arr = np.asarray(values, dtype=float)
            arr[arr < FSCAN_MIN_CLEANED_PRESSURE] = 0.0
            total = float(arr.sum())
            peak = float(arr.max(initial=0.0))
            if total > 0:
                cop_x = float(np.dot(arr, FSCAN_X) / total)
                cop_y = float(np.dot(arr, FSCAN_Y) / total)
            else:
                cop_x = math.nan
                cop_y = math.nan
            if peak > 0:
                peak_idx = int(np.argmax(arr))
                peak_x = float(FSCAN_X[peak_idx])
                peak_y = float(FSCAN_Y[peak_idx])
            else:
                peak_x = math.nan
                peak_y = math.nan

            times.append(time_s)
            totals.append(total)
            peaks.append(peak)
            peak_xs.append(peak_x)
            peak_ys.append(peak_y)
            heels.append(float(arr[FSCAN_HEEL_MASK].sum()))
            cop_xs.append(cop_x)
            cop_ys.append(cop_y)

    return pd.DataFrame(
        {
            "time_s": np.asarray(times, dtype=float),
            "total_pressure": np.asarray(totals, dtype=float),
            "peak_pressure": np.asarray(peaks, dtype=float),
            "peak_x": np.asarray(peak_xs, dtype=float),
            "peak_y": np.asarray(peak_ys, dtype=float),
            "heel_pressure": np.asarray(heels, dtype=float),
            "cop_x": np.asarray(cop_xs, dtype=float),
            "cop_y": np.asarray(cop_ys, dtype=float),
        }
    )


def parse_fscan_with_matrices(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    times: list[float] = []
    totals: list[float] = []
    peaks: list[float] = []
    peak_xs: list[float] = []
    peak_ys: list[float] = []
    heels: list[float] = []
    cop_xs: list[float] = []
    cop_ys: list[float] = []
    matrices: list[np.ndarray] = []
    seconds_per_frame = 0.01
    frame_re = re.compile(r"^Frame\s+\d+,\s*([^,]*)")

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        lines = iter(handle)
        for line in lines:
            if line.startswith("SECONDS_PER_FRAME"):
                try:
                    seconds_per_frame = float(line.strip().split()[-1])
                except ValueError:
                    pass
                continue

            match = frame_re.match(line)
            if not match:
                continue

            time_text = match.group(1).strip()
            time_s = float(time_text) if time_text else len(times) * seconds_per_frame
            values: list[float] = []
            for _ in range(FSCAN_ROWS):
                row = next(lines).strip().split(",")
                for cell in row[:FSCAN_COLS]:
                    if cell == "B" or cell == "":
                        values.append(0.0)
                    else:
                        values.append(float(cell) * FSCAN_RAW_SCALE)

            arr = np.asarray(values, dtype=np.float32)
            arr[arr < FSCAN_MIN_CLEANED_PRESSURE] = 0.0
            total = float(arr.sum())
            peak = float(arr.max(initial=0.0))
            if total > 0:
                cop_x = float(np.dot(arr, FSCAN_X) / total)
                cop_y = float(np.dot(arr, FSCAN_Y) / total)
            else:
                cop_x = math.nan
                cop_y = math.nan
            if peak > 0:
                peak_idx = int(np.argmax(arr))
                peak_x = float(FSCAN_X[peak_idx])
                peak_y = float(FSCAN_Y[peak_idx])
            else:
                peak_x = math.nan
                peak_y = math.nan

            times.append(time_s)
            totals.append(total)
            peaks.append(peak)
            peak_xs.append(peak_x)
            peak_ys.append(peak_y)
            heels.append(float(arr[FSCAN_HEEL_MASK].sum()))
            cop_xs.append(cop_x)
            cop_ys.append(cop_y)
            matrices.append(arr.reshape(FSCAN_ROWS, FSCAN_COLS))

    frame_df = pd.DataFrame(
        {
            "time_s": np.asarray(times, dtype=float),
            "total_pressure": np.asarray(totals, dtype=float),
            "peak_pressure": np.asarray(peaks, dtype=float),
            "peak_x": np.asarray(peak_xs, dtype=float),
            "peak_y": np.asarray(peak_ys, dtype=float),
            "heel_pressure": np.asarray(heels, dtype=float),
            "cop_x": np.asarray(cop_xs, dtype=float),
            "cop_y": np.asarray(cop_ys, dtype=float),
        }
    )
    return frame_df, np.stack(matrices).astype(np.float32)


def parse_sock(path: Path, foot: str, trial: Trial | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    data_cols = [f"Data{i}" for i in range(1, 27)]
    values = df[data_cols].astype(float).to_numpy() * SOCK_RAW_SCALE
    values = clean_sock_values(values, trial)
    values = select_sock_sensor_values(values)
    coords = sock_coords_for_foot(foot)
    heel_mask = coords[:, 1] >= 40.0
    totals = values.sum(axis=1)
    peaks = values.max(axis=1)
    peak_indices = np.argmax(values, axis=1)
    peak_x = coords[peak_indices, 0].astype(float)
    peak_y = coords[peak_indices, 1].astype(float)
    peak_x[peaks <= 0] = math.nan
    peak_y[peaks <= 0] = math.nan
    heels = values[:, heel_mask].sum(axis=1)
    weighted_x = values @ coords[:, 0]
    weighted_y = values @ coords[:, 1]
    cop_x = np.divide(weighted_x, totals, out=np.full_like(totals, np.nan, dtype=float), where=totals > 0)
    cop_y = np.divide(weighted_y, totals, out=np.full_like(totals, np.nan, dtype=float), where=totals > 0)
    time_s = (df["Time(ms)"].astype(float).to_numpy() - float(df["Time(ms)"].iloc[0])) / 1000.0
    return pd.DataFrame(
        {
            "time_s": time_s,
            "total_pressure": totals.astype(float),
            "peak_pressure": peaks.astype(float),
            "peak_x": peak_x,
            "peak_y": peak_y,
            "heel_pressure": heels.astype(float),
            "cop_x": cop_x.astype(float),
            "cop_y": cop_y.astype(float),
        }
    )
