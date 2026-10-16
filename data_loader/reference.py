from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import FSCAN_COLS, FSCAN_ROWS, OUT_DIR, RESAMPLE_POINTS, discover_trials, find_stances
from data_loader.features import read_trial
from data_loader.io import parse_fscan_with_matrices
from utils.resample import resample_matrix

TARGET = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
N_CELLS = FSCAN_ROWS * FSCAN_COLS


def mean_cop_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Mean over stances of the resampled per-stance trajectories, per subject-foot and phase point."""
    grouped = trajectories.groupby(
        ["sex", "device", "subject", "condition", "foot", "point_index", "stance_percent"],
        as_index=False, sort=True,
    )
    return grouped.agg(
        cop_x=("cop_x", "mean"),
        cop_y=("cop_y", "mean"),
        total_pressure=("total_pressure", "mean"),
        total_pressure_norm=("total_pressure_norm", "mean"),
        n_steps=("step_index", "nunique"),
    )


def build_mean_trajectories(trials) -> pd.DataFrame:
    parts = []
    for trial in trials:
        _, trajectories, qc = read_trial(trial)
        parts.append(trajectories)
        print(f"  {trial.subject:12s} {trial.foot} {trial.condition}: {qc['stance_count']:3d} stances")
    return mean_cop_trajectories(pd.concat(parts, ignore_index=True))


def build_mean_fields(trials) -> dict[str, np.ndarray]:
    """Per subject-foot mean stance field (100, 1260): every stance resampled, then averaged."""
    fields = {}
    for trial in trials:
        frame_df, matrices = parse_fscan_with_matrices(Path(trial.path))
        stances, _ = find_stances(frame_df)
        if not stances:
            print(f"  {trial.subject} {trial.foot}: no stances")
            continue
        time_s = frame_df["time_s"].to_numpy()
        flat = matrices.reshape(len(matrices), N_CELLS).astype(np.float64)
        per_stance = [resample_matrix(time_s[s:e], flat[s:e], TARGET) for s, e in stances]
        fields["|".join((trial.sex, trial.subject, trial.foot))] = np.nanmean(per_stance, axis=0)
    return fields


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = [t for t in discover_trials() if t.device == "fscan" and t.condition == "only"]
    print(f"{len(trials)} F-Scan recordings")
    traj = build_mean_trajectories(trials)
    traj.to_csv(OUT_DIR / "mean_cop_trajectories.csv", index=False, encoding="utf-8-sig")
    fields = build_mean_fields(trials)
    np.savez_compressed(OUT_DIR / "fscan_mean_fields.npz", **{k: v.astype(np.float32) for k, v in fields.items()})
    print(f"wrote {OUT_DIR / 'mean_cop_trajectories.csv'} and fscan_mean_fields.npz ({len(fields)} feet)")


if __name__ == "__main__":
    main()
