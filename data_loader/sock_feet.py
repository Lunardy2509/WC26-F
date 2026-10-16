from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from config import (
    MIRROR_SOCK_LEFT_FOOT, OUT_DIR, RESAMPLE_POINTS, SOCK_COORDS_FULL, SOCK_ML_MIDLINE, SOCK_RAW_SCALE,
    WARMUP_SECONDS, clean_sock_values, discover_trials, find_stances, resample_series,
)

TARGET = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
N_CH = len(SOCK_COORDS_FULL)
# The diagram map: Data13 takes the position the legacy table gives Data14, and
# so on. Rows 1-12 and 19-26 are unchanged.
_BLOCK = [12, 13, 14, 15, 16, 17]
_DIAGRAM_SOURCE = [13, 14, 12, 16, 17, 15]


def diagram_coords() -> np.ndarray:
    c = SOCK_COORDS_FULL.copy()
    c[_BLOCK] = SOCK_COORDS_FULL[_DIAGRAM_SOURCE]
    return c


COORD_MAPS = {"legacy": SOCK_COORDS_FULL.copy(), "diagram": diagram_coords()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ loading
class Foot:
    """One subject-foot: cleaned 26-channel frames, stances, and the F-SCAN target."""

    def __init__(self, trial, ref: np.ndarray, field: np.ndarray | None):
        df = pd.read_csv(trial.path)
        v = df[[f"Data{i}" for i in range(1, N_CH + 1)]].astype(float).to_numpy() * SOCK_RAW_SCALE
        v = clean_sock_values(v)
        t = (df["Time(ms)"].astype(float).to_numpy() - float(df["Time(ms)"].iloc[0])) / 1000.0
        keep = t >= WARMUP_SECONDS
        self.values, self.time_s = v[keep], t[keep]
        self.sex, self.subject, self.foot = trial.sex, trial.subject, trial.foot
        self.ref, self.field = ref, field

        # Stances come from all 26 channels and are shared by every rule, so a
        # rule changes the COP decoder and nothing else. Heel (AP >= 40) is
        # Data20..Data26 under either map.
        heel = self.values[:, SOCK_COORDS_FULL[:, 1] >= 40.0].sum(axis=1)
        fdf = pd.DataFrame({"time_s": self.time_s, "total_pressure": self.values.sum(axis=1),
                            "heel_pressure": heel})
        self.stances, _ = find_stances(fdf)

        # Surrogate tensor (S, 100, 26): every channel resampled onto stance time.
        X = []
        for s, e in self.stances:
            tt = self.time_s[s:e]
            X.append(np.stack([resample_series(tt, self.values[s:e, j], TARGET)
                               for j in range(N_CH)], axis=1))
        self.X = np.nan_to_num(np.asarray(X, dtype=float))

    def coords(self, cmap: str) -> np.ndarray:
        c = COORD_MAPS[cmap].copy()
        if MIRROR_SOCK_LEFT_FOOT and self.foot == "L":
            c[:, 0] = 2.0 * SOCK_ML_MIDLINE - c[:, 0]
        return c


def load_feet() -> list[Foot]:
    """Sock feet with an F-Scan reference; run `python -m data_loader.reference` first."""
    traj = pd.read_csv(OUT_DIR / "mean_cop_trajectories.csv")
    traj = traj[(traj.device == "fscan") & (traj.condition == "only")]
    truth = {}
    for key, g in traj.groupby(["sex", "subject", "foot"], sort=True):
        g = g.sort_values("point_index")
        truth[key] = np.column_stack([g["cop_x"].to_numpy(), g["cop_y"].to_numpy()])
    fields_path = OUT_DIR / "fscan_mean_fields.npz"
    fields = np.load(fields_path) if fields_path.exists() else None

    feet = []
    for t in discover_trials():
        if not (t.device == "sock" and t.condition == "only"):
            continue
        key = (t.sex, t.subject, t.foot)
        if key not in truth:
            continue
        fk = "|".join(key)
        field = fields[fk].astype(float) if fields is not None and fk in fields.files else None
        f = Foot(t, truth[key], field)
        if f.stances:
            feet.append(f)
    return feet
