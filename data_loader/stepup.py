from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from config import STEPUP_DIR as STEPUP
from data_loader.stepup_steps import remap

PY_ROOT = STEPUP / "py"


def load_sex_map() -> dict[str, str]:
    """ParticipantID -> 'male' / 'female' from the dataset's own metadata."""
    meta = pd.read_csv(STEPUP / "participant_metadata.csv")
    out = {}
    for _, r in meta.iterrows():
        s = str(r["Sex"]).strip().lower()
        if s in ("male", "female"):
            out[f"{int(r['ParticipantID']):03d}"] = s
    return out


GRID_ROWS, GRID_COLS = 60, 21
_GRID_RC = np.stack(np.meshgrid(np.arange(GRID_ROWS, dtype=float),
                                np.arange(GRID_COLS, dtype=float),
                                indexing="ij"), axis=-1).reshape(-1, 2)


def _regrid_field(frames: np.ndarray, box, ref) -> np.ndarray:
    """Sample a footstep field onto the shared 60x21 frame through the box mapping.

    Done as one 3-D map_coordinates call rather than one per frame; the spatial
    mapping is identical across frames, and the per-frame loop dominated runtime
    over ~11k footsteps.
    """
    src = remap(_GRID_RC, ref, box)
    t = len(frames)
    n = GRID_ROWS * GRID_COLS
    index = np.stack([
        np.repeat(np.arange(t, dtype=float), n),
        np.tile(src[:, 0], t),
        np.tile(src[:, 1], t),
    ], axis=0)
    out = map_coordinates(frames, index, order=1, mode="constant", cval=0.0)
    return out.reshape(t, GRID_ROWS, GRID_COLS)
