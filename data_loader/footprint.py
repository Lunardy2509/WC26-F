from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import DATA_DIR, OUT_DIR, ROOT, FSCAN_COLS, FSCAN_ROWS

CACHE_PATH = OUT_DIR / "fscan_reference_footprint.json"
REFERENCE_BOX_PATH = ROOT / "configs" / "fscan_reference_footprint.json"
# Fraction of a recording's peak cell pressure below which a cell is treated as
# unloaded. Guards against isolated hot cells dragging the box to the frame edge.
ACTIVE_FRACTION = 0.02


def footprint_box(peak_map: np.ndarray, fraction: float = ACTIVE_FRACTION) -> tuple[float, float, float, float]:
    """(row_min, row_max, col_min, col_max) of the loaded region, in grid indices."""
    peak = float(peak_map.max())
    if peak <= 0.0:
        raise ValueError("peak map is empty")
    active = peak_map > peak * fraction
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]
    return float(rows.min()), float(rows.max()), float(cols.min()), float(cols.max())


def measure_cohort(data_root: Path) -> dict:
    from data_loader.io import parse_fscan_with_matrices

    files = sorted(data_root.glob("*/f_scan/*/*.csv"))
    if not files:
        raise FileNotFoundError(f"no F-SCAN csv under {data_root}")

    per_trial = []
    for path in files:
        _, matrices = parse_fscan_with_matrices(path)
        peak_map = matrices.reshape(len(matrices), FSCAN_ROWS, FSCAN_COLS).max(axis=0)
        try:
            box = footprint_box(peak_map)
        except ValueError:
            print(f"  skip (no load): {path}")
            continue
        per_trial.append(
            {
                "subject": path.parent.name,
                "trial": path.name,
                "row_min": box[0],
                "row_max": box[1],
                "col_min": box[2],
                "col_max": box[3],
                "rows": box[1] - box[0],
                "cols": box[3] - box[2],
            }
        )
        print(f"  {path.parent.name:12s} {path.name:22s} rows[{box[0]:.0f}..{box[1]:.0f}] cols[{box[2]:.0f}..{box[3]:.0f}]")

    arr = {key: np.array([t[key] for t in per_trial], dtype=float) for key in ("row_min", "row_max", "col_min", "col_max")}
    return {
        "n_trials": len(per_trial),
        "active_fraction": ACTIVE_FRACTION,
        "mean_box": {key: float(values.mean()) for key, values in arr.items()},
        "sd_box": {key: float(values.std(ddof=1)) for key, values in arr.items()},
        "per_trial": per_trial,
    }


def load_reference_box() -> tuple[float, float, float, float]:
    """Mean real-F-Scan footprint box, as (row_min, row_max, col_min, col_max).

    Reads configs/fscan_reference_footprint.json, the box used in the paper. `python -m
    data_loader.footprint` re-measures it from the data into PPS_OUT_DIR; copy that file over the
    config to use it.
    """
    box = json.loads(REFERENCE_BOX_PATH.read_text())["mean_box"]
    return box["row_min"], box["row_max"], box["col_min"], box["col_max"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(DATA_DIR))
    parser.add_argument("--show", action="store_true", help="print the cached box and exit")
    args = parser.parse_args()

    if args.show:
        box = load_reference_box()
        print(f"rows [{box[0]:.2f} .. {box[1]:.2f}]  cols [{box[2]:.2f} .. {box[3]:.2f}]")
        return

    result = measure_cohort(Path(args.data_root))
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result, indent=2))

    mean, sd = result["mean_box"], result["sd_box"]
    print(f"\n{result['n_trials']} trials")
    print(f"  rows [{mean['row_min']:.2f} .. {mean['row_max']:.2f}]  (SD {sd['row_min']:.2f} / {sd['row_max']:.2f})")
    print(f"  cols [{mean['col_min']:.2f} .. {mean['col_max']:.2f}]  (SD {sd['col_min']:.2f} / {sd['col_max']:.2f})")
    print(f"  wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
