from __future__ import annotations

import argparse
import pickle
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import OUT_DIR
from data_loader.stepup import PY_ROOT, _regrid_field, load_sex_map
from data_loader.footprint import load_reference_box
from data_loader.stepup_steps import contact_box, extract_raw_steps, remap
from data_loader.stepup_steps import field_cop as field_cop_rc
from utils.resample import resample_matrix
from experiments import cop_benchmark as cb

OUT = OUT_DIR / "public_baselines"
FOOTWEAR = ["BF", "ST", "P1", "P2"]
SPEEDS = ["W1", "W3", "W4"]
SAMPLE_HZ = 100.0
FLOOR_KPA = 20.0           # F-SCAN cleaning floor (FSCAN_MIN_CLEANED_PRESSURE / 1000)
MIN_STANCES = 5
N_FIELDS = 8               # resampled reference fields kept per foot for the LSTM / cGAN
FOREST_STANCES = 4         # the cascade forest sees 240 training feet here, not 24


def native_blocks(box, uv, sizes, shape):
    r0, r1, c0, c1 = box
    out = []
    for (u, v), (w, h) in zip(uv, sizes):
        rc, cc = r0 + v * (r1 - r0), c0 + u * (c1 - c0)
        rs = np.clip(np.arange(h) + int(round(rc - (h - 1) / 2)), 0, shape[0] - 1)
        cs = np.clip(np.arange(w) + int(round(cc - (w - 1) / 2)), 0, shape[1] - 1)
        rr, c2 = np.meshgrid(np.unique(rs), np.unique(cs), indexing="ij")
        out.append((rr * shape[1] + c2).ravel())
    return out


def to_ref_xy(rc, box, ref_box):
    m = remap(rc, box, ref_box)
    return np.column_stack([m[:, 1] + 1.0, m[:, 0] + 1.0])


def extract_subject(args):
    subject, sex, footwear, speed, ref_box = args
    d = PY_ROOT / subject / footwear / speed
    if not (d / "trial.npz").exists():
        return []
    trial = np.load(d / "trial.npz")["arr_0"]
    meta = pd.read_csv(d / "metadata.csv")
    layouts = {lay: cb.layout_spec(lay) for lay in cb.layouts_used()}
    r0, r1, c0, c1 = ref_box
    vxy = {lay: np.column_stack([c0 + uv[:, 0] * (c1 - c0) + 1.0, r0 + uv[:, 1] * (r1 - r0) + 1.0])
           for lay, (uv, _) in layouts.items()}
    out = []
    for side in ("Left", "Right"):
        steps = []
        for fs in extract_raw_steps(trial, meta, side, resample=False):
            fs = np.where(fs < FLOOR_KPA, 0.0, fs)
            box = contact_box(fs.max(axis=0))
            if box is not None:
                steps.append((fs, box))
        if len(steps) < MIN_STANCES:
            continue
        pick = np.unique(np.linspace(0, len(steps) - 1, min(N_FIELDS, len(steps))).astype(int))
        vch = {lay: [] for lay in layouts}
        cop, pp, pti, fields, stances, cursor, vcells = [], [], [], [], [], 0, {}
        for i, (fs, box) in enumerate(steps):
            T = len(fs)
            flat = fs.reshape(T, -1)
            tt = np.arange(T) / SAMPLE_HZ
            for lay, (uv, s) in layouts.items():
                blocks = native_blocks(box, uv, s, fs.shape[1:])
                vch[lay].append(np.stack([flat[:, b].mean(axis=1) for b in blocks], axis=1).astype(np.float32))
                vcells.setdefault(lay, len(np.unique(np.concatenate(blocks))))
            cop.append(to_ref_xy(field_cop_rc(fs), box, ref_box))
            pp.append(float(fs.max()))
            pti.append(float(np.trapezoid(flat, tt, axis=0).max()))
            if i in pick:
                g = _regrid_field(fs, box, ref_box).reshape(T, cb.N_CELLS)
                fields.append(resample_matrix(tt, g, cb.TARGET))
            stances.append((cursor, cursor + T))
            cursor += T
        time_s = np.arange(cursor) / SAMPLE_HZ
        traj = cb.stance_trajs(time_s, stances, np.concatenate(cop))
        out.append(dict(
            sex=sex, subject=subject, foot="L" if side == "Left" else "R", time=time_s, stances=stances,
            box=ref_box, vch={lay: np.concatenate(v) for lay, v in vch.items()}, vxy=vxy, vcells=vcells,
            traj=traj, mean=np.nanmean(traj, axis=0), pp=np.asarray(pp), pti=np.asarray(pti),
            fields=np.asarray(fields, dtype=np.float16), field_idx=pick))
    del trial
    return out


def route_foot(r) -> cb.RouteFoot:
    v = cb.RouteFoot(r["sex"], r["subject"], r["foot"], r["time"], r["stances"])
    v.ch, v.xy, v.cells = dict(r["vch"]), dict(r["vxy"]), dict(r["vcells"])
    v.ref_traj, v.ref_mean = r["traj"], r["mean"]
    v.ref_fields, v.field_idx = r["fields"], r["field_idx"]
    v.ref_pp, v.ref_pti = float(np.nanmean(r["pp"])), float(np.nanmean(r["pti"]))
    v.ref_pp_st, v.ref_pti_st = r["pp"], r["pti"]
    tot = v.ch["ours26"].sum(axis=1)
    v.scale = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
    return v


def make_folds(subjects, k, seed=0):
    perm = np.random.default_rng(seed).permutation(sorted(subjects))
    return [sorted(perm[i::k].tolist()) for i in range(k)]


def load_condition(footwear, speed, subjects, sex_map, ref_box, workers, tag, rebuild):
    cache = OUT / "cache" / f"{tag}.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as fh:
            recs = pickle.load(fh)
        if recs and set(cb.layouts_used()) <= set(recs[0]["vch"]):
            return recs
        print(f"    [{tag}] cache predates a layout; re-extracting", flush=True)
    jobs = [(s, sex_map.get(s, "unknown"), footwear, speed, ref_box) for s in subjects]
    recs = []
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap_unordered(extract_subject, jobs), start=1):
            recs.extend(part)
            if i % 25 == 0 or i == len(jobs):
                print(f"    [{tag}] extracted {i}/{len(jobs)} participants, {len(recs)} feet", flush=True)
    recs.sort(key=lambda r: (r["subject"], r["foot"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(recs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return recs


def rebuild_summary(suffix, lim):
    want = {m[0] for m in cb.METHODS if not (suffix == "_fast" and m[3] in cb.DEEP)}
    cops, scals = [], []
    for fw in FOOTWEAR:
        for sp in ("W1", "W2", "W3", "W4"):
            d = OUT / f"{fw}_{sp}{lim}"
            pc, ps = d / f"baseline_cop_virtual{suffix}.csv", d / f"baseline_scalar_virtual{suffix}.csv"
            if not (pc.exists() and ps.exists()):
                continue
            cop, scal = pd.read_csv(pc), pd.read_csv(ps)
            if not want <= set(cop.method):
                continue
            n = int(cop.n_feet.max())
            cops.append(cop.assign(footwear=fw, speed=sp, n_feet_condition=n))
            scals.append(scal.assign(footwear=fw, speed=sp, n_feet_condition=n))
    if cops:
        pd.concat(cops).to_csv(OUT / f"summary_cop{suffix}{lim}.csv", index=False, encoding="utf-8-sig")
        pd.concat(scals).to_csv(OUT / f"summary_scalar{suffix}{lim}.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--footwear", nargs="+", default=FOOTWEAR, choices=FOOTWEAR)
    ap.add_argument("--speeds", nargs="+", default=SPEEDS, choices=["W1", "W2", "W3", "W4"])
    ap.add_argument("--skip-deep", action="store_true", help="skip cGAN / LSTM / deep forest")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3, help="extraction processes (~3.5 GB RAM each)")
    ap.add_argument("--limit-subjects", type=int, default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                    help="run only these cop_benchmark.METHODS names and merge them into each condition's CSVs")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cb.POINT_STANCES["forest"] = FOREST_STANCES

    sex_map, ref_box = load_sex_map(), load_reference_box()
    subjects = sorted([p.name for p in PY_ROOT.iterdir() if p.is_dir()], key=int)
    if a.limit_subjects:
        subjects = subjects[:a.limit_subjects]
    lim = f"_limit{a.limit_subjects}" if a.limit_subjects else ""
    suffix = "_fast" if a.skip_deep else ""
    for fw in a.footwear:
        for sp in a.speeds:
            tag, t0 = f"{fw}_{sp}{lim}", time.time()
            print(f"\n===== {tag} =====", flush=True)
            recs = load_condition(fw, sp, subjects, sex_map, ref_box, a.workers, tag, a.rebuild_cache)
            feet = [route_foot(r) for r in recs]
            n_sub = len({f.subject for f in feet})
            folds = make_folds({f.subject for f in feet}, min(a.folds, n_sub))
            print(f"  {len(feet)} feet, {n_sub} participants, {sum(len(f.stances) for f in feet)} stances",
                  flush=True)
            rows = cb.run_route(feet, "virtual", a.skip_deep, folds, only=a.methods)
            cop, scal, perfoot = cb.summarise(rows, "virtual")
            d = OUT / tag
            d.mkdir(parents=True, exist_ok=True)
            for name, df in (("cop", cop), ("scalar", scal), ("per_foot", perfoot)):
                path = d / f"baseline_{name}_virtual{suffix}.csv"
                if a.methods:
                    df = cb.merge_into(path, df, a.methods)
                df.to_csv(path, index=False, encoding="utf-8-sig")
            rebuild_summary(suffix, lim)
            print(cop[["method", "cop_x_rmse_mm", "cop_y_rmse_mm", "cop_x_r", "cop_y_r", "cop_x_dev_r",
                       "cop_x_ba_bias", "cop_x_ba_lo", "cop_x_ba_hi"]].round(3).to_string(index=False))
            print(scal[["method", "metric", "r", "icc_abs", "rmse", "ba_bias"]].round(3).to_string(index=False))
            print(f"  {tag} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("\nWrote results/public_baselines/")


if __name__ == "__main__":
    main()
