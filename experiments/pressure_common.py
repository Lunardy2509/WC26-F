from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
from scipy import optimize, signal
from scipy import stats as sstats

from data_loader.io import parse_fscan_with_matrices
from utils.agreement import icc21
from config import (
    FSCAN_COLS, FSCAN_ROWS, OUT_DIR, SOCK_COORDS_FULL, SOCK_RAW_SCALE, WARMUP_SECONDS,
    clean_sock_values, discover_trials, find_stances,
)
from data_loader.participants import body_mass_kg
from experiments.cop_benchmark import TEMPLATE_UV, bland_altman_rm, block, canon_field

N_CH = 26
N_CELLS = FSCAN_ROWS * FSCAN_COLS
CELL_M2 = (0.508e-2) ** 2            # F-SCAN sensel area
G = 9.81
SITE_CELLS = (2, 2)                  # reference block under one sock dot (~1 cm)
CAL_TOLERANCE = 0.30                 # R3: implied / measured mass must lie within 1 +- this
FS = 100.0                           # uniform resampling rate, Hz (the sock's nominal rate)
OUT = OUT_DIR / "pressure_validation"
CACHE = OUT / "cache"

# DSP parameters, fixed before any agreement was computed.
HAMPEL_HALF = 3                      # samples each side
HAMPEL_K = 3.0                       # MADs
LOWPASS_HZ = 20.0
LOWPASS_ORDER = 4
BASELINE_WIN_S = 4.0                 # spans 3-4 gait cycles, so it always contains swing
BASELINE_Q = 0.10
GATE_K = 3.0                         # noise-gate multiple of swing-phase sigma
QM_LEVELS = np.linspace(0.0, 1.0, 201)

REGIONS = ["hallux", "lesser_toes", "MTH1", "MTH2_3", "MTH4_5", "midfoot", "heel"]
FOREFOOT = ["MTH1", "MTH2_3", "MTH4_5"]
_trap = np.trapezoid


def region_of(u: float, v: float) -> str:
    if v < 0.20:
        return "hallux" if u < 0.30 else "lesser_toes"
    if v < 0.40:
        return "MTH1" if u < 0.30 else ("MTH2_3" if u < 0.55 else "MTH4_5")
    if v < 0.68:
        return "midfoot"
    return "heel"


SOCK_REGION = np.array([region_of(u, v) for u, v in TEMPLATE_UV])


# ------------------------------------------------------------------ outcomes from a site signal
def stance_outcomes(time: np.ndarray, stances, X: np.ndarray, regions: np.ndarray) -> dict:
    pp, pti, imax, imax_reg, load, dur = [], [], [], [], [], []
    for s, e in stances:
        seg, tt = X[s:e], time[s:e]
        pp.append(seg.max(axis=0))
        pti.append(_trap(seg, tt, axis=0))
        imax.append(_trap(seg.max(axis=1), tt))
        imax_reg.append([_trap(seg[:, regions == g].max(axis=1), tt) if (regions == g).any() else np.nan
                         for g in REGIONS])
        load.append(_trap(seg.sum(axis=1), tt))
        dur.append(tt[-1] - tt[0])
    return dict(site_pp=np.asarray(pp), site_pti=np.asarray(pti), imax=np.asarray(imax),
                imax_reg=np.asarray(imax_reg), load=np.asarray(load), dur=np.asarray(dur))


def regional_max(site_vals: np.ndarray, regions: np.ndarray) -> np.ndarray:
    return np.stack([site_vals[:, regions == g].max(axis=1) if (regions == g).any()
                     else np.full(len(site_vals), np.nan) for g in REGIONS], axis=1)


# ------------------------------------------------------------------ reference
def reference_record(F: np.ndarray, time: np.ndarray, stances) -> dict:
    in_st = np.zeros(len(F), bool)
    for s, e in stances:
        in_st[s:e] = True
    peak = F[in_st].max(axis=0).reshape(FSCAN_ROWS, FSCAN_COLS)
    act = peak > 0.02 * peak.max()
    r_, c_ = np.where(act.any(1))[0], np.where(act.any(0))[0]
    box = (r_.min(), r_.max(), c_.min(), c_.max())
    rr, cc = np.meshgrid(np.arange(FSCAN_ROWS), np.arange(FSCAN_COLS), indexing="ij")
    u = (cc.ravel() - box[2]) / max(box[3] - box[2], 1)
    v = (rr.ravel() - box[0]) / max(box[1] - box[0], 1)
    inbox = (u >= -0.05) & (u <= 1.05) & (v >= -0.05) & (v <= 1.05)
    creg = np.where(inbox, np.array([region_of(a, b) for a, b in zip(u, v)]), "outside")
    full = stance_outcomes(time, stances, F, creg)
    sites = np.stack([F[:, block(box, q, SITE_CELLS)[0]].mean(axis=1) for q in TEMPLATE_UV], axis=1)
    site = stance_outcomes(time, stances, sites, SOCK_REGION)
    starts = np.array([s for s, _ in stances])
    cycle = float(np.median(np.diff(time[starts]))) if len(starts) > 2 else math.nan
    duty = float(np.median(full["dur"])) / cycle if cycle and np.isfinite(cycle) else math.nan
    mean_force = float(np.mean(full["load"] / full["dur"])) * 1000.0 * CELL_M2      # N
    return dict(
        box=box, time=time, stances=stances, sites=sites.astype(np.float32),
        dur=full["dur"],
        pp=full["site_pp"].max(axis=1),                        # whole-foot peak-site PP
        pti=full["site_pti"].max(axis=1),                      # whole-foot max site-specific PTI
        imax=full["imax"], load=full["load"],
        reg_pp=regional_max(full["site_pp"], creg), reg_pti=regional_max(full["site_pti"], creg),
        reg_imax=full["imax_reg"], site_out=site,
        duty=duty, mean_force_n=mean_force, implied_kg=mean_force * 2.0 * duty / G,
    )


def load_reference(refresh: bool = False) -> dict:
    path = CACHE / "reference.pkl"
    if path.exists() and not refresh:
        return pickle.load(open(path, "rb"))
    ref = {}
    for t in discover_trials():
        if not (t.device == "fscan" and t.condition == "only"):
            continue
        fdf, mats = parse_fscan_with_matrices(t.path)
        keep = fdf["time_s"].to_numpy() >= WARMUP_SECONDS
        fdf = fdf[keep].reset_index(drop=True)
        F = canon_field(mats[keep].reshape(-1, N_CELLS) / 1000.0, t.foot).astype(np.float32)
        stances, _ = find_stances(fdf)
        if not stances:
            continue
        ref[(t.sex, t.subject, t.foot)] = reference_record(F, fdf["time_s"].to_numpy(), stances)
        print(f"  F-SCAN {t.subject:12s} {t.foot}: {len(stances)} stances", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(ref, open(path, "wb"))
    return ref


def reference_qc(ref: dict) -> pd.DataFrame:
    rows = [dict(sex=k[0], subject=k[1], foot=k[2], implied_kg=r["implied_kg"], mass_kg=body_mass_kg(k[1]),
                 duty=r["duty"], mean_force_n=r["mean_force_n"], pp=r["pp"].mean(), pti=r["pti"].mean())
            for k, r in ref.items()]
    q = pd.DataFrame(rows)
    q["implied_over_true"] = q.implied_kg / q.mass_kg
    q["R1_implausible_mass"] = q.implied_kg < 30.0
    q["R2_lr_pp_mismatch"] = False
    med = q.pp.median()
    for _, g in q.groupby("subject"):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        if (abs(math.log(a.pp / b.pp)) > math.log(2.0)
                and abs(math.log(a.implied_kg / b.implied_kg)) < math.log(1.5)):
            far = g.index[np.argmax(np.abs(np.log(g.pp / med)).to_numpy())]
            q.loc[far, "R2_lr_pp_mismatch"] = True
    q["qc_pass"] = ~(q.R1_implausible_mass | q.R2_lr_pp_mismatch)
    q["R3_calibration_off_30pct"] = (q.implied_over_true - 1.0).abs() > CAL_TOLERANCE
    q["cal30_pass"] = q.implied_over_true.notna() & ~q.R3_calibration_off_30pct
    return q


_SCALED = ("sites", "pp", "pti", "imax", "load", "reg_pp", "reg_pti", "reg_imax")


def bodyweight_reference(ref: dict) -> dict:
    out = {}
    for k, r in ref.items():
        m = body_mass_kg(k[1])
        if not (np.isfinite(m) and r["implied_kg"] > 0):
            continue
        c = m / r["implied_kg"]
        new = dict(r)
        for f in _SCALED:
            new[f] = r[f] * c
        new["site_out"] = {f: (v if f == "dur" else v * c) for f, v in r["site_out"].items()}
        new.update(mean_force_n=r["mean_force_n"] * c, implied_kg=m, bw_scale=c)
        out[k] = new
    return out


def sock_bodyweight_scale(s: dict, chain: str, subject: str) -> float:
    o = stance_outcomes(s["time"], s["stances"], s[chain].astype(float), SOCK_REGION)
    starts = np.array([a for a, _ in s["stances"]])
    cycle = float(np.median(np.diff(s["time"][starts]))) if len(starts) > 2 else math.nan
    duty = float(np.median(o["dur"])) / cycle
    implied = float(np.mean(o["load"] / o["dur"])) * 2.0 * duty
    return body_mass_kg(subject) / implied if implied > 0 else math.nan


def bodyweight_factors(sock: dict, keys) -> dict:
    return {k: sock_bodyweight_scale(sock[k], "dsp", k[1]) for k in keys}


# ------------------------------------------------------------------ sock
def _hampel(x: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x)
    w = 2 * HAMPEL_HALF + 1
    med = df.rolling(w, center=True, min_periods=1).median()
    mad = (df - med).abs().rolling(w, center=True, min_periods=1).median() * 1.4826
    out = df.where((df - med).abs() <= HAMPEL_K * mad.clip(lower=1e-9), med)
    return out.to_numpy()


def dsp_clean(time: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, dict]:
    grid = np.arange(time[0], time[-1], 1.0 / FS)
    u = np.stack([np.interp(grid, time, raw[:, j]) for j in range(raw.shape[1])], axis=1)
    despiked = _hampel(u)
    spikes = np.mean(despiked != u, axis=0)
    sos = signal.butter(LOWPASS_ORDER, LOWPASS_HZ, fs=FS, output="sos")
    u = signal.sosfiltfilt(sos, despiked, axis=0)
    hf = np.std(despiked - u, axis=0)
    win = int(BASELINE_WIN_S * FS)
    base = (pd.DataFrame(u).rolling(win, center=True, min_periods=win // 4).quantile(BASELINE_Q)
            .rolling(win, center=True, min_periods=1).median().to_numpy())
    d = u - base
    tot = d.sum(axis=1)
    swing = tot <= np.percentile(tot, 25)                 # unloaded frames of the whole foot
    sigma = 1.4826 * np.median(np.abs(d[swing] - np.median(d[swing], axis=0)), axis=0)
    gated = np.where(d > GATE_K * sigma, d, 0.0)
    out = np.stack([np.interp(time, grid, gated[:, j]) for j in range(raw.shape[1])], axis=1)
    info = dict(sigma=sigma, drift=np.ptp(base, axis=0), hf_removed=hf, spike_rate=spikes)
    return np.clip(out, 0.0, None), info


def load_sock(refresh: bool = False) -> dict:
    path = CACHE / "sock.pkl"
    if path.exists() and not refresh:
        return pickle.load(open(path, "rb"))
    out = {}
    for t in discover_trials():
        if not (t.device == "sock" and t.condition == "only"):
            continue
        df = pd.read_csv(t.path)
        raw = df[[f"Data{i}" for i in range(1, N_CH + 1)]].astype(float).to_numpy() * SOCK_RAW_SCALE
        time = (df["Time(ms)"].astype(float).to_numpy() - float(df["Time(ms)"].iloc[0])) / 1000.0
        # Stances exactly as data_loader/sock_feet.Foot derives them.
        pipe = clean_sock_values(raw)
        dsp, info = dsp_clean(time, raw)
        keep = time >= WARMUP_SECONDS
        time, pipe, dsp = time[keep], pipe[keep], dsp[keep]
        heel = pipe[:, SOCK_COORDS_FULL[:, 1] >= 40.0].sum(axis=1)
        stances, _ = find_stances(pd.DataFrame({"time_s": time, "total_pressure": pipe.sum(axis=1),
                                                "heel_pressure": heel}))
        if not stances:
            continue
        out[(t.sex, t.subject, t.foot)] = dict(time=time, stances=stances, pipeline=pipe.astype(np.float32),
                                               dsp=dsp.astype(np.float32), dsp_info=info)
        print(f"  sock   {t.subject:12s} {t.foot}: {len(stances)} stances", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


# ------------------------------------------------------------------ software calibration
def stance_frames(X: np.ndarray, stances) -> np.ndarray:
    return np.concatenate([X[s:e] for s, e in stances], axis=0)


def quantile_functions(X: np.ndarray, stances) -> np.ndarray:
    return np.quantile(stance_frames(X, stances), QM_LEVELS, axis=0).T


def apply_quantile_map(X: np.ndarray, qs: np.ndarray, qf: np.ndarray) -> np.ndarray:
    out = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        xs, fs = np.maximum.accumulate(qs[j]), np.maximum.accumulate(qf[j])
        pos = xs > 0
        if pos.sum() < 2:
            continue
        xs, fs = xs[pos], fs[pos]
        xs = xs + np.arange(len(xs)) * 1e-9 * max(xs[-1], 1.0)            # strictly increasing
        x = X[:, j]
        y = np.interp(x, xs, fs)
        k = max(2, len(xs) // 10)
        slope = (fs[-1] - fs[-k]) / max(xs[-1] - xs[-k], 1e-9)
        y = np.where(x > xs[-1], fs[-1] + slope * (x - xs[-1]), y)
        out[:, j] = np.where(x > 0, y, 0.0)
    return out


def subjects(keys) -> np.ndarray:
    return np.array([k[1] for k in keys])


def loso_linear(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr = (groups != g) & np.isfinite(x) & np.isfinite(y)
        if tr.sum() < 3:
            continue
        b, a = np.polyfit(x[tr], y[tr], 1)
        pred[groups == g] = a + b * x[groups == g]
    return pred


def loso_template(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(y[groups != g]) for g in groups])


def smooth_max(v: np.ndarray, tau: float) -> np.ndarray:
    m = v.max(axis=-1, keepdims=True)
    return (m + tau * np.log(np.exp((v - m) / tau).sum(axis=-1, keepdims=True)))[..., 0]


def learn_channel_weights(site_vals: np.ndarray, target: np.ndarray, lam: float = 0.1) -> np.ndarray:
    tau = 0.05 * float(np.nanmedian(target))

    def loss(theta):
        p = smooth_max(site_vals * np.exp(theta)[None, :], tau)
        return -ccc(p, target) + lam * float(theta @ theta)

    res = optimize.minimize(loss, np.zeros(site_vals.shape[1]), method="L-BFGS-B")
    return np.exp(res.x)


# ------------------------------------------------------------------ variants
# name -> (signal chain, calibration). "raw" outputs keep ADC units and pass the outcome-level
# LOSO linear map (the paper's method); the others are already in kPa.
# The body-weight step calibration is the PRIMARY PP/PTI method; every other variant is kept as a
# documented alternative.
PRIMARY = "DSP + body-weight step calibration"
VARIANTS = {
    PRIMARY: ("dsp", "bw"),
    "pipeline, raw ADC": ("pipeline", "none"),
    "DSP, raw ADC": ("dsp", "none"),
    "DSP + channel gain equalisation": ("dsp", "gain"),
    "pipeline + quantile map": ("pipeline", "qm_channel"),
    "DSP + quantile map (one curve)": ("dsp", "qm_pooled"),
    "DSP + quantile map (per channel)": ("dsp", "qm_channel"),
}


def variant_units(name: str) -> str:
    return "raw" if VARIANTS[name][1] in ("none", "gain", "bw") else "kPa"


def _fold_calibration(train, chain, how, ref, sock, qs, qf):
    if how == "gain":
        num = np.mean([stance_frames(ref[k]["sites"], ref[k]["stances"]).mean(axis=0) for k in train], axis=0)
        den = np.mean([stance_frames(sock[k][chain], sock[k]["stances"]).mean(axis=0) for k in train], axis=0)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    a = np.mean([qs[chain][k] for k in train], axis=0)
    b = np.mean([qf[k] for k in train], axis=0)
    if how == "qm_pooled":
        a = np.repeat(a.mean(axis=0, keepdims=True), N_CH, axis=0)
        b = np.repeat(b.mean(axis=0, keepdims=True), N_CH, axis=0)
    return a, b


def calibrate(X, how, cal):
    if how == "none":
        return X
    if how == "gain":
        return X * cal[None, :]
    return apply_quantile_map(X, *cal)


def build_variants(ref: dict, sock: dict, keys: list) -> tuple[dict, dict]:
    groups = subjects(keys)
    qf = {k: quantile_functions(ref[k]["sites"], ref[k]["stances"]) for k in keys}
    qs = {c: {k: quantile_functions(sock[k][c], sock[k]["stances"]) for k in keys} for c in ("pipeline", "dsp")}
    out, cals = {}, {}
    for name, (chain, how) in VARIANTS.items():
        out[name], cals[name] = {}, {}
        for g in np.unique(groups):
            train = [k for k in keys if k[1] != g]
            cal = None if how in ("none", "bw") else _fold_calibration(train, chain, how, ref, sock, qs, qf)
            cals[name][g] = cal
            for k in (k for k in keys if k[1] == g):
                if how == "bw":
                    X = sock[k][chain].astype(float) * sock_bodyweight_scale(sock[k], chain, k[1])
                else:
                    X = calibrate(sock[k][chain].astype(float), how, cal)
                out[name][k] = stance_outcomes(sock[k]["time"], sock[k]["stances"], X, SOCK_REGION)
    return out, cals


def learned_weight_predictions(ref, sock, keys, cals, quantity: str, lam: float = 0.1):
    site_key = "site_pp" if quantity == "pp" else "site_pti"
    name = "DSP + quantile map (per channel)"
    groups = subjects(keys)
    pred, weights = {}, {}
    for g in np.unique(groups):
        cal = cals[name][g]
        train = [k for k in keys if k[1] != g]
        S = np.array([stance_outcomes(sock[k]["time"], sock[k]["stances"],
                                      calibrate(sock[k]["dsp"].astype(float), "qm_channel", cal),
                                      SOCK_REGION)[site_key].mean(axis=0) for k in train])
        y = np.array([ref[k][quantity].mean() for k in train])
        w = learn_channel_weights(S, y, lam)
        weights[g] = w
        for k in (k for k in keys if k[1] == g):
            o = stance_outcomes(sock[k]["time"], sock[k]["stances"],
                                calibrate(sock[k]["dsp"].astype(float), "qm_channel", cal), SOCK_REGION)
            pred[k] = float((o[site_key] * w[None, :]).max(axis=1).mean())
    return pred, weights


# ------------------------------------------------------------------ statistics
def ccc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return math.nan
    return float(2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def agreement(pred: np.ndarray, ref: np.ndarray, groups: np.ndarray, absolute: bool = True,
              n_boot: int = 0, seed: int = 0) -> dict:
    m = np.isfinite(pred) & np.isfinite(ref)
    p, r, g = pred[m], ref[m], groups[m]
    out = dict(n_feet=int(m.sum()), n_participants=len(np.unique(g)))
    if m.sum() < 4 or np.std(p) == 0:
        return out
    out.update(pearson_r=float(sstats.pearsonr(p, r)[0]), spearman_rho=float(sstats.spearmanr(p, r)[0]))
    if absolute:
        bias, lo, hi = bland_altman_rm(p - r, g)
        out.update(ccc=ccc(p, r), icc_a1=icc21(p, r), rmse=float(np.sqrt(np.mean((p - r) ** 2))),
                   bias=bias, loa_low=lo, loa_high=hi, ref_mean=float(r.mean()))
    if n_boot:
        rng = np.random.default_rng(seed)
        ug = np.unique(g)
        rho, cc = [], []
        for _ in range(n_boot):
            pick = rng.choice(ug, len(ug), replace=True)
            idx = np.concatenate([np.where(g == s)[0] for s in pick])
            if np.std(p[idx]) == 0 or np.std(r[idx]) == 0:
                continue
            rho.append(sstats.spearmanr(p[idx], r[idx])[0])
            if absolute:
                cc.append(ccc(p[idx], r[idx]))
        out.update(rho_ci_low=float(np.nanpercentile(rho, 2.5)), rho_ci_high=float(np.nanpercentile(rho, 97.5)))
        if absolute:
            out.update(ccc_ci_low=float(np.nanpercentile(cc, 2.5)), ccc_ci_high=float(np.nanpercentile(cc, 97.5)))
    return out


def score(pred: np.ndarray, ref_vals: np.ndarray, groups: np.ndarray, units: str,
          n_boot: int = 0, absolute: bool = True) -> dict:
    est = loso_linear(pred, ref_vals, groups) if units == "raw" else pred
    out = agreement(est, ref_vals, groups, absolute=absolute)
    out.update({k: v for k, v in agreement(pred, ref_vals, groups, absolute=False, n_boot=n_boot).items()
                if k in ("pearson_r", "spearman_rho", "rho_ci_low", "rho_ci_high")})
    if n_boot and absolute:
        b = agreement(est, ref_vals, groups, absolute=True, n_boot=n_boot)
        out.update(ccc_ci_low=b.get("ccc_ci_low"), ccc_ci_high=b.get("ccc_ci_high"))
    out["units"] = units
    return out


def existing_baselines(quantity: str, keys: list, ref: dict | None = None,
                       factors: dict | None = None) -> pd.DataFrame:
    path = OUT_DIR / "baseline_per_foot_physical.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[d.decoder != "control"] if "decoder" in d else d
    want = {(k[1], k[2]) for k in keys}
    d = d[[(s, f) in want for s, f in zip(d.subject, d.foot)]].copy()
    scale = {(k[1], k[2]): r["bw_scale"] for k, r in (ref or {}).items() if "bw_scale" in r}
    if scale:
        d[f"ref_{quantity}"] = d[f"ref_{quantity}"] * [scale[(s, f)] for s, f in zip(d.subject, d.foot)]
    fac = {(k[1], k[2]): v for k, v in (factors or {}).items()}
    readouts = [("raw", np.ones(len(d)))]
    if fac:
        readouts.append(("body-weight", np.array([fac.get((s, f), np.nan) for s, f in zip(d.subject, d.foot)])))
    rows = []
    for tag, c in readouts:
        d["_x"] = d[quantity].to_numpy(float) * c
        for m, g in d.groupby("method", sort=False):
            x, y, grp = g["_x"].to_numpy(float), g[f"ref_{quantity}"].to_numpy(float), g.subject.to_numpy()
            a = agreement(loso_linear(x, y, grp), y, grp)
            a.update({k: v for k, v in agreement(x, y, grp, absolute=False).items()
                      if k in ("pearson_r", "spearman_rho")})
            rows.append(dict(method=m, readout=tag, **a))
    return pd.DataFrame(rows)


def subsets(ref: dict, sock: dict) -> tuple[dict, pd.DataFrame]:
    q = reference_qc(ref)
    ok = {(r.sex, r.subject, r.foot) for r in q.itertuples() if r.qc_pass}
    cal = {(r.sex, r.subject, r.foot) for r in q.itertuples() if r.cal30_pass}
    keys = sorted(set(ref) & set(sock))
    ref_bw = bodyweight_reference(ref)
    return {"all": (keys, ref), "qc": ([k for k in keys if k in ok], ref),
            "cal30": ([k for k in keys if k in cal], ref),
            "bw": ([k for k in keys if k in ref_bw], ref_bw)}, q


def md_table(df: pd.DataFrame, cols: list[str], fmt: str = "{:+.3f}") -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c, "")
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                         else ("" if isinstance(v, (float, np.floating)) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body
