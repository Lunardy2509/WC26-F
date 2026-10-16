from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats as sstats

from data_loader.io import parse_fscan_with_matrices
from config import (
    FSCAN_COLS, FSCAN_ROWS, MIX_VALIDATION_FEET, OUT_DIR, RESAMPLE_POINTS, SOCK_COORDS_FULL, SOCK_RAW_SCALE,
    WARMUP_SECONDS,
    clean_sock_values, discover_trials, find_stances, participant_of, resample_series,
)
from experiments.cop_benchmark import TEMPLATE, canon_field, centroid_frames, field_cop

OUT = OUT_DIR / "mix_validation"
FS = 100.0
MATCH_S = 0.15                     # stance-onset agreement after alignment
MAX_LAG_S = 60.0
ARCH_CH = [12, 13, 15, 16]         # 0-indexed sock channels 13, 14, 16, 17 (diagram map: medial arch)
TARGET = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
N_CELLS = FSCAN_ROWS * FSCAN_COLS


def uniform(t, x):
    g = np.arange(t[0], t[-1], 1.0 / FS)
    return g, np.interp(g, t, x)


def align(t_f, load_f, t_s, load_s):
    gf, xf = uniform(t_f, load_f)
    gs, xs = uniform(t_s, load_s)
    xf, xs = (xf - xf.mean()) / xf.std(), (xs - xs.mean()) / xs.std()
    c = signal.correlate(xf, xs, mode="full", method="fft")
    lags = signal.correlation_lags(len(xf), len(xs), mode="full")
    ok = np.abs(lags) <= MAX_LAG_S * FS
    c, lags = c[ok], lags[ok]
    k = int(np.argmax(c))
    overlap = min(len(xf), len(xs) + lags[k]) - max(0, lags[k])
    return gf[0] - gs[0] + lags[k] / FS, float(c[k] / max(overlap, 1))


def traj(t, s, e, xy):
    tt = t[s:e]
    return np.column_stack([resample_series(tt, xy[s:e, 0], TARGET), resample_series(tt, xy[s:e, 1], TARGET)])


def learned_weights():
    wt = pd.read_csv(OUT_DIR / "wc26f_folds.csv", encoding="utf-8-sig")
    wt = wt[wt.route == "physical"]
    W = {}
    for r in wt.itertuples():
        w = np.array([getattr(r, f"w{j}") for j in range(1, 27)], dtype=float)
        W[participant_of(r.held_out)] = w / w.mean()
    return W, np.mean(list(W.values()), axis=0)


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def load_mix():
    by = {}
    trials = discover_trials()                       # fills MIX_VALIDATION_FEET
    for t in trials:
        if t.condition == "mix" or (t.subject, t.foot) in MIX_VALIDATION_FEET:
            by.setdefault((t.sex, t.subject, t.foot), {})[t.device] = t
    return {k: v for k, v in by.items() if set(v) == {"fscan", "sock"}}


def main():
    W, W_mean = learned_weights()
    feet, steps = [], []
    for (sex, subj, foot), tr in sorted(load_mix().items()):
        part = participant_of(subj)
        fdf, mats = parse_fscan_with_matrices(tr["fscan"].path)
        t_f = fdf["time_s"].to_numpy()
        F = canon_field(mats.reshape(-1, N_CELLS) / 1000.0, foot)
        df = pd.read_csv(tr["sock"].path)
        raw = df[[f"Data{i}" for i in range(1, 27)]].astype(float).to_numpy() * SOCK_RAW_SCALE
        t_s = (df["Time(ms)"].astype(float).to_numpy() - float(df["Time(ms)"].iloc[0])) / 1000.0
        v = clean_sock_values(raw)
        lag, peak_c = align(t_f, F.sum(axis=1), t_s, v.sum(axis=1))

        kf = t_f >= WARMUP_SECONDS
        ks = t_s >= WARMUP_SECONDS
        fdf_k = fdf[kf].reset_index(drop=True)
        st_f, _ = find_stances(fdf_k)
        tf, Fk = t_f[kf], F[kf]
        ts, vk = t_s[ks], v[ks]
        heel = vk[:, SOCK_COORDS_FULL[:, 1] >= 40.0].sum(axis=1)
        st_s, _ = find_stances(pd.DataFrame({"time_s": ts, "total_pressure": vk.sum(axis=1), "heel_pressure": heel}))

        w = W.get(part, W_mean)
        cop_f, cop_s = field_cop(Fk), centroid_frames(vk, TEMPLATE, w)
        on_f = np.array([tf[s] for s, _ in st_f])
        pairs = []
        for s, e in st_s:
            d = np.abs(on_f - (ts[s] + lag))
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= MATCH_S:
                pairs.append(((s, e), st_f[j]))
        if not pairs:
            print(f"  {subj:8s} {foot}: no matched stances (peak corr {peak_c:.2f})")
            continue
        TS = np.array([traj(ts, s, e, cop_s) for (s, e), _ in pairs])
        TF = np.array([traj(tf, s, e, cop_f) for _, (s, e) in pairs])
        dS, dF = TS - np.nanmean(TS, 0), TF - np.nanmean(TF, 0)
        for i, ((s, e), (a, b)) in enumerate(pairs):
            steps.append(dict(sex=sex, subject=subj, participant=part, foot=foot, step=i,
                              ml_r=corr(TS[i, :, 0], TF[i, :, 0]), ap_r=corr(TS[i, :, 1], TF[i, :, 1]),
                              ml_dev_r=corr(dS[i, :, 0], dF[i, :, 0]), ap_dev_r=corr(dS[i, :, 1], dF[i, :, 1]),
                              sock_pp=vk[s:e].max(), sock_pti=np.trapezoid(vk[s:e], ts[s:e], axis=0).max(),
                              sock_dur=ts[e - 1] - ts[s],
                              ref_pp=Fk[a:b].max(), ref_pti=np.trapezoid(Fk[a:b], tf[a:b], axis=0).max(),
                              ref_dur=tf[b - 1] - tf[a]))
        st = pd.DataFrame([r for r in steps if r["subject"] == subj and r["foot"] == foot])
        # arch: sock duty of the medial-arch channels; F-SCAN share of stance frames with medial-arch contact
        in_s = np.zeros(len(vk), bool)
        for s, e in st_s:
            in_s[s:e] = True
        in_f = np.zeros(len(Fk), bool)
        for s, e in st_f:
            in_f[s:e] = True
        G = Fk.reshape(-1, FSCAN_ROWS, FSCAN_COLS)
        pk = G[in_f].max(axis=0)
        act = pk > 0.02 * pk.max()
        rr, cc = np.where(act.any(1))[0], np.where(act.any(0))[0]
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        ra, rb = r0 + int(0.45 * (r1 - r0)), r0 + int(0.65 * (r1 - r0))       # midfoot band
        cm = c0 + int(0.40 * (c1 - c0))                                          # medial 40 %
        arch_ref = float(((G[in_f][:, ra:rb, c0:cm] > 5.0).mean(axis=(1, 2))).mean())
        feet.append(dict(sex=sex, subject=subj, participant=part, foot=foot, lag_s=lag, align_corr=peak_c,
                         stances_sock=len(st_s), stances_fscan=len(st_f), matched=len(pairs),
                         ml_r=st.ml_r.median(), ap_r=st.ap_r.median(), ml_dev_r=st.ml_dev_r.median(),
                         ap_dev_r=st.ap_dev_r.median(),
                         pp_within_rho=sstats.spearmanr(st.sock_pp, st.ref_pp)[0] if len(st) > 4 else np.nan,
                         pti_within_rho=sstats.spearmanr(st.sock_pti, st.ref_pti)[0] if len(st) > 4 else np.nan,
                         dur_within_rho=sstats.spearmanr(st.sock_dur, st.ref_dur)[0] if len(st) > 4 else np.nan,
                         sock_pp=st.sock_pp.mean(), ref_pp=st.ref_pp.mean(), sock_pti=st.sock_pti.mean(),
                         ref_pti=st.ref_pti.mean(), sock_arch_duty=float((vk[in_s][:, ARCH_CH] > 0).mean()),
                         fscan_arch_contact=arch_ref, weights="own fold" if part in W else "mean of folds"))
        print(f"  {subj:8s} {foot}: lag {lag:+7.2f} s, corr {peak_c:.2f}, matched {len(pairs)}/{len(st_s)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    P, S = pd.DataFrame(feet), pd.DataFrame(steps)
    P.to_csv(OUT / "mix_per_foot.csv", index=False)
    S.to_csv(OUT / "mix_per_step.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(P.round(3).to_string(index=False))
    good = P[P.align_corr >= 0.5]
    summ = dict(feet=len(P), aligned_feet=len(good), participants=good.participant.nunique(),
                matched_steps=int(good.matched.sum()),
                ml_r=good.ml_r.median(), ap_r=good.ap_r.median(), ml_dev_r=good.ml_dev_r.median(),
                ap_dev_r=good.ap_dev_r.median(), pp_within_rho=good.pp_within_rho.median(),
                pti_within_rho=good.pti_within_rho.median(), dur_within_rho=good.dur_within_rho.median(),
                pp_between_rho=sstats.spearmanr(good.sock_pp, good.ref_pp)[0],
                pti_between_rho=sstats.spearmanr(good.sock_pti, good.ref_pti)[0])
    pd.Series(summ).to_csv(OUT / "mix_summary.csv", header=["value"])
    print(pd.Series(summ).round(3).to_string())


if __name__ == "__main__":
    main()
