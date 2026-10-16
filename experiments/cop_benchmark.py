from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from data_loader.io import parse_fscan_with_matrices
from utils.resample import resample_matrix
from model.linear import ridge_fit, ridge_predict
from utils.agreement import icc21
from config import (
    FSCAN_COLS, FSCAN_ROWS, FSCAN_X, FSCAN_Y, OUT_DIR, RESAMPLE_POINTS, WARMUP_SECONDS,
    discover_trials, find_stances, resample_series,
)
from data_loader import sock_feet as A
from model.layouts import LAYOUTS, layout_sizes, layout_uv, sock_uv

MM = 5.08                       # F-SCAN sensel pitch (header: ROW/COL_SPACING 0.508 cm)
BOX_MM = np.array([88.6, 245.7])   # cohort footprint box, width x length
TARGET = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
N_CELLS = FSCAN_ROWS * FSCAN_COLS
DEVICE = A.DEVICE
OURS_CELLS = 2                  # our sock's sensing dots are ~1 cm
COMMITTED_8 = [0, 8, 10, 12, 19, 20, 21, 25]
TEMPLATE = A.COORD_MAPS["diagram"]          # canonical (right-foot) sock coordinates
TEMPLATE_UV = sock_uv(TEMPLATE)
FIELD_STANCES = 30              # per-stance reference fields kept per foot (virtual decoders)

METHODS = [
    # name, citation, layout, decoder
    ("WalkinSense", "[13],[35]", "walkinsense8", "centroid"),
    ("Zhang et al. (SportSole)", "[14]", "zhang8", "zhang"),
    ("Ciniglio et al.", "[15]", "ciniglio16", "centroid"),
    ("Xian et al. (improved GA)", "[16]", "xian8", "centroid"),
    ("Lin et al.", "[19]", "lin8", "centroid"),
    ("Tseng et al. (ISAC + GPBT)", "[18]", "tseng8", "centroid"),
    ("Gautier et al.", "[23]", "gautier7", "centroid"),
    ("Ho et al.", "[37]", "ho10", "centroid"),
    ("Davidovica et al. (DAid)", "[38]", "daid6", "centroid"),
    ("Najafi et al. (SmartSox)", "[42]", "najafi5", "centroid"),
    ("Hu et al.", "[22]", "hu12", "hu"),
    ("Mun & Choi (LSTM)", "[21]", "mun9", "lstm"),
    ("Chan et al. (cGAN)", "[20]", "ours26", "cgan"),
    ("Gong et al. (deep forest)", "[17]", "ours26", "forest"),
    ("Linear decoder (ridge)", "-", "ours26", "ridge"),
    ("Committed 8 (prior)", "-", "committed8", "centroid"),
    # WC26-F (model/wc26f.py) and its two heads as ablations; all three come from one fit per fold.
    ("WC26-F centroid head", "-", "ours26", "wc26f:centroid"),
    ("WC26-F field head", "-", "ours26", "wc26f:field"),
    ("WC26-F (ours)", "-", "ours26", "wc26f"),
]
DEEP = {"lstm", "cgan", "forest"}


def pearson(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ------------------------------------------------------------------ agreement
def bland_altman_rm(diff: np.ndarray, cluster: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(diff)
    d, g = diff[ok], cluster[ok]
    if d.size < 3:
        return math.nan, math.nan, math.nan
    groups = pd.Series(d).groupby(g)
    m, means = groups.size().to_numpy(), groups.mean().to_numpy()
    n, N = len(m), d.size
    bias = float(d.mean())
    msb = float((m * (means - bias) ** 2).sum() / max(n - 1, 1))
    msw = float(((d - pd.Series(g).map(dict(zip(groups.mean().index, means))).to_numpy()) ** 2).sum()
                / max(N - n, 1))
    div = (N ** 2 - (m ** 2).sum()) / (max(n - 1, 1) * N)
    var = max(0.0, (msb - msw) / div) + msw
    sd = math.sqrt(var)
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


# ------------------------------------------------------------------ helpers
def canon_field(F: np.ndarray, foot: str) -> np.ndarray:
    if foot != "L":
        return F
    sh = F.shape
    return F.reshape(*sh[:-1], FSCAN_ROWS, FSCAN_COLS)[..., ::-1].reshape(sh)


def centroid_frames(ch: np.ndarray, xy: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    v = ch if w is None else ch * w
    tot = v.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.column_stack([np.where(tot > 0, v @ xy[:, 0] / tot, np.nan),
                                np.where(tot > 0, v @ xy[:, 1] / tot, np.nan)])


def field_cop(F: np.ndarray) -> np.ndarray:
    tot = F.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([np.where(tot > 0, F @ FSCAN_X / tot, np.nan),
                         np.where(tot > 0, F @ FSCAN_Y / tot, np.nan)], axis=-1)


def stance_trajs(time, stances, frames_xy) -> np.ndarray:
    out = []
    for s, e in stances:
        tt = time[s:e]
        out.append(np.column_stack([resample_series(tt, frames_xy[s:e, 0], TARGET),
                                    resample_series(tt, frames_xy[s:e, 1], TARGET)]))
    return np.asarray(out)


def stance_scalars(time, stances, frames_ch) -> tuple[np.ndarray, np.ndarray]:
    pp, pti = [], []
    for s, e in stances:
        seg, tt = frames_ch[s:e], time[s:e]
        pp.append(float(seg.max()) if seg.size else math.nan)
        pti.append(float(np.trapezoid(seg, tt, axis=0).max()) if len(tt) > 1 else math.nan)
    return np.asarray(pp), np.asarray(pti)


def stance_inputs(time, stances, frames_ch) -> np.ndarray:
    return np.asarray([resample_matrix(time[s:e], frames_ch[s:e], TARGET) for s, e in stances])


def shares(x: np.ndarray) -> np.ndarray:
    tot = x.sum(axis=-1, keepdims=True)
    return np.divide(x, tot, out=np.zeros_like(x, dtype=float), where=tot > 0)


def with_total(x: np.ndarray, scale: float) -> np.ndarray:
    return np.concatenate([shares(x), x.sum(axis=-1, keepdims=True) / scale], axis=-1)


# ------------------------------------------------------------------ data
class RouteFoot:

    def __init__(self, sex, subject, foot, time, stances):
        self.sex, self.subject, self.foot = sex, subject, foot
        self.time, self.stances = time, stances
        self.ch: dict[str, np.ndarray] = {}        # layout -> (frames, k)
        self.xy: dict[str, np.ndarray] = {}        # layout -> (k, 2) canonical cell coords
        self.cells: dict[str, int] = {}            # layout -> grid cells covered (virtual)
        self.X: dict[str, np.ndarray] = {}         # layout -> (S, 100, k) resampled inputs

    def inputs(self, layout):
        if layout not in self.X:
            self.X[layout] = stance_inputs(self.time, self.stances, self.ch[layout])
        return self.X[layout]


def load_reference() -> dict:
    ref = {}
    for t in discover_trials():
        if not (t.device == "fscan" and t.condition == "only"):
            continue
        frame_df, mats = parse_fscan_with_matrices(t.path)
        keep = frame_df["time_s"].to_numpy() >= WARMUP_SECONDS
        frame_df = frame_df[keep].reset_index(drop=True)
        F = canon_field(mats[keep].reshape(-1, N_CELLS) / 1000.0, t.foot)      # kPa
        stances, _ = find_stances(frame_df)
        if not stances:
            continue
        time = frame_df["time_s"].to_numpy()
        ref[(t.sex, t.subject, t.foot)] = foot_record(F, time, stances)
        print(f"  F-SCAN {t.subject:12s} {t.foot}: {len(stances)} stances", flush=True)
    return ref


def foot_record(F, time, stances, n_fields=None) -> dict:
    in_st = np.zeros(len(F), dtype=bool)
    for s, e in stances:
        in_st[s:e] = True
    peak = F[in_st].max(axis=0).reshape(FSCAN_ROWS, FSCAN_COLS)
    act = peak > 0.02 * peak.max()
    rows, cols = np.where(act.any(1))[0], np.where(act.any(0))[0]
    traj = stance_trajs(time, stances, field_cop(F))
    pp, pti = stance_scalars(time, stances, F)
    n_fields = FIELD_STANCES if n_fields is None else n_fields
    pick = np.unique(np.linspace(0, len(stances) - 1, min(n_fields, len(stances))).astype(int))
    fields = np.asarray([resample_matrix(time[s:e], F[s:e], TARGET) for s, e in
                         [stances[i] for i in pick]], dtype=np.float16)
    box = (rows.min(), rows.max(), cols.min(), cols.max())
    # Virtual sensors are sampled here, so the full frame stack is never kept.
    vch, vxy, vcells = {}, {}, {}
    for lay in layouts_used():
        uv, sizes = layout_spec(lay)
        blocks = [block(box, q, s) for q, s in zip(uv, sizes)]
        vch[lay] = np.stack([F[:, b].mean(axis=1) for b, _ in blocks], axis=1).astype(np.float32)
        vxy[lay] = np.asarray([c for _, c in blocks])
        vcells[lay] = len(np.unique(np.concatenate([b for b, _ in blocks])))
    return dict(time=time, stances=stances, box=box, vch=vch, vxy=vxy, vcells=vcells,
                traj=traj, mean=np.nanmean(traj, axis=0), pp=pp, pti=pti, fields=fields, field_idx=pick)


def block(box, uv, size):
    w, h = size
    r0, r1, c0, c1 = box
    rc, cc = r0 + uv[1] * (r1 - r0), c0 + uv[0] * (c1 - c0)
    rs = np.clip(np.arange(h) + int(round(rc - (h - 1) / 2)), 0, FSCAN_ROWS - 1)
    cs = np.clip(np.arange(w) + int(round(cc - (w - 1) / 2)), 0, FSCAN_COLS - 1)
    rr, cc_ = np.meshgrid(np.unique(rs), np.unique(cs), indexing="ij")
    return (rr * FSCAN_COLS + cc_).ravel(), (cc_.mean() + 1.0, rr.mean() + 1.0)


def layouts_used():
    return sorted({m[2] for m in METHODS if m[2]})


def layout_spec(layout):
    if layout == "ours26":
        return TEMPLATE_UV, [(OURS_CELLS, OURS_CELLS)] * len(TEMPLATE_UV)
    if layout == "committed8":
        return TEMPLATE_UV[COMMITTED_8], [(OURS_CELLS, OURS_CELLS)] * len(COMMITTED_8)
    return layout_uv(layout), layout_sizes(layout)


def physical_channels(layout) -> tuple[np.ndarray, np.ndarray]:
    if layout == "ours26":
        return np.arange(26), np.zeros(26)
    if layout == "committed8":
        return np.asarray(COMMITTED_8), np.zeros(8)
    uv = layout_uv(layout)
    d = np.linalg.norm((uv[:, None, :] - TEMPLATE_UV[None, :, :]) * BOX_MM, axis=2)
    idx = d.argmin(axis=1)
    return np.unique(idx), d.min(axis=1)


def build_routes(ref: dict):
    phys_src = A.load_feet()
    phys, virt = [], []
    for f in phys_src:
        key = (f.sex, f.subject, f.foot)
        if key not in ref:
            continue
        r = ref[key]
        # physical: our sock frames, canonical template coordinates
        p = RouteFoot(f.sex, f.subject, f.foot, f.time_s, f.stances)
        for lay in layouts_used():
            idx, _ = physical_channels(lay)
            p.ch[lay], p.xy[lay] = f.values[:, idx], TEMPLATE[idx]
        ref_mean = f.ref.copy()
        if f.foot == "L":
            ref_mean[:, 0] = (FSCAN_COLS + 1) - ref_mean[:, 0]
        p.ref_mean = ref_mean                                       # pipeline F-SCAN COP
        p.ref_field = canon_field(f.field / 1000.0, f.foot).astype(np.float32)
        p.ref_pp, p.ref_pti = float(np.nanmean(r["pp"])), float(np.nanmean(r["pti"]))
        tot = f.values.sum(axis=1)
        p.scale = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
        phys.append(p)
        # virtual: layouts sampled from this foot's own F-SCAN frames
        v = RouteFoot(f.sex, f.subject, f.foot, r["time"], r["stances"])
        v.ch, v.xy, v.cells = dict(r["vch"]), dict(r["vxy"]), dict(r["vcells"])
        v.ref_traj, v.ref_mean = r["traj"], r["mean"]
        v.ref_fields, v.field_idx = r["fields"], r["field_idx"]
        v.ref_pp, v.ref_pti = float(np.nanmean(r["pp"])), float(np.nanmean(r["pti"]))
        v.ref_pp_st, v.ref_pti_st = r["pp"], r["pti"]
        tot = v.ch["ours26"].sum(axis=1)
        v.scale = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
        virt.append(v)
    return phys, virt


# ------------------------------------------------------------------ training sets
def point_set(feet, layout, route, feature, max_stances=None, rng=None):
    Xs, Ys, G, FI = [], [], [], []
    for gi, f in enumerate(feet):
        X = f.inputs(layout)
        S = len(X)
        sel = np.arange(S)
        if route == "virtual":                          # field targets exist for these only
            sel = f.field_idx if feature == "field" else sel
        if max_stances and len(sel) > max_stances:
            sel = np.sort(rng.choice(sel, max_stances, replace=False))
        for j, si in enumerate(sel):
            x = X[si]
            ok = x.sum(axis=1) > 0
            if route == "physical":
                y = f.ref_mean
                fi = np.stack([np.full(100, gi), np.arange(100)], axis=1)
            else:
                y = f.ref_traj[si]
                k = int(np.where(f.field_idx == si)[0][0]) if feature == "field" else 0
                fi = np.stack([np.full(100, gi), np.full(100, k) * 100 + np.arange(100)], axis=1)
            ok &= np.isfinite(y).all(axis=1)
            Xs.append(x[ok]); Ys.append(y[ok]); FI.append(fi[ok]); G.append(np.full(ok.sum(), gi))
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(G), np.vstack(FI)


# ------------------------------------------------------------------ decoders
class Cascade:

    def __init__(self, n=60):
        self.n = n

    def _ests(self, lvl):
        return [RandomForestRegressor(self.n, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=lvl),
                ExtraTreesRegressor(self.n, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=10 + lvl)]

    def fit(self, X, Y, groups):
        self.l1, self.l2 = self._ests(0), self._ests(1)
        aug = [cross_val_predict(e, X, Y, groups=groups, cv=GroupKFold(3)) for e in self.l1]
        for e in self.l1:
            e.fit(X, Y)
        A2 = np.hstack([X] + aug)
        for e in self.l2:
            e.fit(A2, Y)
        return self

    def predict(self, X):
        A2 = np.hstack([X] + [e.predict(X) for e in self.l1])
        return np.mean([e.predict(A2) for e in self.l2], axis=0)


class Gen(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(256, 512), torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(512, N_CELLS), torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Disc(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d + N_CELLS, 512), torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(512, 256), torch.nn.LeakyReLU(0.2),
                                       torch.nn.Linear(256, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class Seq(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = torch.nn.LSTM(d, 128, num_layers=2, batch_first=True)
        self.out = torch.nn.Sequential(torch.nn.Linear(128, N_CELLS), torch.nn.ReLU())

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h)


FIELD_SCALE = 100.0     # kPa / 100 keeps targets O(1)


def field_bank(feet, route):
    """GPU tensor of field targets, indexed by (foot, row) from point_set."""
    if route == "physical":
        bank = np.stack([f.ref_field for f in feet])                          # (F, 100, 1260)
    else:
        n = max(len(f.field_idx) for f in feet)
        bank = np.zeros((len(feet), n * 100, N_CELLS), dtype=np.float16)
        for i, f in enumerate(feet):
            bank[i, :len(f.field_idx) * 100] = f.ref_fields.reshape(-1, N_CELLS)
    return torch.as_tensor(np.asarray(bank, dtype=np.float32) / FIELD_SCALE, device=DEVICE)


def fit_cgan(train, layout, route, seed):
    torch.manual_seed(seed)
    X, _, _, FI = point_set(train, layout, route, "field")
    Xi = torch.as_tensor(with_total_pts(train, layout, route, X, FI), dtype=torch.float32, device=DEVICE)
    bank = field_bank(train, route)
    fi = torch.as_tensor(FI, device=DEVICE)
    G, D = Gen(Xi.shape[1]).to(DEVICE), Disc(Xi.shape[1]).to(DEVICE)
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(3000):
        b = torch.randint(0, len(Xi), (512,), device=DEVICE)
        x, y = Xi[b], bank[fi[b, 0], fi[b, 1]]
        fake = G(x)
        od.zero_grad()
        ld = bce(D(x, y), torch.ones(len(b), 1, device=DEVICE)) + \
            bce(D(x, fake.detach()), torch.zeros(len(b), 1, device=DEVICE))
        ld.backward()
        od.step()
        og.zero_grad()
        lg = bce(D(x, fake), torch.ones(len(b), 1, device=DEVICE)) + 100.0 * (fake - y).abs().mean()
        lg.backward()
        og.step()
    return G.eval()


def with_total_pts(feet, layout, route, X, FI):
    scale = np.array([f.scale for f in feet])[FI[:, 0]][:, None]
    return np.concatenate([shares(X), X.sum(axis=1, keepdims=True) / scale], axis=1)


class _Indexed:

    def __init__(self, bank, index):
        self.bank, self.index = bank, index

    def __getitem__(self, b):
        return self.bank[self.index[b]]


def fit_lstm(train, layout, route, seed):
    torch.manual_seed(seed)
    xs, ys, fid = [], [], []
    for i, f in enumerate(train):
        X = f.inputs(layout)
        sel = np.arange(len(X)) if route == "physical" else f.field_idx
        for k, si in enumerate(sel):
            xs.append(with_total(X[si], f.scale))
            if route == "physical":
                fid.append(i)
            else:
                ys.append(f.ref_fields[k].astype(np.float32))
    Xt = torch.as_tensor(np.asarray(xs), dtype=torch.float32, device=DEVICE)
    if route == "physical":
        bank = torch.as_tensor(np.stack([f.ref_field for f in train]).astype(np.float32) / FIELD_SCALE, device=DEVICE)
        Yt = _Indexed(bank, torch.as_tensor(np.asarray(fid), device=DEVICE))
    else:
        Yt = torch.as_tensor(np.asarray(ys, dtype=np.float32) / FIELD_SCALE, device=DEVICE)
    m = Seq(Xt.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(1200):
        b = torch.randint(0, len(Xt), (64,), device=DEVICE)
        opt.zero_grad()
        loss = ((m(Xt[b]) - Yt[b]) ** 2).mean()
        loss.backward()
        opt.step()
    return m.eval()


def fit_phase_calibration(train, layout, route):
    xs, ys = [], []
    for f in train:
        traj = stance_trajs(f.time, f.stances, centroid_frames(f.ch[layout], f.xy[layout]))
        if route == "physical":
            xs.append(np.nanmean(traj, axis=0)[None])
            ys.append(f.ref_mean[None])
        else:
            xs.append(traj)
            ys.append(f.ref_traj)
    X, Y = np.concatenate(xs), np.concatenate(ys)
    a, b = np.zeros((len(TARGET), 2)), np.ones((len(TARGET), 2))
    for k in range(len(TARGET)):
        for ax in range(2):
            x, y = X[:, k, ax], Y[:, k, ax]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3 and np.std(x[ok]) > 0:
                b[k, ax], a[k, ax] = np.polyfit(x[ok], y[ok], 1)
            elif ok.any():
                a[k, ax], b[k, ax] = np.mean(y[ok]), 0.0
    return a, b


# ------------------------------------------------------------------ evaluation
def predict(method_decoder, model, f, layout, route):
    ch, xy = f.ch[layout], f.xy[layout]
    if method_decoder in ("centroid", "learned"):
        w = model if method_decoder == "learned" else None
        traj = stance_trajs(f.time, f.stances, centroid_frames(ch, xy, w))
        pp, pti = stance_scalars(f.time, f.stances, ch)
    elif method_decoder == "zhang":
        a, b = model
        traj = a[None] + b[None] * stance_trajs(f.time, f.stances, centroid_frames(ch, xy))
        pp, pti = stance_scalars(f.time, f.stances, ch)
    elif method_decoder == "hu":
        cop = shares(ch) @ model
        cop[ch.sum(axis=1) <= 0] = np.nan
        traj = stance_trajs(f.time, f.stances, cop)
        pp, pti = stance_scalars(f.time, f.stances, ch)
    elif method_decoder == "ridge":
        cop = ridge_predict(model, ch)
        cop[ch.sum(axis=1) <= 0] = np.nan
        traj = stance_trajs(f.time, f.stances, cop)
        pp, pti = stance_scalars(f.time, f.stances, ch)
    elif method_decoder == "forest":
        loaded = ch.sum(axis=1) > 0
        cop = np.full((len(ch), 2), np.nan)
        in_st = np.zeros(len(ch), dtype=bool)
        for s, e in f.stances:
            in_st[s:e] = True
        sel = loaded & in_st
        cop[sel] = model.predict(shares(ch[sel]))
        traj = stance_trajs(f.time, f.stances, cop)
        pp, pti = stance_scalars(f.time, f.stances, ch)
    elif method_decoder == "cgan":
        traj, pp, pti = [], [], []
        with torch.no_grad():
            for s, e in f.stances:
                x = with_total(ch[s:e], f.scale)
                Fh = model(torch.as_tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy() * FIELD_SCALE
                Fh[ch[s:e].sum(axis=1) <= 0] = 0.0
                tt = f.time[s:e]
                c = field_cop(Fh)
                traj.append(np.column_stack([resample_series(tt, c[:, 0], TARGET),
                                             resample_series(tt, c[:, 1], TARGET)]))
                pp.append(float(Fh.max()))
                pti.append(float(np.trapezoid(Fh, tt, axis=0).max()) if len(tt) > 1 else math.nan)
        traj, pp, pti = np.asarray(traj), np.asarray(pp), np.asarray(pti)
    elif method_decoder.startswith("wc26f"):
        import sys
        from model import wc26f
        cb = sys.modules[__name__]
        head = method_decoder.partition(":")[2]
        if head == "centroid":
            traj = wc26f.apply_affine(model.A_c, wc26f.Stances(f).stance_trajs(model.w))
            pp, pti = stance_scalars(f.time, f.stances, ch)
        elif head == "field":
            outs = [predict("lstm", m, f, layout, route) for m in model.fields]
            traj = wc26f.apply_affine(model.A_f, np.nanmean([o[0] for o in outs], axis=0))
            pp, pti = np.mean([o[1] for o in outs], axis=0), np.mean([o[2] for o in outs], axis=0)
        else:
            traj, pp, pti = wc26f.predict(model, f, route, cb)
    elif method_decoder == "lstm" and isinstance(model, list):
        outs = [predict("lstm", m, f, layout, route) for m in model]
        traj = np.nanmean([o[0] for o in outs], axis=0)
        pp, pti = np.mean([o[1] for o in outs], axis=0), np.mean([o[2] for o in outs], axis=0)
    elif method_decoder == "lstm":
        X = f.inputs(layout)
        with torch.no_grad():
            Fh = model(torch.as_tensor(np.stack([with_total(x, f.scale) for x in X]),
                                       dtype=torch.float32, device=DEVICE)).cpu().numpy() * FIELD_SCALE
        traj = field_cop(Fh)
        dur = np.array([f.time[e - 1] - f.time[s] for s, e in f.stances])
        pp = Fh.max(axis=(1, 2))
        pti = np.array([np.trapezoid(Fh[i], TARGET * dur[i], axis=0).max() for i in range(len(Fh))])
    else:
        raise ValueError(method_decoder)
    return traj, pp, pti


POINT_STANCES = {"hu": 30, "ridge": 30, "forest": 15} 
LSTM_SEEDS = 5
WC26F_FOLDS: list[dict] = []


def run_route(feet, route, skip_deep, folds=None, only=None):
    subjects = sorted({f.subject for f in feet})
    groups = folds or [[s] for s in subjects]
    rng = np.random.default_rng(0)

    rows = []
    wc26f_models = {}
    for name, cite, layout, dec in METHODS:
        if (skip_deep and dec in DEEP) or (only and name not in only):
            continue
        print(f"  [{route}] {name}", flush=True)
        for gi, group in enumerate(groups):
            held = set(group)
            train = [f for f in feet if f.subject not in held]
            test = [f for f in feet if f.subject in held]
            if dec == "control":
                mean_traj = np.nanmean([f.ref_mean for f in train], axis=0)
                pp_m = np.nanmean([f.ref_pp for f in train])
                pti_m = np.nanmean([f.ref_pti for f in train])
                for f in test:
                    S = len(f.stances)
                    rows.append(record(name, cite, layout, dec, route, f, np.repeat(mean_traj[None], S, 0),
                                       np.full(S, pp_m), np.full(S, pti_m), units="ref"))
                continue
            model = None
            if dec == "zhang":
                model = fit_phase_calibration(train, layout, route)
            elif dec == "hu":
                X, Y, _, _ = point_set(train, layout, route, "cop", POINT_STANCES["hu"], rng)
                model = np.linalg.lstsq(shares(X), Y, rcond=None)[0]
            elif dec == "ridge":
                X, Y, _, _ = point_set(train, layout, route, "cop", POINT_STANCES["ridge"], rng)
                model = ridge_fit(X, Y, 10.0)
            elif dec == "forest":
                X, Y, G, _ = point_set(train, layout, route, "cop", POINT_STANCES["forest"], rng)
                model = Cascade().fit(shares(X), Y, G)
            elif dec == "cgan":
                model = fit_cgan(train, layout, route, seed=gi)
            elif dec == "lstm":
                model = [fit_lstm(train, layout, route, seed=gi + 1000 * s) for s in range(LSTM_SEEDS)]
            elif dec.startswith("wc26f"):
                if gi not in wc26f_models:
                    import sys
                    from model import wc26f
                    wc26f_models[gi] = wc26f.fit(train, route, gi, sys.modules[__name__])
                    m = wc26f_models[gi]
                    print(f"    fold {gi}: lambda={m.lam} beta={m.beta:.1f} alpha={m.alpha:.1f}", flush=True)
                model = wc26f_models[gi]
            if dec == "wc26f":
                WC26F_FOLDS.append(dict(route=route, fold=gi, held_out=" ".join(group), lam=model.lam,
                                        beta=model.beta, alpha=model.alpha,
                                        **{f"w{j + 1}": model.w[j] for j in range(len(model.w))}))
            for f in test:
                traj, pp, pti = predict(dec, model, f, layout, route)
                field_out = dec in ("cgan", "lstm", "wc26f", "wc26f:field")
                units = "ref" if field_out or route == "virtual" else "raw"
                rows.append(record(name, cite, layout, dec, route, f, traj, pp, pti, units))
    return rows


def record(name, cite, layout, dec, route, f, traj, pp, pti, units):
    return dict(method=name, cite=cite, layout=layout, decoder=dec, route=route, sex=f.sex,
                subject=f.subject, foot=f.foot, traj=traj, mean=np.nanmean(traj, axis=0),
                pp=float(np.nanmean(pp)), pti=float(np.nanmean(pti)), units=units, f=f)


# ------------------------------------------------------------------ summarise
def calibrate_loso(df, col, ref_col):
    out = df[col].copy()
    for held in df.subject.unique():
        tr, te = df.subject != held, df.subject == held
        x, y = df.loc[tr, col].to_numpy(), df.loc[tr, ref_col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) == 0:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        out[te] = b * df.loc[te, col] + a
    return out


def loso_affine(rs):
    out = {}
    for r in rs:
        tr = [q for q in rs if q["subject"] != r["subject"]]
        cal = np.empty_like(r["mean"])
        for ax in range(2):
            x = np.concatenate([q["mean"][:, ax] for q in tr])
            y = np.concatenate([q["f"].ref_mean[:, ax] for q in tr])
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.std(x[ok]) == 0:
                b, a = 0.0, np.nanmean(y)
            else:
                b, a = np.polyfit(x[ok], y[ok], 1)
            cal[:, ax] = b * r["mean"][:, ax] + a
        out[id(r)] = cal
    return out


def template_deviation_r(rs):
    out = {}
    if rs[0]["decoder"] == "control":
        return {id(r): (math.nan, math.nan) for r in rs}
    for r in rs:
        tr = [q for q in rs if q["subject"] != r["subject"]]
        tp = np.nanmean([q["mean"] for q in tr], axis=0)
        tf = np.nanmean([q["f"].ref_mean for q in tr], axis=0)
        dp, dr = r["mean"] - tp, r["f"].ref_mean - tf
        out[id(r)] = (pearson(dp[:, 0], dr[:, 0]), pearson(dp[:, 1], dr[:, 1]))
    return out


def summarise(rows, route):
    cop_rows, foot_rows = [], []
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    for name, rs in by_method.items():
        cal = loso_affine(rs)
        dev = template_deviation_r(rs)
        cdx, cdy, crx, cry = [], [], [], []
        for r in rs:
            d = (cal[id(r)] - r["f"].ref_mean) * MM
            cdx.append(d[:, 0]); cdy.append(d[:, 1])
            rm = np.sqrt(np.nanmean(d ** 2, axis=0))
            crx.append(rm[0]); cry.append(rm[1])
        dx, dy, clx, rmx, rmy, rx, ry = [], [], [], [], [], [], []
        for k, r in enumerate(rs):
            f = r["f"]
            if route == "physical":
                d = (r["mean"] - f.ref_mean) * MM
                rmse = np.sqrt(np.nanmean(d ** 2, axis=0))
                dx.append(d[:, 0]); dy.append(d[:, 1]); clx.append(np.full(len(d), f.subject))
            else:
                d = (r["traj"] - f.ref_traj) * MM 
                rmse = np.nanmean(np.sqrt(np.nanmean(d ** 2, axis=1)), axis=0)
                dx.append(d[..., 0].ravel()); dy.append(d[..., 1].ravel())
                clx.append(np.full(d[..., 0].size, f.subject))
            rmx.append(rmse[0]); rmy.append(rmse[1])
            rx.append(pearson(r["mean"][:, 0], f.ref_mean[:, 0]))
            ry.append(pearson(r["mean"][:, 1], f.ref_mean[:, 1]))
            foot_rows.append(dict(method=name, route=route, subject=f.subject, foot=f.foot,
                                  cop_x_rmse_mm=rmse[0], cop_y_rmse_mm=rmse[1],
                                  cop_x_r=rx[-1], cop_y_r=ry[-1],
                                  cal_cop_x_rmse_mm=crx[k], cal_cop_y_rmse_mm=cry[k],
                                  cop_x_dev_r=dev[id(r)][0], cop_y_dev_r=dev[id(r)][1],
                                  pp=r["pp"], pti=r["pti"],
                                  ref_pp=f.ref_pp, ref_pti=f.ref_pti, units=r["units"]))
        cl = np.concatenate(clx)
        bx, by_ = bland_altman_rm(np.concatenate(dx), cl), bland_altman_rm(np.concatenate(dy), cl)
        clc = np.concatenate([np.full(100, r["subject"]) for r in rs])
        cbx, cby = bland_altman_rm(np.concatenate(cdx), clc), bland_altman_rm(np.concatenate(cdy), clc)
        r0 = rs[0]
        cop_rows.append(dict(method=name, cite=r0["cite"], layout=r0["layout"], decoder=r0["decoder"],
                             route=route, n_feet=len(rs),
                             cop_x_rmse_mm=np.nanmean(rmx), cop_x_rmse_sd=np.nanstd(rmx, ddof=1),
                             cop_y_rmse_mm=np.nanmean(rmy), cop_y_rmse_sd=np.nanstd(rmy, ddof=1),
                             cop_x_r=np.nanmean(rx), cop_y_r=np.nanmean(ry),
                             cop_x_dev_r=np.nanmean([dev[id(r)][0] for r in rs]),
                             cop_y_dev_r=np.nanmean([dev[id(r)][1] for r in rs]),
                             cop_x_ba_bias=bx[0], cop_x_ba_lo=bx[1], cop_x_ba_hi=bx[2],
                             cop_y_ba_bias=by_[0], cop_y_ba_lo=by_[1], cop_y_ba_hi=by_[2],
                             cal_cop_x_rmse_mm=np.nanmean(crx), cal_cop_y_rmse_mm=np.nanmean(cry),
                             cal_cop_x_ba_lo=cbx[1], cal_cop_x_ba_hi=cbx[2],
                             cal_cop_y_ba_lo=cby[1], cal_cop_y_ba_hi=cby[2]))
    feet_df = pd.DataFrame(foot_rows)
    feet_df["decoder"] = feet_df.method.map({m[0]: m[3] for m in METHODS})
    scal, feet_df = scalar_table(feet_df, route)
    return pd.DataFrame(cop_rows), scal, feet_df


def scalar_table(feet_df: pd.DataFrame, route: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scal = []
    for name, g in feet_df.groupby("method", sort=False):
        g = g.copy()
        control = (g.decoder == "control").all()
        cal = route == "physical" and not control
        for q in ("pp", "pti"):
            g[q + "_cal"] = calibrate_loso(g, q, "ref_" + q) if cal else g[q]
            x, y = g[q + "_cal"].to_numpy(), g["ref_" + q].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            ba = bland_altman_rm(x - y, g.subject.to_numpy())
            scal.append(dict(method=name, route=route, metric=q.upper(), n_feet=int(ok.sum()),
                             r=math.nan if control else pearson(g[q].to_numpy(), y),
                             icc_abs=icc21(x[ok], y[ok]),
                             rmse=float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))),
                             ba_bias=ba[0], ba_lo=ba[1], ba_hi=ba[2],
                             ref_mean=float(np.mean(y[ok])), calibrated=cal))
        feet_df.loc[g.index, "pp_cal"] = g["pp_cal"]
        feet_df.loc[g.index, "pti_cal"] = g["pti_cal"]
    return pd.DataFrame(scal), feet_df


def layout_table(phys, virt):
    rows = []
    for name, cite, layout, dec in METHODS:
        if layout is None:
            continue
        idx, snap = physical_channels(layout)
        spec = LAYOUTS.get(layout, {})
        rows.append(dict(method=name, cite=cite, layout=layout, decoder=dec,
                         published_sensors=len(spec.get("sensors", [])) or len(idx),
                         sock_channels=" ".join(str(i + 1) for i in idx), n_sock_channels=len(idx),
                         snap_mean_mm=float(np.mean(snap)), snap_max_mm=float(np.max(snap)),
                         virtual_grid_cells=float(np.mean([v.cells[layout] for v in virt])),
                         fidelity=spec.get("fidelity", "ours"), source=spec.get("source", "this work")))
    return pd.DataFrame(rows)


def merge_into(path, new: pd.DataFrame, names) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig", dtype={"subject": str, "foot": str})
        new = pd.concat([old[~old.method.isin(names)], new], ignore_index=True)
    order = new.method.map({m[0]: i for i, m in enumerate(METHODS)}).fillna(len(METHODS))
    return new.iloc[np.argsort(order.to_numpy(), kind="stable")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-deep", action="store_true", help="skip cGAN / LSTM / deep forest")
    ap.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                    help="run only these METHODS names and merge them into the existing CSVs and "
                         "trajectories (to add a layout without refitting the rest)")
    ap.add_argument("--cache", default=None, help="pickle path for the parsed F-SCAN reference")
    ap.add_argument("--routes", nargs="+", default=["physical", "virtual"], choices=["physical", "virtual"])
    ap.add_argument("--rescore", action="store_true",
                    help="recompute baseline_scalar_*.csv from the per-foot CSVs without refitting")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pd.set_option("display.width", 250, "display.max_columns", 40)

    if a.rescore:
        for route in a.routes:
            pf = pd.read_csv(OUT_DIR / f"baseline_per_foot_{route}.csv")
            pf["decoder"] = pf.method.map({m[0]: m[3] for m in METHODS})
            scal, pf = scalar_table(pf, route)
            scal.to_csv(OUT_DIR / f"baseline_scalar_{route}.csv", index=False, encoding="utf-8-sig")
            pf.to_csv(OUT_DIR / f"baseline_per_foot_{route}.csv", index=False, encoding="utf-8-sig")
            print(f"\n===== {route.upper()} : PP (kPa) and PTI (kPa*s), rescored =====")
            print(scal.round(3).to_string(index=False), flush=True)
        return

    import pickle
    from pathlib import Path
    ref = None
    if a.cache and Path(a.cache).exists():
        ref = pickle.load(open(a.cache, "rb"))
        if not set(layouts_used()) <= set(next(iter(ref.values()))["vch"]):
            print("cached reference predates a layout; re-reading F-SCAN", flush=True)
            ref = None
    if ref is None:
        print("reading F-SCAN reference ...", flush=True)
        ref = load_reference()
        if a.cache:
            pickle.dump(ref, open(a.cache, "wb"))
    phys, virt = build_routes(ref)
    print(f"{len(phys)} subject-feet, {len({f.subject for f in phys})} participants", flush=True)

    lay = layout_table(phys, virt)
    lay.to_csv(OUT_DIR / "baseline_layouts.csv", index=False, encoding="utf-8-sig")
    print(lay[["method", "n_sock_channels", "sock_channels", "snap_mean_mm", "virtual_grid_cells", "fidelity"]]
          .round(1).to_string(index=False), flush=True)

    for route, feet in (("physical", phys), ("virtual", virt)):
        if route not in a.routes:
            continue
        if a.methods and (unknown := set(a.methods) - {m[0] for m in METHODS}):
            raise SystemExit(f"unknown method(s) {sorted(unknown)}")
        rows = run_route(feet, route, a.skip_deep, only=a.methods)
        cop, scal, perfoot = summarise(rows, route)
        trajs = {f"{r['method']}|{r['subject']}|{r['foot']}": np.stack([r["mean"], r["f"].ref_mean]) for r in rows}
        npz = OUT_DIR / f"baseline_trajectories_{route}.npz"
        if a.methods:
            if npz.exists():
                with np.load(npz) as z:
                    trajs = {**{k: z[k] for k in z.files if k.split("|")[0] not in a.methods}, **trajs}
            cop, scal, perfoot = (merge_into(OUT_DIR / f"baseline_{n}_{route}.csv", df, a.methods)
                                  for n, df in (("cop", cop), ("scalar", scal), ("per_foot", perfoot)))
        np.savez_compressed(npz, **trajs)
        cop.to_csv(OUT_DIR / f"baseline_cop_{route}.csv", index=False, encoding="utf-8-sig")
        scal.to_csv(OUT_DIR / f"baseline_scalar_{route}.csv", index=False, encoding="utf-8-sig")
        perfoot.to_csv(OUT_DIR / f"baseline_per_foot_{route}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===== {route.upper()} : COP (mm) =====")
        print(cop[["method", "cite", "cop_x_rmse_mm", "cop_y_rmse_mm", "cop_x_r", "cop_y_r",
                   "cop_x_dev_r", "cop_y_dev_r", "cop_x_ba_bias", "cop_x_ba_lo", "cop_x_ba_hi",
                   "cop_y_ba_bias", "cop_y_ba_lo", "cop_y_ba_hi", "cal_cop_x_rmse_mm",
                   "cal_cop_y_rmse_mm"]].round(3).to_string(index=False))
        print(f"\n===== {route.upper()} : PP (kPa) and PTI (kPa*s) =====")
        print(scal.round(3).to_string(index=False), flush=True)
    if WC26F_FOLDS:
        pd.DataFrame(WC26F_FOLDS).to_csv(OUT_DIR / "wc26f_folds.csv", index=False, encoding="utf-8-sig")
    print("\nWrote baseline_layouts.csv, baseline_cop_*.csv, baseline_scalar_*.csv, baseline_per_foot_*.csv,"
          " wc26f_folds.csv")

if __name__ == "__main__":
    main()
