from __future__ import annotations

import math

import numpy as np
import torch

from config import RESAMPLE_POINTS

TARGET = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDAS = (0.003, 0.01)
AP_WEIGHT = 0.25
STEPS = 400
SEEDS = 5                  # field-head models per fold; readouts are averaged
INNER_K = 5
BETAS = np.linspace(0.0, 1.0, 11)
ALPHAS = np.linspace(0.0, 0.9, 10)


# ------------------------------------------------------------------ exact centroid head
def _interp_entries(t, valid, offset):
    idx = np.where(valid)[0]
    if len(idx) < 2 or t[idx[-1]] == t[idx[0]]:
        return None
    u = (t[idx] - t[idx[0]]) / (t[idx[-1]] - t[idx[0]])
    u, first = np.unique(u, return_index=True)
    idx = idx[first]
    if len(u) < 2:
        return None
    j = np.clip(np.searchsorted(u, TARGET, side="right") - 1, 0, len(u) - 2)
    a = np.clip((TARGET - u[j]) / (u[j + 1] - u[j]), 0, 1)
    p = np.arange(len(TARGET))
    return np.concatenate([p, p]), np.concatenate([idx[j], idx[j + 1]]) + offset, np.concatenate([1 - a, a])


class Stances:
    def __init__(self, f, layout="ours26"):
        ch, frames, P, F, W, o = f.ch[layout], [], [], [], [], 0
        self.per_stance, self.n_stances = [], len(f.stances)
        for i, (s, e) in enumerate(f.stances):
            v = ch[s:e]
            ent = _interp_entries(f.time[s:e], v.sum(1) > 0, o)
            if ent is None:                       # < 2 loaded frames: no trajectory, as resample_series
                continue
            frames.append(v)
            P.append(ent[0]); F.append(ent[1]); W.append(ent[2])
            self.per_stance.append((i, ent))
            o += len(v)
        self.V = np.concatenate(frames).astype(np.float32)
        self.p, self.fi = np.concatenate(P), np.concatenate(F)
        self.w = (np.concatenate(W) / len(frames)).astype(np.float32)
        self.xy = f.xy[layout].astype(float)

    def frame_cop(self, w):
        x = self.V * w
        tot = x.sum(1)
        cop = np.zeros((len(x), 2))
        ok = tot > 0
        cop[ok] = (x[ok] @ self.xy) / tot[ok, None]
        return cop

    def mean_traj(self, w):
        cop = self.frame_cop(w)
        out = np.zeros((len(TARGET), 2))
        np.add.at(out, self.p, cop[self.fi] * self.w[:, None])
        return out

    def stance_trajs(self, w):
        cop = self.frame_cop(w)
        out = np.full((self.n_stances, len(TARGET), 2), np.nan)
        for i, (p, fi, wt) in self.per_stance:
            t = np.zeros((len(TARGET), 2))
            np.add.at(t, p, cop[fi] * wt[:, None])
            out[i] = t
        return out


class _Batch:
    def __init__(self, st: list[Stances], targets: list[np.ndarray]):
        n, e, F = max(len(s.V) for s in st), max(len(s.p) for s in st), len(st)
        V = np.zeros((F, n, st[0].V.shape[1]), np.float32)
        P, FI, W = np.zeros((F, e), np.int64), np.zeros((F, e), np.int64), np.zeros((F, e), np.float32)
        for i, s in enumerate(st):
            V[i, :len(s.V)] = s.V
            P[i, :len(s.p)], FI[i, :len(s.p)], W[i, :len(s.p)] = s.p, s.fi, s.w
        scale = float(np.percentile(V[V > 0], 90)) if (V > 0).any() else 1.0
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEVICE)
        self.V, self.P, self.FI, self.W = t(V / scale), t(P, torch.int64), t(FI, torch.int64), t(W)
        self.R = t(np.stack(targets))
        self.xy, self.F = t(st[0].xy), F


def _corr(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def _traj(B, w):
    x = B.V * w
    tot = x.sum(-1, keepdim=True)
    cop = (x @ B.xy) / torch.where(tot > 0, tot, torch.ones_like(tot))
    g = torch.gather(cop, 1, B.FI.unsqueeze(-1).expand(-1, -1, 2)) * B.W.unsqueeze(-1)
    return torch.zeros(B.F, len(TARGET), 2, device=DEVICE).scatter_add(1, B.P.unsqueeze(-1).expand(-1, -1, 2), g)


def fit_centroid(st: list[Stances], targets: list[np.ndarray], lam: float) -> np.ndarray:
    B = _Batch(st, targets)
    k = st[0].V.shape[1]
    th = torch.zeros(k, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(STEPS):
        opt.zero_grad()
        w = torch.exp(th.clamp(-12, 12))
        T = _traj(B, w)
        loss = -(_corr(T[..., 0], B.R[..., 0]) + AP_WEIGHT * _corr(T[..., 1], B.R[..., 1])).mean() \
            + lam * (th ** 2).sum()
        loss.backward()
        opt.step()
    w = torch.exp(th.clamp(-12, 12)).detach().cpu().numpy().astype(float)
    return w / w.mean()


# ------------------------------------------------------------------ fusion
def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _score(P, Y):
    return float(np.nanmean([_pearson(p[:, 0], y[:, 0]) + AP_WEIGHT * _pearson(p[:, 1], y[:, 1])
                             for p, y in zip(P, Y)]))


def _affine(X, Y):
    ab = []
    for ax in range(2):
        x, y = X[..., ax].ravel(), Y[..., ax].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        b, a = np.polyfit(x[ok], y[ok], 1) if ok.sum() > 2 and np.std(x[ok]) > 0 else (0.0, np.nanmean(y))
        ab.append((a, b))
    return np.array(ab)


def apply_affine(ab, P):
    return np.stack([ab[ax, 0] + ab[ax, 1] * P[..., ax] for ax in range(2)], -1)


class Model:
    def __init__(self, w, lam, fields, A_c, A_f, beta, alpha, T):
        self.w, self.lam, self.fields = w, lam, fields
        self.A_c, self.A_f, self.beta, self.alpha, self.T = A_c, A_f, beta, alpha, T

    def fuse(self, c, f):
        """c, f: (..., 100, 2) centroid-head and field-head trajectories -> fused COP."""
        mix = (1 - self.beta) * apply_affine(self.A_c, c) + self.beta * apply_affine(self.A_f, f)
        return (1 - self.alpha) * mix + self.alpha * self.T


def fit(train, route, seed, cb, seeds: int = SEEDS):
    st = {id(f): Stances(f) for f in train}
    Y = {id(f): f.ref_mean for f in train}
    subj = np.array(sorted({f.subject for f in train}))
    order = np.random.default_rng(seed).permutation(subj)
    folds = [set(order[i::INNER_K]) for i in range(INNER_K)]
    oof_c = {lam: {} for lam in LAMBDAS}
    oof_f = {}
    for k, fs in enumerate(folds):
        fit_f = [f for f in train if f.subject not in fs]
        val_f = [f for f in train if f.subject in fs]
        for lam in LAMBDAS:
            w = fit_centroid([st[id(f)] for f in fit_f], [Y[id(f)] for f in fit_f], lam)
            for f in val_f:
                oof_c[lam][id(f)] = st[id(f)].mean_traj(w)
        m = cb.fit_lstm(fit_f, "ours26", route, seed=seed * 100 + k)
        for f in val_f:
            traj, _, _ = cb.predict("lstm", m, f, "ours26", route)
            oof_f[id(f)] = np.nanmean(traj, axis=0)
    ids = [id(f) for f in train]
    Yt = np.stack([Y[i] for i in ids])
    lam = max(LAMBDAS, key=lambda l: _score([oof_c[l][i] for i in ids], Yt))
    Xc, Xf = np.stack([oof_c[lam][i] for i in ids]), np.stack([oof_f[i] for i in ids])
    A_c, A_f = _affine(Xc, Yt), _affine(Xf, Yt)
    Cc, Cf = apply_affine(A_c, Xc), apply_affine(A_f, Xf)
    T = Yt.mean(0)
    beta = max(BETAS, key=lambda b: _score((1 - b) * Cc + b * Cf, Yt))
    mix = (1 - beta) * Cc + beta * Cf
    alpha = max(ALPHAS, key=lambda a: _score((1 - a) * mix + a * T, Yt))
    w = fit_centroid([st[i] for i in ids], list(Yt), lam)
    fields = [cb.fit_lstm(train, "ours26", route, seed=seed + 1000 * s) for s in range(seeds)]   # = LSTM baseline seeds
    return Model(w, lam, fields, A_c, A_f, beta, alpha, T)


def predict(model: Model, f, route, cb):
    st = Stances(f)
    c = st.stance_trajs(model.w)
    outs = [cb.predict("lstm", m, f, "ours26", route) for m in model.fields]
    ftraj = np.nanmean([o[0] for o in outs], axis=0)
    pp = np.mean([o[1] for o in outs], axis=0)
    pti = np.mean([o[2] for o in outs], axis=0)
    return model.fuse(c, ftraj), pp, pti
