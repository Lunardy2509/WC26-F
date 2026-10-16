from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats

from config import OUT_DIR, STEPUP_DIR

ROOT_PUB = STEPUP_DIR
OUT = OUT_DIR / "public_baselines"
CONDITIONS = ("BF_W1", "BF_W3", "BF_W4")


def per_foot(recs, mass) -> pd.DataFrame:
    rows = []
    for r in recs:
        X, t = r["vch"]["ours26"].astype(float), r["time"]
        pp, pti, load = [], [], []
        for s, e in r["stances"]:
            seg, tt = X[s:e], t[s:e]
            pp.append(seg.max())
            pti.append(np.trapezoid(seg, tt, axis=0).max())
            load.append(np.trapezoid(seg.sum(1), tt) / max(tt[-1] - tt[0], 1e-9))
        c = mass.get(r["subject"], np.nan) / np.mean(load)
        rows.append(dict(subject=r["subject"], foot=r["foot"], mass_kg=mass.get(r["subject"], np.nan),
                         ref_pp=np.mean(r["pp"]), ref_pti=np.mean(r["pti"]), pp=np.mean(pp), pti=np.mean(pti),
                         pp_bw=c * np.mean(pp), pti_bw=c * np.mean(pti)))
    return pd.DataFrame(rows)


def main():
    meta = pd.read_csv(ROOT_PUB / "participant_metadata.csv")
    mass = {f"{int(i):03d}": float(m) for i, m in zip(meta.ParticipantID, meta["Weight (Kg)"])}
    feet, summ = [], []
    for cond in CONDITIONS:
        with open(OUT / "cache" / f"{cond}.pkl", "rb") as fh:
            recs = pickle.load(fh)
        d = per_foot(recs, mass).assign(condition=cond)
        del recs
        d = d[d.mass_kg.notna()]
        feet.append(d)
        for q in ("pp", "pti"):
            for tag, col in (("26 sites, kPa", q), ("26 sites + body-weight step", f"{q}_bw")):
                summ.append(dict(condition=cond, metric=q.upper(), readout=tag, n_feet=len(d),
                                 n_participants=d.subject.nunique(), r=stats.pearsonr(d[col], d[f"ref_{q}"])[0],
                                 rho=stats.spearmanr(d[col], d[f"ref_{q}"])[0]))
        print(pd.DataFrame(summ)[lambda x: x.condition == cond].round(3).to_string(index=False), flush=True)
    pd.concat(feet).to_csv(OUT / "pressure_ceiling_26_per_foot.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(OUT / "pressure_ceiling_26.csv", index=False, encoding="utf-8-sig")
    print("wrote", OUT / "pressure_ceiling_26.csv")


if __name__ == "__main__":
    main()
