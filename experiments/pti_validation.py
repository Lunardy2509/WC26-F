from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats as sstats

from experiments import pressure_common as C

FF = np.isin(C.SOCK_REGION, C.FOREFOOT)
HEEL = C.SOCK_REGION == "heel"
FF_IDX = [C.REGIONS.index(g) for g in C.FOREFOOT]
HEEL_IDX = C.REGIONS.index("heel")


def sock_values(o: dict) -> dict:
    pti = o["site_pti"]
    whole = pti.max(axis=1)
    return dict(whole=whole.mean(), imax=o["imax"].mean(), dur=o["dur"].mean(), amp=(whole / o["dur"]).mean(),
                share=(whole / o["load"]).mean(), load=o["load"].mean(),
                ffheel=(pti[:, FF].max(axis=1) / np.maximum(pti[:, HEEL].max(axis=1), 1e-9)).mean(),
                regional=C.regional_max(pti, C.SOCK_REGION).mean(axis=0), reg_imax=o["imax_reg"].mean(axis=0),
                site=pti.mean(axis=0))


def ref_values(r: dict) -> dict:
    so = r["site_out"]
    return dict(whole=r["pti"].mean(), imax=r["imax"].mean(), dur=r["dur"].mean(),
                amp=(r["pti"] / r["dur"]).mean(), share=(r["pti"] / r["load"]).mean(), load=r["load"].mean(),
                ffheel=(r["reg_pti"][:, FF_IDX].max(axis=1) / r["reg_pti"][:, HEEL_IDX]).mean(),
                regional=r["reg_pti"].mean(axis=0), reg_imax=r["reg_imax"].mean(axis=0),
                site=so["site_pti"].mean(axis=0),
                virtual=so["site_pti"].max(axis=1).mean(), virtual_imax=so["imax"].mean(),
                virtual_reg=C.regional_max(so["site_pti"], C.SOCK_REGION).mean(axis=0),
                virtual_reg_imax=so["imax_reg"].mean(axis=0))


def partial_r(x, y, cx, cy):
    """r(log x, log y) after regressing both on log durations of both devices."""
    Z = np.c_[np.ones(len(x)), np.log(cx), np.log(cy)]
    res = lambda v: np.log(v) - Z @ np.linalg.lstsq(Z, np.log(v), rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def evaluate(subset: str, keys: list, ref: dict, sock: dict, n_boot: int):
    groups = C.subjects(keys)
    print(f"[{subset}] {len(keys)} feet, building variants ...", flush=True)
    outs, cals = C.build_variants(ref, sock, keys)
    R = {k: ref_values(ref[k]) for k in keys}
    S = {v: {k: sock_values(outs[v][k]) for k in keys} for v in outs}
    get = lambda D, f: np.array([D[k][f] for k in keys])
    whole, decomposition, regional, sites = [], [], [], []

    # ---- whole foot, both definitions
    lw, weights = C.learned_weight_predictions(ref, sock, keys, cals, "pti")
    for field, label in (("whole", "max site-specific PTI"), ("imax", "integral of instantaneous max")):
        y = get(R, field)
        vfield = "virtual" if field == "whole" else "virtual_imax"
        rows = [("Layout ceiling: F-SCAN at the 26 sites", "virtual", get(R, vfield), "kPa")]
        rows += [(v, "sock", get(S[v], field), C.variant_units(v)) for v in C.VARIANTS]
        if field == "whole":
            rows.append(("DSP + quantile map + learned channel weights", "sock", np.array([lw[k] for k in keys]), "kPa"))
        for m, kind, p, u in rows:
            whole.append(dict(subset=subset, definition=label, method=m, kind=kind,
                              **C.score(p, y, groups, u, n_boot)))

    # ---- what carries the agreement: duration vs amplitude, plus scale-free outcomes
    y_t = get(R, "dur")
    for v in ("pipeline, raw ADC", "DSP, raw ADC", "DSP + quantile map (per channel)"):
        sv = S[v]
        for field, label, absolute in (("dur", "stance duration T (s)", True),
                                       ("amp", "amplitude A = PTI / T", C.variant_units(v) == "kPa"),
                                       ("share", "peak-site share of load impulse", False),
                                       ("ffheel", "forefoot / heel PTI", False),
                                       ("load", "load impulse int sum p dt (repo pressure_impulse analogue)", False)):
            units = "s" if field == "dur" else ("kPa" if absolute else C.variant_units(v))
            a = C.score(get(sv, field), get(R, field), groups, units, n_boot if field in ("share", "amp") else 0,
                        absolute=absolute)
            decomposition.append(dict(subset=subset, method=v, outcome=label, **a))
        decomposition.append(dict(subset=subset, method=v, outcome="partial r(log PTI | log T, both devices)",
                                  pearson_r=partial_r(get(sv, "whole"), get(R, "whole"), get(sv, "dur"), y_t),
                                  n_feet=len(keys)))
        decomposition.append(dict(subset=subset, method=v, outcome="r(sock T, reference PTI)",
                                  pearson_r=float(sstats.pearsonr(get(sv, "dur"), get(R, "whole"))[0]),
                                  spearman_rho=float(sstats.spearmanr(get(sv, "dur"), get(R, "whole"))[0]),
                                  n_feet=len(keys)))

    # ---- regional, both definitions
    for i, g in enumerate(C.REGIONS):
        for field, vfield, label in (("regional", "virtual_reg", "max site-specific"),
                                     ("reg_imax", "virtual_reg_imax", "integral of instantaneous max")):
            yr = np.array([R[k][field][i] for k in keys])
            vr = np.array([R[k][vfield][i] for k in keys])
            cand = {"Layout ceiling (virtual)": (vr, "kPa")}
            for v in ("pipeline, raw ADC", "DSP + quantile map (per channel)"):
                cand[v] = (np.array([S[v][k][field][i] for k in keys]), C.variant_units(v))
            for m, (p, u) in cand.items():
                regional.append(dict(subset=subset, region=g, definition=label,
                                     n_sites=int((C.SOCK_REGION == g).sum()), method=m,
                                     capture=float(np.nanmean(vr / yr)), **C.score(p, yr, groups, u)))

    # ---- matched site
    for j in range(C.N_CH):
        yj = np.array([R[k]["site"][j] for k in keys])
        row = dict(subset=subset, channel=j + 1, region=C.SOCK_REGION[j], ref_mean_kpa_s=float(yj.mean()))
        for v, tag in (("pipeline, raw ADC", "pipeline"), ("DSP, raw ADC", "dsp"),
                       ("DSP + quantile map (per channel)", "dsp_qm")):
            a = C.score(np.array([S[v][k]["site"][j] for k in keys]), yj, groups, C.variant_units(v))
            row.update({f"{tag}_{m}": a.get(m) for m in ("spearman_rho", "pearson_r", "ccc", "rmse")})
        sites.append(row)

    per_foot = pd.DataFrame([dict(subset=subset, sex=k[0], subject=k[1], foot=k[2], ref_pti=R[k]["whole"],
                                  ref_dur=R[k]["dur"], virtual_pti=R[k]["virtual"],
                                  **{f"pti[{v}]": S[v][k]["whole"] for v in C.VARIANTS},
                                  **{f"dur[{v}]": S[v][k]["dur"] for v in ("pipeline, raw ADC", "DSP, raw ADC")},
                                  **{"pti[learned weights]": lw[k]}) for k in keys])
    w = pd.DataFrame(weights).T
    w.columns = [f"ch{j + 1}" for j in range(C.N_CH)]
    w.insert(0, "subset", subset)
    return (pd.DataFrame(whole), pd.DataFrame(decomposition), pd.DataFrame(regional), pd.DataFrame(sites),
            per_foot, w)


def write_md(whole, decomp, regional, sites, base, qc):
    cols = ["method", "units", "n_feet", "pearson_r", "spearman_rho", "rho_ci_low", "rho_ci_high",
            "ccc", "icc_a1", "rmse", "bias", "loa_low", "loa_high"]
    L = ["# PTI validation, private cohort (physical sock vs F-SCAN)", "",
         "Generated by `python -m experiments.pti_validation`. Between subject-feet; every fit LOSO by participant.",
         "PTI in kPa*s. Raw-ADC rows reach kPa*s through an outcome-level LOSO linear map.",
         f"Primary method: `{C.PRIMARY}`. Every published method is scored raw and after the same",
         "body-weight step (`readout` column). Subset `cal30` keeps the reference sessions whose implied mass is",
         "within 30 % of the measured mass (rule R3).", "",
         "## Reference QC", "", C.md_table(qc, ["subject", "foot", "implied_kg", "mass_kg", "implied_over_true", "pti", "qc_pass",
                                               "R3_calibration_off_30pct", "cal30_pass"],
                                               "{:.2f}"),
         "", "Subset `bw` scores every foot with a measured mass against the body-weight-calibrated reference",
         "(each recording scaled by measured mass / implied mass; see experiments/pressure_common.py)."]
    for sub in ("all", "cal30", "qc", "bw"):
        for d in whole.definition.unique():
            t = whole[(whole.subset == sub) & (whole.definition == d)]
            L += [f"## Whole-foot PTI ({d}), subset `{sub}`", "", C.md_table(t, cols), ""]
        b = base[base.subset == sub]
        if len(b):
            L += [f"### Existing baselines (experiments/cop_benchmark.py), subset `{sub}`", "",
                  C.md_table(b, ["method", "readout", "n_feet", "pearson_r", "spearman_rho", "ccc", "icc_a1", "rmse"]), ""]
        L += [f"## Duration vs amplitude, and scale-free PTI, subset `{sub}`", "",
              C.md_table(decomp[decomp.subset == sub], ["method", "outcome", "units", "pearson_r", "spearman_rho",
                                                        "rho_ci_low", "rho_ci_high", "ccc", "rmse"]), ""]
        piv = regional[regional.subset == sub].pivot_table(index=["region", "definition", "n_sites", "capture"],
                                                           columns="method", values="spearman_rho",
                                                           sort=False).reset_index()
        L += [f"## Regional PTI, Spearman rho, subset `{sub}`", "", C.md_table(piv, list(piv.columns), "{:+.2f}"), ""]
        L += [f"## Matched-site PTI, subset `{sub}`", "",
              C.md_table(sites[sites.subset == sub], ["channel", "region", "ref_mean_kpa_s", "pipeline_spearman_rho",
                                                      "dsp_spearman_rho", "dsp_qm_spearman_rho", "dsp_qm_ccc",
                                                      "dsp_qm_rmse"], "{:.2f}"), ""]
    (C.OUT / "pti_validation.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-parse recordings instead of using the caches")
    ap.add_argument("--n-boot", type=int, default=2000, help="participant bootstrap resamples (0 = none)")
    args = ap.parse_args()
    ref = C.load_reference(args.refresh)
    sock = C.load_sock(args.refresh)
    sets, qc = C.subsets(ref, sock)
    parts = [evaluate(s, k, r, sock, args.n_boot) for s, (k, r) in sets.items()]
    whole, decomp, regional, sites, per_foot, weights = (pd.concat(p, ignore_index=True) for p in zip(*parts))
    fac = C.bodyweight_factors(sock, sets["all"][0])
    base = pd.concat([C.existing_baselines("pti", k, r, fac).assign(subset=s) for s, (k, r) in sets.items()],
                     ignore_index=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    for name, df in (("wholefoot", whole), ("decomposition", decomp), ("regional", regional),
                     ("matched_site", sites), ("per_foot", per_foot), ("learned_weights", weights),
                     ("existing_baselines", base)):
        df.to_csv(C.OUT / f"pti_{name}.csv", index=False)
    write_md(whole, decomp, regional, sites, base, qc)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["subset", "definition", "method", "units", "pearson_r", "spearman_rho", "ccc", "icc_a1", "rmse"]
    print(whole[show].round(3).to_string(index=False))
    print(decomp[["subset", "method", "outcome", "pearson_r", "spearman_rho", "ccc"]].round(3).to_string(index=False))
    print(f"wrote {C.OUT / 'pti_validation.md'}")


if __name__ == "__main__":
    main()
