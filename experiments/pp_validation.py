from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments import pressure_common as C

FF = np.isin(C.SOCK_REGION, C.FOREFOOT)
HEEL = C.SOCK_REGION == "heel"
FF_IDX = [C.REGIONS.index(g) for g in C.FOREFOOT]
HEEL_IDX = C.REGIONS.index("heel")


def sock_values(o: dict) -> dict:
    pp = o["site_pp"]
    reg = C.regional_max(pp, C.SOCK_REGION)
    load = o["load"] / o["dur"]
    return dict(whole=pp.max(axis=1).mean(), regional=reg.mean(axis=0), site=pp.mean(axis=0),
                conc=(pp.max(axis=1) / load).mean(),
                ffheel=(pp[:, FF].max(axis=1) / np.maximum(pp[:, HEEL].max(axis=1), 1e-9)).mean())


def ref_values(r: dict) -> dict:
    load = r["load"] / r["dur"]
    return dict(whole=r["pp"].mean(), regional=r["reg_pp"].mean(axis=0),
                site=r["site_out"]["site_pp"].mean(axis=0), conc=(r["pp"] / load).mean(),
                ffheel=(r["reg_pp"][:, FF_IDX].max(axis=1) / r["reg_pp"][:, HEEL_IDX]).mean(),
                virtual=r["site_out"]["site_pp"].max(axis=1).mean(),
                virtual_reg=C.regional_max(r["site_out"]["site_pp"], C.SOCK_REGION).mean(axis=0))


def evaluate(subset: str, keys: list, ref: dict, sock: dict, n_boot: int):
    groups = C.subjects(keys)
    print(f"[{subset}] {len(keys)} feet, building variants ...", flush=True)
    outs, cals = C.build_variants(ref, sock, keys)
    R = {k: ref_values(ref[k]) for k in keys}
    S = {v: {k: sock_values(outs[v][k]) for k in keys} for v in outs}
    y = np.array([R[k]["whole"] for k in keys])
    whole, regional, sites, scalefree = [], [], [], []

    # ---- whole foot
    def add(label, kind, pred, units):
        whole.append(dict(subset=subset, method=label, kind=kind, **C.score(pred, y, groups, units, n_boot)))

    add("Layout ceiling: F-SCAN at the 26 sites", "virtual", np.array([R[k]["virtual"] for k in keys]), "kPa")
    for v in C.VARIANTS:
        add(v, "sock", np.array([S[v][k]["whole"] for k in keys]), C.variant_units(v))
    lw, weights = C.learned_weight_predictions(ref, sock, keys, cals, "pp")
    add("DSP + quantile map + learned channel weights", "sock", np.array([lw[k] for k in keys]), "kPa")

    # ---- regional
    for i, g in enumerate(C.REGIONS):
        yr = np.array([R[k]["regional"][i] for k in keys])
        cand = {"Layout ceiling (virtual)": (np.array([R[k]["virtual_reg"][i] for k in keys]), "kPa")}
        for v in ("pipeline, raw ADC", "DSP + quantile map (per channel)"):
            cand[v] = (np.array([S[v][k]["regional"][i] for k in keys]), C.variant_units(v))
        for m, (p, u) in cand.items():
            regional.append(dict(subset=subset, region=g, n_sites=int((C.SOCK_REGION == g).sum()), method=m,
                                 capture=float(np.nanmean(np.array([R[k]["virtual_reg"][i] for k in keys]) / yr)),
                                 **C.score(p, yr, groups, u)))

    # ---- matched site (measurement question; quantile map cannot change rho, by construction)
    for j in range(C.N_CH):
        yj = np.array([R[k]["site"][j] for k in keys])
        row = dict(subset=subset, channel=j + 1, region=C.SOCK_REGION[j], ref_mean_kpa=float(yj.mean()),
                   sock_alive=float(np.mean([S["pipeline, raw ADC"][k]["site"][j] > 0 for k in keys])))
        for v, tag in (("pipeline, raw ADC", "pipeline"), ("DSP, raw ADC", "dsp"),
                       ("DSP + quantile map (per channel)", "dsp_qm")):
            a = C.score(np.array([S[v][k]["site"][j] for k in keys]), yj, groups, C.variant_units(v))
            row.update({f"{tag}_{m}": a.get(m) for m in ("spearman_rho", "pearson_r", "ccc", "rmse")})
        sites.append(row)

    # ---- scale-free
    for metric, label in (("conc", "PP concentration (PP / own mean load)"), ("ffheel", "forefoot / heel PP")):
        yr = np.array([R[k][metric] for k in keys])
        for v in ("pipeline, raw ADC", "DSP, raw ADC", "DSP + quantile map (per channel)"):
            p = np.array([S[v][k][metric] for k in keys])
            scalefree.append(dict(subset=subset, outcome=label, method=v,
                                  **C.score(p, yr, groups, "kPa", n_boot, absolute=False)))

    per_foot = pd.DataFrame([dict(subset=subset, sex=k[0], subject=k[1], foot=k[2], ref_pp=R[k]["whole"],
                                  virtual_pp=R[k]["virtual"],
                                  **{f"pp[{v}]": S[v][k]["whole"] for v in C.VARIANTS},
                                  **{"pp[learned weights]": lw[k]}) for k in keys])
    w = pd.DataFrame(weights).T
    w.columns = [f"ch{j + 1}" for j in range(C.N_CH)]
    w.insert(0, "subset", subset)
    return (pd.DataFrame(whole), pd.DataFrame(regional), pd.DataFrame(sites), pd.DataFrame(scalefree),
            per_foot, w)


def dsp_report(sock: dict, keys: list) -> pd.DataFrame:
    rows = []
    for k in keys:
        i = sock[k]["dsp_info"]
        rows.append(dict(subject=k[1], foot=k[2], noise_sigma_adc=np.median(i["sigma"]) / 10,
                         drift_median_adc=np.median(i["drift"]) / 10, drift_max_adc=np.max(i["drift"]) / 10,
                         hf_removed_adc=np.median(i["hf_removed"]) / 10, spike_rate=np.max(i["spike_rate"])))
    return pd.DataFrame(rows)


def write_md(whole, regional, sites, scalefree, base, qc, dsp):
    cols = ["method", "units", "n_feet", "pearson_r", "spearman_rho", "rho_ci_low", "rho_ci_high",
            "ccc", "icc_a1", "rmse", "bias", "loa_low", "loa_high"]
    L = ["# PP validation, private cohort (physical sock vs F-SCAN)", "",
         "Generated by `python -m experiments.pp_validation`. Between subject-feet; every fit LOSO by participant.",
         "Rank metrics (r, rho) use each prediction as produced. Absolute metrics (CCC, ICC(A,1), RMSE,",
         "limits of agreement) are in kPa; raw-ADC rows reach kPa through an outcome-level LOSO linear map.",
         f"Primary method: `{C.PRIMARY}`. Every published method is scored raw and after the same",
         "body-weight step (`readout` column). Subset `cal30` keeps the reference sessions whose implied mass is",
         "within 30 % of the measured mass (rule R3).", "",
         "## Reference QC", "", C.md_table(qc, ["subject", "foot", "implied_kg", "mass_kg", "implied_over_true", "pp",
                                               "R1_implausible_mass", "R2_lr_pp_mismatch", "R3_calibration_off_30pct",
                                               "qc_pass", "cal30_pass"], "{:.2f}"),
         "", "Subset `bw` scores every foot with a measured mass against the body-weight-calibrated reference",
         "(each recording scaled by measured mass / implied mass; see experiments/pressure_common.py).", ""]
    for sub in ("all", "cal30", "qc", "bw"):
        L += [f"## Whole-foot PP, subset `{sub}`", "", C.md_table(whole[whole.subset == sub], cols), ""]
        b = base[base.subset == sub]
        if len(b):
            L += [f"### Existing baselines (experiments/cop_benchmark.py), subset `{sub}`", "",
                  C.md_table(b, ["method", "readout", "n_feet", "pearson_r", "spearman_rho", "ccc", "icc_a1", "rmse"]), ""]
        piv = regional[regional.subset == sub].pivot_table(index=["region", "n_sites", "capture"], columns="method",
                                                           values="spearman_rho", sort=False).reset_index()
        L += [f"## Regional PP, Spearman rho, subset `{sub}`", "",
              C.md_table(piv, list(piv.columns), "{:+.2f}"), ""]
        L += [f"## Matched-site PP, subset `{sub}`", "",
              C.md_table(sites[sites.subset == sub], ["channel", "region", "ref_mean_kpa", "sock_alive",
                                                      "pipeline_spearman_rho", "dsp_spearman_rho",
                                                      "dsp_qm_spearman_rho", "dsp_qm_ccc", "dsp_qm_rmse"], "{:.2f}"), ""]
        L += [f"## Scale-free PP outcomes, subset `{sub}`", "",
              C.md_table(scalefree[scalefree.subset == sub], ["outcome", "method", "n_feet", "pearson_r",
                                                              "spearman_rho", "rho_ci_low", "rho_ci_high"]), ""]
    L += ["## DSP diagnostics (ADC counts)", "", C.md_table(dsp, list(dsp.columns), "{:.3f}")]
    (C.OUT / "pp_validation.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-parse recordings instead of using the caches")
    ap.add_argument("--n-boot", type=int, default=2000, help="participant bootstrap resamples (0 = none)")
    args = ap.parse_args()
    ref = C.load_reference(args.refresh)
    sock = C.load_sock(args.refresh)
    sets, qc = C.subsets(ref, sock)
    parts = [evaluate(s, k, r, sock, args.n_boot) for s, (k, r) in sets.items()]
    whole, regional, sites, scalefree, per_foot, weights = (pd.concat(p, ignore_index=True) for p in zip(*parts))
    fac = C.bodyweight_factors(sock, sets["all"][0])
    base = pd.concat([C.existing_baselines("pp", k, r, fac).assign(subset=s) for s, (k, r) in sets.items()],
                     ignore_index=True)
    dsp = dsp_report(sock, sets["all"][0])
    C.OUT.mkdir(parents=True, exist_ok=True)
    for name, df in (("wholefoot", whole), ("regional", regional), ("matched_site", sites),
                     ("scale_free", scalefree), ("per_foot", per_foot), ("learned_weights", weights),
                     ("existing_baselines", base)):
        df.to_csv(C.OUT / f"pp_{name}.csv", index=False)
    qc.to_csv(C.OUT / "reference_qc.csv", index=False)
    dsp.to_csv(C.OUT / "dsp_diagnostics.csv", index=False)
    write_md(whole, regional, sites, scalefree, base, qc, dsp)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["subset", "method", "units", "pearson_r", "spearman_rho", "ccc", "icc_a1", "rmse"]
    print(whole[show].round(3).to_string(index=False))
    print(scalefree[["subset", "outcome", "method", "pearson_r", "spearman_rho"]].round(3).to_string(index=False))
    print(f"wrote {C.OUT / 'pp_validation.md'}")


if __name__ == "__main__":
    main()
