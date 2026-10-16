from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ our sock
# The 26-point template (diagram map, see data_loader/sock_feet.py) placed in the
# footprint box by the per-axis linear map that sends sensor 1 (hallux) to the
# cohort hallux and sensors 5 / 26 to the cohort lateral-toe / posterior-heel
# extremes. Linear, so it keeps the sock's own geometry exactly.
def sock_uv(coords: np.ndarray) -> np.ndarray:
    u = 0.15 + (coords[:, 0] - 4.5) * (0.65 / 15.0)
    v = 0.08 + (coords[:, 1] - 4.5) * (0.85 / 54.0)
    return np.column_stack([u, v])


LAYOUTS: dict[str, dict] = {
    "walkinsense8": dict(
        ref="[13], [35]", name="Healy et al. / de Castro et al. (WalkinSense)", cells=3,
        fidelity="anatomical",
        source="8 discrete sensors: hallux, a row of four across the metatarsal heads, lateral "
               "midfoot, medial and lateral heel (WalkinSense insole photo, Lin et al. [19] Fig. A1)",
        sensors=[("hallux", 0.15, 0.09), ("MTH1", 0.15, 0.27), ("MTH2", 0.32, 0.26),
                 ("MTH3-4", 0.55, 0.26), ("MTH5", 0.78, 0.30), ("lateral midfoot", 0.80, 0.50),
                 ("medial heel", 0.36, 0.87), ("lateral heel", 0.60, 0.86)]),
    "ciniglio16": dict(
        ref="[15]", name="Ciniglio et al.", cells=3, fidelity="figure",
        source="16 sensors 15x15 mm: 2 toes, 6 forefoot block + 1 lateral forefoot, 2 lateral + "
               "1 medial midfoot, 2x2 hindfoot (Fig. 2, size 38-39)",
        sensors=[("hallux", 0.20, 0.10), ("toe 2", 0.45, 0.11),
                 ("forefoot m-d", 0.20, 0.17), ("forefoot c-d", 0.36, 0.17), ("forefoot l-d", 0.52, 0.17),
                 ("forefoot m-p", 0.20, 0.22), ("forefoot c-p", 0.36, 0.22), ("forefoot l-p", 0.52, 0.22),
                 ("lateral forefoot", 0.72, 0.20), ("lateral midfoot dist", 0.72, 0.39),
                 ("medial midfoot", 0.45, 0.57), ("lateral midfoot prox", 0.68, 0.57),
                 ("heel med-ant", 0.42, 0.81), ("heel lat-ant", 0.62, 0.81),
                 ("heel med-post", 0.42, 0.89), ("heel lat-post", 0.62, 0.89)]),
    "lin8": dict(
        ref="[19]", name="Lin et al. (DRL)", cells=3, fidelity="exact",
        source="8 cells on the footprint-normalised 7x20 grid (Fig. 8c): (x,y) = (4,1) (6,1) (4,4) "
               "(1,6) (4,6) (1,11) (2,16) (4,17), medial at x = 6",
        sensors=[(f"cell({x},{y})", (6 - x + 0.5) / 7.0, (y + 0.5) / 20.0)
                 for x, y in [(4, 1), (6, 1), (4, 4), (1, 6), (4, 6), (1, 11), (2, 16), (4, 17)]]),
    "gautier7": dict(
        ref="[23]", name="Gautier et al. (exhaustive, k = 7)", cells=4, fidelity="figure",
        source="Global-optimal 7 of 16 zones, Table 3: S1, S4, S6, S10, S11, S13, S14",
        sensors=[("S1 post-medial heel", 0.38, 0.93), ("S4 ant-lateral heel", 0.62, 0.78),
                 ("S6 lateral midfoot", 0.72, 0.62), ("S10 MTH2", 0.30, 0.28),
                 ("S11 MTH3", 0.46, 0.28), ("S13 MTH5", 0.80, 0.32), ("S14 hallux", 0.18, 0.08)]),
    "ho10": dict(
        ref="[37]", name="Ho et al. (textile capacitive)", cells=4, fidelity="figure",
        source="10 sensors 20x20 mm: 5 forefoot, 2 lateral midfoot, 3 rearfoot (Fig. 1)",
        sensors=[("hallux", 0.25, 0.12), ("lesser toes", 0.55, 0.12), ("MTH1", 0.17, 0.28),
                 ("MTH3", 0.48, 0.28), ("MTH5", 0.79, 0.28), ("lateral midfoot dist", 0.76, 0.42),
                 ("lateral midfoot prox", 0.67, 0.57), ("anterior heel", 0.60, 0.72),
                 ("posterior heel lat", 0.57, 0.87), ("posterior heel med", 0.27, 0.87)]),
    "daid6": dict(
        ref="[38]", name="Davidovica et al. (DAid smart sock)", cells=4, fidelity="anatomical",
        source="6 textile sock sensors e0..e5 = (+-50, 100), (+-50, 0), (+-25, -100) mm; forefoot "
               "pair under the metatarsal heads, heel pair on the calcaneus",
        sensors=[("forefoot medial", 0.20, 0.27), ("forefoot lateral", 0.78, 0.30),
                 ("midfoot medial", 0.22, 0.55), ("midfoot lateral", 0.80, 0.55),
                 ("heel medial", 0.32, 0.88), ("heel lateral", 0.64, 0.88)]),
    "najafi5": dict(
        ref="[42]", name="Najafi et al. (SmartSox, optical-fiber sock)", cells=3, fidelity="anatomical",
        source="5 fiber-Bragg-grating sites woven into the sock under the big toe, first MTH, fifth MTH, "
               "midfoot (under the arch) and hind foot (Methods, Fig. 1B); sensor size not reported, so a "
               "15 mm square like the other unsized layouts",
        sensors=[("big toe", 0.15, 0.09), ("MTH1", 0.15, 0.27), ("MTH5", 0.80, 0.32),
                 ("midfoot (arch)", 0.25, 0.53), ("heel", 0.47, 0.87)]),
    "hu12": dict(
        ref="[22]", name="Hu et al. (12 FSR)", cells=3, fidelity="exact",
        source="12 FSR coordinates (mm) on a 90x270 mm insole, Fig. 2 table; medial = +x",
        sensors=[(f"FSR{i}", (85.0 - x) / 85.0, (270.0 - y) / 270.0) for i, (x, y) in enumerate(
            [(30, 236), (60, 239), (37, 194), (62, 197), (10, 176), (9, 148), (9, 122), (49, 109),
             (42, 69), (14, 40), (37, 31), (52, 45)], start=1)]),
    "mun9": dict(
        ref="[21]", name="Mun & Choi (9 Pedar sites)", cells=3, fidelity="exact",
        source="Pedar-X sensors #8, #10, #22, #24 (heel), #39 (lateral arch), #67, #70, #72 "
               "(metatarsal heads), #84 (hallux base), Fig. 1 grid",
        sensors=[("#8", 0.36, 0.90), ("#10", 0.64, 0.90), ("#22", 0.36, 0.77), ("#24", 0.64, 0.77),
                 ("#39", 0.79, 0.63), ("#67", 0.79, 0.37), ("#70", 0.21, 0.30), ("#72", 0.50, 0.30),
                 ("#84", 0.21, 0.17)]),
    "zhang8": dict(
        ref="[14]", name="Zhang et al. (SportSole, IEE 8-cell insole)", cells=3, fidelity="figure",
        source="8 piezoresistive cells under the hallux, lesser toes, MTH1, MTH3, MTH5, lateral arch and "
               "medial / lateral calcaneus (Sec. II); positions and sizes from Tseng et al. [18] Fig. 8, "
               "which maps the insole onto the 60 x 21 footprint grid (medial = left). The toe cluster "
               "(~20 cells) is one rectangle of the same area",
        sensors=[(lab, c / 21.0, r / 60.0, w, h) for lab, c, r, w, h in [
            ("hallux", 5.5, 6.5, 3, 7), ("lesser toes", 11.8, 7.9, 5, 4), ("MTH1", 4.5, 16.5, 3, 7),
            ("MTH3", 10.0, 17.5, 4, 7), ("MTH5", 16.0, 19.5, 4, 7), ("lateral arch", 15.5, 33.5, 3, 7),
            ("calcaneus medial", 10.0, 54.0, 4, 8), ("calcaneus lateral", 15.0, 54.0, 4, 8)]]),
    "xian8": dict(
        ref="[16]", name="Xian et al. (improved genetic algorithm)", cells=3, fidelity="exact",
        source="Table II: 8 sensors on the footprint scaled to a 34 x 14 matrix, (x, y) = (10,31) (10,24) "
               "(6,24) (4,21) (6,14) (7,9) (12,8) (10,4); y runs toward the toes and x toward the medial "
               "edge (Fig. 10: lateral midfoot band at low x, CoP ends at the hallux). Size 3 x 3 from "
               "Tseng et al. [18] Table IV (Tseng drew Xian's 10-sensor variant; this is the 8-sensor "
               "insole Xian built and validated)",
        sensors=[(f"S{i}", (14.5 - x) / 14.0, (34.5 - y) / 34.0) for i, (x, y) in enumerate(
            [(10, 31), (10, 24), (6, 24), (4, 21), (6, 14), (7, 9), (12, 8), (10, 4)], start=1)]),
    "tseng8": dict(
        ref="[18]", name="Tseng et al. (ISAC-Discrete + GPBT)", cells=3, fidelity="figure",
        source="Fig. 11: 8 locations on the 20 x 7 design grid, each a 3 x 3 block on the 60 x 21 "
               "footprint grid (72 cells, medial = left); block centres digitised from the figure",
        sensors=[(lab, c / 21.0, r / 60.0) for lab, c, r in [
            ("hallux", 4.45, 4.65), ("MTH1-2", 7.5, 16.95), ("MTH2-3", 10.45, 16.95),
            ("lateral forefoot", 16.55, 20.05), ("lateral midfoot", 16.55, 26.35),
            ("heel anterior", 10.45, 45.9), ("heel centre", 10.45, 48.85),
            ("posterior heel", 7.4, 55.15)]]),
}


def layout_uv(name: str) -> np.ndarray:
    return np.array([s[1:3] for s in LAYOUTS[name]["sensors"]], dtype=float)


def layout_sizes(name: str) -> list[tuple[int, int]]:
    """(w, h) in cells per sensor: its own size when given, else the layout's square `cells`."""
    lay = LAYOUTS[name]
    return [(int(s[3]), int(s[4])) if len(s) == 5 else (lay["cells"], lay["cells"]) for s in lay["sensors"]]
