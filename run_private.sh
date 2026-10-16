#!/usr/bin/env sh
# Private cohort: 26-channel sock against F-Scan. Steps run in order; each reads the previous output.
set -e
python main.py prepare        # F-Scan mean COP and mean fields per subject-foot
python main.py physical       # COP benchmark (physical + virtual routes), WC26-F and 16 comparators
python main.py pressure       # PP and PTI agreement (needs results/baseline_per_foot_physical.csv)
python main.py mix            # simultaneous recordings (needs results/wc26f_folds.csv)
