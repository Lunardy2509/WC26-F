#!/usr/bin/env sh
# StepUP-P150: every method on the virtual sock, footwear BF / ST / P1 / P2 x speed W1 / W3 / W4.
set -e
python main.py stepup                       # all 12 conditions, every method (long; add --skip-deep for a fast run)
python main.py stepup-ceiling               # 26-site PP / PTI ceiling (reads the stepup extraction cache)
