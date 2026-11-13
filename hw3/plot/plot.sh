#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 plot/plot_cfg.py
python3 plot/plot_apg_1.py
python3 plot/plot_apg_2.py
python3 plot/plot_apg_3.py
