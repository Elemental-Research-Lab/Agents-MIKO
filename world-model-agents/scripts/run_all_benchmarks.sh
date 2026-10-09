#!/usr/bin/env bash
# Reproduce every benchmark in results/. About 40 minutes on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/../bench"
python simulator_sp500.py --validate     # 1996-1999: choose the world-model configuration
python simulator_sp500.py                 # 2000-2022 out-of-sample test
python simulator_exchange.py              # public GluonTS exchange_rate_nips benchmark
python state_reader.py                    # S&P 500 2000-2022
python scenarios_crisis.py                # crisis fill-in 2000-2022
python planner_portfolio.py               # 20 stocks, monthly 1995-2022
