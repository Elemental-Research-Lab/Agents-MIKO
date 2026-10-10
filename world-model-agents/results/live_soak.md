# Live soak test (replayed closes 2022-10-03 to 2022-12-28, 61 sessions)

Each simulated evening: pull the new close, rebuild and gate the models, hot-swap, then call every endpoint.

Failures: 0

| step | mean | median | p95 | max |
|---|---|---|---|---|
| refresh_s | 4.81 | 4.83 | 5.29 | 5.84 |
| state_s | 0.02 | 0.02 | 0.02 | 0.04 |
| simulate_s | 0.49 | 0.48 | 0.58 | 0.64 |
| scenario_s | 0.11 | 0.11 | 0.14 | 0.16 |
| plan_s | 0.47 | 0.45 | 0.57 | 0.62 |

Times in seconds on 2 CPU cores. Refresh includes refitting every world model and running all gates.
First call after a swap may include fitting models for a new asset set.
