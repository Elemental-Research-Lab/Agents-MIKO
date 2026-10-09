# State Reader benchmark: S&P 500, 2000-01-03 to 2022-11-28 (5747 days, refitted every January)

## 1. Next-21-day volatility (QLIKE; lower is better)
| next-21-day volatility forecast | QLIKE | MSE log-vol |
|---|---|---|
| ewma | 0.4254 | 0.1478 |
| garch | 0.3377 | 0.1397 |
| gjr | 0.3272 | 0.1334 |
| har | 0.3721 | 0.1445 |
| vix_only | 0.3343 | 0.1180 |
| analogs | 0.5481 | 0.1485 |
| state_reader | 0.3350 | 0.1186 |

Diebold-Mariano on QLIKE, State Reader vs each model (negative = State Reader better; Newey-West lag 21):

| vs | DM stat | p-value |
|---|---|---|
| ewma | -3.93 | 8.45e-05 |
| garch | -0.15 | 0.882 |
| gjr | 0.39 | 0.695 |
| har | -2.64 | 0.00828 |
| vix_only | 0.10 | 0.918 |
| analogs | -3.88 | 0.000105 |

Diebold-Mariano on MSE of log volatility:

| vs | DM stat | p-value |
|---|---|---|
| ewma | -4.93 | 8.24e-07 |
| garch | -4.28 | 1.9e-05 |
| gjr | -2.66 | 0.0078 |
| har | -6.03 | 1.63e-09 |
| vix_only | 0.49 | 0.624 |
| analogs | -3.19 | 0.00143 |

## 2. Drawdown early warning
| P(7% drawdown within 21 days); base rate 0.176 | AUC | Brier | log loss |
|---|---|---|---|
| State Reader (5 state features) | 0.7063 | 0.1369 | 0.4588 |
| VIX only | 0.7294 | 0.1339 | 0.4281 |
| 21-day realized vol only | 0.7203 | 0.1326 | 0.4292 |
| HMM stress-regime probability | 0.7109 | 0.3222 | 2.8516 |
| below 200-day moving average | 0.6878 | 0.2641 | 2.4329 |

## 3. Regime reading vs NBER recessions (public label)
| NBER recession days in test: 647 of 5747 (3 recessions) | AUC (recession days) | mean in recessions | mean otherwise | HMM AUC minus this: 90% CI (year-block bootstrap) |
|---|---|---|---|---|
| HMM stress-regime probability | 0.910 | 0.972 | 0.340 | nan |
| VIX level | 0.857 | 32.437 | 18.627 | [-0.030, +0.099] |
| GJR-GARCH volatility | 0.866 | 0.118 | 0.027 | [-0.042, +0.102] |
| below 200-day moving average | 0.849 | 0.923 | 0.226 | [+0.023, +0.115] |

## 4. Novelty: are unusual days really different?
| novelty quintile | novelty_pct | mean_abs_next_day_return_pct | realized_vol_next_21d | gjr_qlike | state_reader_qlike | gjr_normal_1pct_var_breach | drawdown_event_rate |
|---|---|---|---|---|---|---|---|
| Q1 (most familiar) | 0.0572 | 0.4719 | 0.1047 | 0.2779 | 0.2448 | 0.0157 | 0.0400 |
| Q2 | 0.2324 | 0.5693 | 0.1245 | 0.4919 | 0.5106 | 0.0174 | 0.0757 |
| Q3 | 0.4637 | 0.6920 | 0.1476 | 0.3982 | 0.4361 | 0.0226 | 0.1488 |
| Q4 | 0.7197 | 0.9758 | 0.1926 | 0.2601 | 0.2564 | 0.0209 | 0.2724 |
| Q5 (most novel) | 0.9338 | 1.4251 | 0.2646 | 0.2080 | 0.2270 | 0.0148 | 0.3435 |
