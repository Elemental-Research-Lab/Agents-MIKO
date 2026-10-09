# Simulator benchmark: Exchange-rate CRPS-sum (GluonTS exchange_rate_nips)

5 rolling windows x 30 business days, 100 sample paths, 10 seeds. Lower is better.
World model settings carried over from the S&P 500 validation: kernel=False, widen=0.0; the VIX head is off (no VIX for FX).

## This repo
| model (this repo) | CRPS-sum mean | CRPS-sum std | own implementation |
|---|---|---|---|
| random-walk (Gaussian) | 0.00576 | 0.00020 | 0.00579 |
| historical-sim-500d | 0.00546 | 0.00013 | 0.00548 |
| block-bootstrap | 0.00531 | 0.00021 | 0.00533 |
| garch-t | 0.00495 | 0.00017 | 0.00495 |
| gjr-garch-t | 0.00494 | 0.00018 | 0.00494 |
| fhs (no state) | 0.00474 | 0.00019 | 0.00474 |
| statepath (world model) | 0.00476 | 0.00019 | 0.00476 |

## Published results on the same split
| published (Rasul et al. 2021 Table 2; ScoreGrad rows: Yan et al. 2021 Table III) | CRPS-sum mean | CRPS-sum std |
|---|---|---|
| VES | 0.005 | 0.000 |
| VAR | 0.005 | 0.000 |
| VAR-Lasso | 0.012 | 0.000 |
| GARCH | 0.023 | 0.000 |
| KVAE | 0.014 | 0.002 |
| Vec-LSTM ind-scaling | 0.008 | 0.001 |
| Vec-LSTM lowrank-Copula | 0.007 | 0.000 |
| GP scaling | 0.009 | 0.000 |
| GP Copula | 0.007 | 0.000 |
| Transformer MAF | 0.005 | 0.003 |
| TimeGrad | 0.006 | 0.001 |
| ScoreGrad VP SDE | 0.006 | 0.001 |
| ScoreGrad VE SDE | 0.007 | 0.001 |
| ScoreGrad sub-VP SDE | 0.006 | 0.001 |

Best published mean: 0.005. Published values are rounded to 3 decimals.
