# Simulator benchmark: S&P 500 risk bands (2000-01-03 to 2022-12-28, 5758 test days)

World model configuration chosen on 1996-1999 validation data, before this test: `statepath[kernel=0,head=1,widen=0.0]`.

## 1-day ahead
| model | breach 1% (target 1%) | Kupiec p 1% | Christoffersen p 1% | pinball 1% (x1e4) | breach 5% (target 5%) | Kupiec p 5% | Christoffersen p 5% | pinball 5% (x1e4) | FZ0 2.5% | CRPS (x1e4) |
|---|---|---|---|---|---|---|---|---|---|---|
| random-walk | 0.03126 | 0 | 3.035e-06 | 6.249 | 0.06495 | 6.13e-07 | 2.349e-07 | 15.85 | -2.995 | 63.65 |
| historical-sim-500d | 0.01685 | 1.983e-06 | 3.161e-05 | 5.194 | 0.05471 | 0.1063 | 1.138e-06 | 15.26 | -3.272 | 62.48 |
| block-bootstrap | 0.0191 | 6.893e-10 | 4.259e-05 | 5.529 | 0.06721 | 1.153e-08 | 3.574e-07 | 15.81 | -3.173 | 62.96 |
| garch-t | 0.0165 | 5.841e-06 | 0.09357 | 3.834 | 0.0627 | 2.048e-05 | 0.3332 | 13.04 | -3.517 | 60 |
| gjr-garch-t | 0.01494 | 0.0004521 | 0.1833 | 3.7 | 0.0594 | 0.001467 | 0.8615 | 12.7 | -3.555 | 59.6 |
| fhs (statepath, no state) | 0.01268 | 0.04989 | 0.3183 | 3.653 | 0.0528 | 0.3345 | 0.5904 | 12.62 | -3.583 | 59.59 |
| statepath[kernel=0,head=1,widen=0.0] <- selected world model | 0.01407 | 0.003469 | 0.4503 | 3.538 | 0.06026 | 0.0005267 | 0.3552 | 12.51 | -3.625 | 59.39 |
| statepath[kernel=1,head=1,widen=0.0] | 0.01407 | 0.003469 | 0.4503 | 3.57 | 0.05801 | 0.0065 | 0.7475 | 12.51 | -3.622 | 59.38 |
| statepath[kernel=0,head=0,widen=0.0] | 0.01233 | 0.08634 | 0.2889 | 3.658 | 0.05262 | 0.3651 | 0.7721 | 12.63 | -3.585 | 59.59 |

## 10-day ahead (overlapping windows; Kupiec on non-overlapping)
| model | breach 1% | Kupiec p 1% (non-overlapping) | pinball 1% (x1e4) | breach 5% | Kupiec p 5% (non-overlapping) | pinball 5% (x1e4) | FZ0 2.5% | CRPS (x1e4) |
|---|---|---|---|---|---|---|---|---|
| random-walk | 0.0264 | 0.02254 | 18.28 | 0.05731 | 0.4321 | 47.02 | -1.962 | 181.1 |
| historical-sim-500d | 0.02414 | 0.003538 | 17.33 | 0.05419 | 0.9695 | 47.28 | -2.014 | 181.4 |
| block-bootstrap | 0.01928 | 0.3757 | 17.28 | 0.06461 | 0.09289 | 47.08 | -2.03 | 178 |
| garch-t | 0.02015 | 0.6154 | 13.38 | 0.06877 | 0.09289 | 40.74 | -2.328 | 173.6 |
| gjr-garch-t | 0.01494 | 0.7447 | 12.77 | 0.05262 | 0.8779 | 40.27 | -2.374 | 171.9 |
| fhs (statepath, no state) | 0.01077 | 0.4354 | 12.57 | 0.04481 | 0.1179 | 40.21 | -2.404 | 171.6 |
| statepath[kernel=0,head=1,widen=0.0] <- selected world model | 0.01094 | 0.2032 | 12.72 | 0.0528 | 0.9695 | 40.67 | -2.405 | 171.3 |
| statepath[kernel=1,head=1,widen=0.0] | 0.01094 | 0.4354 | 12.66 | 0.05158 | 0.9695 | 40.67 | -2.401 | 171.1 |
| statepath[kernel=0,head=0,widen=0.0] | 0.01146 | 0.2032 | 12.63 | 0.04568 | 0.07569 | 40.35 | -2.406 | 171.8 |

## Diebold-Mariano: world model vs GARCH baselines (negative stat = world model has lower loss)
| test | DM stat | p-value |
|---|---|---|
| 1d pinball 1% vs garch-t | -4.42 | 9.83e-06 |
| 1d pinball 5% vs garch-t | -5.60 | 2.19e-08 |
| 1d FZ0 2.5% vs garch-t | -5.76 | 8.46e-09 |
| 1d CRPS vs garch-t | -9.69 | 0 |
| 10d pinball 1% vs garch-t | -1.05 | 0.295 |
| 10d pinball 5% vs garch-t | -0.14 | 0.892 |
| 10d FZ0 2.5% vs garch-t | -1.47 | 0.142 |
| 10d CRPS vs garch-t | -5.10 | 3.36e-07 |
| 1d pinball 1% vs gjr-garch-t | -2.48 | 0.0132 |
| 1d pinball 5% vs gjr-garch-t | -2.46 | 0.0138 |
| 1d FZ0 2.5% vs gjr-garch-t | -3.38 | 0.000733 |
| 1d CRPS vs gjr-garch-t | -3.98 | 6.99e-05 |
| 10d pinball 1% vs gjr-garch-t | -0.17 | 0.863 |
| 10d pinball 5% vs gjr-garch-t | 1.03 | 0.303 |
| 10d FZ0 2.5% vs gjr-garch-t | -1.11 | 0.267 |
| 10d CRPS vs gjr-garch-t | -2.04 | 0.041 |
| 1d pinball 1% vs fhs (statepath, no state) | -1.76 | 0.0782 |
| 1d pinball 5% vs fhs (statepath, no state) | -1.58 | 0.114 |
| 1d FZ0 2.5% vs fhs (statepath, no state) | -2.33 | 0.0199 |
| 1d CRPS vs fhs (statepath, no state) | -3.96 | 7.4e-05 |
| 10d pinball 1% vs fhs (statepath, no state) | 0.65 | 0.515 |
| 10d pinball 5% vs fhs (statepath, no state) | 1.18 | 0.237 |
| 10d FZ0 2.5% vs fhs (statepath, no state) | -0.04 | 0.967 |
| 10d CRPS vs fhs (statepath, no state) | -1.08 | 0.282 |
