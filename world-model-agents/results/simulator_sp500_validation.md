# Simulator model selection on validation period 1996-01-02 to 1999-12-31

Selected: **statepath[kernel=0,head=1,widen=0.0]**

## 1-day
| model | breach 1% (target 1%) | Kupiec p 1% | Christoffersen p 1% | pinball 1% (x1e4) | breach 5% (target 5%) | Kupiec p 5% | Christoffersen p 5% | pinball 5% (x1e4) | FZ0 2.5% | CRPS (x1e4) |
|---|---|---|---|---|---|---|---|---|---|---|
| gjr-garch-t | 0.0201 | 0.004864 | 0.4157 | 4.143 | 0.06734 | 0.01691 | 0.4188 | 12.89 | -3.443 | 58.31 |
| fhs (statepath, no state) | 0.01709 | 0.04133 | 0.4418 | 4.215 | 0.07035 | 0.005387 | 0.6429 | 12.81 | -3.457 | 58.27 |
| statepath[kernel=0,head=0,widen=0.0] | 0.01809 | 0.02124 | 0.3306 | 4.138 | 0.07136 | 0.003578 | 0.5966 | 12.88 | -3.457 | 58.33 |
| statepath[kernel=0,head=0,widen=0.25] | 0.01709 | 0.04133 | 0.2913 | 4.081 | 0.06332 | 0.0637 | 0.5809 | 13.11 | -3.459 | 58.45 |
| statepath[kernel=0,head=1,widen=0.0] | 0.01307 | 0.3534 | 0.1579 | 3.916 | 0.0603 | 0.1481 | 0.8349 | 12.57 | -3.522 | 58.04 |
| statepath[kernel=0,head=1,widen=0.25] | 0.01206 | 0.5269 | 0.131 | 3.933 | 0.05729 | 0.3022 | 0.8732 | 12.82 | -3.512 | 58.2 |
| statepath[kernel=1,head=0,widen=0.0] | 0.01809 | 0.02124 | 0.3306 | 4.048 | 0.06734 | 0.01691 | 0.4188 | 12.81 | -3.48 | 58.27 |
| statepath[kernel=1,head=0,widen=0.25] | 0.01608 | 0.0765 | 0.4693 | 3.967 | 0.06231 | 0.08562 | 0.626 | 13.05 | -3.481 | 58.34 |
| statepath[kernel=1,head=1,widen=0.0] | 0.01206 | 0.5269 | 0.5881 | 3.853 | 0.05427 | 0.5418 | 0.9674 | 12.53 | -3.53 | 57.99 |
| statepath[kernel=1,head=1,widen=0.25] | 0.01106 | 0.7422 | 0.6198 | 3.896 | 0.05226 | 0.7452 | 0.8601 | 12.77 | -3.52 | 58.12 |

## 10-day
| model | breach 1% | Kupiec p 1% (non-overlapping) | pinball 1% (x1e4) | breach 5% | Kupiec p 5% (non-overlapping) | pinball 5% (x1e4) | FZ0 2.5% | CRPS (x1e4) |
|---|---|---|---|---|---|---|---|---|
| gjr-garch-t | 0.0191 | 0.3763 | 10.93 | 0.05226 | 0.635 | 37.73 | -2.476 | 173.1 |
| fhs (statepath, no state) | 0.01307 | 1 | 10.58 | 0.04724 | 0.635 | 37.1 | -2.52 | 173.6 |
| statepath[kernel=0,head=0,widen=0.0] | 0.01307 | 1 | 10.46 | 0.05729 | 0.656 | 37.16 | -2.522 | 172.7 |
| statepath[kernel=0,head=0,widen=0.25] | 0.009045 | 1 | 10.6 | 0.05528 | 1 | 38.1 | -2.514 | 173.8 |
| statepath[kernel=0,head=1,widen=0.0] | 0.007035 | 1 | 10.15 | 0.05025 | 0.635 | 36.95 | -2.55 | 172.7 |
| statepath[kernel=0,head=1,widen=0.25] | 0.00603 | 1 | 10.67 | 0.04925 | 0.635 | 37.79 | -2.537 | 173.9 |
| statepath[kernel=1,head=0,widen=0.0] | 0.01709 | 1 | 10.58 | 0.04925 | 1 | 37.32 | -2.508 | 173 |
| statepath[kernel=1,head=0,widen=0.25] | 0.01206 | 1 | 10.8 | 0.04724 | 1 | 38.35 | -2.5 | 174 |
| statepath[kernel=1,head=1,widen=0.0] | 0.01106 | 1 | 10.32 | 0.04523 | 1 | 37.16 | -2.534 | 173 |
| statepath[kernel=1,head=1,widen=0.25] | 0.009045 | 1 | 10.82 | 0.04422 | 1 | 38.1 | -2.515 | 174.1 |

## Mean rank (lower is better)
| model | mean rank |
|---|---|
| statepath[kernel=0,head=1,widen=0.0] | 1.667 |
| statepath[kernel=1,head=1,widen=0.0] | 1.833 |
| statepath[kernel=0,head=1,widen=0.25] | 4.333 |
| statepath[kernel=1,head=1,widen=0.25] | 4.833 |
| statepath[kernel=0,head=0,widen=0.0] | 5.167 |
| statepath[kernel=1,head=0,widen=0.0] | 5.333 |
| statepath[kernel=0,head=0,widen=0.25] | 6.333 |
| statepath[kernel=1,head=0,widen=0.25] | 6.5 |
