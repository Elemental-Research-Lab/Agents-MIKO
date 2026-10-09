# Planner benchmark: 20 US large caps, monthly, 1995-01-31 to 2022-11-30 (335 months)

## A. Fully invested classics (DeMiguel-Garlappi-Uppal style)
| fully invested (no risk goal) | ann. return | ann. vol | Sharpe | Sharpe net 10bps | CEQ (gamma=1) | max drawdown | monthly turnover | avg invested | realized monthly CVaR95 | Sharpe vs 1/N: z | Sharpe vs 1/N: p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/N | 0.165 | 0.161 | 1.025 | 1.020 | 0.152 | -0.446 | 0.059 | 1.000 | 0.093 | nan | nan |
| inverse volatility | 0.150 | 0.142 | 1.060 | 1.055 | 0.140 | -0.402 | 0.052 | 1.000 | 0.084 | 0.859 | 0.390 |
| min variance (Ledoit-Wolf) | 0.130 | 0.127 | 1.018 | 1.007 | 0.121 | -0.312 | 0.113 | 1.000 | 0.079 | -0.057 | 0.955 |
| mean-variance max Sharpe (sample) | 0.208 | 0.199 | 1.043 | 1.021 | 0.188 | -0.473 | 0.372 | 1.000 | 0.106 | 0.121 | 0.904 |
| equal risk contribution | 0.156 | 0.144 | 1.086 | 1.081 | 0.146 | -0.410 | 0.057 | 1.000 | 0.084 | 1.910 | 0.056 |
| HRP | 0.142 | 0.132 | 1.072 | 1.057 | 0.133 | -0.361 | 0.170 | 1.000 | 0.081 | 0.720 | 0.472 |
| min CVaR (historical) | 0.143 | 0.131 | 1.092 | 1.070 | 0.134 | -0.239 | 0.240 | 1.000 | 0.076 | 0.508 | 0.612 |

## B. Risk goal cvar95 <= 6%
| risk goal: cvar95 <= 6%, max 20% per name | ann. return | ann. vol | Sharpe | Sharpe net 10bps | CEQ (gamma=1) | max drawdown | monthly turnover | avg invested | realized monthly CVaR95 | Sharpe vs 1/N: z | Sharpe vs 1/N: p | Sharpe vs 1/N sized by historical risk: p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| planner (world model) | 0.120 | 0.108 | 1.112 | 1.076 | 0.114 | -0.235 | 0.322 | 0.881 | 0.063 | 0.847 | 0.397 | 0.142 |
| 1/N scaled by world-model risk | 0.123 | 0.109 | 1.124 | 1.115 | 0.117 | -0.249 | 0.085 | 0.780 | 0.059 | 1.698 | 0.089 | 0.003 |
| 1/N scaled by FHS risk (world model without state) | 0.122 | 0.109 | 1.121 | 1.111 | 0.116 | -0.250 | 0.084 | 0.772 | 0.058 | 1.775 | 0.076 | 0.009 |
| 1/N scaled by GARCH-t risk | 0.136 | 0.122 | 1.121 | 1.113 | 0.129 | -0.280 | 0.080 | 0.855 | 0.066 | 1.687 | 0.092 | 0.014 |
| 1/N scaled by GJR-GARCH-t risk | 0.128 | 0.116 | 1.102 | 1.093 | 0.121 | -0.270 | 0.084 | 0.822 | 0.064 | 1.380 | 0.168 | 0.038 |
| scaled 1/N (historical risk) | 0.116 | 0.119 | 0.971 | 0.965 | 0.108 | -0.304 | 0.055 | 0.767 | 0.071 | -0.777 | 0.437 | nan |
| scaled min CVaR (historical risk) | 0.131 | 0.118 | 1.113 | 1.090 | 0.125 | -0.202 | 0.218 | 0.883 | 0.070 | 0.624 | 0.532 | 0.261 |
| planner + GARCH-t futures | 0.136 | 0.138 | 0.987 | 0.966 | 0.126 | -0.302 | 0.233 | 0.953 | 0.076 | -0.362 | 0.718 | 0.882 |
| planner + block-bootstrap futures | 0.114 | 0.113 | 1.011 | 0.985 | 0.108 | -0.261 | 0.245 | 0.821 | 0.068 | -0.145 | 0.885 | 0.671 |

| goal compliance (21-day returns after each rebalance; 45 crisis months) | realized cvar95 (all months) | realized cvar95 (crisis months) | goal | share of months losing more than 6.0% (goal implies <= 5%) |
|---|---|---|---|---|
| planner (world model) | 0.067 | 0.086 | 0.060 | 0.024 |
| 1/N scaled by world-model risk | 0.062 | 0.063 | 0.060 | 0.018 |
| 1/N scaled by FHS risk (world model without state) | 0.061 | 0.064 | 0.060 | 0.021 |
| 1/N scaled by GARCH-t risk | 0.069 | 0.070 | 0.060 | 0.033 |
| 1/N scaled by GJR-GARCH-t risk | 0.066 | 0.072 | 0.060 | 0.030 |
| scaled 1/N (historical risk) | 0.075 | 0.094 | 0.060 | 0.033 |
| scaled min CVaR (historical risk) | 0.072 | 0.099 | 0.060 | 0.024 |
| planner + GARCH-t futures | 0.081 | 0.105 | 0.060 | 0.042 |
| planner + block-bootstrap futures | 0.070 | 0.101 | 0.060 | 0.027 |

## B. Risk goal vol <= 10%
| risk goal: vol <= 10%, max 20% per name | ann. return | ann. vol | Sharpe | Sharpe net 10bps | CEQ (gamma=1) | max drawdown | monthly turnover | avg invested | realized monthly CVaR95 | Sharpe vs 1/N: z | Sharpe vs 1/N: p | Sharpe vs 1/N sized by historical risk: p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| planner (world model) | 0.104 | 0.095 | 1.087 | 1.047 | 0.099 | -0.195 | 0.319 | 0.807 | 0.057 | 0.466 | 0.641 | 0.545 |
| 1/N scaled by world-model risk | 0.100 | 0.090 | 1.116 | 1.106 | 0.096 | -0.200 | 0.072 | 0.651 | 0.049 | 1.462 | 0.144 | 0.046 |
| 1/N scaled by FHS risk (world model without state) | 0.101 | 0.090 | 1.117 | 1.108 | 0.097 | -0.206 | 0.064 | 0.642 | 0.049 | 1.618 | 0.106 | 0.059 |
| 1/N scaled by GARCH-t risk | 0.102 | 0.087 | 1.179 | 1.160 | 0.099 | -0.190 | 0.134 | 0.620 | 0.048 | 1.574 | 0.115 | 0.055 |
| 1/N scaled by GJR-GARCH-t risk | 0.106 | 0.089 | 1.198 | 1.179 | 0.102 | -0.157 | 0.134 | 0.638 | 0.050 | 1.972 | 0.049 | 0.020 |
| scaled 1/N (historical risk) | 0.094 | 0.093 | 1.011 | 1.006 | 0.090 | -0.249 | 0.041 | 0.620 | 0.056 | -0.204 | 0.839 | nan |
| scaled min CVaR (historical risk) | 0.108 | 0.094 | 1.149 | 1.126 | 0.104 | -0.175 | 0.181 | 0.749 | 0.056 | 0.847 | 0.397 | 0.285 |
| planner + GARCH-t futures | 0.122 | 0.110 | 1.110 | 1.065 | 0.116 | -0.204 | 0.418 | 0.813 | 0.065 | 0.637 | 0.524 | 0.445 |
| planner + block-bootstrap futures | 0.104 | 0.097 | 1.071 | 1.058 | 0.099 | -0.236 | 0.105 | 0.744 | 0.059 | 0.436 | 0.663 | 0.582 |

| goal compliance (21-day returns after each rebalance; 45 crisis months) | realized vol (all months) | realized vol (crisis months) | goal | share of months losing more than 4.7% (goal implies <= 5%) |
|---|---|---|---|---|
| planner (world model) | 0.097 | 0.109 | 0.100 | 0.036 |
| 1/N scaled by world-model risk | 0.091 | 0.106 | 0.100 | 0.030 |
| 1/N scaled by FHS risk (world model without state) | 0.092 | 0.112 | 0.100 | 0.027 |
| 1/N scaled by GARCH-t risk | 0.089 | 0.099 | 0.100 | 0.027 |
| 1/N scaled by GJR-GARCH-t risk | 0.090 | 0.102 | 0.100 | 0.027 |
| scaled 1/N (historical risk) | 0.095 | 0.130 | 0.100 | 0.042 |
| scaled min CVaR (historical risk) | 0.097 | 0.110 | 0.100 | 0.024 |
| planner + GARCH-t futures | 0.112 | 0.143 | 0.100 | 0.045 |
| planner + block-bootstrap futures | 0.100 | 0.147 | 0.100 | 0.036 |

Crisis months (holding months): 2000-09 to 2002-09, 2008-09 to 2009-03, 2020-02 to 2020-04, 2022-01 to 2022-10.
Realized CVaR95 over crisis months averages only the worst ~5% of them (2-3 months), so read it as indicative.
Sharpe ratios use raw returns with cash at 0% (no risk-free rate); the Jobson-Korkie-Memmel test compares Sharpe ratios.
Survivorship caveat: all 20 stocks survived to 2022; absolute returns are flattered, comparisons are fair.
