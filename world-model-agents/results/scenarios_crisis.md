# Scenario Generator benchmark: filling in real crises (2000-2022)

Shocks given: S&P 500, VIX and WTI cumulative 21-day moves. Filled in: 20 US large caps.
Each window uses an engine fitted only on data before it.

World-model variant chosen by lowest mean CRPS on the 7 crisis windows up to 2007-12-31: **wm v3: stress covariance**
(selection CRPS: wm v3: stress covariance 7.576, wm v1: local-linear 8.499, wm v2: local slopes 9.073).
The 2008-2022 crisis windows are the hold-out.

## hold-out crises 2008-2022 (S&P 500 down 10% or more in 21 days): 11 windows
| method | RMSE_% | MAE_% | IC | CRPS_% | coverage90 | calib_err90 |
|---|---|---|---|---|---|---|
| wm v1: local-linear | 13.131 | 9.041 | 0.421 | 6.853 | 0.814 | 0.114 |
| wm v2: local slopes | 13.944 | 9.769 | 0.345 | 7.407 | 0.795 | 0.114 |
| wm v3: stress covariance <- selected | 12.705 | 9.045 | 0.402 | 6.727 | 0.891 | 0.073 |
| gauss (full-history cov) | 12.717 | 8.779 | 0.483 | 6.783 | 0.782 | 0.155 |
| gauss1y (trailing 1y cov) | 13.178 | 8.961 | 0.532 | 6.942 | 0.786 | 0.159 |
| beta (CAPM on S&P) | 11.848 | 8.616 | 0.460 | 6.460 | 0.827 | 0.127 |
| zero (no fill-in) | 22.414 | 18.242 | nan | 14.296 | 0.536 | 0.364 |

| comparison | world model better in | median diff (pp) | Wilcoxon p |
|---|---|---|---|
| stresscov vs gauss: RMSE_% | 7/11 (ties 0) | -0.136 | 0.638 |
| stresscov vs gauss: CRPS_% | 6/11 (ties 0) | -0.015 | 0.765 |
| stresscov vs gauss: coverage90 | 7/11 (ties 4) | 0.100 | 0.0156 |
| stresscov vs gauss: calib_err90 | 6/11 (ties 4) | -0.050 | 0.0625 |
| stresscov vs gauss1y: RMSE_% | 5/11 (ties 0) | 0.696 | 0.638 |
| stresscov vs gauss1y: CRPS_% | 2/11 (ties 0) | 0.469 | 0.32 |
| stresscov vs gauss1y: coverage90 | 5/11 (ties 4) | 0.000 | 0.0625 |
| stresscov vs gauss1y: calib_err90 | 5/11 (ties 5) | 0.000 | 0.0938 |
| stresscov vs beta: RMSE_% | 3/11 (ties 0) | 0.479 | 0.083 |
| stresscov vs beta: CRPS_% | 4/11 (ties 0) | 0.238 | 0.206 |
| stresscov vs beta: coverage90 | 6/11 (ties 4) | 0.050 | 0.0469 |
| stresscov vs beta: calib_err90 | 6/11 (ties 4) | -0.050 | 0.0938 |
| stresscov vs zero: RMSE_% | 11/11 (ties 0) | -8.753 | 0.000977 |
| stresscov vs zero: CRPS_% | 11/11 (ties 0) | -4.357 | 0.000977 |
| stresscov vs zero: coverage90 | 10/11 (ties 1) | 0.200 | 0.00195 |
| stresscov vs zero: calib_err90 | 9/11 (ties 2) | -0.200 | 0.00391 |

## validation crises 2000-2007 (used to choose the variant): 7 windows
| method | RMSE_% | MAE_% | IC | CRPS_% | coverage90 | calib_err90 |
|---|---|---|---|---|---|---|
| wm v1: local-linear | 15.488 | 11.481 | 0.134 | 8.499 | 0.757 | 0.171 |
| wm v2: local slopes | 16.472 | 12.264 | 0.103 | 9.073 | 0.736 | 0.193 |
| wm v3: stress covariance <- selected | 13.830 | 10.420 | 0.263 | 7.576 | 0.829 | 0.114 |
| gauss (full-history cov) | 14.014 | 10.499 | 0.209 | 7.784 | 0.764 | 0.164 |
| gauss1y (trailing 1y cov) | 14.211 | 10.624 | 0.264 | 7.786 | 0.757 | 0.200 |
| beta (CAPM on S&P) | 13.639 | 10.482 | 0.287 | 7.636 | 0.771 | 0.171 |
| zero (no fill-in) | 20.075 | 15.263 | nan | 11.767 | 0.571 | 0.329 |

| comparison | world model better in | median diff (pp) | Wilcoxon p |
|---|---|---|---|
| stresscov vs gauss: RMSE_% | 5/7 (ties 0) | -0.339 | 0.688 |
| stresscov vs gauss: CRPS_% | 4/7 (ties 0) | -0.281 | 0.297 |
| stresscov vs gauss: coverage90 | 4/7 (ties 2) | 0.050 | 0.188 |
| stresscov vs gauss: calib_err90 | 4/7 (ties 2) | -0.000 | 0.25 |
| stresscov vs gauss1y: RMSE_% | 2/7 (ties 0) | 0.228 | 0.938 |
| stresscov vs gauss1y: CRPS_% | 4/7 (ties 0) | -0.302 | 0.688 |
| stresscov vs gauss1y: coverage90 | 3/7 (ties 2) | 0.000 | 0.625 |
| stresscov vs gauss1y: calib_err90 | 4/7 (ties 2) | -0.050 | 0.438 |
| stresscov vs beta: RMSE_% | 3/7 (ties 0) | 0.163 | 0.688 |
| stresscov vs beta: CRPS_% | 4/7 (ties 0) | -0.028 | 0.578 |
| stresscov vs beta: coverage90 | 3/7 (ties 4) | 0.000 | 0.25 |
| stresscov vs beta: calib_err90 | 3/7 (ties 4) | 0.000 | 0.25 |
| stresscov vs zero: RMSE_% | 6/7 (ties 0) | -6.649 | 0.0312 |
| stresscov vs zero: CRPS_% | 7/7 (ties 0) | -4.028 | 0.0156 |
| stresscov vs zero: coverage90 | 6/7 (ties 1) | 0.200 | 0.0312 |
| stresscov vs zero: calib_err90 | 6/7 (ties 1) | -0.200 | 0.0312 |

## all non-overlapping 21-day windows 2000-2022: 274 windows
| method | RMSE_% | MAE_% | IC | CRPS_% | coverage90 | calib_err90 |
|---|---|---|---|---|---|---|
| wm v1: local-linear | 7.494 | 5.404 | 0.214 | 4.025 | 0.914 | 0.079 |
| wm v2: local slopes | 7.797 | 5.654 | 0.179 | 4.166 | 0.909 | 0.077 |
| wm v3: stress covariance <- selected | 7.393 | 5.328 | 0.210 | 4.023 | 0.940 | 0.076 |
| gauss (full-history cov) | 7.359 | 5.302 | 0.230 | 3.990 | 0.929 | 0.078 |
| gauss1y (trailing 1y cov) | 7.307 | 5.269 | 0.241 | 3.837 | 0.890 | 0.071 |
| beta (CAPM on S&P) | 7.479 | 5.411 | 0.159 | 4.100 | 0.942 | 0.079 |
| zero (no fill-in) | 8.587 | 6.352 | nan | 4.711 | 0.932 | 0.080 |

| comparison | world model better in | median diff (pp) | Wilcoxon p |
|---|---|---|---|
| stresscov vs gauss: RMSE_% | 121/274 (ties 0) | 0.028 | 0.0612 |
| stresscov vs gauss: CRPS_% | 106/274 (ties 0) | 0.031 | 2.07e-05 |
| stresscov vs gauss: coverage90 | 61/274 (ties 191) | 0.000 | 4.21e-05 |
| stresscov vs gauss: calib_err90 | 44/274 (ties 191) | 0.000 | 0.647 |
| stresscov vs gauss1y: RMSE_% | 110/274 (ties 0) | 0.062 | 0.00781 |
| stresscov vs gauss1y: CRPS_% | 64/274 (ties 0) | 0.210 | 1.79e-18 |
| stresscov vs gauss1y: coverage90 | 162/274 (ties 64) | 0.050 | 8.97e-16 |
| stresscov vs gauss1y: calib_err90 | 87/274 (ties 66) | 0.000 | 0.0518 |
| stresscov vs beta: RMSE_% | 155/274 (ties 0) | -0.039 | 0.00376 |
| stresscov vs beta: CRPS_% | 189/274 (ties 0) | -0.070 | 2.46e-11 |
| stresscov vs beta: coverage90 | 30/274 (ties 195) | 0.000 | 0.141 |
| stresscov vs beta: calib_err90 | 45/274 (ties 195) | 0.000 | 0.313 |
| stresscov vs zero: RMSE_% | 218/274 (ties 0) | -0.716 | 9.76e-30 |
| stresscov vs zero: CRPS_% | 240/274 (ties 0) | -0.374 | 1.26e-36 |
| stresscov vs zero: coverage90 | 61/274 (ties 143) | 0.000 | 0.632 |
| stresscov vs zero: calib_err90 | 69/274 (ties 144) | 0.000 | 0.864 |

## Per crisis window
| window_start | S&P 500 move % | RMSE_% beta | RMSE_% gauss | RMSE_% stresscov | CRPS_% beta | CRPS_% gauss | CRPS_% stresscov |
|---|---|---|---|---|---|---|---|
| 2000-09-14 | -10.45 | 27.71 | 28.55 | 27.87 | 16.21 | 16.49 | 16.21 |
| 2001-02-15 | -12.57 | 10.26 | 11.11 | 10.56 | 5.66 | 6.06 | 5.58 |
| 2001-08-17 | -18.27 | 9.58 | 10.50 | 9.07 | 5.70 | 6.04 | 5.30 |
| 2002-06-24 | -19.35 | 12.99 | 14.53 | 14.36 | 7.25 | 8.05 | 8.10 |
| 2002-09-11 | -14.60 | 17.12 | 15.66 | 16.80 | 9.03 | 8.23 | 8.24 |
| 2003-01-15 | -12.27 | 7.70 | 6.82 | 7.55 | 4.04 | 3.63 | 4.01 |
| 2007-12-26 | -11.08 | 10.13 | 10.93 | 10.59 | 5.57 | 5.99 | 5.59 |
| 2008-06-06 | -10.81 | 10.76 | 9.98 | 11.58 | 5.50 | 4.97 | 5.84 |
| 2008-09-29 | -30.03 | 12.51 | 14.12 | 12.76 | 7.93 | 8.33 | 7.89 |
| 2008-11-05 | -15.96 | 13.02 | 12.85 | 16.45 | 7.05 | 6.94 | 8.92 |
| 2008-12-18 | -10.97 | 22.90 | 23.02 | 22.89 | 9.86 | 9.70 | 9.50 |
| 2009-02-06 | -20.02 | 12.17 | 14.83 | 13.07 | 6.77 | 8.38 | 7.05 |
| 2010-04-26 | -11.80 | 3.75 | 4.06 | 3.24 | 2.77 | 2.66 | 2.78 |
| 2011-07-11 | -16.69 | 8.85 | 10.76 | 9.33 | 4.61 | 5.30 | 4.59 |
| 2018-12-03 | -11.31 | 8.89 | 9.27 | 9.30 | 4.49 | 4.76 | 4.86 |
| 2020-02-24 | -32.97 | 19.17 | 23.01 | 23.00 | 12.17 | 14.00 | 12.68 |
| 2022-04-21 | -12.53 | 10.22 | 11.17 | 10.75 | 5.87 | 6.12 | 6.11 |
| 2022-08-26 | -12.96 | 8.09 | 6.82 | 7.39 | 4.04 | 3.44 | 3.78 |

