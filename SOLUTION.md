# SMILES-2026 — my report

What I got: **9.79 dB** average (baseline gives ~4 dB, target is ">8 dB"). Per-channel:

| ch0 | ch1 | ch2 | ch3 | avg |
|---|---|---|---|---|
| 11.01 | 8.30 | 12.42 | 7.44 | 9.79 |

## Reproducibility

```bash
pip install numpy scipy gdown
python applicant_solution.py
```

The script downloads `challenge.mat` (~393 MB) from Google Drive via `gdown`, runs the baseline, runs my canceller, writes `results.json`. I ran it on macOS with Python 3.9, numpy 2.0.2, scipy 1.13.1. Total runtime is around 1.5–2 minutes, almost all of it the dataset download. The canceller itself is fast (the auto-tune is purely algebraic, no FFTs inside the search loop).

I did not touch `task_and_baseline.py`. Only modified the body of `your_canceller` in `applicant_solution.py`.

## What I ended up doing

The idea is pretty simple: the validator clearly says you can subtract "TX nonlinearity + one rank-1 spatial component". So ideally you should just do exactly that — and do it in a way that when the validator decomposes our removed signal it finds almost exactly that structure (so err ≈ 0).

Three steps:

1. **Subtract TX nonlinearity.** Call `helpers["fit_tx_prediction"](rx)` — this is the same LSQ projection onto 10 cross-terms × 13 lags that the validator later uses for verification. No need to invent anything here.

2. **Find the rank-1 direction.** Take `R = rx - tx_pred`, run it through `score_filter` per channel, compute the 4×4 covariance, take the top eigenvector `v` via `eigh`. Then form a shared time profile `shared_R = R @ v` (important — wideband, not the band-filtered version) and per-channel coefficients `alphas[c] = <shared_band, R_band[:,c]> / ||shared_band||²`.

3. **Per-channel scale via binary search.** At full scale (`scale = 1.0`) the validator returns INVALID — `fit_tx_prediction` only fits on a 200k subset out of 2.46M samples, and the passband ripple introduces a small leakage that hits the per-channel guard `err / residual ≤ 0.80`. So I replicated the validator's checks algebraically (it all reduces to `eigh` on a 4×4 matrix plus scalars) and binary-searched the maximum valid scale per channel. ch2 is the binding channel with scale ≈ 0.87, the others can run at 1.0.

Final: subtract `rx - tx_pred - outer(shared_R, alphas * scales)`.

## What contributed most to the metric

If you do exactly what I described above WITHOUT the two tricks below, you only get around 7 dB. The two things that push it to 9.79:

- **Replacing `band @ v` with `R @ v` (wideband shared).** In the naive version I was building rank-1 as `α * (band_R @ v)` — but that's already a band-filtered signal, and when I subtract it in time domain the validator applies `score_filter` once more when computing the score. So you get double filtering, and because of the passband factor (~0.75 of their filter) you lose almost 3 dB. With wideband `shared_R = R @ v` the filter is only applied once and the cancellation is full-strength.

- **Per-channel scale auto-tune.** You can't just use full-scale wideband — INVALID. I figured out the boundary is around 0.86, but instead of hardcoding I wrote the auto-tune: inside `is_valid(scales)` I replicated the two validator checks (explainability ≥ 0.95 and err/residual ≤ 0.80 per channel) — both reduce to a 4×4 `eigh`, takes microseconds. Then binary search per channel separately. That's another ~0.15 dB on top of uniform scaling.

## Experiments that didn't work

- **Iterating TX ↔ rank-1.** Classic idea — alternating optimization: refit TX on (rx − rank1), then refit rank-1 on (rx − new_tx), and so on. In theory it should monotonically improve F-norm. In practice the score drops (7.0 → 5.3 → 5.0 dB), and from 5+ iterations it goes INVALID. The reason: `fit_tx_prediction` is fit on a 200k subset, and feeding it (rx − rank1) makes the LSQ overfit the subset while the full-N residual gets worse.

- **Reverse order: rank-1 first, then TX.** Immediately INVALID. The rank-1 in that case grabs TX-correlated content (because in raw rx the TX leakage is the dominant variance direction), and the validator's later TX projection then double-counts — explainability drops below 0.95.

- **Full-N TX LSQ.** I tried doing my own LSQ over all 2.46M samples instead of the validator's subset. INVALID. The reason: on the full sample set the LSQ catches spurious correlations the subset LSQ doesn't see, and that difference goes into err.

- **Wideband without auto-tune (just scale = 1.0).** INVALID around scale ≈ 0.87. Without binary search you can't squeeze more than a hardcoded 0.80–0.85, which is 0.2–0.4 dB worse.

- **Projecting `shared_R` onto the orthogonal complement of TX-span on subset** (via `fit_tx(shared_R)` and subtracting). The idea was to kill the leakage analytically. Almost worked, but explainability landed at 0.947 < 0.95 — the passband mismatch between X_s (single filter) and X_full² (double filter) leaves a residual leakage this trick can't fix.

- **Other choices of `v`.** I tried uniform `(1,1,1,1)/2` — gave 5.3 dB, bad. Numerical optimization of avg-dB over `v` via scipy.optimize blew up to numerically degenerate solutions. The top eigenvector remains the best choice.

- **Richer TX features** (more cross-terms, more lags). Anything outside the validator's fixed feature set goes into err. The constraint is hard.

## What I didn't do and where it's weak

- ch3 is stuck at 7.44 dB. It only has ~4% of the energy on the top eigenvector (`|v[3]|² ≈ 0.039`), and with a budget of one rank-1 that's the ceiling. The validator doesn't allow a second rank-1 component.
- My analysis of why exactly the iteration breaks is mostly empirical, I didn't derive it strictly.
- I didn't look at the spectra or the time-domain structure of the actual signals — there might be non-stationarity or some other structure that could be exploited. This is the weak point — I was fitting to the validator's allowed form rather than really digging into the data.
