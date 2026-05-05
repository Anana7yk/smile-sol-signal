# SMILES-2026 — my report

What I got: **10.09 dB** average, every channel ≥ 9.49 dB (baseline gives ~4 dB, target is ">8 dB"). Per-channel:

| ch0 | ch1 | ch2 | ch3 | avg | min |
|---|---|---|---|---|---|
| 11.90 | 9.49 | 9.49 | 9.49 | 10.09 | 9.49 |

## Reproducibility

```bash
pip install numpy scipy gdown
python applicant_solution.py
```

The script downloads `challenge.mat` (~393 MB) from Google Drive via `gdown`, runs the baseline, runs my canceller, writes `results.json`. I ran it on macOS with Python 3.9, numpy 2.0.2, scipy 1.13.1. Total runtime is around 2 minutes — most of it is the dataset download. The numerical optimization of the spatial direction `v` adds ~30 seconds, the rest of the canceller is fast (purely algebraic, no FFTs in the hot loop).

I did not touch `task_and_baseline.py`. Only modified the body of `your_canceller` in `applicant_solution.py`.

## What I ended up doing

The validator clearly says you can subtract "TX nonlinearity + one rank-1 spatial component". So the solution is built around exactly that — and arranged so that when the validator decomposes our removed signal it finds almost exactly that structure (so err ≈ 0).

Four steps:

1. **Subtract TX nonlinearity.** Call `helpers["fit_tx_prediction"](rx)` — the same LSQ projection onto 10 cross-terms × 13 lags that the validator later uses for verification. Nothing to invent here.

2. **Set up the rank-1 problem.** Compute `R = rx - tx_pred`, `band_R = score_filter(R)` per channel, and several aggregates (cov, fit_tx applied to each column of R, etc.). These are independent of the spatial direction `v`, so they get computed once.

3. **Optimize the spatial direction `v`.** Initialize from the top eigenvector of `cov(band_R)`, then run Nelder-Mead on the **max-min objective**: maximize the minimum per-channel reduction in dB. For each candidate `v`, the inner loop computes per-channel coefficients, runs a binary search for the maximum validator-safe scale per channel, and returns the resulting per-channel dB. The whole inner loop is algebraic (no FFTs).

4. **Final cancellation.** Subtract `outer(shared_R, alphas * scales)` for the optimal `v`.

## What contributed most to the metric

If you do the naive thing (top eigenvector + in-band rank-1) you only get ~7 dB. Three things lift it to 10.09:

- **Replacing `band @ v` with `R @ v` (wideband shared)** — almost +3 dB. In the naive version rank-1 is built as `α * (band_R @ v)`, but that's already a band-filtered signal, and when subtracted in time domain the validator applies `score_filter` once more when scoring. Double filtering with passband factor ~0.75 means we lose ~3 dB. With wideband `shared_R = R @ v` the filter is only applied once and the cancellation is full-strength.

- **Per-channel scale auto-tune** — extra +0.1 to +0.3 dB. You can't just use full-scale wideband; the validator's TX projection runs on a 200k subset and the passband mismatch leaks a tiny amount that grows with cancellation magnitude, eventually hitting the per-channel guard `err / residual ≤ 0.80`. I replicated the validator's two checks algebraically (it all reduces to one 4×4 `eigh` per evaluation) and binary-search the maximum valid scale per channel.

- **Numerical optimization of `v`** — last +0.3 dB on average and a huge boost on the weak channel. The top eigenvector maximises F-norm reduction, which is biased toward channels with strong rank-1 coupling. ch3 has only ~4% of the energy on the top eigvec and ends up at 7.4 dB. By instead optimising `v` for the **max-min** objective via Nelder-Mead, the optimum equalises the three weakest channels at 9.49 dB and ch0 climbs to 11.9 dB. The avg goes from 9.79 to 10.09.

## Experiments that didn't work

- **Iterating TX ↔ rank-1.** Classic alternating optimization: refit TX on (rx − rank1), then refit rank-1, etc. Theoretically should monotonically improve F-norm. In practice the score drops (7.0 → 5.3 → 5.0 dB), and from 5+ iterations it goes INVALID. Reason: `fit_tx_prediction` is fit on a 200k subset, and feeding it (rx − rank1) makes the LSQ overfit the subset while the full-N residual gets worse.

- **Reverse order: rank-1 first, then TX.** Immediately INVALID. The rank-1 grabs TX-correlated content (the dominant variance direction in raw rx), and the validator's later TX projection double-counts it — explainability drops below 0.95.

- **Full-N TX LSQ.** I tried doing my own LSQ over all 2.46M samples instead of the validator's subset. INVALID. The full-N LSQ catches spurious correlations the subset LSQ doesn't see, and the difference goes into err.

- **Wideband without auto-tune (just scale = 1.0).** INVALID around scale ≈ 0.87 (with the top eigenvector). Without binary search you can't squeeze more than a hardcoded 0.80–0.85, which is 0.2–0.4 dB worse.

- **Projecting `shared_R` onto the orthogonal complement of TX-span on subset** (via `fit_tx(shared_R)` and subtracting). Almost worked — explainability landed at 0.947 < 0.95 — but the passband mismatch between X_s (single filter) and X_full² (double filter) leaves a residual leakage this trick can't fix.

- **Other choices of `v`.** Uniform `(1,1,1,1)/2` gives 5.3 dB. Maximising avg-dB instead of max-min gives 10.46 dB but with min = 7.5 dB (one channel below the Good tier). Max-min wins on robustness; the small avg loss (10.46 → 10.09) buys all-channels ≥ 9.49.

- **Richer TX features** (more cross-terms, more lags). Anything outside the validator's fixed feature set goes into err. The constraint is hard.

## What I didn't do and where it's weak

- The `v` optimization is pure Nelder-Mead from a single starting point. There's no guarantee of global optimum. Multi-start could help but the solution is already comfortably above target.
- My analysis of why exactly the iteration breaks is mostly empirical, I didn't derive it strictly.
- I didn't look at the spectra or the time-domain structure of the actual signals — there might be non-stationarity or some other structure that could be exploited. This is the weak point — I was fitting to the validator's allowed form rather than really digging into the data.
