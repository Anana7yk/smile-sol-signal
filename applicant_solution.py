import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    del tx_n

    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]

    tx_pred = fit_tx(rx)
    R = rx - tx_pred

    band_R = np.column_stack([score_filter(R[:, c]) for c in range(R.shape[1])])
    cov = band_R.conj().T @ band_R / band_R.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared_R = R @ v
    shared_band = band_R @ v
    denom = np.vdot(shared_band, shared_band) + 1e-30
    alphas = np.array([np.vdot(shared_band, band_R[:, c]) / denom
                       for c in range(R.shape[1])])

    fit_tx_pred = fit_tx(tx_pred)
    fit_tx_shared = fit_tx(shared_R[:, None])[:, 0]
    f_tx_pred = np.column_stack([score_filter(tx_pred[:, c]) for c in range(R.shape[1])])

    A = f_tx_pred - fit_tx_pred
    B = shared_band - fit_tx_shared
    G_AA = A.conj().T @ A / N
    g_AB = (A.conj().T @ B) / N
    G_BB = (np.vdot(B, B).real) / N
    R_band_norm = np.array([np.vdot(band_R[:, c], band_R[:, c]).real / N
                            for c in range(R.shape[1])])
    proj_band = np.array([np.vdot(shared_band, band_R[:, c])
                          for c in range(R.shape[1])]) / N
    sb_norm = np.vdot(shared_band, shared_band).real / N
    t_norm = np.array([np.vdot(f_tx_pred[:, c], f_tx_pred[:, c]).real / N
                       for c in range(R.shape[1])])
    t_proj = np.array([np.vdot(shared_band, f_tx_pred[:, c])
                       for c in range(R.shape[1])]) / N

    def is_valid(scales):
        c_vec = alphas * scales
        cov_r = G_AA.copy()
        cov_r += np.outer(g_AB, c_vec)
        cov_r += np.outer(c_vec.conj(), g_AB.conj())
        cov_r += np.outer(c_vec.conj(), c_vec) * G_BB
        cov_r = (cov_r + cov_r.conj().T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov_r)
        lam = eigvals[-1].real
        v_r = eigvecs[:, -1]
        err_per_ch = np.real(np.diag(cov_r)) - lam * np.abs(v_r) ** 2
        err_per_ch = np.maximum(err_per_ch, 0.0)
        p_after = (R_band_norm
                   + np.abs(c_vec) ** 2 * sb_norm
                   - 2 * (c_vec.conj() * proj_band).real)
        rmd_per_ch = (t_norm
                      + np.abs(c_vec) ** 2 * sb_norm
                      + 2 * (c_vec.conj() * t_proj).real)
        explain = 1.0 - np.mean(err_per_ch) / max(np.mean(rmd_per_ch), 1e-30)
        guard = err_per_ch / np.maximum(p_after, 1e-30)
        return (explain >= 0.95) and (guard.max() <= 0.80)

    n_ch = R.shape[1]
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if is_valid(np.full(n_ch, mid)): lo = mid
        else: hi = mid
    scales = np.full(n_ch, lo)
    for ch in range(n_ch):
        lo_c, hi_c = scales[ch], 1.0
        for _ in range(20):
            mid = (lo_c + hi_c) / 2
            test = scales.copy(); test[ch] = mid
            if is_valid(test): lo_c = mid
            else: hi_c = mid
        scales[ch] = lo_c

    rank1 = np.outer(shared_R, alphas * scales)
    return rx - tx_pred - rank1


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
