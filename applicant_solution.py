import json
import gdown

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

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
    n_ch = R.shape[1]

    band_R = np.column_stack([score_filter(R[:, c]) for c in range(n_ch)])
    fit_tx_R = np.column_stack([fit_tx(R[:, c:c+1])[:, 0] for c in range(n_ch)])
    fit_tx_pred = fit_tx(tx_pred)
    f_tx_pred = np.column_stack([score_filter(tx_pred[:, c]) for c in range(n_ch)])

    A = f_tx_pred - fit_tx_pred
    G_AA = A.conj().T @ A / N
    R_band_norm = np.array([np.vdot(band_R[:, c], band_R[:, c]).real / N for c in range(n_ch)])
    p_rx_band = np.array([np.mean(np.abs(score_filter(rx[:, c])) ** 2) for c in range(n_ch)])
    t_norm = np.array([np.vdot(f_tx_pred[:, c], f_tx_pred[:, c]).real / N for c in range(n_ch)])

    cov_band_R = band_R.conj().T @ band_R / N
    AH_R = A.conj().T @ band_R / N
    AH_fit = A.conj().T @ fit_tx_R / N
    fit_R_norm_mat = fit_tx_R.conj().T @ fit_tx_R / N
    f_tx_band_R = f_tx_pred.conj().T @ band_R / N

    def evaluate(v):
        shared_R = R @ v
        shared_band = band_R @ v
        fit_tx_shared = fit_tx_R @ v
        sb_norm = (v.conj() @ cov_band_R @ v).real
        sb_norm = max(sb_norm, 1e-30)
        proj_band = (cov_band_R @ v).conj()
        alphas = proj_band / sb_norm

        g_AB = (AH_R @ v) - (AH_fit @ v)
        G_BB = sb_norm + (v.conj() @ fit_R_norm_mat @ v).real - 2 * (v.conj() @ (band_R.conj().T @ fit_tx_R / N) @ v).real
        G_BB = max(G_BB, 1e-30)
        t_proj = (f_tx_band_R @ v).conj()

        def is_valid(scales):
            c_vec = alphas * scales
            cov_r = G_AA + np.outer(g_AB, c_vec) + np.outer(c_vec.conj(), g_AB.conj()) + np.outer(c_vec.conj(), c_vec) * G_BB
            cov_r = (cov_r + cov_r.conj().T) / 2
            eigvals, eigvecs = np.linalg.eigh(cov_r)
            lam = eigvals[-1].real
            v_r = eigvecs[:, -1]
            err = np.maximum(np.real(np.diag(cov_r)) - lam * np.abs(v_r) ** 2, 0)
            p_after = R_band_norm + np.abs(c_vec) ** 2 * sb_norm - 2 * (c_vec.conj() * proj_band).real
            rmd = t_norm + np.abs(c_vec) ** 2 * sb_norm + 2 * (c_vec.conj() * t_proj).real
            explain = 1 - err.mean() / max(rmd.mean(), 1e-30)
            guard = err / np.maximum(p_after, 1e-30)
            return (explain >= 0.95) and (guard.max() <= 0.80)

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

        c_vec = alphas * scales
        p_after = R_band_norm + np.abs(c_vec) ** 2 * sb_norm - 2 * (c_vec.conj() * proj_band).real
        p_after = np.maximum(p_after, 1e-30)
        db = 10 * np.log10(p_rx_band / p_after)
        return db, scales, shared_R, alphas

    _, vecs0 = np.linalg.eigh(cov_band_R)
    v_init = vecs0[:, -1]
    v_init = v_init / v_init[0]

    def parametrize(x):
        v = np.array([1.0, x[0] + 1j * x[1], x[2] + 1j * x[3], x[4] + 1j * x[5]])
        return v / np.linalg.norm(v)

    def neg_min_db(x):
        db, _, _, _ = evaluate(parametrize(x))
        return -db.min()

    x0 = np.array([v_init[1].real, v_init[1].imag,
                   v_init[2].real, v_init[2].imag,
                   v_init[3].real, v_init[3].imag])
    res = minimize(neg_min_db, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})

    v_best = parametrize(res.x)
    _, scales, shared_R, alphas = evaluate(v_best)

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
