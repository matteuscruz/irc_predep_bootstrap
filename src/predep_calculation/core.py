import numpy as np
import scipy.stats as ss
from astropy.stats import bayesian_blocks


def _ecdf(data):
    s = np.sort(data)
    n = len(s)
    return lambda x: np.searchsorted(s, x, side="right") / n


def predep(X, Y, n_boot=10000):
    """
    PREDEP α_{Y|X}: mede quanto o conhecimento de X (preditor) reduz a incerteza sobre Y (resposta).

    Convenção: X = preditor (ex: índice MoV), Y = resposta (ex: precipitação).
    α = (s(Y|X) − s(Y)) / s(Y|X)
    - 0: X e Y independentes
    - 1: X prediz Y perfeitamente
    """
    try:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        BY1 = np.random.choice(Y, n_boot)
        BY2 = np.random.choice(Y, n_boot)
        diff_y = BY1 - BY2
        if np.std(diff_y) == 0:
            return np.nan
        s_y = float(np.squeeze(ss.gaussian_kde(diff_y).pdf(0)))

        edges_x = bayesian_blocks(X)
        ecdf_x = _ecdf(X)

        s_y_mid_x = 0.0
        if edges_x.shape[0] > 1:
            for i in range(1, edges_x.shape[0]):
                bg, ed = edges_x[i - 1], edges_x[i]
                y_mid_x = Y[(X >= bg) & (X < ed)]
                if y_mid_x.shape[0] < 2 or len(np.unique(y_mid_x)) < 2:
                    continue
                dy_c = (
                    np.random.choice(y_mid_x, n_boot)
                    - np.random.choice(y_mid_x, n_boot)
                )
                if np.std(dy_c) == 0:
                    continue
                p_range = ecdf_x(ed) - ecdf_x(bg)
                kde_val = float(np.squeeze(ss.gaussian_kde(dy_c).pdf(0)))
                s_y_mid_x += p_range * kde_val

            if s_y_mid_x == 0:
                return np.nan
            return float((s_y_mid_x - s_y) / s_y_mid_x)
        return np.nan
    except Exception:
        return np.nan
