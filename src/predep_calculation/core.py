import numpy as np
import scipy.stats as ss
from astropy.stats import bayesian_blocks


def _ecdf(data):
    s = np.sort(data)
    n = len(s)
    return lambda x: np.searchsorted(s, x, side="right") / n


def predep(X, Y, n_boot=10000):
    """
    Calcula PREDEP α_{Y|X}: quanto o conhecimento de X (preditor)
    reduz a incerteza sobre Y (resposta).

    Interpretação: α = (S_{Y|X} - S_Y) / S_{Y|X}
    - 0: X e Y independentes
    - 1: X prediz Y perfeitamente
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)

        BY1 = np.random.choice(Y, n_boot)
        BY2 = np.random.choice(Y, n_boot)
        s_y = float(ss.gaussian_kde(BY1 - BY2).pdf(0))

        edges_x = bayesian_blocks(X)
        ecdf_x = _ecdf(X)

        s_y_mid_x = 0.0
        if edges_x.shape[0] > 1:
            for i in range(1, edges_x.shape[0]):
                bg = edges_x[i - 1]
                ed = edges_x[i]
                Y_mid_X = Y[(X >= bg) & (X < ed)]
                if Y_mid_X.shape[0] == 0:
                    continue

                dy_c = np.random.choice(Y_mid_X, n_boot) - np.random.choice(Y_mid_X, n_boot)
                p_range = ecdf_x(ed) - ecdf_x(bg)
                s_y_mid_x += p_range * float(ss.gaussian_kde(dy_c).pdf(0))

            return float((s_y_mid_x - s_y) / s_y_mid_x)
        return np.nan
    except Exception:
        return np.nan
