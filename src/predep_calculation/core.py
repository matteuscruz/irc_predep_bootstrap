import numpy as np
import scipy.stats as ss
from statsmodels.distributions import ECDF


def predep(X, Y, n_boot=10000):
    """
    PREDEP α_{Y|X}: quanto X (MoV) prediz Y (precipitação).

    α = (S_{Y|X} - S_Y) / S_{Y|X}
    - 0: X e Y independentes
    - 1: X prediz Y perfeitamente
    """
    try:
        from astropy.stats import bayesian_blocks

        rng = np.random.default_rng()
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # S_Y: densidade marginal de Y
        diff_y = rng.choice(Y, n_boot) - rng.choice(Y, n_boot)
        if np.std(diff_y) == 0:
            return np.nan
        s_y = float(np.squeeze(ss.gaussian_kde(diff_y).pdf(0)))

        # Particiona X (preditor) com Bayesian Blocks
        edges_x = bayesian_blocks(X)
        if len(edges_x) < 2:
            return np.nan
        ecdf_x = ECDF(X)

        s_y_mid_x = 0.0
        for i in range(1, len(edges_x)):
            bg, ed = edges_x[i - 1], edges_x[i]
            mask = (X >= bg) & (X < ed)
            if i == len(edges_x) - 1:
                mask = mask | (X == ed)  # último bin fechado à direita
            y_bin = Y[mask]
            if len(y_bin) < 2 or len(np.unique(y_bin)) < 2:
                continue
            p_range = ecdf_x(ed) - ecdf_x(bg)
            dy_c = rng.choice(y_bin, n_boot) - rng.choice(y_bin, n_boot)
            if np.std(dy_c) == 0:
                continue
            kde_val = float(np.squeeze(ss.gaussian_kde(dy_c).pdf(0)))
            s_y_mid_x += p_range * kde_val

        if s_y_mid_x == 0:
            return np.nan
        return float((s_y_mid_x - s_y) / s_y_mid_x)
    except Exception:
        return np.nan
