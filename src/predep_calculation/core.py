import numpy as np
import scipy.stats as ss
from scipy.cluster.hierarchy import fcluster, linkage


def predep(X, Y, n_boot=10000, method="ward"):
    """
    PREDEP α_{Y|X}: mede quanto X prediz Y.

    method='ward'            : Hierarchical Clustering Ward k=√N (Algorithm 1 do paper)
    method='bayesian_blocks' : Bayesian Blocks (astropy) para particionamento de X
    """
    try:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        N = len(X)

        BY1 = np.random.choice(Y, n_boot)
        BY2 = np.random.choice(Y, n_boot)
        diff_y = BY1 - BY2
        if np.std(diff_y) == 0:
            return np.nan
        s_y = float(np.squeeze(ss.gaussian_kde(diff_y).pdf(0)))

        if method == "ward":
            k = max(2, int(np.sqrt(N)))
            Z = linkage(X.reshape(-1, 1), method="ward")
            labels = fcluster(Z, k, criterion="maxclust") - 1
            bins = [labels == c for c in range(k)]
        elif method == "bayesian_blocks":
            from astropy.stats import bayesian_blocks
            edges = bayesian_blocks(X)
            if len(edges) < 2:
                return np.nan
            bins = [(X >= edges[i - 1]) & (X < edges[i]) for i in range(1, len(edges))]
            bins[-1] = bins[-1] | (X == edges[-1])  # último bin fechado à direita
        else:
            raise ValueError(f"method deve ser 'ward' ou 'bayesian_blocks': {method!r}")

        s_y_mid_x = 0.0
        for idx in bins:
            y_bin = Y[idx]
            if len(y_bin) < 2 or len(np.unique(y_bin)) < 2:
                continue
            p = idx.sum() / N
            dy_c = np.random.choice(y_bin, n_boot) - np.random.choice(y_bin, n_boot)
            if np.std(dy_c) == 0:
                continue
            kde_val = float(np.squeeze(ss.gaussian_kde(dy_c).pdf(0)))
            s_y_mid_x += p * kde_val

        if s_y_mid_x == 0:
            return np.nan
        return float((s_y_mid_x - s_y) / s_y_mid_x)
    except Exception:
        return np.nan
