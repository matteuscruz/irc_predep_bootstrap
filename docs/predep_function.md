predep(X, Y, n_boot=10000):
    """
    Calcula PREDEP α_{Y|X}: quanto o conhecimento de X (preditor)
    reduz a incerteza sobre Y (resposta). 

    Interpretação: α = (S_{Y|X} - S_Y) / S_{Y|X}
    - 0: X e Y independentes
    - 1: X prediz Y perfeitamente
    """
    # S_Y: densidade marginal de Y estimada por bootstrap
    BY1 = np.random.choice(Y, n_boot)
    BY2 = np.random.choice(Y, n_boot)
    DY = BY1 - BY2
    s_y = ss.gaussian_kde(DY).pdf(0)

    # Particiona X (o condicionante/preditor)
    edges_x = bayesian_blocks(X)
    ecdf_x = ECDF(X)

    s_y_mid_x = 0
    if edges_x.shape[0] > 1:
        for i in range(1, edges_x.shape[0]):
            bg = edges_x[i - 1]
            ed = edges_x[i]

            # Filtra Y nos índices onde X está no bin [bg, ed)
            Y_mid_X = Y[(X >= bg) & (X < ed)]
            if Y_mid_X.shape[0] == 0:
                continue

            BY_mid_X1 = np.random.choice(Y_mid_X, n_boot)
            BY_mid_X2 = np.random.choice(Y_mid_X, n_boot)
            DY_mid_X = BY_mid_X1 - BY_mid_X2

            p_range = ecdf_x(ed) - ecdf_x(bg)
            p_y_mid_x = ss.gaussian_kde(DY_mid_X).pdf(0)
            s_y_mid_x += p_range * p_y_mid_x

        alpha_est = (s_y_mid_x - s_y) / s_y_mid_x
        return alpha_est
    else:
        return np.nan