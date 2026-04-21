"""
Módulo para cálculo de regressão linear simples.
"""

import numpy as np
import scipy.stats as ss


def regression_calc(X, Y):
    """
    Calcula o coeficiente de determinação (r²) de uma regressão linear
    simples entre X e Y.

    Parameters
    ----------
    X : array-like
        Variável preditora (e.g., índice MoV)
    Y : array-like
        Variável resposta (e.g., precipitação)

    Returns
    -------
    float
        r² ∈ [0, 1], onde 0 indica ausência de relação linear e
        1 indica relação perfeita.
        Retorna np.nan se não for possível calcular.
    """
    try:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if len(X) != len(Y):
            raise ValueError(
                f"X e Y devem ter o mesmo tamanho (X={len(X)}, Y={len(Y)})"
            )
        if len(X) < 3 or len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
            return float('nan')

        result = ss.linregress(X, Y)
        return float(result.rvalue ** 2)
    except Exception:
        return float('nan')
