

def regression_calc(X, Y):
    """
    Calcula o coeficiente de determinação (r²) de uma regressão linear
    simples entre X e Y.

    Parameters
    ----------
    X : array-like
        Variável preditora (e.g., precipitação)
    Y : array-like
        Variável resposta (e.g., índice TNA)

    Returns
    -------
    float
        r² ∈ [0, 1], onde 0 indica ausência de relação linear e
        1 indica relação perfeita.
        Retorna np.nan se não for possível calcular.

    Notes
    -----
    O algoritmo:
    1. Valida que X e Y têm o mesmo tamanho (ValueError explícito se não)
    2. Verifica mínimo de 3 observações e 3 valores únicos em cada variável
    3. Calcula regressão linear via scipy.stats.linregress(X, Y)
    4. Retorna r² = rvalue²
    """
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
