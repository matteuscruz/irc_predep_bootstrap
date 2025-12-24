"""
Módulo principal para cálculo do PREDEP (Predictive Dependence).
"""

import numpy as np
import scipy.stats as ss
from astropy.stats import bayesian_blocks


def ecdf(data):
    """Implementação simples de ECDF."""
    sorted_data = np.sort(data)
    n = len(data)
    
    def _ecdf(x):
        return np.searchsorted(sorted_data, x, side='right') / n
    
    return _ecdf


def predep(X, Y, n_boot=10000):
    """
    Calcula PREDEP (Predictive Dependence) entre duas variáveis contínuas.
    
    VERSÃO ORIGINAL - SEM MODIFICAÇÕES
    
    PREDEP mede quanto o conhecimento de Y reduz a incerteza sobre X,
    usando estimação de densidade de kernel e particionamento adaptativo.
    
    Parameters
    ----------
    X : array-like
        Primeira variável (geralmente o preditor - MoV index)
    Y : array-like
        Segunda variável (geralmente a resposta - precipitação)
    n_boot : int, optional
        Número de amostras de bootstrap (padrão: 10000)
    
    Returns
    -------
    float
        Valor PREDEP α ∈ [0, 1]
        - 0: variáveis independentes
        - 1: Y prediz perfeitamente X
        - np.nan: se não for possível calcular
    
    Notes
    -----
    O algoritmo:
    1. Estima s(X) usando diferenças bootstrap e KDE
    2. Particiona Y usando Bayesian Blocks
    3. Calcula s(X|Y) para cada partição
    4. Retorna α = (s(X|Y) - s(X)) / s(X|Y)
    """
    BX1 = np.random.choice(X, n_boot)
    BX2 = np.random.choice(X, n_boot)
    DX = BX1 - BX2
    s_x = ss.gaussian_kde(DX).pdf(0)

    edges_y = bayesian_blocks(Y)
    ecdf_y = ecdf(Y)

    s_x_mid_y = 0
    if edges_y.shape[0] > 1:
        for i in range(1, edges_y.shape[0]):
            bg = edges_y[i - 1]
            ed = edges_y[i]
            X_mid_Y = X[(Y >= bg) & (Y < ed)]
            if X_mid_Y.shape[0] == 0:
                continue

            BX_mid_Y1 = np.random.choice(X_mid_Y, n_boot)
            BX_mid_Y2 = np.random.choice(X_mid_Y, n_boot)
            DX_mid_Y = BX_mid_Y1 - BX_mid_Y2

            p_range = ecdf_y(ed) - ecdf_y(bg)

            p_x_mid_y = ss.gaussian_kde(DX_mid_Y).pdf(0)
            s_x_mid_y += p_range * (p_x_mid_y)
        alpha_est = (s_x_mid_y - s_x) / s_x_mid_y
        return alpha_est
    else:
        return np.nan