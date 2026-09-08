import numpy as np


def precompute_mov_alignment(mov_lookup, s_years, s_months, lag):
    """
    Alinha o índice do MoV a cada observação de precipitação de uma estação,
    aplicando o lag em meses.

    Para cada mês/ano da estação, busca o valor do MoV `lag` meses ANTES
    (`mês - lag`), com rollback de ano quando o mês resultante for <= 0
    (ex.: mês=1, lag=3 -> mês=-2 -> +12 -> mês=10, ano-1). `mov_lookup` é um
    dict {(ano, mês): valor}; combinações sem MoV disponível viram NaN
    (filtradas depois, antes de chamar `predep()`).

    Usada de forma idêntica pelos três experimentos de cálculo granular de
    PREDEP (`experiments/06,07,08_*.py`) — extraída aqui para não duplicar a
    mesma lógica de alinhamento em cada script.
    """
    n = len(s_years)
    aligned = np.full(n, np.nan)
    for k in range(n):
        tm = int(s_months[k]) - lag
        ty = int(s_years[k])
        while tm <= 0:
            tm += 12
            ty -= 1
        aligned[k] = mov_lookup.get((ty, tm), np.nan)
    return aligned
