"""Calibração nula do PREDEP por surrogates de deslocamento anual.

Motivação
---------
Uma estimativa pontual de α não diz se um valor é sinal ou flutuação amostral. O
artigo original propõe intervalos de confiança bootstrap (Seç. 5); aqui resolvemos a
metade complementar do problema — **qual valor de α seria obtido se não houvesse
relação alguma** — que é o que permite afirmar magnitude nos Resultados.

Desenho do surrogate
--------------------
Deslocar o índice de MoV por um número inteiro de **anos**. Isso:

- preserva **exatamente** a autocorrelação do índice e o seu ciclo sazonal, porque a
  série continua sendo ela mesma, apenas realinhada;
- destrói o pareamento com a precipitação;
- não exige nenhuma hipótese paramétrica (ao contrário de randomização de fase).

Duas limitações a declarar no texto:

1. preserva a estrutura do **índice**, não a da precipitação — o nulo é, portanto,
   levemente otimista;
2. o limiar resultante é por (índice, estação, defasagem), não por célula. A hipótese
   de que o nulo varia pouco no espaço deve ser testada numa subamostra de células.
"""

from __future__ import annotations

import numpy as np

from .core import predep

#: Deslocamentos pequenos demais deixam autocorrelação residual entre índice e chuva.
MIN_SHIFT_YEARS = 3


def year_shift_surrogate(mov_lookup: dict, shift_years: int) -> dict:
    """Devolve o lookup do MoV com as chaves deslocadas em `shift_years` anos."""
    return {(y + shift_years, m): v for (y, m), v in mov_lookup.items()}


def align_shifted(mov_lookup, years, months, lag, shift_years):
    """Alinha o MoV deslocado à série de precipitação, com o mesmo `lag`.

    Espelha `lag_alignment.precompute_mov_alignment`, aplicando o deslocamento antes.
    """
    out = np.empty(len(years), dtype=float)
    for k, (ty, tm) in enumerate(zip(years, months)):
        m2, y2 = tm - lag, ty - shift_years
        while m2 <= 0:
            m2 += 12
            y2 -= 1
        out[k] = mov_lookup.get((y2, m2), np.nan)
    return out


def null_distribution(
    mov_lookup,
    precip,
    years,
    months,
    lag,
    shifts=None,
    *,
    min_pairs: int = 10,
    **predep_kwargs,
):
    """α sob a hipótese nula, para uma célula e uma (estação, defasagem).

    `precip` é a série de anomalias já restrita à estação. Devolve o vetor de α dos
    surrogates que puderam ser calculados.
    """
    if shifts is None:
        span = max(years) - min(years)
        shifts = [s for s in range(MIN_SHIFT_YEARS, span - MIN_SHIFT_YEARS + 1)]

    out = []
    for s in shifts:
        x = align_shifted(mov_lookup, years, months, lag, s)
        ok = np.isfinite(x) & np.isfinite(precip)
        if ok.sum() < min_pairs:
            continue
        a = predep(x[ok], precip[ok], **predep_kwargs)
        if np.isfinite(a):
            out.append(a)
    return np.asarray(out)


def null_threshold(samples, q: float = 95.0) -> float:
    """Limiar α_q da distribuição nula. `nan` se houver amostras de menos."""
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size < 10:
        return float("nan")
    return float(np.percentile(samples, q))
