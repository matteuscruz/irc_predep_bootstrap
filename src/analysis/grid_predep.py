"""
Cubo de α (PREDEP) e r² por pixel, para uma bacia.

O laço é o do `experiments/15_modal_grid_pixel.py`, promovido a módulo: a
partição em bins depende só de X — a série do MoV — então ela é construída UMA
vez por (estação, lag, MoV) e reaproveitada nos milhares de pixels daquela
combinação. É o que torna o cálculo viável; refazer a partição por pixel custa
5× mais e dá exatamente o mesmo número.

Os dois valores saem do MESMO alinhamento de amostras, e as duas fontes usam a
MESMA semente por (pixel, lag, estação, MoV). Sem isso, a diferença entre
Xavier e CHIRPS incluiria ruído de bootstrap, que é da mesma ordem do sinal
(ver `results/exp09_bootstrap_defeito.md`).

Convenções do artigo, já auditadas em `results/conformidade_final.md`:
partição `ward` com k=√N, `n_boot = ⌈n·log n⌉`, pares `i == j` permitidos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.predep_calculation.core import (build_partition,  # noqa: E402
                                         predep_with_partition)
from src.regression_calculation.core import regression_calc  # noqa: E402

SEASONS = ["DJF", "MAM", "JJA", "SON"]
MESES = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
         "JJA": (6, 7, 8), "SON": (9, 10, 11)}

# Estado por processo: o Pool não deve serializar o cubo de precipitação a cada
# tarefa (são centenas de MB).
_G: dict = {}


def _init(pr, mov, meses, seed):
    # Uma thread BLAS por processo. Sem isto cada worker abre N threads e os
    # processos brigam pelos mesmos núcleos: medido `sys` 13m contra `user`
    # 10m numa corrida de 5 min — mais tempo em troca de contexto que em conta.
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    _G.update(pr=pr, mov=mov, meses=meses, seed=seed)


def alinhar(mov_serie, pr, meses, estacao, lag):
    """Amostras (X, Y) da estação, com o MoV lido em `mês − lag`.

    A amostra é MENSAL, não a média sazonal: os 348 meses da janela comum dão
    87 amostras por estação (29 anos × 3 meses). Por isso não há agrupamento
    por ano-estação — cada mês da estação é um par (X, Y) independente para o
    estimador, e o dezembro de DJF não precisa ser reetiquetado.

    O lag encurta a amostra no começo da série: com lag 3, DJF cai para 85 e
    MAM para 86, porque os primeiros meses não têm de onde olhar para trás.
    """
    idx = np.where(np.isin(meses, MESES[estacao]))[0]
    if len(idx) == 0:
        return np.empty(0), np.empty((0, pr.shape[1]))
    src = idx - lag
    ok = src >= 0
    idx, src = idx[ok], src[ok]
    x = mov_serie[src]
    y = pr[idx]
    fin = np.isfinite(x)
    return x[fin], y[fin]


def deslocar_anos(mov_serie, anos):
    """Surrogate por deslocamento anual: rola a série do MoV em `anos` anos.

    Preserva a distribuição e a autocorrelação do índice e destrói só o
    pareamento com a chuva — é o nulo certo aqui. Embaralhar seria um nulo mais
    fraco: destruiria também a autocorrelação, e aí qualquer estrutura temporal
    do índice apareceria como "sinal".

    A rolagem é circular, o que cria uma emenda no ponto de dobra. Com 29 anos
    de série e deslocamentos de 3 a 13 anos, é uma emenda em 348 meses.
    """
    import numpy as np
    return np.roll(np.asarray(mov_serie, dtype=float), 12 * int(anos))


def _tarefa(si, li, mi):
    """α e r² de todos os pixels para uma combinação (estação, lag, MoV)."""
    pr, mov = _G["pr"], _G["mov"]                    # pr: (2, tempo, pixel)
    meses, seed0 = _G["meses"], _G["seed"]
    n_src, _, n_px = pr.shape
    out = np.full((2, n_src, n_px), np.nan, dtype=np.float32)

    x, _ = alinhar(mov[:, mi], pr[0], meses, SEASONS[si], li)
    if len(x) < 10:
        return si, li, mi, out
    part = build_partition(x, "ward")
    for s in range(n_src):
        xs, ys = alinhar(mov[:, mi], pr[s], meses, SEASONS[si], li)
        for p in range(n_px):
            col = ys[:, p]
            fin = np.isfinite(col)
            if fin.sum() < 10:
                continue
            # Semente igual para as duas fontes no mesmo ponto do grid: a
            # diferença entre elas passa a ser diferença de dado, não de sorteio.
            semente = seed0 + p + 1000003 * (si + 4 * (li + 7 * mi))
            if fin.all():
                out[0, s, p] = predep_with_partition(part, col, seed=semente)
            else:
                out[0, s, p] = predep_with_partition(
                    build_partition(xs[fin], "ward"), col[fin], seed=semente)
            out[1, s, p] = regression_calc(xs[fin], col[fin])
    return si, li, mi, out


def calcular_grid(ds, pixels, movs_idx, *, lags=range(7), seasons=SEASONS,
                  seed=20260908, n_proc=None, progresso=True):
    """α e r² por pixel para uma bacia.

    Devolve `(alpha, r2)`, ambos `(fonte, estação, lag, MoV, pixel)`.

    `ds` é o dataset da janela comum (`predep_common_window.nc`); `pixels` são
    os índices devolvidos por `bacias.pixels_da_bacia`.
    """
    import pandas as pd
    from multiprocessing import Pool

    meses = pd.DatetimeIndex(ds.time.values).month.values
    pr = np.stack([ds["pr_xavier"].values[:, pixels],
                   ds["pr_chirps"].values[:, pixels]]).astype(np.float32)
    mov = ds["mov"].values.astype(float)

    seasons = list(seasons)
    lags = list(lags)
    movs_idx = list(movs_idx)
    forma = (2, len(seasons), len(lags), len(movs_idx), len(pixels))
    alpha = np.full(forma, np.nan, dtype=np.float32)
    r2 = np.full(forma, np.nan, dtype=np.float32)

    tarefas = [(si, li, mi) for si in range(len(seasons))
               for li in lags for mi in movs_idx]
    total = len(tarefas)
    if progresso:
        print(f"    {len(pixels):,} pixels × {len(movs_idx)} MoVs × {len(lags)} lags "
              f"× {len(seasons)} estações = {total} combinações", flush=True)

    args = (pr, mov, meses, seed)
    feito = 0
    with Pool(processes=n_proc, initializer=_init, initargs=args) as pool:
        for si, li, mi, res in pool.starmap(_tarefa, tarefas, chunksize=1):
            alpha[:, si, lags.index(li), movs_idx.index(mi)] = res[0]
            r2[:, si, lags.index(li), movs_idx.index(mi)] = res[1]
            feito += 1
            if progresso and feito % max(1, total // 20) == 0:
                print(f"      {feito}/{total}", flush=True)
    return alpha, r2


def escolher_por_predominancia(alpha):
    """O (lag, MoV) de cada estação, pela predominância com concordância.

    `argmax` da média espacial de `min(α_xavier, α_chirps)`: um candidato só
    pontua alto se AMBAS as fontes o acharem forte. A escolha é única para a
    bacia — é o que remove o viés de selecionar o máximo de 203 candidatos em
    cada pixel (ver METODOLOGIA_PREDOMINANCIA.md, seção 1).

    Devolve `(sel_lag, sel_mov, concordancia)`.
    """
    concord = np.nanmean(np.minimum(alpha[0], alpha[1]), axis=-1)   # (est, lag, mov)
    n_se, _, n_mov = concord.shape
    sel_lag = np.zeros(n_se, dtype=int)
    sel_mov = np.zeros(n_se, dtype=int)
    for si in range(n_se):
        plano = concord[si].ravel()
        k = int(np.nanargmax(np.where(np.isfinite(plano), plano, -np.inf)))
        sel_lag[si], sel_mov[si] = divmod(k, n_mov)
    return sel_lag, sel_mov, concord
