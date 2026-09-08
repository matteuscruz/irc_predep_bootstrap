"""
Nós da pipeline de demonstração/validação do PREDEP.

Cada nó chama a função REAL do repo principal (não duplica lógica) — o
objetivo é visualizar, com Kedro-Viz, os passos verdadeiros de construção
dos dados e cálculo do PREDEP, para validação contra o paper de referência
(``references/PREDEP.pdf``), não reimplementar nada.

Roda numa fatia pequena (1 MoV, 1 pixel real, 4 estações, alguns lags) —
o pipeline de produção roda a bacia inteira (7503 pixels × 23 MoVs) no
Modal; replicar isso aqui não é o objetivo (Kedro-Viz mostra a ESTRUTURA
do fluxo de dados, não cada iteração de um laço em escala de produção).
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# kedro/src/predep_viz/pipelines/predep/nodes.py -> repo root
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

SOURCE_FILES = {
    "xavier": ROOT / "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc",
    "chirps": ROOT / "data/processed/chirps_monthly/pr_monthly_1981_2026_CHIRPS_0p1deg.nc",
}


def load_precip_source(source: str, time_start: str, time_end: str) -> xr.DataArray:
    """
    Nó 1 — Ingestão: abre a precipitação mensal já processada (Xavier ou
    CHIRPS), já regridada para a mesma grade de 0.1° (ver
    ``tools/convert_chirps_daily_to_monthly.py``) e recortada para a janela
    temporal pareada 1981-2019 usada em
    ``experiments/08_predep_parana_recompute.py``.
    """
    if source not in SOURCE_FILES:
        raise ValueError(f"source inválido: {source!r} (use 'xavier' ou 'chirps')")
    ds = xr.open_dataset(SOURCE_FILES[source]).sel(time=slice(time_start, time_end))
    return ds["pr"]


def load_pixel_mask(mov_name: str) -> dict:
    """
    Nó 2 — escolhe um pixel real e válido da bacia do Paraná, a partir da
    máscara combinada (polígono do Paraná ∩ cobertura CHIRPS completa) já
    construída em
    ``experiments/08_predep_parana_recompute.py:build_combined_mask``.

    Retorna o mesmo pixel usado na investigação de ruído de bootstrap desta
    sessão (lat=-21.45, lon=-54.05) — permite comparar o α produzido aqui
    com o valor já auditado manualmente.

    ``mov_name`` não é usado no cálculo (a máscara é a mesma para todos os
    MoVs) — está no assinatura só para o Kedro-Viz deixar explícito que a
    escolha do pixel é parte do mesmo contexto de execução do MoV.
    """
    npz = np.load(ROOT / "data/processed/chirps_monthly/parana_combined_mask.npz")
    lats, lons = npz["lats"], npz["lons"]
    valid_idx = np.argwhere(npz["mask"])
    i, j = valid_idx[len(valid_idx) // 2]
    return {"lat": float(lats[i]), "lon": float(lons[j])}


def extract_pixel_series(precip: xr.DataArray, pixel: dict) -> pd.Series:
    """
    Nó 3 — extrai a série temporal de precipitação mensal em UM pixel
    (vizinho mais próximo do lat/lon escolhido), como ``pandas.Series``
    indexada por tempo.
    """
    da = precip.sel(latitude=pixel["lat"], longitude=pixel["lon"], method="nearest")
    series = da.to_series()
    series.index.name = "time"
    return series


def compute_monthly_climatology(pixel_series: pd.Series) -> pd.Series:
    """
    Nó 4 — climatologia mensal: média de cada mês-calendário (1-12) sobre
    toda a janela 1981-2019. Base para a anomalia (nó 5). Mesma convenção de
    ``experiments/08_predep_parana_recompute.py`` (climatologia por mês
    calendário, não por estação).
    """
    months = pixel_series.index.month
    clim = pixel_series.groupby(months).mean()
    clim.index.name = "month"
    return clim


def compute_anomaly(pixel_series: pd.Series, climatology: pd.Series) -> pd.Series:
    """
    Nó 5 — anomalia mensal = precipitação − climatologia[mês]. É essa
    anomalia (não a precipitação bruta) que entra no ``predep()`` como Y.

    Contexto de validação (medido nesta sessão, ver
    ``notebooks/exploratory/chirps_vs_xavier_divergence_investigation.ipynb``):
    a correlação entre CHIRPS e Xavier cai de ~0.95 (precipitação bruta,
    dominada pelo ciclo sazonal) para ~0.90 na anomalia mensal — uma das
    causas identificadas da divergência entre fontes no α/SOM.
    """
    months = pixel_series.index.month
    clim_per_obs = climatology.reindex(months).to_numpy()
    return pd.Series(
        pixel_series.to_numpy() - clim_per_obs,
        index=pixel_series.index,
        name="anomaly",
    )


def load_mov_index(mov_name: str) -> dict:
    """
    Nó 6 — carrega o índice do MoV
    (``data/raw/mov_indices/{mov}_indice*.csv``) e monta um lookup
    ``{(ano, mês): valor}``. Mesmo parsing de
    ``experiments/08_predep_parana_recompute.py:upload_mov_index``.
    """
    matches = glob.glob(str(ROOT / f"data/raw/mov_indices/{mov_name}_indice*.csv"))
    if not matches:
        raise FileNotFoundError(f"Índice não encontrado para MoV '{mov_name}'")
    df = pd.read_csv(matches[0])
    df.columns = [c.strip() for c in df.columns]
    df = df[[df.columns[0], df.columns[1]]]
    df.columns = ["time", "value"]
    df["time"] = pd.to_datetime(df["time"])
    return {
        (d.year, d.month): v
        for d, v in zip(df["time"], df["value"])
        if not pd.isna(v)
    }


def filter_season(anomaly: pd.Series, season: str) -> pd.Series:
    """
    Nó 7 (repetido por estação — DJF/MAM/JJA/SON) — recorta a anomalia
    mensal para os meses da estação climática (``SEASON_MONTHS``, idêntico
    ao usado em ``experiments/08_predep_parana_recompute.py``).
    """
    months = anomaly.index.month
    return anomaly[np.isin(months, SEASON_MONTHS[season])]


def align_mov_all_lags(season_anomaly: pd.Series, mov_lookup: dict, lags: list) -> pd.DataFrame:
    """
    Nó 8 (repetido por estação; **lag tratado como dimensão interna, não
    como um nó por valor** — colapsado a pedido do usuário para não
    explodir o grafo em 1 nó por lag) — alinha o MoV a cada observação de
    precipitação da estação, para CADA lag em ``lags`` (parâmetro
    ``params:lags``), ``lag`` meses ANTES (rollback de ano quando
    mês−lag ≤ 0, ex.: mês=1, lag=3 → mês=10 do ano anterior).

    Usa a função real extraída nesta sessão para eliminar a duplicação que
    existia entre ``experiments/06,07,08_*.py``:
    ``src.predep_calculation.lag_alignment.precompute_mov_alignment``.

    Retorna um DataFrame "tidy" (formato longo) com colunas ``lag``, ``X``
    (MoV alinhado) e ``Y`` (anomalia de precipitação) — uma linha por
    observação×lag, ainda não filtrado por validade. Formato longo (em vez
    de um dict por lag) de propósito: permite ao Kedro-Viz mostrar uma
    prévia tabular real deste passo (dict não é previsualizável).
    """
    from src.predep_calculation.lag_alignment import precompute_mov_alignment

    years = season_anomaly.index.year.to_numpy()
    months = season_anomaly.index.month.to_numpy()
    frames = [
        pd.DataFrame(
            {
                "lag": lag,
                "X": precompute_mov_alignment(mov_lookup, years, months, lag),
                "Y": season_anomaly.to_numpy(),
            }
        )
        for lag in lags
    ]
    return pd.concat(frames, ignore_index=True)


def build_valid_xy_all_lags(aligned: pd.DataFrame) -> dict:
    """
    Nó 9 (por estação, todos os lags de uma vez) — filtro de validade antes
    de chamar o ``predep()``: exige ≥10 observações completas (sem NaN) e
    ≥3 valores distintos em X e em Y. Mesmo critério de
    ``experiments/08_predep_parana_recompute.py:_compute_lag_worker`` —
    evita degenerar o particionamento (bayesian_blocks) e o KDE em amostras
    pequenas ou quase-constantes.

    Recebe o DataFrame tidy (``lag, X, Y``) produzido por
    ``align_mov_all_lags`` e agrupa por ``lag`` internamente. Retorna
    ``{lag: {"X", "Y", "ok", "n"}}`` — estrutura interna (não catalogada
    para preview: já é o que os dois nós seguintes consomem diretamente).
    """
    result = {}
    for lag, group in aligned.groupby("lag"):
        valid = group.dropna()
        ok = bool(
            len(valid) >= 10
            and valid["X"].nunique() >= 3
            and valid["Y"].nunique() >= 3
        )
        result[int(lag)] = {
            "X": valid["X"].to_numpy(),
            "Y": valid["Y"].to_numpy(),
            "ok": ok,
            "n": int(len(valid)),
        }
    return result


def compute_predep_all_lags(xy_by_lag: dict, n_boot: int, seed: int) -> dict:
    r"""
    Nó 10 (por estação, todos os lags de uma vez) — a equação do PREDEP
    (Assunção, Figueiredo, Tinoco Jr., de Sá-Freire, Silva — "An
    Interpretable Measure for Quantifying Predictive Dependence between
    Continuous Random Variables", ``references/PREDEP.pdf``):

        α_{Y|X} = (S_{Y|X} - S_Y) / S_{Y|X}                          (Eq. 3.3)

        S_Y     = E[f_Y(Y)]            = ∫ f_Y(y)² dy                (Eq. 3.1)
        S_{Y|X} = E_X[ E_{Y|X} f(Y|X) ]                              (Eq. 3.2)

    S_Y (e S_{Y|X}) são estimados via bootstrap explorando que, se
    W = Y₁ − Y₂ para duas cópias independentes de Y, a densidade de W em
    w=0 é exatamente E[f_Y(Y)] — "densidade de convolução avaliada em zero"
    (Seção 5 do paper). O mesmo procedimento é repetido POR BIN de X
    (Algoritmo 1 do paper) para estimar S_{Y|X}, ponderando cada bin pela
    massa de probabilidade de X naquele intervalo (via ECDF).

    Chama o ``predep()`` REAL de ``src/predep_calculation/core.py`` — nada
    é reimplementado aqui. Aplicado a CADA lag de ``xy_by_lag`` (um lag por
    valor do dict); o lag em si não gera um nó próprio no grafo (colapsado
    a pedido do usuário — o Kedro-Viz mostraria 1 nó idêntico por lag, sem
    ganho de clareza).

    **Notas de validação contra o paper (auditoria feita em 2026-08-31,
    NÃO corrigidas no código de produção — decisão do usuário):**

    - *Convenção X/Y*: nosso ``predep(X, Y)`` bina X e testa Y → α_{Y|X}.
      Isso bate exatamente com o Algoritmo 1 (pseudocódigo) do paper. Os
      scripts soltos de referência (``references/predep.py`` e
      ``predep_paper.py``) usam a convenção OPOSTA (binam Y, testam X) —
      não é um erro nosso, o próprio Algoritmo 1 confirma nossa convenção.
    - *Discrepância 1 — binning*: o texto principal do paper e o Algoritmo 1
      especificam Hierarchical Clustering com k=√n para particionar X.
      Nosso código usa ``astropy.stats.bayesian_blocks(X)`` (igual ao
      script de referência ``predep.py``, mas diferente do texto principal
      do paper).
    - *Discrepância 2 — n_boot*: o Algoritmo 1 especifica
      ``n_b = ⌈n·log(n)⌉`` (dependente do tamanho da amostra). Aqui
      ``n_boot`` é um valor fixo (parâmetro ``n_boot`` desta pipeline). Para
      uma estação típica (~117 observações), a fórmula do paper daria
      ~557 — bem menor que os 5000/10000/20000 testados na investigação de
      hoje.
    """
    from src.predep_calculation.core import predep

    result = {}
    for lag, xy in xy_by_lag.items():
        result[lag] = predep(xy["X"], xy["Y"], n_boot=n_boot, seed=seed) if xy["ok"] else float("nan")
    return result


def compute_regression_all_lags(xy_by_lag: dict) -> dict:
    """
    Nó 11 (por estação, todos os lags de uma vez) — baseline linear (r²),
    calculado com a mesma função real usada em produção
    (``src/regression_calculation/core.py``), para comparação lado a lado
    com o α do PREDEP (o PREDEP captura dependência não-linear que o r²
    não vê). Aplicado a cada lag de ``xy_by_lag``.
    """
    from src.regression_calculation.core import regression_calc

    result = {}
    for lag, xy in xy_by_lag.items():
        result[lag] = regression_calc(xy["X"], xy["Y"]) if xy["ok"] else float("nan")
    return result


def assemble_season_result(
    alpha_by_lag: dict, r2_by_lag: dict, xy_by_lag: dict, season: str
) -> pd.DataFrame:
    """
    Nó 12 (por estação) — monta uma linha de resultado (estação, lag, α,
    r², n_obs) para CADA lag desta estação, num único DataFrame.
    """
    return pd.DataFrame(
        [
            {
                "season": season,
                "lag": lag,
                "alpha_core": alpha_by_lag[lag],
                "r2": r2_by_lag[lag],
                "n_obs": xy_by_lag[lag]["n"],
            }
            for lag in sorted(alpha_by_lag)
        ]
    )


def combine_results(*season_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Nó final — concatena os DataFrames de todas as estações (cada um já com
    uma linha por lag) num único resultado. Persistido via catálogo
    (``conf/base/catalog.yml:predep_results``, ``pandas.CSVDataset``) para
    inspeção manual (comparação com os NCs de produção em
    ``results/predep_granular_brazil/``) e preview no Kedro-Viz.
    """
    return pd.concat(season_dfs, ignore_index=True).sort_values(["season", "lag"]).reset_index(drop=True)
