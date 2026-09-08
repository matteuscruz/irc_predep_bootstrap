"""
Stages 4/5 — análise a jusante do PREDEP, sobre os dados REAIS de produção
(os 23 NCs por fonte carregados pela pipeline `predep_production`).

Duas cadeias:
- **PREDEP máximo**: α máximo por pixel (sobre MoV × lag) + MoV responsável,
  comparando as duas fontes por trimestre.
- **SOM**: matriz de features (pixel × MoV×lag×estação) → treino → regimes →
  importância local por MoV → concordância entre as fontes.

Todos os nós chamam as funções reais de `src/analysis/som_insights.py`,
`src/visualization/cluster_plots.py` e `src/visualization/som_plots.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# kedro/src/predep_viz/pipelines/analysis/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

SEASONS = ["DJF", "MAM", "JJA", "SON"]


def plot_predep_max_comparison(
    stack_chirps: dict, stack_xavier: dict, out_dir: str, margin
) -> pd.DataFrame:
    """
    Stage 5 — os 4 plots comparativos de PREDEP máximo (um por trimestre),
    cada um 3×2: α máximo de cada fonte + diferença absoluta (linha 1) e MoV
    responsável de cada fonte + concordância espacial (linha 2).

    Reusa `src/visualization/cluster_plots.py:plot_predep_max_comparison_season`
    (a mesma função por trás de `tools/gen_predep_max_comparison_plots.py`).

    `margin` (ex.: 0.05, calibrada empiricamente contra o ruído de bootstrap
    do `predep()`): abaixo dela, diferenças de α viram "não significativo" e
    disputas entre o 1º e o 2º MoV viram "ambíguo". `null` desliga a margem.

    Retorna um DataFrame com as estatísticas por estação (nº de pixels
    conclusivos e % de concordância entre eles) — o número que resume a
    comparação.
    """
    from src.visualization.cluster_plots import (
        _max_and_best_mov,
        SIGNAL_THRESHOLD,
        plot_predep_max_comparison_season,
    )

    if stack_chirps["mov_names"] != stack_xavier["mov_names"]:
        raise ValueError("Lista de MoVs difere entre as fontes — comparação inválida.")

    out_path = ROOT / out_dir
    rows = []
    for season_idx, season in enumerate(SEASONS):
        plot_predep_max_comparison_season(
            stack_chirps["alpha_core"],
            stack_xavier["alpha_core"],
            stack_chirps["mov_names"],
            stack_chirps["lats"],
            stack_chirps["lons"],
            season_idx,
            season,
            out_path,
            margin=margin,
        )

        val_c, mov_c, valid_c, gap_c = _max_and_best_mov(
            stack_chirps["alpha_core"][:, season_idx]
        )
        val_x, mov_x, valid_x, gap_x = _max_and_best_mov(
            stack_xavier["alpha_core"][:, season_idx]
        )
        both = valid_c & valid_x
        if margin is None:
            conclusive = both
        else:
            low_c = valid_c & (val_c < SIGNAL_THRESHOLD)
            low_x = valid_x & (val_x < SIGNAL_THRESHOLD)
            conclusive = (
                both
                & ~low_c & ~low_x
                & (gap_c >= margin) & (gap_x >= margin)
            )
        n_conc = int(conclusive.sum())
        agree = int((mov_c[conclusive] == mov_x[conclusive]).sum()) if n_conc else 0
        rows.append(
            {
                "season": season,
                "pixels_footprint": int(both.sum()),
                "pixels_conclusivos": n_conc,
                "concordancia_pct": round(100 * agree / n_conc, 1) if n_conc else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_som_features(stack: dict, exp_dir: str, som_params: dict) -> dict:
    """
    Stage 4 — matriz de features do SOM: cada **pixel** vira uma amostra
    descrita pelo α de cada (MoV × lag × estação), z-scored.

    Chama `src/analysis/som_insights.py:build_feature_matrix` — a mesma
    função usada por `generate_som_insights.py` e
    `tools/gen_som_comparison_plot.py`. Com 23 MoVs × 6 lags × 4 estações
    são 552 features por pixel, sobre 7503 pixels.
    """
    from src.analysis.som_insights import build_feature_matrix

    source = stack["source"]
    exp_path = ROOT / "results/predep_granular_brazil" / exp_dir / source
    bundle = build_feature_matrix(
        exp_path,
        mov_filter=None,
        lags=som_params["lags"],
        standardize=True,
        variable="alpha_core",
        nc_suffix=f"_predep_granular_seasonal_{source}",
    )
    return {"bundle": bundle, "source": source}


def train_som_and_regimes(features: dict, som_params: dict) -> dict:
    """
    Stage 4 — treina o SOM (MiniSom) e agrupa os neurônios em regimes
    (KMeans), com `seed` fixa para reprodutibilidade.

    Chama as funções reais `train_som`, `assign_regimes`, `regime_to_grid`,
    `regime_fingerprints` e `local_mov_importance` de
    `src/analysis/som_insights.py`.

    `local_mov_importance` decompõe a distância pixel→neurônio vencedor por
    MoV: para cada pixel, quais MoVs mais explicam sua alocação. É a base da
    comparação entre fontes (ids de regime não são comparáveis entre dois
    SOMs treinados independentemente — MoV é vocabulário compartilhado).
    """
    from src.analysis.som_insights import (
        assign_regimes,
        local_mov_importance,
        regime_fingerprints,
        regime_to_grid,
        train_som,
    )

    bundle = features["bundle"]
    som, metrics = train_som(
        bundle.X,
        som_size=som_params.get("som_size") or None,
        num_iteration=som_params["iterations"],
        seed=som_params["seed"],
    )
    regimes = assign_regimes(
        som,
        bundle.X,
        n_regimes=som_params["n_regimes"],
        seed=som_params["seed"],
        max_k=len(bundle.mov_names),
    )
    return {
        "source": features["source"],
        "bundle": bundle,
        "som": som,
        "metrics": metrics,
        "regimes": regimes,
        "regime_grid": regime_to_grid(regimes, bundle),
        "fingerprints": regime_fingerprints(regimes, bundle),
        "importance": local_mov_importance(bundle, som, regimes, top_k=som_params["top_k"]),
    }


def compute_som_concordance(som_a: dict, som_b: dict, som_params: dict) -> dict:
    """
    Stage 4 — concordância local entre os dois SOMs: para cada pixel, quantos
    dos top-k MoVs mais explicativos são os MESMOS nas duas fontes (0..k).

    Mesma lógica de `tools/gen_som_comparison_plot.py:overlap_grid_from` —
    comparação por MoV (vocabulário compartilhado), não por id de regime.
    """
    top_k = som_params["top_k"]
    bundle_a, bundle_b = som_a["bundle"], som_b["bundle"]
    n_lat, n_lon = len(bundle_a.lats), len(bundle_a.lons)
    if (n_lat, n_lon) != (len(bundle_b.lats), len(bundle_b.lons)):
        raise ValueError("Grades diferentes entre as duas fontes.")

    valid_a = np.zeros((n_lat, n_lon), dtype=bool)
    valid_a[bundle_a.pixel_rc[:, 0], bundle_a.pixel_rc[:, 1]] = True
    valid_b = np.zeros((n_lat, n_lon), dtype=bool)
    valid_b[bundle_b.pixel_rc[:, 0], bundle_b.pixel_rc[:, 1]] = True
    both = valid_a & valid_b

    top_a = np.full((n_lat, n_lon, top_k), None, dtype=object)
    top_a[bundle_a.pixel_rc[:, 0], bundle_a.pixel_rc[:, 1], :] = som_a["importance"]["top_movs"]
    top_b = np.full((n_lat, n_lon, top_k), None, dtype=object)
    top_b[bundle_b.pixel_rc[:, 0], bundle_b.pixel_rc[:, 1], :] = som_b["importance"]["top_movs"]

    overlap = np.full((n_lat, n_lon), np.nan)
    for i, j in zip(*np.where(both)):
        overlap[i, j] = len(set(top_a[i, j, :]) & set(top_b[i, j, :]))

    return {
        "overlap": overlap,
        "top_k": top_k,
        "mean": float(np.nanmean(overlap)),
        "n_pixels": int(np.sum(~np.isnan(overlap))),
    }


def plot_som_comparison_figure(
    som_xavier: dict, som_chirps: dict, concordance: dict, out_dir: str, caveat
) -> pd.DataFrame:
    """
    Stage 5 — a figura 1×3 do SOM: mapa de regimes de cada fonte + a
    intensidade de concordância local (nº de MoVs em comum no top-k).

    Reusa `src/visualization/som_plots.py:plot_som_comparison` (a mesma
    função por trás de `tools/gen_som_comparison_plot.py`).

    Retorna as métricas resumidas (QE/TE de cada SOM, K, concordância média)
    para checagem numérica rápida.
    """
    from src.visualization.cluster_plots import _load_basin_outlines
    from src.visualization.som_plots import plot_som_comparison

    bundle_x = som_xavier["bundle"]
    mov0 = bundle_x.mov_names[0]
    outlines = _load_basin_outlines(
        ROOT / "results/predep_granular_brazil" / "exp01" / mov0, mov0
    )

    plot_som_comparison(
        som_xavier["regime_grid"], bundle_x, som_xavier["fingerprints"], "Xavier",
        som_chirps["regime_grid"], som_chirps["bundle"], som_chirps["fingerprints"], "CHIRPS",
        concordance["overlap"], concordance["top_k"], ROOT / out_dir,
        "som_comparison.png",
        basin_outlines=outlines,
        caveat=caveat,
    )

    return pd.DataFrame(
        [
            {
                "fonte": s["source"],
                "som_size": s["metrics"]["som_size"],
                "qe": round(s["metrics"]["quantization_error"], 3),
                "te": round(s["metrics"]["topographic_error"], 4),
                "n_regimes": s["regimes"].n_regimes,
                "concordancia_media": round(concordance["mean"], 3),
                "top_k": concordance["top_k"],
                "pixels": concordance["n_pixels"],
            }
            for s in (som_xavier, som_chirps)
        ]
    )
