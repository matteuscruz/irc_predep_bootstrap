#!/usr/bin/env python3
"""
Junta os dois SOMs (Xavier e CHIRPS) em uma única imagem 1x3: mapa de
regimes de cada fonte + um terceiro painel de concordância local — para
cada pixel, quantos dos top-k MoVs mais explicativos (local_mov_importance)
são os MESMOS nas duas fontes (0 a top_k), uma medida contínua de
intensidade de concordância em vez de um match binário de regime (os ids de
regime não são comparáveis entre dois SOMs treinados independentemente).

Retreina os dois SOMs com os MESMOS parâmetros de generate_som_insights.py
(seed=123, lags 0,1,3,6,9,12, n_regimes=7, variable=alpha_core) — treino é
determinístico e rápido, então não depende de reusar objetos de uma rodada
anterior.

Uso:
  python tools/gen_som_comparison_plot.py \\
      --xavier-dir results/predep_granular_brazil/exp08_parana_recompute/xavier \\
      --xavier-suffix _predep_granular_seasonal_xavier \\
      --chirps-dir results/predep_granular_brazil/exp08_parana_recompute/chirps \\
      --chirps-suffix _predep_granular_seasonal_chirps \\
      --out-dir plots/weekly_20260831_chirps_vs_xavier \\
      --out-name som_comparison.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.analysis.som_insights import (  # noqa: E402
    build_feature_matrix, train_som, assign_regimes, regime_to_grid,
    regime_fingerprints, local_mov_importance,
)
from src.visualization.cluster_plots import _load_basin_outlines  # noqa: E402
from src.visualization.som_plots import plot_som_comparison  # noqa: E402

DEFAULT_LAGS = [0, 1, 3, 6, 9, 12]
N_REGIMES = 7
SEED = 123
TOP_K = 3


def run_som(exp_dir, nc_suffix, label, optimize=False, auto_k=False,
           search_sample_size=None, search_epochs=None):
    import time

    print(f"[{label}] Construindo matriz de features ({exp_dir})...")
    bundle = build_feature_matrix(
        exp_dir, mov_filter=None, lags=DEFAULT_LAGS,
        standardize=True, variable="alpha_core", nc_suffix=nc_suffix,
    )
    print(f"[{label}] Amostras: {bundle.X.shape[0]}  Features: {bundle.X.shape[1]}")

    tag = " (otimização Pareto QE×TE)" if optimize else ""
    print(f"[{label}] Treinando SOM (seed={SEED}){tag}...")
    t0 = time.time()
    som, metrics = train_som(
        bundle.X, som_size=None, num_iteration=10000, seed=SEED,
        optimize=optimize, search_sample_size=search_sample_size,
        search_epochs=search_epochs,
    )
    print(f"[{label}]   som_size={metrics['som_size']}  qe={metrics['quantization_error']:.3f}"
         f"  ({time.time() - t0:.1f}s)")

    k_tag = " (busca de K via cotovelo)" if auto_k else ""
    print(f"[{label}] Atribuindo regimes{k_tag}...")
    regimes = assign_regimes(som, bundle.X, n_regimes=N_REGIMES, seed=SEED,
                             auto_k=auto_k, max_k=len(bundle.mov_names))
    if auto_k:
        print(f"[{label}]   K escolhido: {regimes.n_regimes}")
    regime_grid = regime_to_grid(regimes, bundle)
    fps = regime_fingerprints(regimes, bundle)
    imp = local_mov_importance(bundle, som, regimes, top_k=TOP_K)

    return bundle, regime_grid, fps, imp


def overlap_grid_from(bundle_a, imp_a, bundle_b, imp_b, top_k):
    n_lat, n_lon = len(bundle_a.lats), len(bundle_a.lons)
    assert (n_lat, n_lon) == (len(bundle_b.lats), len(bundle_b.lons)), (
        "Grades diferentes entre as duas fontes — precisam ter o mesmo footprint."
    )

    valid_a = np.zeros((n_lat, n_lon), dtype=bool)
    valid_a[bundle_a.pixel_rc[:, 0], bundle_a.pixel_rc[:, 1]] = True
    valid_b = np.zeros((n_lat, n_lon), dtype=bool)
    valid_b[bundle_b.pixel_rc[:, 0], bundle_b.pixel_rc[:, 1]] = True
    both = valid_a & valid_b

    top_a = np.full((n_lat, n_lon, top_k), None, dtype=object)
    top_a[bundle_a.pixel_rc[:, 0], bundle_a.pixel_rc[:, 1], :] = imp_a["top_movs"]
    top_b = np.full((n_lat, n_lon, top_k), None, dtype=object)
    top_b[bundle_b.pixel_rc[:, 0], bundle_b.pixel_rc[:, 1], :] = imp_b["top_movs"]

    overlap = np.full((n_lat, n_lon), np.nan)
    for i, j in zip(*np.where(both)):
        overlap[i, j] = len(set(top_a[i, j, :]) & set(top_b[i, j, :]))
    return overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xavier-dir", required=True)
    parser.add_argument("--xavier-suffix", default="_predep_granular_seasonal_xavier")
    parser.add_argument("--chirps-dir", required=True)
    parser.add_argument("--chirps-suffix", default="_predep_granular_seasonal_chirps")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-name", default="som_comparison.png")
    parser.add_argument("--caveat", default=None,
                        help="Texto de ressalva exibido no rodapé da figura "
                             "(ex.: aviso sobre bootstrap não-seedado nos dados de entrada).")
    parser.add_argument("--optimize", action="store_true",
                        help="Busca de topologia via fronteira de Pareto QE×TE "
                             "(metodologia de referência do exp01), independente "
                             "por fonte — Xavier e CHIRPS podem sair com grades "
                             "diferentes entre si.")
    parser.add_argument("--auto-k", action="store_true",
                        help="Escolhe o nº de regimes via cotovelo/silhueta em "
                             "vez do --n-regimes fixo, independente por fonte.")
    parser.add_argument("--search-sample-size", type=int, default=None,
                        help="[--optimize] Subamostra usada na varredura de "
                             "topologias (modelo final sempre retreina na base "
                             "completa). Ex.: 3000 — acelera bastante a busca.")
    parser.add_argument("--search-epochs", type=int, default=None,
                        help="[--optimize] Épocas por candidata na varredura "
                             "(padrão: mesmo valor do treino final). Ex.: 2.")
    args = parser.parse_args()

    som_kwargs = dict(
        optimize=args.optimize, auto_k=args.auto_k,
        search_sample_size=args.search_sample_size,
        search_epochs=args.search_epochs,
    )
    bundle_x, grid_x, fps_x, imp_x = run_som(args.xavier_dir, args.xavier_suffix, "Xavier", **som_kwargs)
    bundle_c, grid_c, fps_c, imp_c = run_som(args.chirps_dir, args.chirps_suffix, "CHIRPS", **som_kwargs)

    print(f"Calculando concordância local (top-{TOP_K} MoVs por pixel)...")
    overlap = overlap_grid_from(bundle_x, imp_x, bundle_c, imp_c, TOP_K)
    n_both = int(np.sum(~np.isnan(overlap)))
    print(f"  Pixels comparados: {n_both}  |  concordância média: {np.nanmean(overlap):.2f}/{TOP_K}")

    mov0 = bundle_x.mov_names[0]
    outlines = _load_basin_outlines(Path(args.xavier_dir) / mov0, mov0)

    plot_som_comparison(
        grid_x, bundle_x, fps_x, "Xavier",
        grid_c, bundle_c, fps_c, "CHIRPS",
        overlap, TOP_K, args.out_dir, args.out_name,
        basin_outlines=outlines, caveat=args.caveat,
    )


if __name__ == "__main__":
    main()
