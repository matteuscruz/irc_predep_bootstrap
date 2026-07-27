#!/usr/bin/env python3
"""
Gera os insights espaciais de SOM (MiniSom) para um experimento PREDEP.

Cada pixel do Brasil vira uma amostra descrita por α (alpha_core) de cada
MoV × lag, concatenado nas 4 estações. Um único SOM agrupa pixels com
assinatura parecida em "regimes", que são pintados no mapa e resumidos.

Plots gerados em plots/exp{NN}/som/ (NÃO substituem os plots existentes):
  - som_regime_map_exp{NN}.png          mapa de regimes (principal)
  - som_umatrix_exp{NN}.png             U-matrix
  - som_hitmap_exp{NN}.png              mapa de ativação
  - som_component_planes_exp{NN}.png    α por MoV na grade SOM
  - som_regime_fingerprints_exp{NN}.png assinatura (MoV/lag/estação) por regime
  - som_regime_mov_table_exp{NN}.png    heatmap regime × MoV
  - som_insights_exp{NN}.pdf            tudo num PDF

Uso:
  python src/analysis/generate_som_insights.py
  python src/analysis/generate_som_insights.py --exp-n 1
  python src/analysis/generate_som_insights.py --exp-n 1 --movs "ONI,NAO" --som-size 10 --n-regimes 5
  python src/analysis/generate_som_insights.py --exp-n 1 --lags all --no-standardize
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.som_insights import (  # noqa: E402
    build_feature_matrix, train_som, assign_regimes,
    regime_to_grid, regime_fingerprints,
)
from src.visualization import som_plots  # noqa: E402
from src.visualization.cluster_plots import _load_basin_outlines  # noqa: E402
from src.visualization.generate_plots_exp import latest_exp, _make_pdf  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-n", type=int, default=0,
                        help="Número do experimento (padrão: mais recente)")
    parser.add_argument("--movs", type=str, default="",
                        help="MoVs separados por vírgula (padrão: todos)")
    parser.add_argument("--lags", type=str, default="0,1,3,6,9,12",
                        help="Lags (vírgula) ou 'all' (padrão: 0,1,3,6,9,12)")
    parser.add_argument("--som-size", type=int, default=0,
                        help="Lado da grade SOM (padrão: heurística)")
    parser.add_argument("--n-regimes", type=int, default=7,
                        help="Nº de regimes (clusters de neurônios; padrão: 7)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Iterações de treino do SOM (padrão: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente (padrão: 42)")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Treina o SOM em α na escala nativa (sem z-score)")
    parser.add_argument("--som-diagnostics", action="store_true",
                        help="Inclui os diagnósticos quadrados do SOM "
                             "(U-matrix, hitmap, component planes na grade)")
    args = parser.parse_args()

    results_base = ROOT / "results" / "predep_granular_brazil"
    exp_n = args.exp_n if args.exp_n > 0 else latest_exp(results_base)
    exp_dir = results_base / f"exp{exp_n:02d}"
    if not exp_dir.exists():
        print(f"[ERRO] Experimento não encontrado: {exp_dir}")
        sys.exit(1)

    mov_filter = [m.strip() for m in args.movs.split(",") if m.strip()] or None
    lags = (None if args.lags.strip().lower() == "all"
            else [int(x.strip()) for x in args.lags.split(",") if x.strip()])

    print(f"Experimento : exp{exp_n:02d}")
    print("Construindo matriz de features...")
    bundle = build_feature_matrix(exp_dir, mov_filter, lags,
                                  standardize=not args.no_standardize)
    print(f"  Amostras (pixels): {bundle.X.shape[0]}  |  Features: {bundle.X.shape[1]}")
    print(f"  MoVs ({len(bundle.mov_names)}): {bundle.mov_names}")
    print(f"  Lags: {bundle.lags}  |  Estações: {bundle.seasons}")

    print("Treinando SOM...")
    som, metrics = train_som(bundle.X, som_size=args.som_size or None,
                             num_iteration=args.iterations, seed=args.seed)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"Atribuindo {args.n_regimes} regimes (KMeans nos neurônios)...")
    regimes = assign_regimes(som, bundle.X, n_regimes=args.n_regimes, seed=args.seed)
    regime_grid = regime_to_grid(regimes, bundle)
    fps = regime_fingerprints(regimes, bundle)

    mov0 = bundle.mov_names[0]
    fallback = sorted(results_base.glob(f"exp*/{mov0}"))
    outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)

    plots_dir = ROOT / "plots" / f"exp{exp_n:02d}" / "som"
    if plots_dir.exists():           # limpa saídas antigas (evita arquivos órfãos)
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gerando plots em {plots_dir}/ ...")
    pngs = [
        som_plots.plot_regime_map(regime_grid, bundle, fps, plots_dir, exp_n,
                                  basin_outlines=outlines),
        som_plots.plot_regime_panels(regime_grid, bundle, fps, plots_dir, exp_n,
                                     basin_outlines=outlines),
        som_plots.plot_geographic_component_planes(bundle, plots_dir, exp_n,
                                                   basin_outlines=outlines),
        som_plots.plot_atypicality_map(som, regimes, bundle, plots_dir, exp_n,
                                       basin_outlines=outlines),
        som_plots.plot_boundary_map(som, regimes, bundle, plots_dir, exp_n,
                                    basin_outlines=outlines),
        som_plots.plot_regime_fingerprints(fps, bundle, plots_dir, exp_n),
        som_plots.plot_regime_mov_heatmap(fps, bundle, plots_dir, exp_n),
    ]
    if args.som_diagnostics:
        pngs += [
            som_plots.plot_umatrix(som, plots_dir, exp_n),
            som_plots.plot_hitmap(som, bundle.X, plots_dir, exp_n),
            som_plots.plot_component_planes(som, bundle, plots_dir, exp_n),
        ]
    _make_pdf(pngs, plots_dir / f"som_insights_exp{exp_n:02d}.pdf")
    print("Concluído.")


if __name__ == "__main__":
    main()
