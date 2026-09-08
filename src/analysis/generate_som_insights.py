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

  # Fonte Brasil-todo já unificada (1 arquivo por MoV×estação, sem quebras
  # nas fronteiras de bacia vindas do merge pós-cálculo por bacia):
  python src/analysis/generate_som_insights.py --exp-n 99 --source parquet \
      --parquet-dir ../predep_visualization/results/predep_granular_brazil/exp_brasil
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.som_insights import (  # noqa: E402
    build_feature_matrix, build_feature_matrix_from_parquet, train_som,
    assign_regimes, regime_to_grid, regime_fingerprints, movs_above_threshold,
)
from src.visualization import som_plots  # noqa: E402
from src.visualization.cluster_plots import _load_basin_outlines  # noqa: E402
from src.visualization.generate_plots_exp import latest_exp, _make_pdf  # noqa: E402
from src.som_modeling.som_visualization import (  # noqa: E402
    plot_pareto_curve, plot_elbow_curve, plot_dendrogram,
)

VALUE_LABELS = {"alpha_core": "α", "r2": "r²"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-n", type=int, default=0,
                        help="Número do experimento (padrão: mais recente). "
                             "Com --source nc, também rotula a saída "
                             "(plots/exp{N:02d}/...) mesmo quando --exp-dir "
                             "aponta para um diretório com outro nome.")
    parser.add_argument("--exp-dir", type=str, default=None,
                        help="[--source nc] Override do diretório de entrada "
                             "dos NCs (ex.: results/predep_granular_brazil/"
                             "exp06_chirps_check). Se omitido, usa "
                             "results/predep_granular_brazil/exp{--exp-n:02d}. "
                             "Use junto de --exp-n para rotular a saída.")
    parser.add_argument("--nc-suffix", type=str,
                        default="_predep_granular_seasonal",
                        help="[--source nc] Sufixo do arquivo NC de cada MoV, "
                             "antes de '.nc' (padrão: "
                             "'_predep_granular_seasonal'). Use "
                             "'_predep_granular_seasonal_chirps' para o "
                             "experimento CHIRPS.")
    parser.add_argument("--movs", type=str, default="",
                        help="MoVs separados por vírgula (padrão: todos)")
    parser.add_argument("--lags", type=str, default="0,1,3,6,9,12",
                        help="Lags (vírgula) ou 'all' (padrão: 0,1,3,6,9,12)")
    parser.add_argument("--som-size", type=int, default=0,
                        help="Lado da grade SOM (padrão: heurística)")
    parser.add_argument("--n-regimes", type=int, default=7,
                        help="Nº de regimes (clusters de neurônios; padrão: 7)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Iterações de treino do SOM (padrão: 10000). "
                             "Sem --use-epochs, é um total FIXO de "
                             "atualizações de peso — em datasets grandes "
                             "(ex.: Brasil-todo via --source parquet, "
                             "~70k+ pixels) isso sub-treina o SOM (10000 "
                             "atualizações vê só ~14% dos dados uma vez). "
                             "Use --use-epochs para que este número vire "
                             "nº de ÉPOCAS completas (e reduza-o de acordo, "
                             "ex.: 8).")
    parser.add_argument("--use-epochs", action="store_true",
                        help="Interpreta --iterations como nº de ÉPOCAS "
                             "completas (len(X) atualizações cada) em vez "
                             "de total fixo de atualizações. Recomendado "
                             "para datasets grandes (--source parquet).")
    parser.add_argument("--search-sample-size", type=int, default=None,
                        help="[--optimize] Nº de pixels usados na varredura "
                             "de topologias do Pareto (subamostra; o modelo "
                             "vencedor é sempre retreinado na base "
                             "completa). Sem isso, cada uma das ~30-50 "
                             "topologias candidatas treina na base inteira "
                             "— caro em datasets grandes. Ex.: 15000.")
    parser.add_argument("--search-epochs", type=int, default=None,
                        help="[--optimize] Nº de épocas por candidata na "
                             "varredura de topologias (padrão: mesmo valor "
                             "de --iterations). Ex.: 1.")
    parser.add_argument("--seed", type=int, default=42, help="Semente (padrão: 42)")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Treina o SOM em α na escala nativa (sem z-score)")
    parser.add_argument("--som-diagnostics", action="store_true",
                        help="Inclui os diagnósticos quadrados do SOM "
                             "(U-matrix, hitmap, component planes na grade)")
    parser.add_argument("--variable", choices=["alpha_core", "r2"],
                        default="alpha_core",
                        help="Variável do NC usada como feature (padrão: "
                             "alpha_core). Use 'r2' para treinar o SOM "
                             "equivalente sobre a linha de base linear e "
                             "comparar lado a lado com o de PREDEP.")
    parser.add_argument("--optimize", action="store_true",
                        help="Otimiza a topologia da grade SOM via "
                             "fronteira de Pareto (QE x TE) — mesma "
                             "metodologia usada no exp01 de referência.")
    parser.add_argument("--auto-k", action="store_true",
                        help="Escolhe o nº de regimes via cotovelo do WSS "
                             "(K-Means nos neurônios) em vez do --n-regimes "
                             "fixo — mesma metodologia usada no exp01 de "
                             "referência.")
    parser.add_argument("--mov-alpha-threshold", type=float, default=None,
                        help="Inclui no SOM apenas MoVs cujo score de α "
                             "(PREDEP; média entre pixels do melhor "
                             "lag×estação por pixel) exceda este limiar "
                             "(ex.: 0.15 ou 0.3). Sempre calculado sobre "
                             "alpha_core, mesmo se --variable=r2. "
                             "Ignora --movs se ambos forem passados.")
    parser.add_argument("--source", choices=["nc", "parquet"], default="nc",
                        help="Fonte dos dados. 'nc' (padrão): NCs Brazil-wide "
                             "em results/predep_granular_brazil, mesclados a "
                             "partir do cálculo PREDEP feito bacia a bacia — "
                             "pode deixar quebras visíveis nas fronteiras de "
                             "bacia. 'parquet': experimento Brasil-todo já "
                             "unificado (1 arquivo por MoV×estação, colunas "
                             "basin/latitude/longitude/lag/r2/alpha), via "
                             "--parquet-dir.")
    parser.add_argument("--parquet-dir", type=str, default=None,
                        help="[--source parquet] Caminho do experimento "
                             "parquet Brasil-todo (ex.: ../predep_visualization"
                             "/results/predep_granular_brazil/exp_brasil). "
                             "Obrigatório com --source parquet.")
    parser.add_argument("--standardize-method", choices=["zscore", "quantile"],
                        default="quantile",
                        help="[--source parquet] Normalização das features: "
                             "'quantile' (RankGauss, padrão — evita que a "
                             "cauda longa de α/r² domine o SOM) ou 'zscore'. "
                             "Ignorado com --source nc (sempre zscore).")
    args = parser.parse_args()

    results_base = ROOT / "results" / "predep_granular_brazil"
    if args.source == "parquet":
        if not args.parquet_dir:
            print("[ERRO] --source parquet requer --parquet-dir.")
            sys.exit(1)
        exp_dir = Path(args.parquet_dir)
        if not exp_dir.exists():
            print(f"[ERRO] Diretório parquet não encontrado: {exp_dir}")
            sys.exit(1)
        exp_n = args.exp_n
    else:
        exp_n = args.exp_n if args.exp_n > 0 else latest_exp(results_base)
        exp_dir = Path(args.exp_dir) if args.exp_dir else results_base / f"exp{exp_n:02d}"
        if not exp_dir.exists():
            print(f"[ERRO] Experimento não encontrado: {exp_dir}")
            sys.exit(1)

    mov_filter = [m.strip() for m in args.movs.split(",") if m.strip()] or None
    lags = (None if args.lags.strip().lower() == "all"
            else [int(x.strip()) for x in args.lags.split(",") if x.strip()])

    value_label = VALUE_LABELS[args.variable]

    thr_tag = ""
    if args.mov_alpha_threshold is not None:
        if mov_filter:
            print("[AVISO] --movs ignorado: --mov-alpha-threshold define o filtro.")
        print(f"Selecionando MoVs com α (PREDEP) > {args.mov_alpha_threshold} "
              f"(score = média entre pixels do melhor lag×estação por pixel)...")
        mov_filter, scores = movs_above_threshold(
            exp_dir, args.mov_alpha_threshold, lags=lags, source=args.source
        )
        for name in sorted(scores, key=scores.get, reverse=True):
            flag = "OK " if scores[name] > args.mov_alpha_threshold else "   "
            print(f"  [{flag}] {name:10s} score={scores[name]:.4f}")
        print(f"  → {len(mov_filter)}/{len(scores)} MoVs mantidos: {mov_filter}")
        if not mov_filter:
            print("[ERRO] Nenhum MoV passou do limiar.")
            sys.exit(1)
        thr_tag = f"_gt{args.mov_alpha_threshold:g}".replace(".", "p")

    print(f"Experimento : exp{exp_n:02d}  |  fonte: {args.source}  |  "
          f"variável: {args.variable}")
    print("Construindo matriz de features...")
    if args.source == "parquet":
        parquet_variable = "alpha" if args.variable == "alpha_core" else "r2"
        bundle = build_feature_matrix_from_parquet(
            exp_dir, mov_filter, lags,
            standardize=not args.no_standardize,
            variable=parquet_variable,
            standardize_method=args.standardize_method,
        )
    else:
        bundle = build_feature_matrix(exp_dir, mov_filter, lags,
                                      standardize=not args.no_standardize,
                                      variable=args.variable,
                                      nc_suffix=args.nc_suffix)
    print(f"  Amostras (pixels): {bundle.X.shape[0]}  |  Features: {bundle.X.shape[1]}")
    print(f"  MoVs ({len(bundle.mov_names)}): {bundle.mov_names}")
    print(f"  Lags: {bundle.lags}  |  Estações: {bundle.seasons}")

    print("Treinando SOM..." + (" (otimização Pareto QE×TE)" if args.optimize else "")
          + (" (por épocas)" if args.use_epochs else ""))
    som, metrics = train_som(bundle.X, som_size=args.som_size or None,
                             num_iteration=args.iterations, seed=args.seed,
                             optimize=args.optimize, use_epochs=args.use_epochs,
                             search_sample_size=args.search_sample_size,
                             search_epochs=args.search_epochs)
    for k, v in metrics.items():
        if k != "pareto_results":
            print(f"  {k}: {v}")

    if args.auto_k:
        print("Buscando nº de regimes via cotovelo do WSS (KMeans nos neurônios)...")
    else:
        print(f"Atribuindo {args.n_regimes} regimes (KMeans nos neurônios)...")
    regimes = assign_regimes(som, bundle.X, n_regimes=args.n_regimes,
                             seed=args.seed, auto_k=args.auto_k,
                             max_k=len(bundle.mov_names))
    if args.auto_k:
        print(f"  Nº de regimes escolhido pelo cotovelo: {regimes.n_regimes}")
    regime_grid = regime_to_grid(regimes, bundle)
    fps = regime_fingerprints(regimes, bundle)

    mov0 = bundle.mov_names[0]
    fallback = sorted(results_base.glob(f"exp*/{mov0}"))
    outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)

    som_subdir = ("som" if args.variable == "alpha_core" else f"som_{args.variable}") + thr_tag
    plots_dir = ROOT / "plots" / f"exp{exp_n:02d}" / som_subdir
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
                                                   basin_outlines=outlines,
                                                   value_label=value_label),
        som_plots.plot_atypicality_map(som, regimes, bundle, plots_dir, exp_n,
                                       basin_outlines=outlines),
        som_plots.plot_boundary_map(som, regimes, bundle, plots_dir, exp_n,
                                    basin_outlines=outlines),
        som_plots.plot_regime_fingerprints(fps, bundle, plots_dir, exp_n,
                                           value_label=value_label),
        som_plots.plot_regime_mov_heatmap(fps, bundle, plots_dir, exp_n,
                                          value_label=value_label),
    ]
    if args.optimize and "pareto_results" in metrics:
        pareto_path = plots_dir / f"som_pareto_curve_exp{exp_n:02d}.png"
        plot_pareto_curve(metrics["pareto_results"], save_path=str(pareto_path))
        pngs.append(pareto_path)
    if args.auto_k and regimes.kmeans_results:
        elbow_path = plots_dir / f"som_elbow_curve_exp{exp_n:02d}.png"
        plot_elbow_curve(
            regimes.kmeans_results["k_values"],
            regimes.kmeans_results["wss_values"],
            elbow_k=regimes.n_regimes,
            save_path=str(elbow_path),
        )
        pngs.append(elbow_path)
    if args.auto_k and regimes.hierarchical_model:
        dendro_path = plots_dir / f"som_dendrogram_exp{exp_n:02d}.png"
        plot_dendrogram(regimes.hierarchical_model, save_path=str(dendro_path))
        pngs.append(dendro_path)
    if args.som_diagnostics:
        pngs += [
            som_plots.plot_umatrix(som, plots_dir, exp_n),
            som_plots.plot_hitmap(som, bundle.X, plots_dir, exp_n),
            som_plots.plot_component_planes(som, bundle, plots_dir, exp_n,
                                            value_label=value_label),
        ]
    _make_pdf(pngs, plots_dir / f"som_insights_exp{exp_n:02d}.pdf")
    print("Concluído.")


if __name__ == "__main__":
    main()
