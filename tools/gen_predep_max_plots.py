#!/usr/bin/env python3
"""
Gera, para um dataset (Xavier ou CHIRPS), um PNG por trimestre climático
(DJF/MAM/JJA/SON) com o PREDEP (alpha_core) máximo espacial — máximo sobre
todos os MoVs e lags — e o MoV responsável por esse máximo em cada pixel.

Usado para a comparação CHIRPS vs Xavier da Weekly de 31/08/2026: rodar uma
vez por dataset para gerar os 8 plots (4 estações x 2 datasets).

Uso:
  python tools/gen_predep_max_plots.py \\
      --exp-dir results/predep_granular_brazil/exp01 \\
      --nc-suffix _predep_granular_seasonal.nc \\
      --dataset-label Xavier \\
      --out-dir plots/weekly_20260831_chirps_vs_xavier/predep_max

  python tools/gen_predep_max_plots.py \\
      --exp-dir results/predep_granular_brazil/exp06_chirps_check \\
      --nc-suffix _predep_granular_seasonal_chirps.nc \\
      --dataset-label CHIRPS \\
      --out-dir plots/weekly_20260831_chirps_vs_xavier/predep_max
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization.cluster_plots import (  # noqa: E402
    SEASONS, _load_mov_stack, plot_predep_max_season,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", required=True,
                        help="Diretório do experimento (contém um subdiretório por MoV)")
    parser.add_argument("--nc-suffix", default="_predep_granular_seasonal.nc",
                        help="Sufixo do arquivo NC de cada MoV (padrão: "
                             "_predep_granular_seasonal.nc)")
    parser.add_argument("--dataset-label", required=True,
                        help="Rótulo do dataset usado no título/nome dos arquivos (ex.: Xavier, CHIRPS)")
    parser.add_argument("--out-dir", required=True, help="Diretório de saída dos PNGs")
    parser.add_argument("--movs", default="",
                        help="MoVs separados por vírgula (padrão: todos)")
    args = parser.parse_args()

    mov_filter = [m.strip() for m in args.movs.split(",") if m.strip()] or None

    ac_stack, r2_stack, mov_names, lats, lons, nc_lags = _load_mov_stack(
        args.exp_dir, mov_filter=mov_filter, nc_suffix=args.nc_suffix,
    )
    if not mov_names:
        print(f"[ERRO] Nenhum NC ('{{MoV}}{args.nc_suffix}') encontrado em {args.exp_dir}")
        sys.exit(1)

    print(f"Dataset: {args.dataset_label}  |  MoVs ({len(mov_names)}): {mov_names}")

    out_dir = Path(args.out_dir)
    for season_idx, season in enumerate(SEASONS):
        plot_predep_max_season(
            ac_stack, mov_names, lats, lons, season_idx, season,
            args.dataset_label, out_dir,
        )


if __name__ == "__main__":
    main()
