#!/usr/bin/env python3
"""
Gera, para cada trimestre climático (DJF/MAM/JJA/SON), uma única imagem 3x2
comparando CHIRPS vs Xavier: PREDEP (alpha_core) máximo espacial de cada
fonte + diferença absoluta (linha 1), e o MoV responsável de cada fonte +
concordância espacial entre os dois (linha 2).

Substitui os 8 plots de tools/gen_predep_max_plots.py (4 por fonte) por 4
plots comparativos — um por trimestre.

Os dois datasets precisam ter o mesmo grid e a mesma lista de MoVs (ex.:
saída de experiments/08_predep_parana_recompute.py, que usa a mesma máscara
de pixels nos dois lados).

Uso:
  python tools/gen_predep_max_comparison_plots.py \\
      --chirps-dir results/predep_granular_brazil/exp08_parana_recompute/chirps \\
      --chirps-suffix _predep_granular_seasonal_chirps.nc \\
      --xavier-dir results/predep_granular_brazil/exp08_parana_recompute/xavier \\
      --xavier-suffix _predep_granular_seasonal_xavier.nc \\
      --out-dir plots/weekly_20260831_chirps_vs_xavier/predep_max
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization.cluster_plots import (  # noqa: E402
    SEASONS, _load_mov_stack, plot_predep_max_comparison_season,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chirps-dir", required=True)
    parser.add_argument("--chirps-suffix", default="_predep_granular_seasonal_chirps.nc")
    parser.add_argument("--xavier-dir", required=True)
    parser.add_argument("--xavier-suffix", default="_predep_granular_seasonal_xavier.nc")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--movs", default="",
                        help="MoVs separados por vírgula (padrão: todos)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Margem de ruído (ex.: 0.05) — diferenças/gaps "
                             "de α menores que isso viram 'não significativo' "
                             "/ 'ambíguo' em vez de coloridos normalmente. "
                             "Padrão: sem margem (comportamento original).")
    args = parser.parse_args()

    mov_filter = [m.strip() for m in args.movs.split(",") if m.strip()] or None

    ac_c, r2_c, movs_c, lats_c, lons_c, _ = _load_mov_stack(
        args.chirps_dir, mov_filter=mov_filter, nc_suffix=args.chirps_suffix,
    )
    ac_x, r2_x, movs_x, lats_x, lons_x, _ = _load_mov_stack(
        args.xavier_dir, mov_filter=mov_filter, nc_suffix=args.xavier_suffix,
    )
    if not movs_c or not movs_x:
        print("[ERRO] Nenhum NC encontrado em um dos diretórios.")
        sys.exit(1)
    if movs_c != movs_x:
        print(f"[ERRO] Lista de MoVs diferente entre CHIRPS e Xavier:\n"
              f"  CHIRPS: {movs_c}\n  Xavier: {movs_x}")
        sys.exit(1)
    if ac_c.shape[-2:] != ac_x.shape[-2:] or lats_c.shape != lats_x.shape:
        print(f"[ERRO] Grades diferentes entre CHIRPS {ac_c.shape} e Xavier {ac_x.shape} "
              "— os dois datasets precisam ter o mesmo footprint de pixels.")
        sys.exit(1)

    print(f"MoVs ({len(movs_c)}): {movs_c}")

    out_dir = Path(args.out_dir)
    for season_idx, season in enumerate(SEASONS):
        plot_predep_max_comparison_season(
            ac_c, ac_x, movs_c, lats_c, lons_c, season_idx, season, out_dir,
            margin=args.margin,
        )


if __name__ == "__main__":
    main()
