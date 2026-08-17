#!/usr/bin/env python3
"""
Gera figuras publication-quality para a seção de resultados do paper PREDEP.
Usa dados do experimento `exp_brasil` (22 MoVs, Brasil inteiro).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np

# Adjust imports for local package structure
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.visualization.paper_utils import (
    load_all_movs, load_basin_outlines, draw_basin_outlines,
    make_alpha_cmap, make_lag_cmap, make_mov_cmap, setup_geo_ax,
    SEASONS, SEASON_LABELS, BRAZIL_EXTENT
)

def plot_fig1_overview(data_dict, outlines, output_dir, lang="en"):
    """
    Figura 1: Overview sazonal do PREDEP.
    4 colunas (Estações).
    Linha 1: PREDEP Max).
    Linha 2: Responsible MoV.
    """
    print("Gerando Figura 1: Overview...")
    mov_names = list(data_dict.keys())
    
    # Check if we have data
    if not mov_names:
        print("  Sem dados para Fig 1.")
        return
        
    ds0 = data_dict[mov_names[0]]
    n_seasons = len(SEASONS)
    lats, lons = ds0["lats"], ds0["lons"]
    
    # Calculate overall max and argmax
    max_alpha = np.full((n_seasons, len(lats), len(lons)), np.nan)
    best_mov_idx = np.full((n_seasons, len(lats), len(lons)), np.nan)
    
    for i, mov in enumerate(mov_names):
        ac = data_dict[mov]["alpha_core"]  # (seasons, lags, lat, lon)
        # Max over lags for this MoV
        ac_max_lag = np.nanmax(ac, axis=1)  # (seasons, lat, lon)
        
        # Update overall max
        for s in range(n_seasons):
            valid_new = ~np.isnan(ac_max_lag[s])
            mask = valid_new & (np.isnan(max_alpha[s]) | (ac_max_lag[s] > max_alpha[s]))
            max_alpha[s][mask] = ac_max_lag[s][mask]
            best_mov_idx[s][mask] = i

    # Low signal threshold
    low_signal = ~np.isnan(max_alpha) & (max_alpha < 0.1)

    # Plotting
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, n_seasons, width_ratios=[1, 1, 1, 1], 
                  hspace=0.1, wspace=0.05, left=0.05, right=0.95, top=0.92, bottom=0.30)

    cmap_alpha, norm_alpha, _ = make_alpha_cmap()
    cmap_alpha.set_bad("white")
    
    cmap_mov, norm_mov, mov_colors = make_mov_cmap(mov_names)
    cmap_mov_plot = copy_cmap_set_bad(cmap_mov, (0,0,0,0))
    
    im_alpha = im_mov = None
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

    for s, season in enumerate(SEASONS):
        # Row 1: Max Alpha
        ax1 = setup_geo_ax(fig, gs[0, s], extent)
        ax1.set_facecolor("white")
        im = ax1.imshow(np.ma.masked_invalid(max_alpha[s]), cmap=cmap_alpha, norm=norm_alpha,
                        origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax1, outlines)
        ax1.set_title(SEASON_LABELS[season], fontweight="bold")
        if s == 0:
            ylabel = "Max PREDEP" if lang == "en" else "PREDEP Max"
            ax1.text(-0.15, 0.5, ylabel, transform=ax1.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
        if im_alpha is None: im_alpha = im

        # Row 2: Best MoV
        ax2 = setup_geo_ax(fig, gs[1, s], extent)
        ax2.set_facecolor("white")
        
        # Gray background for low signal
        gray_layer = np.where(low_signal[s], 0.75, np.nan)
        ax2.imshow(np.ma.masked_invalid(gray_layer), cmap="gray", vmin=0, vmax=1,
                   origin="lower", extent=extent, interpolation="nearest")
        
        best_mov_plot = np.where(low_signal[s], np.nan, best_mov_idx[s])
        im = ax2.imshow(np.ma.masked_invalid(best_mov_plot), cmap=cmap_mov_plot, norm=norm_mov,
                        origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax2, outlines)
        
        if s == 0:
            ylabel = "Best MoV" if lang == "en" else "Melhor MoV"
            ax2.text(-0.15, 0.5, ylabel, transform=ax2.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
        if im_mov is None: im_mov = im

    # Colorbars and Legends
    # Add horizontal colorbar for Alpha
    cb1_ax = fig.add_axes([0.15, 0.18, 0.7, 0.02])
    cb1 = fig.colorbar(im_alpha, cax=cb1_ax, orientation='horizontal', ticks=np.arange(0, 1.1, 0.2))
    cb1.set_label("PREDEP", fontweight="bold")
    
    # Legend for MoVs at the bottom
    handles = [Patch(color=mov_colors[i], label=mov_names[i]) for i in range(len(mov_names))]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(mov_names), 11), 
               bbox_to_anchor=(0.5, 0.04), frameon=False)
    
    out_file = Path(output_dir) / "Fig1_PREDEP_Overview.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_file.name}")


def copy_cmap_set_bad(cmap, color):
    import copy
    cmap_plot = copy.copy(cmap)
    cmap_plot.set_bad(color=color)
    return cmap_plot


def main():
    parser = argparse.ArgumentParser(description="Gera figuras para o paper PREDEP")
    parser.add_argument("--exp-dir", type=str, default="results/predep_granular_brazil/exp_brasil",
                        help="Diretório do experimento")
    parser.add_argument("--out-dir", type=str, default="plots/paper_figures",
                        help="Diretório de saída")
    parser.add_argument("--lang", type=str, choices=["en", "pt"], default="en",
                        help="Idioma das figuras (en/pt)")
    parser.add_argument("--movs", type=str, default="ONI,TNA,PSA2,NIN03,AAO,AMO",
                        help="MoVs principais para análise detalhada (separados por vírgula)")
    args = parser.parse_args()

    exp_dir = ROOT / args.exp_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    main_movs = [m.strip() for m in args.movs.split(",") if m.strip()]

    print(f"Lendo dados de {exp_dir}...")
    # Load ALL MoVs for Fig 1 overview
    all_data = load_all_movs(exp_dir)
    
    if not all_data:
        print("Erro: Nenhum dado encontrado.")
        return

    # Load basin outlines from the first available MoV
    first_mov = list(all_data.keys())[0]
    outlines = load_basin_outlines(exp_dir, first_mov)
    
    plot_fig1_overview(all_data, outlines, out_dir, lang=args.lang)

if __name__ == "__main__":
    main()
