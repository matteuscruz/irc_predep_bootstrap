#!/usr/bin/env python3
"""
Gera Figura 2: Lag Ótimo por Estação.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.visualization.paper_utils import (
    load_all_movs, load_basin_outlines, draw_basin_outlines,
    make_alpha_cmap, make_lag_cmap, setup_geo_ax,
    SEASONS, SEASON_LABELS
)

def plot_fig2_optimal_lag(data_dict, outlines, output_dir, lang="en"):
    """
    Figura 2: Lag Ótimo Sazonal (Envelope Multi-MoV).
    4 colunas (Estações).
    Linha 1: Máximo PREDEP α (sobre todos MoVs e lags).
    Linha 2: Lag do máximo PREDEP.
    """
    print("Gerando Figura 2: Optimal Lag...")
    mov_names = list(data_dict.keys())
    ds0 = data_dict[mov_names[0]]
    n_seasons = len(SEASONS)
    lats, lons = ds0["lats"], ds0["lons"]
    lags = ds0["lags"]
    
    max_alpha = np.full((n_seasons, len(lats), len(lons)), np.nan)
    best_lag_idx = np.full((n_seasons, len(lats), len(lons)), np.nan)
    
    for i, mov in enumerate(mov_names):
        ac = data_dict[mov]["alpha_core"]  # (seasons, lags, lat, lon)
        # Max over lags for this MoV
        ac_max_lag_vals = np.nanmax(ac, axis=1)  # (seasons, lat, lon)
        ac_max_lag_idx = np.nanargmax(np.where(np.isnan(ac), -np.inf, ac), axis=1).astype(float)
        
        for s in range(n_seasons):
            mask = np.isnan(max_alpha[s]) | (~np.isnan(ac_max_lag_vals[s]) & (ac_max_lag_vals[s] > max_alpha[s]))
            max_alpha[s][mask] = ac_max_lag_vals[s][mask]
            # store actual lag value, not just index
            best_lag_idx[s][mask] = np.array(lags)[ac_max_lag_idx[s][mask].astype(int)]

    low_signal = ~np.isnan(max_alpha) & (max_alpha < 0.1)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, n_seasons + 1, width_ratios=[1, 1, 1, 1, 0.05], 
                  hspace=0.1, wspace=0.05, left=0.05, right=0.92, top=0.9, bottom=0.15)

    cmap_alpha, norm_alpha, _ = make_alpha_cmap()
    cmap_alpha.set_bad("white")
    
    cmap_lag, norm_lag = make_lag_cmap(lags)
    import copy
    cmap_lag_plot = copy.copy(cmap_lag)
    cmap_lag_plot.set_bad(color=(0,0,0,0))
    
    im_alpha = im_lag = None
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
            ylabel = "Max PREDEP ($\\alpha$)" if lang == "en" else "PREDEP ($\\alpha$) Máximo"
            ax1.text(-0.15, 0.5, ylabel, transform=ax1.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
        if im_alpha is None: im_alpha = im

        # Row 2: Best Lag
        ax2 = setup_geo_ax(fig, gs[1, s], extent)
        ax2.set_facecolor("white")
        
        gray_layer = np.where(low_signal[s], 0.75, np.nan)
        ax2.imshow(np.ma.masked_invalid(gray_layer), cmap="gray", vmin=0, vmax=1,
                   origin="lower", extent=extent, interpolation="nearest")
        
        best_lag_plot = np.where(low_signal[s], np.nan, best_lag_idx[s])
        im = ax2.imshow(np.ma.masked_invalid(best_lag_plot), cmap=cmap_lag_plot, norm=norm_lag,
                        origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax2, outlines)
        
        if s == 0:
            ylabel = "Optimal Lag (months)" if lang == "en" else "Lag Ótimo (meses)"
            ax2.text(-0.15, 0.5, ylabel, transform=ax2.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
        if im_lag is None: im_lag = im

    cb1 = fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, 4]), ticks=np.arange(0, 1.1, 0.2))
    cb1.set_label("PREDEP ($\\alpha$)")
    
    cb2 = fig.colorbar(im_lag, cax=fig.add_subplot(gs[1, 4]), ticks=lags)
    cb2.set_label("Lag (months)" if lang == "en" else "Lag (meses)")

    title = "Optimal Lead Time (Lag) for Maximum Predictability" if lang == "en" else "Tempo de Antecedência Ótimo (Lag) para Previsibilidade Máxima"
    plt.suptitle(title, fontweight="bold", y=0.98)
    
    out_file = Path(output_dir) / "Fig2_Optimal_Lag.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="results/predep_granular_brazil/exp_brasil")
    parser.add_argument("--out-dir", type=str, default="plots/paper_figures")
    parser.add_argument("--lang", type=str, choices=["en", "pt"], default="en")
    args = parser.parse_args()

    exp_dir = ROOT / args.exp_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = load_all_movs(exp_dir)
    if not all_data: return
    
    first_mov = list(all_data.keys())[0]
    outlines = load_basin_outlines(exp_dir, first_mov)
    
    plot_fig2_optimal_lag(all_data, outlines, out_dir, lang=args.lang)

if __name__ == "__main__":
    main()
