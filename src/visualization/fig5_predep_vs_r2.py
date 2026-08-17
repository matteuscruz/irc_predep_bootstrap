#!/usr/bin/env python3
"""
Gera Figura 5: Comparação PREDEP vs R².
"""
import argparse
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.visualization.paper_utils import (
    load_all_movs, load_basin_outlines, draw_basin_outlines,
    make_alpha_cmap, setup_geo_ax, SEASONS, SEASON_LABELS
)

def make_diff_cmap():
    """Colormap divergente centrado em zero para (alpha_core - r2)."""
    norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    return plt.cm.RdBu_r, norm

def plot_fig5_comparison(mov_name, mov_data, outlines, output_dir, lang="en"):
    """
    Figura 5: PREDEP vs R².
    3 linhas × 4 colunas (estações).
    L1: PREDEP máximo; L2: R² máximo; L3: Diferença.
    """
    print(f"Gerando Figura 5: PREDEP vs R2 para {mov_name}...")
    
    ac = mov_data["alpha_core"]  # (seasons, lags, lat, lon)
    r2 = mov_data["r2"]
    lats, lons = mov_data["lats"], mov_data["lons"]
    
    ac_max = np.nanmax(ac, axis=1)
    r2_max = np.nanmax(r2, axis=1)
    diff = ac_max - r2_max
    
    n_rows = 3
    n_cols = len(SEASONS)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(n_rows, n_cols + 2, width_ratios=[1, 1, 1, 1, 0.05, 0.05],
                  hspace=0.1, wspace=0.05, left=0.05, right=0.92, top=0.9, bottom=0.05)

    cmap_alpha, norm_alpha, _ = make_alpha_cmap()
    cmap_alpha.set_bad("white")
    cmap_diff, norm_diff = make_diff_cmap()
    cmap_diff.set_bad("white")
    
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    
    im_alpha = im_r2 = im_diff = None

    for col, season in enumerate(SEASONS):
        # Row 1: PREDEP
        ax1 = setup_geo_ax(fig, gs[0, col], extent)
        ax1.set_facecolor("white")
        im_alpha = ax1.imshow(np.ma.masked_invalid(ac_max[col]), cmap=cmap_alpha, norm=norm_alpha,
                              origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax1, outlines)
        ax1.set_title(SEASON_LABELS[season], fontweight="bold")
        if col == 0:
            ax1.text(-0.15, 0.5, "Max PREDEP", transform=ax1.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

        # Row 2: R2
        ax2 = setup_geo_ax(fig, gs[1, col], extent)
        ax2.set_facecolor("white")
        im_r2 = ax2.imshow(np.ma.masked_invalid(r2_max[col]), cmap=cmap_alpha, norm=norm_alpha,
                           origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax2, outlines)
        if col == 0:
            ax2.text(-0.15, 0.5, "Max R²", transform=ax2.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

        # Row 3: Diff
        ax3 = setup_geo_ax(fig, gs[2, col], extent)
        ax3.set_facecolor("white")
        im_diff = ax3.imshow(np.ma.masked_invalid(diff[col]), cmap=cmap_diff, norm=norm_diff,
                             origin="lower", extent=extent, interpolation="nearest")
        draw_basin_outlines(ax3, outlines)
        if col == 0:
            ax3.text(-0.15, 0.5, "Difference\n(PREDEP - R²)", transform=ax3.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

    # Colorbars
    cb1 = fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0:2, 4]), ticks=np.arange(0, 1.1, 0.2))
    cb1.set_label("Predictive Skill")
    
    cb2 = fig.colorbar(im_diff, cax=fig.add_subplot(gs[2, 5]))
    cb2.set_label("Difference (PREDEP better $\\rightarrow$)")

    title = f"Non-linear Predictability vs Linear Regression: {mov_name}" if lang == "en" else f"Previsibilidade Não-linear vs Regressão Linear: {mov_name}"
    plt.suptitle(title, fontweight="bold", y=0.95)

    out_file = Path(output_dir) / f"Fig5_Comparison_{mov_name}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="results/predep_granular_brazil/exp_brasil")
    parser.add_argument("--out-dir", type=str, default="plots/paper_figures")
    parser.add_argument("--lang", type=str, choices=["en", "pt"], default="en")
    parser.add_argument("--mov", type=str, default="ONI")
    args = parser.parse_args()

    exp_dir = ROOT / args.exp_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    movs_to_plot = [m.strip() for m in args.mov.split(",") if m.strip()]
    all_data = load_all_movs(exp_dir, mov_list=movs_to_plot)
    if not all_data: return
    
    first_mov = list(all_data.keys())[0]
    outlines = load_basin_outlines(exp_dir, first_mov)
    
    for mov in movs_to_plot:
        if mov in all_data:
            plot_fig5_comparison(mov, all_data[mov], outlines, out_dir, lang=args.lang)

if __name__ == "__main__":
    main()
