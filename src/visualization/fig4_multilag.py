#!/usr/bin/env python3
"""
Gera Figura 4: Painéis Multi-Lag para MoVs selecionados.
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
    make_alpha_cmap, setup_geo_ax, SEASONS, SEASON_LABELS
)

def plot_fig4_multilag(mov_name, mov_data, outlines, output_dir, lang="en", subset_lags=None):
    """
    Figura 4: Decaimento de previsibilidade por lag.
    N linhas (lags) × 4 colunas (estações).
    """
    print(f"Gerando Figura 4: Multi-lag para {mov_name}...")
    
    ac = mov_data["alpha_core"]  # (seasons, lags, lat, lon)
    lats, lons = mov_data["lats"], mov_data["lons"]
    all_lags = mov_data["lags"]
    
    if subset_lags is not None:
        valid_idx = [all_lags.index(l) for l in subset_lags if l in all_lags]
        lags_to_plot = [all_lags[i] for i in valid_idx]
        ac = ac[:, valid_idx]
    else:
        lags_to_plot = all_lags

    n_rows = len(lags_to_plot)
    n_cols = len(SEASONS)

    fig = plt.figure(figsize=(3.5 * n_cols, 2.8 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                  hspace=0.05, wspace=0.05, left=0.08, right=0.92, top=0.92, bottom=0.05)

    cmap_alpha, norm_alpha, _ = make_alpha_cmap()
    cmap_alpha.set_bad("white")
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

    im = None
    for li, lag in enumerate(lags_to_plot):
        for col, season in enumerate(SEASONS):
            ax = setup_geo_ax(fig, gs[li, col], extent)
            ax.set_facecolor("white")
            
            data = ac[col, li]
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap_alpha, norm=norm_alpha,
                           origin="lower", extent=extent, interpolation="nearest")
            draw_basin_outlines(ax, outlines)
            
            if li == 0:
                ax.set_title(SEASON_LABELS[season], fontweight="bold", pad=8)
            if col == 0:
                ax.text(-0.15, 0.5, f"Lag {lag}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontweight="bold")

    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]), ticks=np.arange(0, 1.1, 0.2))
    cb.set_label("PREDEP ($\\alpha$)")

    title = f"Predictability Decay: {mov_name}" if lang == "en" else f"Decaimento da Previsibilidade: {mov_name}"
    plt.suptitle(title, fontweight="bold", y=0.98)

    suffix = "_subset" if subset_lags else ""
    out_file = Path(output_dir) / f"Fig4_MultiLag_{mov_name}{suffix}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, default="results/predep_granular_brazil/exp_brasil")
    parser.add_argument("--out-dir", type=str, default="plots/paper_figures")
    parser.add_argument("--lang", type=str, choices=["en", "pt"], default="en")
    parser.add_argument("--movs", type=str, default="ONI,TNA,PSA2")
    parser.add_argument("--lags", type=str, default="1,3,6,9", help="Lags to plot (comma separated)")
    args = parser.parse_args()

    exp_dir = ROOT / args.exp_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    movs_to_plot = [m.strip() for m in args.movs.split(",") if m.strip()]
    lags_to_plot = [int(l.strip()) for l in args.lags.split(",") if l.strip()]
    
    all_data = load_all_movs(exp_dir, mov_list=movs_to_plot)
    if not all_data: return
    
    first_mov = list(all_data.keys())[0]
    outlines = load_basin_outlines(exp_dir, first_mov)
    
    for mov in movs_to_plot:
        if mov in all_data:
            plot_fig4_multilag(mov, all_data[mov], outlines, out_dir, lang=args.lang, subset_lags=lags_to_plot)

if __name__ == "__main__":
    main()
