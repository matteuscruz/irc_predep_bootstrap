#!/usr/bin/env python3
"""
Gera Figura 3: Heatmap Sazonal (MoV x Estação).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.visualization.paper_utils import load_all_movs, SEASONS

def plot_fig3_heatmap(data_dict, output_dir, lang="en"):
    """
    Figura 3: Heatmap de α (P90) por MoV e Estação.
    Mostra a força de cada sinal climático ao longo das estações.
    """
    print("Gerando Figura 3: Heatmap...")
    mov_names = list(data_dict.keys())
    n_seasons = len(SEASONS)
    
    # We will use P90 to represent the "regional strength" of the signal
    heatmap_data = np.zeros((len(mov_names), n_seasons))
    
    for i, mov in enumerate(mov_names):
        ac = data_dict[mov]["alpha_core"]  # (seasons, lags, lat, lon)
        for s in range(n_seasons):
            season_data = ac[s]
            valid = season_data[~np.isnan(season_data)]
            if len(valid) > 0:
                heatmap_data[i, s] = np.percentile(valid, 95)  # Use P95 for a stronger signal indicator
            else:
                heatmap_data[i, s] = np.nan

    # Sort MoVs by overall mean strength for better visualization
    mean_strength = np.nanmean(heatmap_data, axis=1)
    sort_idx = np.argsort(mean_strength)[::-1]
    sorted_movs = [mov_names[i] for i in sort_idx]
    sorted_data = heatmap_data[sort_idx]

    plt.figure(figsize=(10, 12))
    
    ax = sns.heatmap(sorted_data, cmap="YlOrRd", annot=True, fmt=".2f",
                     xticklabels=SEASONS, yticklabels=sorted_movs,
                     cbar_kws={'label': 'PREDEP P95 ($\\alpha$)'},
                     vmin=0, vmax=0.3)
                     
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    title = "Climate Signal Strength (95th Percentile PREDEP) by Season" if lang == "en" else "Força do Sinal Climático (Percentil 95 do PREDEP) por Estação"
    plt.title(title, pad=15, fontweight="bold")
    
    if lang == "en":
        plt.xlabel("Season", fontweight="bold")
        plt.ylabel("Mode of Variability", fontweight="bold")
    else:
        plt.xlabel("Estação", fontweight="bold")
        plt.ylabel("Modo de Variabilidade", fontweight="bold")
        
    plt.tight_layout()
    
    out_file = Path(output_dir) / "Fig3_Heatmap.pdf"
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
    
    plot_fig3_heatmap(all_data, out_dir, lang=args.lang)

if __name__ == "__main__":
    main()
