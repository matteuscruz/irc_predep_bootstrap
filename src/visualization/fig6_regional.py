#!/usr/bin/env python3
"""
Gera Figura 6: Perfis regionais (skills por bacia e lag).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.visualization.paper_utils import load_all_movs, load_basin_outlines, BASIN_LABELS

def plot_fig6_regional_profiles(data_dict, outlines, output_dir, lang="en", top_k=3):
    """
    Figura 6: Perfis de decaimento por lag para cada bacia.
    Painel grid, cada célula é uma bacia, mostrando o decaimento dos K melhores MoVs.
    """
    print("Gerando Figura 6: Perfis Regionais...")
    mov_names = list(data_dict.keys())
    lags = data_dict[mov_names[0]]["lags"]
    
    # Precompute basin masks
    # outlines is [(number, basin_name, blats, blons, mask), ...]
    basin_masks = {name: mask for num, name, blats, blons, mask in outlines if mask.any()}
    
    if not basin_masks:
        print("  Erro: sem bacias válidas.")
        return

    # Aggregate data: mean alpha per MoV, per basin, per lag (averaged over seasons)
    # Shape of ac: (seasons, lags, lat, lon)
    full_lats = data_dict[mov_names[0]]["lats"]
    full_lons = data_dict[mov_names[0]]["lons"]

    records = []
    for num, basin_name, blats, blons, mask in outlines:
        if not mask.any(): continue
        
        # Find indices of basin bounding box in the full grid
        lat_idx = [np.where(full_lats == lat)[0][0] for lat in blats if lat in full_lats]
        lon_idx = [np.where(full_lons == lon)[0][0] for lon in blons if lon in full_lons]
        
        if not lat_idx or not lon_idx:
            continue
            
        lat_min, lat_max = min(lat_idx), max(lat_idx)
        lon_min, lon_max = min(lon_idx), max(lon_idx)
        
        # Ensure mask shape matches the extracted slice (in case of floating point precision issues)
        slice_shape = (lat_max - lat_min + 1, lon_max - lon_min + 1)
        if mask.shape != slice_shape:
            # Simple fallback if shapes don't exactly match due to float comparison
            # We just use the raw basin NC file to get the mean
            pass
            
        for mov in mov_names:
            ac = data_dict[mov]["alpha_core"]
            # average across seasons -> (lags, lat, lon)
            ac_mean_seasons = np.nanmean(ac, axis=0)
            
            # Extract the bounding box from the full grid
            ac_subset = ac_mean_seasons[:, lat_min:lat_max+1, lon_min:lon_max+1]
            
            if ac_subset.shape[1:] != mask.shape:
                continue # Skip if shapes don't align
                
            for li, lag in enumerate(lags):
                lag_data = ac_subset[li]
                basin_pixels = lag_data[mask]
                valid = basin_pixels[~np.isnan(basin_pixels)]
                mean_alpha = np.mean(valid) if len(valid) > 0 else np.nan
                records.append({
                    "Basin": basin_name,
                    "MoV": mov,
                    "Lag": lag,
                    "Alpha": mean_alpha
                })
                
    df = pd.DataFrame(records)
    
    # Determine top K MoVs per basin based on overall mean across all lags
    top_movs_per_basin = {}
    basin_names = df["Basin"].unique()
    for basin in basin_names:
        bdf = df[df["Basin"] == basin]
        mean_per_mov = bdf.groupby("MoV")["Alpha"].mean()
        top_movs = mean_per_mov.nlargest(top_k).index.tolist()
        top_movs_per_basin[basin] = top_movs

    # Set up grid of plots (e.g., 4x3 for 12 basins)
    n_basins = len(basin_names)
    n_cols = 3
    n_rows = int(np.ceil(n_basins / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    basin_names = sorted(list(basin_names), key=lambda x: BASIN_LABELS.get(x, 99))
    
    for i, basin in enumerate(basin_names):
        ax = axes[i]
        bdf = df[df["Basin"] == basin]
        top_movs = top_movs_per_basin[basin]
        
        for mov in top_movs:
            mdf = bdf[bdf["MoV"] == mov].sort_values("Lag")
            ax.plot(mdf["Lag"], mdf["Alpha"], marker='o', linewidth=2, label=mov)
            
        ax.set_title(basin.replace("_", " ").title(), fontweight="bold")
        ax.set_xticks(lags)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        if i % n_cols == 0:
            ax.set_ylabel("Mean PREDEP ($\\alpha$)" if lang == "en" else "PREDEP ($\\alpha$) Médio")
        if i >= n_basins - n_cols:
            ax.set_xlabel("Lag (months)" if lang == "en" else "Lag (meses)")
            
    # Hide empty subplots
    for i in range(n_basins, len(axes)):
        axes[i].set_visible(False)
        
    title = "Regional Predictability Profiles by River Basin" if lang == "en" else "Perfis Regionais de Previsibilidade por Bacia Hidrográfica"
    plt.suptitle(title, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_file = Path(output_dir) / "Fig6_Regional_Profiles.pdf"
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
    
    plot_fig6_regional_profiles(all_data, outlines, out_dir, lang=args.lang)

if __name__ == "__main__":
    main()
