"""
Render a single CHIRPS-vs-Xavier PREDEP comparison page (one MoV, one lag,
all 4 climatic seasons) and save it as a one-page PDF.

Meant to be called as its own subprocess, one call per (MoV, lag) — running
the full 12 lags x 23 MoVs loop in a single long-lived Python process was
observed to accumulate >5GB RSS and get OOM-killed (see
notebooks/exploratory/chirps_vs_xavier_predep_sanity_check.ipynb). Spawning
a fresh process per page bounds peak memory to ~800MB and lets the OS fully
reclaim it between pages. See gen_chirps_vs_xavier_pages.sh for the
orchestrator that calls this for every (MoV, lag) and merges the pages with
qpdf.

Usage: python gen_chirps_vs_xavier_page.py <MOV> <LAG> <OUT_PDF>
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOV = sys.argv[1]
LAG = int(sys.argv[2])
OUT_PDF = sys.argv[3]

ROOT = Path(__file__).resolve().parent.parent
CHIRPS_TEMPLATE = str(ROOT / "results/predep_granular_brazil/exp06_chirps_check/{mov}/{mov}_predep_granular_seasonal_chirps.nc")
XAVIER_TEMPLATE = str(ROOT / "results/predep_granular_brazil/exp01/{mov}/{mov}_parana_predep_granular_seasonal.nc")
RASTER_DPI = 150

c = xr.open_dataset(CHIRPS_TEMPLATE.format(mov=MOV))
x = xr.open_dataset(XAVIER_TEMPLATE.format(mov=MOV)).sel(latitude=c.latitude, longitude=c.longitude)
ref = c
season_order = list(ref.season.values)

overlap_mask = (
    ~np.isnan(c["alpha_core"].isel(season=0, lag=0))
    & ~np.isnan(x["alpha_core"].isel(season=0, lag=0))
).values

fig, axes = plt.subplots(len(season_order), 3, figsize=(16, 4 * len(season_order)))
for row, season in enumerate(season_order):
    x_map = np.where(overlap_mask, x["alpha_core"].sel(season=season, lag=LAG).values, np.nan)
    c_map = np.where(overlap_mask, c["alpha_core"].sel(season=season, lag=LAG).values, np.nan)
    diff = c_map - x_map

    valid = ~np.isnan(c_map) & ~np.isnan(x_map)
    c_v, x_v = c_map[valid], x_map[valid]
    if len(c_v) >= 3:
        r_panel = stats.pearsonr(c_v, x_v)[0]
        mae_panel = np.mean(np.abs(c_v - x_v))
    else:
        r_panel, mae_panel = np.nan, np.nan

    for col, (data, panel_title, cmap, vmin, vmax) in enumerate([
        (x_map, "Xavier", "viridis", 0, 1),
        (c_map, "CHIRPS", "viridis", 0, 1),
        (diff, "Diferença (CHIRPS - Xavier)", "RdBu_r", -0.5, 0.5),
    ]):
        ax = axes[row, col]
        im = ax.pcolormesh(ref.longitude, ref.latitude, data, cmap=cmap,
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(f"{panel_title}\n{MOV} {season} lag={LAG}")
        plt.colorbar(im, ax=ax, shrink=0.8)
        if col == 2:
            ax.text(0.02, 0.02, f"r={r_panel:.3f}  MAE={mae_panel:.3f}  n={valid.sum()}",
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))
plt.tight_layout()
fig.savefig(OUT_PDF, dpi=RASTER_DPI)
