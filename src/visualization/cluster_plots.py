"""
Plot functions for Uruguai cluster PREDEP results.
Used locally (after Modal download) by experiments 04 and 05.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path

SEASONS = ["DJF", "MAM", "JJA", "SON"]
N_LAGS = 12
ALPHA_LEVELS = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]


def _make_alpha_cmap():
    """
    Colormap para α ∈ [0, 1]:
      [0.0, 0.1] → cinza (valores não-significativos)
      (0.1, 1.0] → gradiente laranja pálido → laranja intenso
    """
    gray = (0.65, 0.65, 0.65)
    light_orange = (1.0, 0.88, 0.70)
    orange = (1.0, 0.40, 0.0)

    n_gradient = 9  # bins 1..9 → laranja pálido → laranja intenso
    colors = [gray]
    for i in range(n_gradient):
        t = i / (n_gradient - 1)
        colors.append(tuple(light_orange[c] + t * (orange[c] - light_orange[c]) for c in range(3)))

    cmap = ListedColormap(colors, name="alpha_cmap")
    norm = BoundaryNorm(ALPHA_LEVELS, ncolors=cmap.N)
    return cmap, norm


def load_cluster_masks(masks_path="data/processed/uruguai_cluster_masks.npz"):
    npz = np.load(masks_path, allow_pickle=True)
    return {
        "cluster_names": list(npz["cluster_names"]),
        "masks_flat": npz["masks_flat"].astype(bool),
        "lats": npz["lats"],
        "lons": npz["lons"],
        "n_lat": len(npz["lats"]),
        "n_lon": len(npz["lons"]),
    }


def make_alpha_map(alpha_vals, cluster_names, mask_idx, masks_flat, n_lat, n_lon):
    alpha_map = np.full((n_lat, n_lon), np.nan)
    for ci, cname in enumerate(cluster_names):
        if cname not in mask_idx:
            continue
        m2d = masks_flat[mask_idx[cname]].reshape(n_lat, n_lon)
        alpha_map[m2d] = alpha_vals[ci]
    return alpha_map


def plot_4x4_optimal_lags_clusters(
    alpha, label, cluster_names, output_dir, masks_flat, mask_idx, n_lat, n_lon
):
    """
    4 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: α máximo  |  Linhas 2-4: lags 1°, 2°, 3° maiores α.
    alpha: (n_seasons, n_clusters, N_LAGS)
    """
    n_c = len(cluster_names)
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))

    fig = plt.figure(figsize=(5 * n_cols, 22))
    gs = GridSpec(
        4, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.25, wspace=0.15, left=0.05, right=0.95, top=0.94, bottom=0.05,
    )

    cmap_alpha, norm_alpha = _make_alpha_cmap()
    lag_boundaries = np.arange(0.5, N_LAGS + 1.5, 1.0)
    lag_norm = BoundaryNorm(lag_boundaries, ncolors=plt.cm.jet.N, clip=True)

    im_alpha = im_lag = None

    for col, season in enumerate(SEASONS):
        a = alpha[col]  # (n_clusters, N_LAGS)
        order = np.argsort(np.where(np.isnan(a), -999.0, a), axis=1)[:, ::-1]
        alpha_max = a[np.arange(n_c), order[:, 0]].copy()
        lag_1st = (order[:, 0] + 1).astype(float)
        lag_2nd = (order[:, 1] + 1).astype(float)
        lag_3rd = (order[:, 2] + 1).astype(float)

        all_nan = np.all(np.isnan(a), axis=1)
        for arr in (alpha_max, lag_1st, lag_2nd, lag_3rd):
            arr[all_nan] = np.nan

        rows_data = [
            (alpha_max, norm_alpha, f"{season} - α máximo"),
            (lag_1st, lag_norm, f"{season} - Lag 1°"),
            (lag_2nd, lag_norm, f"{season} - Lag 2°"),
            (lag_3rd, lag_norm, f"{season} - Lag 3°"),
        ]

        for row, (vals, norm, _) in enumerate(rows_data):
            ax = fig.add_subplot(gs[row, col])
            amap = make_alpha_map(vals, cluster_names, mask_idx, masks_flat, n_lat, n_lon)
            im = ax.imshow(amap, cmap=cmap_alpha if row == 0 else plt.cm.jet,
                           norm=norm, aspect="auto", origin="lower",
                           interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=12, fontweight="bold")

            for ci, cname in enumerate(cluster_names):
                if cname not in mask_idx or np.isnan(vals[ci]):
                    continue
                m2d = masks_flat[mask_idx[cname]].reshape(n_lat, n_lon)
                rows_m, cols_m = np.where(m2d)
                cy, cx = rows_m.mean(), cols_m.mean()
                txt = f"{vals[ci]:.2f}" if row == 0 else f"{int(vals[ci])}"
                color = "white" if (row == 0 and vals[ci] > 0.7) else "black"
                ax.text(cx, cy, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

            if row == 0 and im_alpha is None:
                im_alpha = im
            if row == 1 and im_lag is None:
                im_lag = im

    fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, n_cols]),
                 ticks=np.arange(0.0, 1.1, 0.1)).set_label("α", fontsize=12, fontweight="bold")
    fig.colorbar(im_lag, cax=fig.add_subplot(gs[1:4, n_cols]),
                 ticks=LAGS).set_label("Lag (meses)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Lags Ótimos por Estação — Clusters da Bacia do Uruguai: {label}\n"
        f"Linha 1: α máximo | Linhas 2–4: Lags do 1°, 2° e 3° maiores α",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"optimal_lags_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_12x4_all_lags_clusters(
    alpha, label, cluster_names, output_dir, masks_flat, mask_idx, n_lat, n_lon
):
    """
    N_LAGS linhas × 4 colunas (DJF, MAM, JJA, SON).
    alpha: (n_seasons, n_clusters, N_LAGS)
    """
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))

    fig = plt.figure(figsize=(3 * n_cols, 30))
    gs = GridSpec(
        N_LAGS, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05, left=0.08, right=0.95, top=0.96, bottom=0.02,
    )

    cmap, norm = _make_alpha_cmap()
    im = None

    for li, lag in enumerate(LAGS):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            vals = alpha[col, :, li]
            amap = make_alpha_map(vals, cluster_names, mask_idx, masks_flat, n_lat, n_lon)
            im = ax.imshow(amap, cmap=cmap, norm=norm, aspect="auto", origin="lower",
                           interpolation="nearest")
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=11, fontweight="bold", pad=3)
            if col == 0:
                ax.text(-0.15, 0.5, f"Lag {lag}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")

    fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]),
                 ticks=ALPHA_LEVELS).set_label(
        "α (Predictive Dependence)\n0=Independente, 1=Perfeito",
        fontsize=10, fontweight="bold",
    )
    plt.suptitle(
        f"PREDEP α — Clusters da Bacia do Uruguai: {label}\n"
        f"Linhas = Lags (1–{N_LAGS} meses) | Colunas = Estações",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"all_lags_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def generate_plots_predep(nc_file, mov_name, plots_dir,
                           masks_path="data/processed/uruguai_cluster_masks.npz"):
    """Gera 2 plots (lags ótimos + todos os lags) para o experimento 04."""
    import netCDF4 as nc4

    md = load_cluster_masks(masks_path)
    mask_cnames = md["cluster_names"]
    masks_flat = md["masks_flat"]
    n_lat, n_lon = md["n_lat"], md["n_lon"]

    with nc4.Dataset(str(nc_file)) as nc:
        cnames = [c.decode() if isinstance(c, bytes) else str(c)
                  for c in nc.variables["cluster"][:]]
        s_nc = [s.decode() if isinstance(s, bytes) else str(s)
                for s in nc.variables["season"][:]]
        alpha = np.ma.filled(nc.variables["alpha"][:], np.nan).astype(float)

    idx = [s_nc.index(s) for s in SEASONS if s in s_nc]
    alpha = alpha[idx]
    mask_idx = {n: mask_cnames.index(n) for n in cnames if n in mask_cnames}

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_4x4_optimal_lags_clusters(alpha, mov_name, cnames, plots_dir,
                                    masks_flat, mask_idx, n_lat, n_lon)
    plot_12x4_all_lags_clusters(alpha, mov_name, cnames, plots_dir,
                                 masks_flat, mask_idx, n_lat, n_lon)


def generate_plots_comparison(nc_file, mov_name, plots_dir,
                               masks_path="data/processed/uruguai_cluster_masks.npz"):
    """Gera 4 plots (2 por métrica: core, r2) para o experimento 05."""
    import netCDF4 as nc4

    md = load_cluster_masks(masks_path)
    mask_cnames = md["cluster_names"]
    masks_flat = md["masks_flat"]
    n_lat, n_lon = md["n_lat"], md["n_lon"]

    with nc4.Dataset(str(nc_file)) as nc:
        cnames = [c.decode() if isinstance(c, bytes) else str(c)
                  for c in nc.variables["cluster"][:]]
        s_nc = [s.decode() if isinstance(s, bytes) else str(s)
                for s in nc.variables["season"][:]]
        ac = np.ma.filled(nc.variables["alpha_core"][:], np.nan).astype(float)
        r2 = np.ma.filled(nc.variables["r2"][:], np.nan).astype(float)

    idx = [s_nc.index(s) for s in SEASONS if s in s_nc]
    ac, r2 = ac[idx], r2[idx]
    mask_idx = {n: mask_cnames.index(n) for n in cnames if n in mask_cnames}

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    for metric_name, alpha in [("core", ac), ("r2", r2)]:
        label = f"{mov_name}_{metric_name}"
        plot_4x4_optimal_lags_clusters(alpha, label, cnames, plots_dir,
                                        masks_flat, mask_idx, n_lat, n_lon)
        plot_12x4_all_lags_clusters(alpha, label, cnames, plots_dir,
                                     masks_flat, mask_idx, n_lat, n_lon)


# ── Granular (exp 06): mapas lat×lon por ponto ────────────────────────────────

def plot_12x4_granular_all_lags(alpha, label, lats, lons, output_dir):
    """
    12 linhas (Lag 1–12) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa lat/lon de α.
    alpha: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    cmap, norm = _make_alpha_cmap()

    fig = plt.figure(figsize=(3 * n_cols, 30))
    gs = GridSpec(
        N_LAGS, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05, left=0.08, right=0.95, top=0.96, bottom=0.02,
    )

    im = None
    for li, lag in enumerate(LAGS):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            im = ax.imshow(
                alpha[col, li], cmap=cmap, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=11, fontweight="bold", pad=3)
            if col == 0:
                ax.text(-0.15, 0.5, f"Lag {lag}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=9, fontweight="bold")

    fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]),
                 ticks=ALPHA_LEVELS).set_label(
        "α (Predictive Dependence)\n0=Independente, 1=Perfeito",
        fontsize=10, fontweight="bold",
    )
    plt.suptitle(
        f"PREDEP α granular — Bacia do Uruguai: {label}\n"
        f"Linhas = Lags (1–{N_LAGS} meses) | Colunas = Estações",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"granular_all_lags_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_granular_optimal(alpha, label, lats, lons, output_dir):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: α máximo por pixel  |  Linha 2: lag do α máximo.
    alpha: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    cmap_alpha, norm_alpha = _make_alpha_cmap()
    lag_boundaries = np.arange(0.5, N_LAGS + 1.5, 1.0)
    lag_norm = BoundaryNorm(lag_boundaries, ncolors=plt.cm.jet.N, clip=True)

    fig = plt.figure(figsize=(5 * n_cols, 11))
    gs = GridSpec(
        2, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.2, wspace=0.15, left=0.06, right=0.95, top=0.92, bottom=0.05,
    )

    im_alpha = im_lag = None

    for col, season in enumerate(SEASONS):
        a = alpha[col]  # (N_LAGS, n_lat, n_lon)
        all_nan = np.all(np.isnan(a), axis=0)

        alpha_max = np.where(all_nan, np.nan, np.nanmax(a, axis=0))
        best_lag_idx = np.nanargmax(
            np.where(np.isnan(a), -np.inf, a), axis=0
        ).astype(float)
        best_lag_idx[all_nan] = np.nan
        best_lag = best_lag_idx + 1.0  # 1-based

        for row, (vals, norm, cmap_use) in enumerate([
            (alpha_max, norm_alpha, cmap_alpha),
            (best_lag, lag_norm, plt.cm.jet),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                vals, cmap=cmap_use, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=12, fontweight="bold")
            if col == 0:
                label_row = "α máximo" if row == 0 else "Lag ótimo"
                ax.text(-0.1, 0.5, label_row, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=10, fontweight="bold")

            if row == 0 and im_alpha is None:
                im_alpha = im
            if row == 1 and im_lag is None:
                im_lag = im

    fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, n_cols]),
                 ticks=np.arange(0.0, 1.1, 0.1)).set_label(
        "α", fontsize=12, fontweight="bold")
    fig.colorbar(im_lag, cax=fig.add_subplot(gs[1, n_cols]),
                 ticks=LAGS).set_label("Lag (meses)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Lags Ótimos (granular) — Bacia do Uruguai: {label}",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"granular_optimal_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def generate_plots_granular(nc_file, mov_name, plots_dir):
    """Gera 4 plots granulares (2 por métrica: core, r2) para o exp 06."""
    import netCDF4 as nc4

    with nc4.Dataset(str(nc_file)) as nc:
        lats = np.array(nc.variables["latitude"][:])
        lons = np.array(nc.variables["longitude"][:])
        s_nc = [s.decode() if isinstance(s, bytes) else str(s)
                for s in nc.variables["season"][:]]
        ac = np.ma.filled(nc.variables["alpha_core"][:], np.nan).astype(float)
        r2 = np.ma.filled(nc.variables["r2"][:], np.nan).astype(float)

    idx = [s_nc.index(s) for s in SEASONS if s in s_nc]
    ac, r2 = ac[idx], r2[idx]

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    for metric_name, alpha in [("core", ac), ("r2", r2)]:
        label = f"{mov_name}_{metric_name}"
        plot_2x4_granular_optimal(alpha, label, lats, lons, plots_dir)
        plot_12x4_granular_all_lags(alpha, label, lats, lons, plots_dir)
