"""
Plot functions for Uruguai cluster PREDEP results.
Used locally (after Modal download) by experiments 04 and 05.
"""

import copy
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


def _make_diff_cmap():
    """
    Colormap divergente centrado em zero para (alpha_core - r2):
      negativo → azul (R² melhor)
      zero     → branco
      positivo → laranja/vermelho (PREDEP melhor)
    """
    norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    return plt.cm.RdBu_r, norm


def _make_winner_cmap():
    """Colormap discreto: cinza=NaN/empate, azul=R² vence, laranja=PREDEP vence."""
    colors = [(0.65, 0.65, 0.65), (0.20, 0.45, 0.72), (1.0, 0.40, 0.0)]
    cmap = ListedColormap(colors, name="winner_cmap")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)
    return cmap, norm


def _winner_vals(ac_vals, r2_vals):
    """
    Retorna array de códigos: 0=NaN/empate, 1=R² melhor, 2=PREDEP melhor.
    ac_vals, r2_vals: arrays 1D por cluster.
    """
    out = np.zeros(len(ac_vals))
    for i in range(len(ac_vals)):
        a, r = ac_vals[i], r2_vals[i]
        if np.isnan(a) and np.isnan(r):
            out[i] = np.nan
        elif np.isnan(a):
            out[i] = 1
        elif np.isnan(r):
            out[i] = 2
        elif r > a:
            out[i] = 1
        else:
            out[i] = 2
    return out


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
    norm = BoundaryNorm(ALPHA_LEVELS, ncolors=cmap.N, clip=True)
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
                 ticks=np.arange(0.0, 1.1, 0.1)).set_label("PREDEP", fontsize=12, fontweight="bold")
    fig.colorbar(im_lag, cax=fig.add_subplot(gs[1:4, n_cols]),
                 ticks=LAGS).set_label("Lag (meses)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Lags Ótimos por Estação — Clusters da Bacia do Uruguai: {label}\n"
        f"Linha 1: PREDEP máximo | Linhas 2–4: Lags do 1°, 2° e 3° maiores PREDEP",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"optimal_lags_{label}.png"
    plt.savefig(out, dpi=150)
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
        "PREDEP\n0=Independente, 1=Perfeito",
        fontsize=10, fontweight="bold",
    )
    plt.suptitle(
        f"PREDEP — Clusters da Bacia do Uruguai: {label}\n"
        f"Linhas = Lags (1–{N_LAGS} meses) | Colunas = Estações",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"all_lags_{label}.png"
    plt.savefig(out, dpi=150)
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
        plot_4x4_optimal_lags_clusters(
            alpha, label, cnames, plots_dir,
            masks_flat, mask_idx, n_lat, n_lon,
        )
        plot_12x4_all_lags_clusters(
            alpha, label, cnames, plots_dir,
            masks_flat, mask_idx, n_lat, n_lon,
        )
    plot_12x4_diff_clusters(
        ac, r2, mov_name, cnames, plots_dir,
        masks_flat, mask_idx, n_lat, n_lon,
    )
    plot_2x4_winner_clusters(
        ac, r2, mov_name, cnames, plots_dir,
        masks_flat, mask_idx, n_lat, n_lon,
    )


# ── Granular (exp 06): mapas lat×lon por ponto ────────────────────────────────

def plot_12x4_granular_all_lags(alpha, label, lats, lons, output_dir, basin_name="Brasil"):
    """
    12 linhas (Lag 1–12) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa lat/lon de α.
    alpha: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))

    cmap, norm = _make_alpha_cmap()
    cmap.set_bad(color="white")
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

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
                np.ma.masked_invalid(alpha[col, li]),
                cmap=cmap, norm=norm,
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
        "PREDEP\n0=Independente, 1=Perfeito",
        fontsize=10, fontweight="bold",
    )
    plt.suptitle(
        f"PREDEP granular — {basin_name}: {label}\n"
        f"Linhas = Lags (1–{N_LAGS} meses) | Colunas = Estações",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"granular_all_lags_{label}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_granular_optimal(alpha, label, lats, lons, output_dir, basin_name="Brasil"):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: α máximo por pixel  |  Linha 2: lag do α máximo.
    alpha: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))

    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad(color="white")
    lag_boundaries = np.arange(0.5, N_LAGS + 1.5, 1.0)
    lag_norm = BoundaryNorm(lag_boundaries, ncolors=plt.cm.jet.N, clip=True)
    jet_cmap = copy.copy(plt.cm.jet)
    jet_cmap.set_bad(color="white")
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

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
            (best_lag, lag_norm, jet_cmap),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                np.ma.masked_invalid(vals),
                cmap=cmap_use, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=12, fontweight="bold")
            if col == 0:
                label_row = "PREDEP máximo" if row == 0 else "Lag ótimo"
                ax.text(-0.1, 0.5, label_row, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=10, fontweight="bold")

            if row == 0 and im_alpha is None:
                im_alpha = im
            if row == 1 and im_lag is None:
                im_lag = im

    fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, n_cols]),
                 ticks=np.arange(0.0, 1.1, 0.1)).set_label(
        "PREDEP", fontsize=12, fontweight="bold")
    fig.colorbar(im_lag, cax=fig.add_subplot(gs[1, n_cols]),
                 ticks=LAGS).set_label("Lag (meses)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Lags Ótimos (granular) — {basin_name}: {label}",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"granular_optimal_{label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_12x4_diff_clusters(
    alpha_core, r2, label, cluster_names, output_dir,
    masks_flat, mask_idx, n_lat, n_lon,
):
    """
    12 linhas (Lag 1–12) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa de (alpha_core - r2) por cluster.
    Laranja/vermelho = PREDEP melhor | Azul = Regressão melhor.
    alpha_core, r2: (n_seasons, n_clusters, N_LAGS)
    """
    diff = alpha_core - r2
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))
    cmap, norm = _make_diff_cmap()

    fig = plt.figure(figsize=(3 * n_cols, 30))
    gs = GridSpec(
        N_LAGS, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.95, top=0.96, bottom=0.02,
    )

    im = None
    for li, lag in enumerate(LAGS):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            vals = diff[col, :, li]
            amap = make_alpha_map(
                vals, cluster_names, mask_idx, masks_flat, n_lat, n_lon
            )
            im = ax.imshow(
                amap, cmap=cmap, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
            )
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=11, fontweight="bold", pad=3)
            if col == 0:
                ax.text(
                    -0.15, 0.5, f"Lag {lag}",
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=9, fontweight="bold",
                )
            for ci, cname in enumerate(cluster_names):
                if cname not in mask_idx or np.isnan(vals[ci]):
                    continue
                m2d = masks_flat[mask_idx[cname]].reshape(n_lat, n_lon)
                ry, cx = np.where(m2d)
                cy, cxx = ry.mean(), cx.mean()
                ax.text(
                    cxx, cy, f"{vals[ci]:+.2f}",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black",
                )

    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]))
    cb.set_label(
        "PREDEP − R²\n← Regressão melhor | PREDEP melhor →",
        fontsize=9, fontweight="bold",
    )
    plt.suptitle(
        f"Diferença PREDEP − Regressão ({label})\n"
        "Laranja = PREDEP melhor | Azul = Regressão melhor",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"diff_all_lags_{label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_winner_clusters(
    alpha_core, r2, label, cluster_names, output_dir,
    masks_flat, mask_idx, n_lat, n_lon,
):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: melhor valor entre alpha_core e r2 (colormap cinza→laranja).
    Linha 2: método vencedor (laranja=PREDEP, azul=Regressão).
    alpha_core, r2: (n_seasons, n_clusters, N_LAGS)
    """
    n_cols = len(SEASONS)
    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_win, norm_win = _make_winner_cmap()

    fig = plt.figure(figsize=(5 * n_cols, 11))
    gs = GridSpec(
        2, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.2, wspace=0.15,
        left=0.06, right=0.95, top=0.92, bottom=0.05,
    )

    im_best = im_win = None

    for col, season in enumerate(SEASONS):
        ac = alpha_core[col]  # (n_clusters, N_LAGS)
        rv = r2[col]

        ac_max = np.where(
            np.all(np.isnan(ac), axis=1), np.nan, np.nanmax(ac, axis=1)
        )
        r2_max = np.where(
            np.all(np.isnan(rv), axis=1), np.nan, np.nanmax(rv, axis=1)
        )

        best_val = np.where(
            np.isnan(ac_max) & np.isnan(r2_max),
            np.nan,
            np.fmax(ac_max, r2_max),
        )

        winner = _winner_vals(ac_max, r2_max)

        for row, (vals, cmap_use, norm_use) in enumerate([
            (best_val, cmap_alpha, norm_alpha),
            (winner, cmap_win, norm_win),
        ]):
            ax = fig.add_subplot(gs[row, col])
            amap = make_alpha_map(
                vals, cluster_names, mask_idx, masks_flat, n_lat, n_lon
            )
            im = ax.imshow(
                amap, cmap=cmap_use, norm=norm_use,
                aspect="auto", origin="lower", interpolation="nearest",
            )
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=12, fontweight="bold")
            if col == 0:
                row_lbl = "Melhor valor" if row == 0 else "Método vencedor"
                ax.text(
                    -0.1, 0.5, row_lbl,
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=10, fontweight="bold",
                )
            if row == 0 and im_best is None:
                im_best = im
            if row == 1 and im_win is None:
                im_win = im

            for ci, cname in enumerate(cluster_names):
                if cname not in mask_idx or np.isnan(vals[ci]):
                    continue
                m2d = masks_flat[mask_idx[cname]].reshape(n_lat, n_lon)
                ry, cx = np.where(m2d)
                cy, cxx = ry.mean(), cx.mean()
                if row == 0:
                    txt = f"{vals[ci]:.2f}"
                    color = "white" if vals[ci] > 0.7 else "black"
                else:
                    txt = "P" if vals[ci] == 2 else "R"
                    color = "white"
                ax.text(
                    cxx, cy, txt,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color,
                )

    fig.colorbar(
        im_best, cax=fig.add_subplot(gs[0, n_cols]),
        ticks=np.arange(0.0, 1.1, 0.1),
    ).set_label("PREDEP / R² máximo", fontsize=11, fontweight="bold")

    cb_win = fig.colorbar(
        im_win, cax=fig.add_subplot(gs[1, n_cols]),
        ticks=[0, 1, 2],
    )
    cb_win.set_ticklabels(["NaN", "Regressão", "PREDEP"])
    cb_win.set_label("Método vencedor", fontsize=11, fontweight="bold")

    plt.suptitle(
        f"Método vencedor — Clusters Uruguai: {label}\n"
        "P = PREDEP  |  R = Regressão  |  Linha 1: melhor valor",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"winner_optimal_{label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_12x4_diff_granular(alpha_core, r2, label, lats, lons, output_dir, basin_name="Brasil"):
    """
    12 linhas (Lag 1–12) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa lat/lon de (alpha_core - r2).
    alpha_core, r2: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    diff = alpha_core - r2
    n_cols = len(SEASONS)
    LAGS = list(range(1, N_LAGS + 1))
    cmap, norm = _make_diff_cmap()
    cmap = copy.copy(cmap)
    cmap.set_bad(color="white")
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    fig = plt.figure(figsize=(3 * n_cols, 30))
    gs = GridSpec(
        N_LAGS, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.95, top=0.96, bottom=0.02,
    )

    im = None
    for li, lag in enumerate(LAGS):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            im = ax.imshow(
                np.ma.masked_invalid(diff[col, li]),
                cmap=cmap, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=11, fontweight="bold", pad=3)
            if col == 0:
                ax.text(
                    -0.15, 0.5, f"Lag {lag}",
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=9, fontweight="bold",
                )

    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]))
    cb.set_label(
        "PREDEP − R²\n← Regressão melhor | PREDEP melhor →",
        fontsize=9, fontweight="bold",
    )
    plt.suptitle(
        f"Diferença PREDEP − Regressão — {basin_name} ({label})\n"
        "Laranja = PREDEP melhor | Azul = Regressão melhor",
        fontsize=14, fontweight="bold", y=0.99,
    )

    out = Path(output_dir) / f"diff_all_lags_{label}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_winner_granular(alpha_core, r2, label, lats, lons, output_dir, basin_name="Brasil"):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: melhor valor (max entre alpha_core e r2) por pixel.
    Linha 2: método vencedor (laranja=PREDEP, azul=Regressão).
    alpha_core, r2: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad(color="white")
    cmap_win, norm_win = _make_winner_cmap()
    cmap_win.set_bad(color="white")
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    fig = plt.figure(figsize=(5 * n_cols, 11))
    gs = GridSpec(
        2, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.2, wspace=0.15,
        left=0.06, right=0.95, top=0.92, bottom=0.05,
    )

    im_best = im_win = None

    for col, season in enumerate(SEASONS):
        ac = alpha_core[col]  # (N_LAGS, n_lat, n_lon)
        rv = r2[col]

        all_nan_ac = np.all(np.isnan(ac), axis=0)
        all_nan_rv = np.all(np.isnan(rv), axis=0)

        ac_max = np.where(all_nan_ac, np.nan, np.nanmax(ac, axis=0))
        r2_max = np.where(all_nan_rv, np.nan, np.nanmax(rv, axis=0))

        best_val = np.where(
            all_nan_ac & all_nan_rv, np.nan, np.fmax(ac_max, r2_max)
        )

        winner = np.where(
            all_nan_ac & all_nan_rv, np.nan,
            np.where(
                np.isnan(ac_max), 1,
                np.where(np.isnan(r2_max), 2,
                         np.where(r2_max > ac_max, 1, 2)),
            ),
        )

        for row, (vals, cmap_use, norm_use) in enumerate([
            (best_val, cmap_alpha, norm_alpha),
            (winner, cmap_win, norm_win),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                np.ma.masked_invalid(vals),
                cmap=cmap_use, norm=norm_use,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=12, fontweight="bold")
            if col == 0:
                row_lbl = "Melhor valor" if row == 0 else "Método vencedor"
                ax.text(
                    -0.1, 0.5, row_lbl,
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=10, fontweight="bold",
                )
            if row == 0 and im_best is None:
                im_best = im
            if row == 1 and im_win is None:
                im_win = im

    fig.colorbar(
        im_best, cax=fig.add_subplot(gs[0, n_cols]),
        ticks=np.arange(0.0, 1.1, 0.1),
    ).set_label("PREDEP / R² máximo", fontsize=11, fontweight="bold")

    cb_win = fig.colorbar(
        im_win, cax=fig.add_subplot(gs[1, n_cols]),
        ticks=[0, 1, 2],
    )
    cb_win.set_ticklabels(["NaN", "Regressão", "PREDEP"])
    cb_win.set_label("Método vencedor", fontsize=11, fontweight="bold")

    plt.suptitle(
        f"Método vencedor granular — {basin_name}: {label}\n"
        "Laranja = PREDEP  |  Azul = Regressão  |  Linha 1: melhor valor",
        fontsize=14, fontweight="bold",
    )

    out = Path(output_dir) / f"winner_optimal_{label}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def generate_plots_granular(nc_file, mov_name, plots_dir, basin_name="Brasil"):
    """
    Gera 6 plots granulares para o exp 06.

    Estrutura de saída dentro de plots_dir:
      PREDEP/BRAZIL/  ou  PREDEP/BACIAS/{basin_name}/
      REGRESSAO/BRAZIL/  ou  REGRESSAO/BACIAS/{basin_name}/
      COMPARACAO/BRAZIL/  ou  COMPARACAO/BACIAS/{basin_name}/
    """
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

    base = Path(plots_dir)
    sub = "BRAZIL" if basin_name == "Brasil" else Path("BACIAS") / basin_name
    predep_dir = base / "PREDEP" / sub
    regr_dir   = base / "REGRESSAO" / sub
    comp_dir   = base / "COMPARACAO" / sub
    for d in (predep_dir, regr_dir, comp_dir):
        d.mkdir(parents=True, exist_ok=True)

    for metric_name, alpha, out_dir in [
        ("core", ac, predep_dir),
        ("r2",   r2, regr_dir),
    ]:
        label = f"{mov_name}_{metric_name}"
        plot_2x4_granular_optimal(alpha, label, lats, lons, out_dir, basin_name=basin_name)
        plot_12x4_granular_all_lags(alpha, label, lats, lons, out_dir, basin_name=basin_name)
    plot_12x4_diff_granular(ac, r2, mov_name, lats, lons, comp_dir, basin_name=basin_name)
    plot_2x4_winner_granular(ac, r2, mov_name, lats, lons, comp_dir, basin_name=basin_name)
