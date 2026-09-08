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
from matplotlib.patches import Patch
from pathlib import Path

SEASONS = ["DJF", "MAM", "JJA", "SON"]
N_LAGS = 12
ALPHA_LEVELS = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]
SIGNAL_THRESHOLD = 0.1  # pixels com ambos PREDEP e r² abaixo deste valor → cinza

BASIN_LABELS = {
    "amazonica":                   1,
    "atlantico_leste":             2,
    "atlantico_nordeste":          3,
    "atlantico_nordeste_oriental": 11,
    "atlantico_sudeste":           4,
    "atlantico_sul":               5,
    "paraguai":                    6,
    "parana":                      12,
    "parnaiba":                    7,
    "sao_francisco":               8,
    "tocantins":                   9,
    "uruguai":                     10,
}


def _make_diff_cmap(vmin=-1.0, vmax=1.0):
    """
    Colormap divergente centrado em zero para (alpha_core - r2):
      negativo → azul (R² melhor)
      zero     → branco
      positivo → laranja/vermelho (PREDEP melhor)
    vmin/vmax derivados dos dados reais para evitar saturação artificial.
    """
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
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


def _make_lag_cmap(lags: list):
    """
    Colormap discreto laranja (pálido → escuro) para os lags fornecidos.
    Funciona com qualquer conjunto de lags (ex: [0,1,3,6,9,12]).
    """
    n = len(lags)
    colors = [plt.cm.Oranges(0.25 + 0.75 * i / max(n - 1, 1)) for i in range(n)]
    cmap = ListedColormap(colors, name="lag_cmap")
    lags_arr = np.array(lags, dtype=float)
    if n == 1:
        boundaries = [lags_arr[0] - 0.5, lags_arr[0] + 0.5]
    else:
        mids = (lags_arr[:-1] + lags_arr[1:]) / 2
        boundaries = np.concatenate([[lags_arr[0] - 0.5], mids, [lags_arr[-1] + 0.5]])
    norm = BoundaryNorm(boundaries, ncolors=n, clip=True)
    return cmap, norm


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


def _load_basin_outlines(nc_dir, mov_name, fallback_dirs=None):
    """
    Lê os NCs por bacia para extrair máscaras de pixels válidos.
    Retorna lista de (número, blats_asc, blons, mask_2d).

    fallback_dirs: lista de dirs adicionais onde procurar arquivos de bacia
    ausentes em nc_dir (usada quando resultados estão em exps separados).
    """
    import netCDF4 as nc4

    def _exp_num(p):
        try:
            return int(p.parent.parent.name.replace("exp", ""))
        except (ValueError, IndexError):
            return -1

    search_dirs = [Path(nc_dir)]
    if fallback_dirs:
        # Ordena do exp mais recente para o mais antigo
        search_dirs += sorted(fallback_dirs, key=_exp_num, reverse=True)

    outlines = []
    for basin_name, number in BASIN_LABELS.items():
        fname = f"{mov_name}_{basin_name}_predep_granular_seasonal.nc"
        nc_file = None
        for d in search_dirs:
            candidate = Path(d) / fname
            if candidate.exists():
                nc_file = candidate
                break
        if nc_file is None:
            continue
        with nc4.Dataset(str(nc_file)) as nc:
            blats = np.array(nc.variables["latitude"][:])
            blons = np.array(nc.variables["longitude"][:])
            ac = np.ma.filled(nc.variables["alpha_core"][:], np.nan)
        # Pixel válido = pelo menos uma season/lag não-NaN
        mask = ~np.all(np.isnan(ac), axis=(0, 1))
        if len(blats) > 1 and blats[0] > blats[-1]:
            blats = blats[::-1]
            mask = mask[::-1, :]
        outlines.append((number, blats, blons, mask))
    return outlines


def _draw_basin_outlines(ax, basin_outlines):
    """Sobrepõe bordas e números de bacias nos eixos atuais (coords geográficas)."""
    from scipy.ndimage import binary_fill_holes
    for number, blats, blons, mask in basin_outlines:
        if not mask.any():
            continue
        # Preenche buracos internos (rios, lagos, pixels sem dado) para
        # obter apenas a borda exterior de cada bacia
        filled = binary_fill_holes(mask)
        try:
            ax.contour(blons, blats, filled.astype(float),
                       levels=[0.5], colors="#111111", linewidths=0.6)
        except Exception:
            pass
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        clat = float(blats[int(rows.mean())])
        clon = float(blons[int(cols.mean())])
        ax.text(clon, clat, str(number),
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.7, edgecolor="none"))


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
    plt.savefig(out, dpi=80)
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
    plt.savefig(out, dpi=80)
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

def plot_12x4_granular_all_lags(alpha, label, lats, lons, output_dir,
                                basin_name="Brasil", basin_outlines=None,
                                lags=None, is_subset=False):
    """
    N linhas (lags selecionados) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa lat/lon de α.
    alpha: (n_seasons, N_LAGS, n_lat, n_lon)
    lags: lista de lags a exibir (None = 1-based indices derivados do shape do array).
    """
    if lags is None:
        lags = list(range(1, alpha.shape[1] + 1))
    n_rows = len(lags)
    n_cols = len(SEASONS)

    cmap, norm = _make_alpha_cmap()
    cmap.set_bad(color="white")
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    fig = plt.figure(figsize=(3 * n_cols, 2.5 * n_rows))
    gs = GridSpec(
        n_rows, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05, left=0.08, right=0.95, top=0.90, bottom=0.02,
    )

    im = None
    for li, lag in enumerate(lags):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            im = ax.imshow(
                np.ma.masked_invalid(alpha[col, li]),
                cmap=cmap, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            if basin_outlines:
                _draw_basin_outlines(ax, basin_outlines)
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=21, fontweight="bold", pad=3)
            if col == 0:
                ax.text(-0.15, 0.5, f"Lag {lag}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=18, fontweight="bold")

    fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]),
                 ticks=ALPHA_LEVELS).set_label(
        "PREDEP\n0=Independente, 1=Perfeito",
        fontsize=20, fontweight="bold",
    )
    lags_str = ",".join(map(str, lags))
    plt.suptitle(
        f"PREDEP — {basin_name}: {label}  (lags: {lags_str} meses)",
        fontsize=26, fontweight="bold", y=0.97,
    )

    suffix = "sel_lags" if is_subset else "all_lags"
    out = Path(output_dir) / f"granular_{suffix}_{label}.png"
    plt.savefig(out, dpi=80)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_granular_optimal(alpha, label, lats, lons, output_dir,
                              basin_name="Brasil", basin_outlines=None, lags=None):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: α máximo por pixel  |  Linha 2: lag do α máximo.
    alpha: (n_seasons, n_lags, n_lat, n_lon)
    lags: lista com os valores reais de lag (ex: [0, 1, 3, 6, 9, 12]).
    """
    if lags is None:
        lags = list(range(alpha.shape[1]))
    lag_arr = np.array(lags, dtype=float)

    n_cols = len(SEASONS)
    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad(color="white")
    lag_cmap, lag_norm = _make_lag_cmap(lags)
    lag_cmap_plot = copy.copy(lag_cmap)
    lag_cmap_plot.set_bad(color=(0, 0, 0, 0))  # transparente: deixa camada cinza aparecer
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
        a = alpha[col]  # (n_lags, n_lat, n_lon)
        all_nan = np.all(np.isnan(a), axis=0)

        alpha_max = np.where(all_nan, np.nan, np.nanmax(a, axis=0))
        best_idx = np.nanargmax(np.where(np.isnan(a), -np.inf, a), axis=0)
        best_lag = np.full(best_idx.shape, np.nan, dtype=float)
        valid = ~all_nan
        best_lag[valid] = lag_arr[best_idx[valid]]

        # Pixels com sinal abaixo do threshold: sem lag ótimo significativo
        low_signal = ~np.isnan(alpha_max) & (alpha_max < SIGNAL_THRESHOLD)
        best_lag_display = np.where(low_signal, np.nan, best_lag)

        for row, (vals, norm, cmap_use) in enumerate([
            (alpha_max,        norm_alpha, cmap_alpha),
            (best_lag_display, lag_norm,   lag_cmap_plot),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("white")

            if row == 1:
                gray_layer = np.where(low_signal, 0.75, np.nan)
                ax.imshow(
                    np.ma.masked_invalid(gray_layer),
                    cmap="gray", vmin=0, vmax=1,
                    aspect="auto", origin="lower", interpolation="nearest",
                    extent=extent,
                )

            im = ax.imshow(
                np.ma.masked_invalid(vals),
                cmap=cmap_use, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            if basin_outlines:
                _draw_basin_outlines(ax, basin_outlines)
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=21, fontweight="bold")
            if col == 0:
                label_row = "PREDEP máximo" if row == 0 else "Lag ótimo"
                ax.text(-0.1, 0.5, label_row, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=18, fontweight="bold")

            if row == 0 and im_alpha is None:
                im_alpha = im
            if row == 1 and im_lag is None:
                im_lag = im

    fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, n_cols]),
                 ticks=np.arange(0.0, 1.1, 0.1)).set_label(
        "PREDEP", fontsize=20, fontweight="bold")
    cb_lag = fig.colorbar(im_lag, cax=fig.add_subplot(gs[1, n_cols]), ticks=lags)
    cb_lag.set_ticklabels([str(l) for l in lags])
    cb_lag.set_label("Lag (meses)", fontsize=20, fontweight="bold")

    plt.suptitle(
        f"Lag ótimo — {basin_name}: {label}",
        fontsize=26, fontweight="bold",
    )

    out = Path(output_dir) / f"granular_optimal_{label}.png"
    plt.savefig(out, dpi=80)
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

    finite = diff[np.isfinite(diff)]
    if finite.size > 0:
        d_min = float(np.percentile(finite, 1))
        d_max = float(np.percentile(finite, 99))
        vabs = max(abs(d_min), abs(d_max), 0.05)
        vmin_plot = max(d_min, -vabs)
        vmax_plot = min(d_max,  vabs)
    else:
        vmin_plot, vmax_plot = -1.0, 1.0

    cmap, norm = _make_diff_cmap(vmin_plot, vmax_plot)

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

    cb_ticks = np.round(np.linspace(vmin_plot, vmax_plot, 5), 2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]), ticks=cb_ticks)
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
    plt.savefig(out, dpi=80)
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
    plt.savefig(out, dpi=80)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_12x4_diff_granular(alpha_core, r2, label, lats, lons, output_dir,
                            basin_name="Brasil", basin_outlines=None,
                            lags=None, is_subset=False):
    """
    N linhas (lags selecionados) × 4 colunas (DJF, MAM, JJA, SON).
    Cada célula: mapa lat/lon de (alpha_core - r2).
    alpha_core, r2: (n_seasons, N_LAGS, n_lat, n_lon)
    lags: lista de lags a exibir (None = 1-based indices derivados do shape do array).
    """
    diff = alpha_core - r2
    if lags is None:
        lags = list(range(1, diff.shape[1] + 1))
    n_rows = len(lags)
    n_cols = len(SEASONS)

    finite = diff[np.isfinite(diff)]
    if finite.size > 0:
        d_min = float(np.percentile(finite, 1))
        d_max = float(np.percentile(finite, 99))
        vabs = max(abs(d_min), abs(d_max), 0.05)
        vmin_plot = max(d_min, -vabs)
        vmax_plot = min(d_max,  vabs)
    else:
        vmin_plot, vmax_plot = -1.0, 1.0

    cmap, norm = _make_diff_cmap(vmin_plot, vmax_plot)
    cmap_plot = copy.copy(cmap)
    cmap_plot.set_bad(color=(0, 0, 0, 0))  # NaN transparente: deixa camada cinza aparecer
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    fig = plt.figure(figsize=(3 * n_cols, 2.5 * n_rows))
    gs = GridSpec(
        n_rows, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.05, wspace=0.05,
        left=0.08, right=0.95, top=0.90, bottom=0.02,
    )

    im = None
    for li, lag in enumerate(lags):
        for col, season in enumerate(SEASONS):
            ax = fig.add_subplot(gs[li, col])
            ax.set_facecolor("white")

            low_signal = (
                ~np.isnan(alpha_core[col, li]) & (alpha_core[col, li] < SIGNAL_THRESHOLD) &
                ~np.isnan(r2[col, li])          & (r2[col, li]          < SIGNAL_THRESHOLD)
            )

            # Camada 1: cinza onde nenhum método tem sinal
            gray_layer = np.where(low_signal, 0.75, np.nan)
            ax.imshow(
                np.ma.masked_invalid(gray_layer),
                cmap="gray", vmin=0, vmax=1,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )

            # Camada 2: diff colormap (NaN transparente onde low_signal)
            diff_layer = np.where(low_signal, np.nan, diff[col, li])
            im = ax.imshow(
                np.ma.masked_invalid(diff_layer),
                cmap=cmap_plot, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            if basin_outlines:
                _draw_basin_outlines(ax, basin_outlines)
            ax.axis("off")
            if li == 0:
                ax.set_title(season, fontsize=21, fontweight="bold", pad=3)
            if col == 0:
                ax.text(
                    -0.15, 0.5, f"Lag {lag}",
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=18, fontweight="bold",
                )

    cb_ticks = np.round(np.linspace(vmin_plot, vmax_plot, 5), 2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, n_cols]), ticks=cb_ticks)
    cb.set_label(
        "PREDEP − R²\n← Regressão melhor | PREDEP melhor →",
        fontsize=20, fontweight="bold",
    )
    lags_str = ",".join(map(str, lags))
    plt.suptitle(
        f"PREDEP − R² — {basin_name} ({label})  (lags: {lags_str} meses)",
        fontsize=26, fontweight="bold", y=0.97,
    )

    suffix = "sel_lags" if is_subset else "all_lags"
    out = Path(output_dir) / f"diff_{suffix}_{label}.png"
    plt.savefig(out, dpi=80)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_2x4_winner_granular(alpha_core, r2, label, lats, lons, output_dir,
                             basin_name="Brasil", basin_outlines=None):
    """
    2 linhas × 4 colunas (DJF, MAM, JJA, SON).
    Linha 1: melhor valor (max entre alpha_core e r2) por pixel.
    Linha 2: método vencedor (laranja=PREDEP, azul=Regressão).
    alpha_core, r2: (n_seasons, N_LAGS, n_lat, n_lon)
    """
    n_cols = len(SEASONS)
    cmap_alpha_plot = copy.copy(_make_alpha_cmap()[0])
    cmap_alpha_plot.set_bad(color=(0, 0, 0, 0))  # NaN transparente
    _, norm_alpha = _make_alpha_cmap()
    cmap_win, norm_win = _make_winner_cmap()
    cmap_win_plot = copy.copy(cmap_win)
    cmap_win_plot.set_bad(color=(0, 0, 0, 0))  # NaN transparente
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

        low_signal = (
            ~np.isnan(ac_max) & (ac_max < SIGNAL_THRESHOLD) &
            ~np.isnan(r2_max) & (r2_max < SIGNAL_THRESHOLD)
        )

        best_val = np.where(
            (all_nan_ac & all_nan_rv) | low_signal, np.nan, np.fmax(ac_max, r2_max)
        )

        winner = np.where(
            (all_nan_ac & all_nan_rv) | low_signal, np.nan,
            np.where(
                np.isnan(ac_max), 1,
                np.where(np.isnan(r2_max), 2,
                         np.where(r2_max > ac_max, 1, 2)),
            ),
        )

        for row, (vals, cmap_use, norm_use) in enumerate([
            (best_val, cmap_alpha_plot, norm_alpha),
            (winner, cmap_win_plot, norm_win),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("white")

            # Camada cinza onde nenhum método tem sinal (ambas as linhas)
            gray_layer = np.where(low_signal, 0.75, np.nan)
            ax.imshow(
                np.ma.masked_invalid(gray_layer),
                cmap="gray", vmin=0, vmax=1,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )

            im = ax.imshow(
                np.ma.masked_invalid(vals),
                cmap=cmap_use, norm=norm_use,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=extent,
            )
            if basin_outlines:
                _draw_basin_outlines(ax, basin_outlines)
            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=21, fontweight="bold")
            if col == 0:
                row_lbl = "Melhor valor" if row == 0 else "Método vencedor"
                ax.text(
                    -0.1, 0.5, row_lbl,
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="center",
                    fontsize=18, fontweight="bold",
                )
            if row == 0 and im_best is None:
                im_best = im
            if row == 1 and im_win is None:
                im_win = im

    fig.colorbar(
        im_best, cax=fig.add_subplot(gs[0, n_cols]),
        ticks=np.arange(0.0, 1.1, 0.1),
    ).set_label("PREDEP / R² máximo", fontsize=20, fontweight="bold")

    cb_win = fig.colorbar(
        im_win, cax=fig.add_subplot(gs[1, n_cols]),
        ticks=[0, 1, 2],
    )
    cb_win.set_ticklabels(["NaN", "Regressão", "PREDEP"])
    cb_win.set_label("Método vencedor", fontsize=20, fontweight="bold")

    plt.suptitle(
        f"Método vencedor — {basin_name}: {label}",
        fontsize=26, fontweight="bold",
    )

    out = Path(output_dir) / f"winner_optimal_{label}.png"
    plt.savefig(out, dpi=80)
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def generate_plots_granular(nc_file, mov_name, plots_dir, basin_name="Brasil", lags=None):
    """
    Gera 6 plots granulares para o exp 06.

    Estrutura de saída dentro de plots_dir:
      PREDEP/BRAZIL/  ou  PREDEP/BACIAS/{basin_name}/
      REGRESSAO/BRAZIL/  ou  REGRESSAO/BACIAS/{basin_name}/
      COMPARACAO/BRAZIL/  ou  COMPARACAO/BACIAS/{basin_name}/
    """
    import netCDF4 as nc4

    nc_file = Path(nc_file)
    if not nc_file.exists():
        print(f"  [AVISO] Arquivo não encontrado, pulando: {nc_file.name}")
        return

    with nc4.Dataset(str(nc_file)) as nc:
        lats = np.array(nc.variables["latitude"][:])
        lons = np.array(nc.variables["longitude"][:])
        s_nc = [s.decode() if isinstance(s, bytes) else str(s)
                for s in nc.variables["season"][:]]
        nc_lags = list(map(int, nc.variables["lag"][:]))
        ac = np.ma.filled(nc.variables["alpha_core"][:], np.nan).astype(float)
        r2 = np.ma.filled(nc.variables["r2"][:], np.nan).astype(float)

    idx = [s_nc.index(s) for s in SEASONS if s in s_nc]
    ac, r2 = ac[idx], r2[idx]

    if lags is not None:
        display_lags = [l for l in lags if l in nc_lags]
        valid_idx = [nc_lags.index(l) for l in display_lags]
        is_subset = display_lags != nc_lags
    else:
        display_lags = nc_lags
        valid_idx = list(range(len(nc_lags)))
        is_subset = False
    ac = ac[:, valid_idx]
    r2 = r2[:, valid_idx]

    # Garante lats em ordem ascendente (sul→norte) para imshow com origin="lower"
    if len(lats) > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        ac = ac[:, :, ::-1, :]
        r2 = r2[:, :, ::-1, :]

    base = Path(plots_dir)
    sub = "BRAZIL" if basin_name == "Brasil" else Path("BACIAS") / basin_name
    predep_dir = base / "PREDEP" / sub
    regr_dir = base / "REGRESSAO" / sub
    comp_dir = base / "COMPARACAO" / sub
    for d in (predep_dir, regr_dir, comp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Carrega bordas das bacias apenas para o mapa Brasil completo.
    # Busca também em outros exps para cobrir bacias rodadas em runs separados.
    basin_outlines = None
    if basin_name == "Brasil":
        nc_dir = Path(nc_file).parent
        results_base = nc_dir.parent.parent  # …/predep_granular_brazil/
        fallback_dirs = sorted(results_base.glob(f"exp*/{mov_name}"))
        basin_outlines = _load_basin_outlines(nc_dir, mov_name, fallback_dirs)

    for metric_name, alpha, out_dir in [
        ("core", ac, predep_dir),
        ("r2", r2, regr_dir),
    ]:
        label = f"{mov_name}_{metric_name}"
        plot_2x4_granular_optimal(
            alpha, label, lats, lons, out_dir,
            basin_name=basin_name, basin_outlines=basin_outlines,
            lags=display_lags,
        )
        plot_12x4_granular_all_lags(
            alpha, label, lats, lons, out_dir,
            basin_name=basin_name, basin_outlines=basin_outlines,
            lags=display_lags, is_subset=is_subset,
        )
    plot_12x4_diff_granular(
        ac, r2, mov_name, lats, lons, comp_dir,
        basin_name=basin_name, basin_outlines=basin_outlines,
        lags=display_lags, is_subset=is_subset,
    )
    plot_2x4_winner_granular(
        ac, r2, mov_name, lats, lons, comp_dir,
        basin_name=basin_name, basin_outlines=basin_outlines,
    )


def generate_brazil_all_lags_granular(nc_file, mov_name, plots_dir):
    """
    Gera apenas o plot 12x4 (todos os lags) para Brasil completo,
    com contornos das bacias, para PREDEP (alpha_core) e R2.
    """
    import netCDF4 as nc4

    nc_file = Path(nc_file)
    if not nc_file.exists():
        print(f"  [AVISO] Arquivo não encontrado, pulando: {nc_file.name}")
        return []

    with nc4.Dataset(str(nc_file)) as nc:
        lats = np.array(nc.variables["latitude"][:])
        lons = np.array(nc.variables["longitude"][:])
        s_nc = [s.decode() if isinstance(s, bytes) else str(s)
                for s in nc.variables["season"][:]]
        ac = np.ma.filled(nc.variables["alpha_core"][:], np.nan).astype(float)
        r2 = np.ma.filled(nc.variables["r2"][:], np.nan).astype(float)

    idx = [s_nc.index(s) for s in SEASONS if s in s_nc]
    if not idx:
        print(f"  [AVISO] Seasons ausentes em: {nc_file.name}")
        return []
    ac, r2 = ac[idx], r2[idx]

    # Garante lats em ordem ascendente (sul→norte) para imshow com origin="lower"
    if len(lats) > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        ac = ac[:, :, ::-1, :]
        r2 = r2[:, :, ::-1, :]

    base = Path(plots_dir)
    predep_dir = base / "PREDEP" / "BRAZIL"
    regr_dir = base / "REGRESSAO" / "BRAZIL"
    for d in (predep_dir, regr_dir):
        d.mkdir(parents=True, exist_ok=True)

    nc_dir = Path(nc_file).parent
    results_base = nc_dir.parent.parent
    fallback_dirs = sorted(results_base.glob(f"exp*/{mov_name}"))
    basin_outlines = _load_basin_outlines(nc_dir, mov_name, fallback_dirs)
    if not basin_outlines:
        print("  [AVISO] Contornos de bacias não encontrados para sobrepor.")

    outputs = []
    label_core = f"{mov_name}_core"
    outputs.append(
        plot_12x4_granular_all_lags(
            ac, label_core, lats, lons, predep_dir,
            basin_name="Brasil", basin_outlines=basin_outlines,
        )
    )
    label_r2 = f"{mov_name}_r2"
    outputs.append(
        plot_12x4_granular_all_lags(
            r2, label_r2, lats, lons, regr_dir,
            basin_name="Brasil", basin_outlines=basin_outlines,
        )
    )
    return outputs


# ── Best MoV por pixel (multi-MoV) ───────────────────────────────────────────

def _make_mov_cmap(mov_names: list):
    """Colormap categórico: uma cor distinta por MoV."""
    base = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
            + list(plt.cm.tab20c.colors))
    colors = [base[i % len(base)] for i in range(len(mov_names))]
    cmap = ListedColormap(colors, name="mov_cmap")
    norm = BoundaryNorm(np.arange(-0.5, len(mov_names) + 0.5), ncolors=len(mov_names))
    return cmap, norm, colors


def plot_best_mov_granular(ac_stack, r2_stack, mov_names, lats, lons,
                           lag_idx, lag_label, exp_n, plots_dir):
    """
    4 linhas × 4 colunas (DJF, MAM, JJA, SON).
      Linha 1: melhor α (PREDEP) por pixel
      Linha 2: qual MoV teve melhor α  (cinza onde sem sinal)
      Linha 3: melhor r² por pixel
      Linha 4: qual MoV teve melhor r²  (cinza onde sem sinal)
    ac_stack: (n_movs, n_seasons, n_lags, n_lat, n_lon)
    lag_idx: int (índice no eixo lag) ou None (max sobre todos os lags)
    """
    n_movs, n_seasons, n_lags, n_lat, n_lon = ac_stack.shape
    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad("white")
    cmap_mov, norm_mov, mov_colors = _make_mov_cmap(mov_names)
    cmap_mov_plot = copy.copy(cmap_mov)
    cmap_mov_plot.set_bad(color=(0, 0, 0, 0))

    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]
    n_cols = len(SEASONS)

    fig = plt.figure(figsize=(5 * n_cols + 1.5, 5 * 4 + 3.0))
    gs = GridSpec(
        4, n_cols + 1, figure=fig,
        width_ratios=[1] * n_cols + [0.04],
        hspace=0.08, wspace=0.05,
        left=0.13, right=0.97, top=0.91, bottom=0.15,
    )

    row_labels = [
        "PREDEP máximo", "MoV — melhor PREDEP",
        "R² máximo",     "MoV — melhor R²",
    ]
    im_alpha = im_mov = None

    for row in range(4):
        is_value_row = row in (0, 2)
        stack = ac_stack if row < 2 else r2_stack

        for col, season in enumerate(SEASONS):
            if lag_idx is None:
                slc = stack[:, col, :, :, :]           # (n_movs, n_lags, n_lat, n_lon)
                best_val = np.nanmax(
                    slc.reshape(n_movs * n_lags, n_lat, n_lon), axis=0
                )
                per_mov_max = np.nanmax(slc, axis=1)   # (n_movs, n_lat, n_lon)
            else:
                slc = stack[:, col, lag_idx, :, :]     # (n_movs, n_lat, n_lon)
                best_val = np.nanmax(slc, axis=0)
                per_mov_max = slc

            all_nan = np.all(np.isnan(per_mov_max), axis=0)
            best_val = np.where(all_nan, np.nan, best_val)
            best_mov_idx = np.full((n_lat, n_lon), np.nan)
            valid = ~all_nan
            best_mov_idx[valid] = np.nanargmax(
                np.where(np.isnan(per_mov_max[:, valid]), -np.inf,
                         per_mov_max[:, valid]), axis=0
            ).astype(float)

            low_signal = ~np.isnan(best_val) & (best_val < SIGNAL_THRESHOLD)

            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("white")

            if is_value_row:
                im = ax.imshow(
                    np.ma.masked_invalid(best_val),
                    cmap=cmap_alpha, norm=norm_alpha,
                    origin="lower", aspect="auto",
                    interpolation="nearest", extent=extent,
                )
                if im_alpha is None:
                    im_alpha = im
            else:
                gray_layer = np.where(low_signal, 0.75, np.nan)
                ax.imshow(
                    np.ma.masked_invalid(gray_layer),
                    cmap="gray", vmin=0, vmax=1,
                    origin="lower", aspect="auto",
                    interpolation="nearest", extent=extent,
                )
                im = ax.imshow(
                    np.ma.masked_invalid(np.where(low_signal, np.nan, best_mov_idx)),
                    cmap=cmap_mov_plot, norm=norm_mov,
                    origin="lower", aspect="auto",
                    interpolation="nearest", extent=extent,
                )
                if im_mov is None:
                    im_mov = im

            ax.axis("off")
            if row == 0:
                ax.set_title(season, fontsize=21, fontweight="bold")
            if col == 0:
                ax.text(-0.18, 0.5, row_labels[row], transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=18, fontweight="bold")

    if im_alpha is not None:
        fig.colorbar(im_alpha, cax=fig.add_subplot(gs[0, n_cols]),
                     ticks=np.arange(0, 1.1, 0.2)).set_label(
            "α", fontsize=20, fontweight="bold")
        fig.colorbar(im_alpha, cax=fig.add_subplot(gs[2, n_cols]),
                     ticks=np.arange(0, 1.1, 0.2)).set_label(
            "r²", fontsize=20, fontweight="bold")

    handles = [Patch(color=mov_colors[i], label=mov_names[i])
               for i in range(len(mov_names))]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(mov_names), 8), fontsize=22,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    plt.suptitle(
        f"Melhor MoV — exp{exp_n}  |  {lag_label}",
        fontsize=26, fontweight="bold",
    )

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    slug = lag_label.replace(" ", "_").replace(",", "")
    out = plots_dir / f"best_mov_{slug}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def _load_mov_stack(exp_dir, mov_filter=None,
                     nc_suffix="_predep_granular_seasonal.nc"):
    """
    Carrega e alinha os NCs merged de todos os MoVs de exp_dir.

    `nc_suffix` permite apontar para NCs com nome diferente (ex.:
    "_predep_granular_seasonal_chirps.nc" para o experimento CHIRPS).

    Retorna (ac_stack, r2_stack, mov_names, lats, lons, nc_lags), com
    ac_stack/r2_stack de shape (n_movs, n_seasons, n_lags, n_lat, n_lon) e
    latitude em ordem ascendente. mov_names é [] se nada for encontrado.
    """
    import xarray as xr

    exp_dir = Path(exp_dir)
    datasets, mov_names = [], []
    for mov_dir in sorted(exp_dir.iterdir()):
        if not mov_dir.is_dir():
            continue
        if mov_filter and mov_dir.name not in mov_filter:
            continue
        nc = mov_dir / f"{mov_dir.name}{nc_suffix}"
        if nc.exists():
            datasets.append(xr.open_dataset(nc))
            mov_names.append(mov_dir.name)

    if not datasets:
        return None, None, [], None, None, None

    aligned = xr.align(*datasets, join="outer", fill_value=np.nan)
    ac_stack = np.stack([ds["alpha_core"].values for ds in aligned])
    r2_stack = np.stack([ds["r2"].values for ds in aligned])

    ds0 = aligned[0]
    nc_lags = list(map(int, ds0.lag.values))
    lats = ds0.latitude.values
    lons = ds0.longitude.values
    for ds in aligned:
        ds.close()

    if len(lats) > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        ac_stack = ac_stack[:, :, :, ::-1, :]
        r2_stack = r2_stack[:, :, :, ::-1, :]

    return ac_stack, r2_stack, mov_names, lats, lons, nc_lags


def generate_best_mov_plots(exp_dir, lags, plots_base, mov_filter=None):
    """
    Carrega os merged NCs de todos os MoVs de exp_dir, gera os plots de melhor
    MoV por pixel (geral + um por lag) e salva um PDF consolidado.
    Saída em: plots_base/best_mov/
    """
    from PIL import Image

    exp_dir = Path(exp_dir)
    exp_n = int(exp_dir.name.replace("exp", ""))

    ac_stack, r2_stack, mov_names, lats, lons, nc_lags = _load_mov_stack(
        exp_dir, mov_filter=mov_filter
    )
    if not mov_names:
        print(f"  [AVISO] Nenhum NC merged encontrado em {exp_dir}")
        return

    print(f"  MoVs ({len(mov_names)}): {mov_names}")

    if lags is not None:
        idx = [nc_lags.index(l) for l in lags if l in nc_lags]
        display_lags = [nc_lags[i] for i in idx]
        ac_stack = ac_stack[:, :, idx]
        r2_stack = r2_stack[:, :, idx]
    else:
        display_lags = nc_lags

    plots_dir = Path(plots_base) / "best_mov"
    pngs = []

    pngs.append(plot_best_mov_granular(
        ac_stack, r2_stack, mov_names, lats, lons,
        lag_idx=None, lag_label="Geral", exp_n=exp_n, plots_dir=plots_dir,
    ))
    for li, lag in enumerate(display_lags):
        pngs.append(plot_best_mov_granular(
            ac_stack, r2_stack, mov_names, lats, lons,
            lag_idx=li, lag_label=f"Lag {lag}", exp_n=exp_n, plots_dir=plots_dir,
        ))

    pdf_path = plots_dir / f"best_mov_exp{exp_n}.pdf"
    imgs = [Image.open(str(p)).convert("RGB") for p in pngs]
    imgs[0].save(str(pdf_path), save_all=True, append_images=imgs[1:])
    for img in imgs:
        img.close()
    print(f"  PDF: {pdf_path.name}  ({len(pngs)} páginas)")


# ── PREDEP máximo por trimestre, um arquivo por estação (comparação datasets) ─

def _max_and_best_mov(slc):
    """
    slc: (n_movs, n_lags, n_lat, n_lon) — α de uma estação, todos os MoVs/lags.
    Retorna (best_val, best_mov_idx, valid, gap), todos (n_lat, n_lon): o α
    máximo por pixel (sobre MoV e lag), o índice do MoV responsável por esse
    máximo (NaN onde não há sinal em nenhum MoV/lag), a máscara de pixels
    válidos, e `gap` = diferença entre o 1º e o 2º melhor MoV (por α máximo
    sobre lag) — usado para marcar pixels "ambíguos" quando o gap é menor
    que a margem de ruído do bootstrap (ver plot_predep_max_comparison_season).
    """
    n_movs, n_lags, n_lat, n_lon = slc.shape
    best_val = np.nanmax(slc.reshape(n_movs * n_lags, n_lat, n_lon), axis=0)
    per_mov_max = np.nanmax(slc, axis=1)            # (n_movs, n_lat, n_lon)

    all_nan = np.all(np.isnan(per_mov_max), axis=0)
    best_val = np.where(all_nan, np.nan, best_val)
    best_mov_idx = np.full((n_lat, n_lon), np.nan)
    gap = np.full((n_lat, n_lon), np.nan)
    valid = ~all_nan

    filled = np.where(np.isnan(per_mov_max[:, valid]), -np.inf, per_mov_max[:, valid])
    order = np.argsort(filled, axis=0)
    best_mov_idx[valid] = order[-1].astype(float)
    sorted_vals = np.sort(filled, axis=0)
    top1 = sorted_vals[-1]
    top2 = sorted_vals[-2] if n_movs >= 2 else np.full_like(top1, -np.inf)
    gap[valid] = top1 - top2

    return best_val, best_mov_idx, valid, gap


def plot_predep_max_season(ac_stack, mov_names, lats, lons, season_idx,
                           season_name, dataset_label, plots_dir):
    """
    Uma estação, um dataset: painel esquerdo com o PREDEP (α) máximo espacial
    (máximo sobre MoV e lag), painel direito com o MoV responsável por esse
    máximo. Salva em plots_dir/predep_max_{dataset_label}_{season_name}.png.

    ac_stack: (n_movs, n_seasons, n_lags, n_lat, n_lon)
    """
    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad("white")
    cmap_mov, norm_mov, mov_colors = _make_mov_cmap(mov_names)
    cmap_mov_plot = copy.copy(cmap_mov)
    cmap_mov_plot.set_bad(color=(0, 0, 0, 0))

    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    best_val, best_mov_idx, valid, _gap = _max_and_best_mov(ac_stack[:, season_idx, :, :, :])
    low_signal = valid & (best_val < SIGNAL_THRESHOLD)

    fig, (ax_val, ax_mov) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    ax_val.set_facecolor("white")
    im_alpha = ax_val.imshow(
        np.ma.masked_invalid(best_val),
        cmap=cmap_alpha, norm=norm_alpha,
        origin="lower", aspect="auto",
        interpolation="nearest", extent=extent,
    )
    ax_val.axis("off")
    ax_val.set_title("PREDEP máximo (α)", fontsize=14, fontweight="bold")
    fig.colorbar(im_alpha, ax=ax_val, fraction=0.046, pad=0.04,
                 ticks=np.arange(0, 1.1, 0.2)).set_label("α", fontsize=12)

    ax_mov.set_facecolor("white")
    gray_layer = np.where(low_signal, 0.75, np.nan)
    ax_mov.imshow(
        np.ma.masked_invalid(gray_layer),
        cmap="gray", vmin=0, vmax=1,
        origin="lower", aspect="auto",
        interpolation="nearest", extent=extent,
    )
    ax_mov.imshow(
        np.ma.masked_invalid(np.where(low_signal, np.nan, best_mov_idx)),
        cmap=cmap_mov_plot, norm=norm_mov,
        origin="lower", aspect="auto",
        interpolation="nearest", extent=extent,
    )
    ax_mov.axis("off")
    ax_mov.set_title("MoV responsável", fontsize=14, fontweight="bold")

    handles = [Patch(color=mov_colors[i], label=mov_names[i])
               for i in range(len(mov_names))]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(mov_names), 8), fontsize=9,
               bbox_to_anchor=(0.5, -0.05), frameon=True)

    plt.suptitle(f"PREDEP máximo — {dataset_label} — {season_name}",
                 fontsize=16, fontweight="bold")

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"predep_max_{dataset_label}_{season_name}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out


def plot_predep_max_comparison_season(ac_stack_chirps, ac_stack_xavier, mov_names,
                                      lats, lons, season_idx, season_name,
                                      plots_dir, margin=None):
    """
    Uma estação, CHIRPS vs Xavier lado a lado (3x2): linha 1 = PREDEP máximo
    (α) de cada fonte + diferença absoluta; linha 2 = MoV responsável de cada
    fonte + concordância espacial.

    `margin`: se `None` (padrão), comportamento original — diferença bruta e
    concordância binária (verde = mesmo MoV, cinza = MoV diferente ou sinal
    fraco). Se um float (ex.: 0.05, calibrado empiricamente a partir do
    ruído de bootstrap não-seedado do predep() — ver
    predep-chirps-vs-xavier-weekly-comparison na memória do projeto):
      - painel de diferença: pixels com |Δα| < margin viram "não
        significativo" (cinza claro) em vez de coloridos pela escala;
      - painéis de MoV: pixels onde o gap entre o 1º e o 2º MoV é menor que
        `margin` viram "ambíguo" (tom terroso), categoria separada do cinza
        de "sem sinal";
      - concordância: 3 categorias — coincidente/não coincidente (só entre
        pixels "confiáveis" nos dois lados) e inconclusivo (sem sinal ou
        ambíguo em algum lado).

    Os dois datasets precisam ter o mesmo grid e a mesma lista/ordem de
    mov_names (ex.: saída de experiments/08_predep_parana_recompute.py, que
    usa a mesma máscara de pixels nos dois lados). Salva em
    plots_dir/predep_max_comparison_{season_name}.png (ou
    predep_max_comparison_{season_name}_margin.png se `margin` for dado).

    ac_stack_*: (n_movs, n_seasons, n_lags, n_lat, n_lon)
    """
    AMBIG_COLOR = "#c9a876"
    NONSIG_COLOR = "#d9d9d9"

    cmap_alpha, norm_alpha = _make_alpha_cmap()
    cmap_alpha.set_bad("white")
    cmap_mov, norm_mov, mov_colors = _make_mov_cmap(mov_names)
    cmap_mov_plot = copy.copy(cmap_mov)
    cmap_mov_plot.set_bad(color=(0, 0, 0, 0))
    cmap_diff = copy.copy(plt.cm.Purples)
    cmap_diff.set_bad("white")

    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    val_c, mov_c, valid_c, gap_c = _max_and_best_mov(ac_stack_chirps[:, season_idx, :, :, :])
    val_x, mov_x, valid_x, gap_x = _max_and_best_mov(ac_stack_xavier[:, season_idx, :, :, :])
    both_valid = valid_c & valid_x

    diff = np.full(val_c.shape, np.nan)
    diff[both_valid] = np.abs(val_c[both_valid] - val_x[both_valid])

    low_c = valid_c & (val_c < SIGNAL_THRESHOLD)
    low_x = valid_x & (val_x < SIGNAL_THRESHOLD)

    if margin is not None:
        ambig_c = valid_c & ~low_c & (gap_c < margin)
        ambig_x = valid_x & ~low_x & (gap_x < margin)
    else:
        ambig_c = np.zeros_like(low_c)
        ambig_x = np.zeros_like(low_x)
    confident_c = valid_c & ~low_c & ~ambig_c
    confident_x = valid_x & ~low_x & ~ambig_x

    if margin is not None:
        both_confident = confident_c & confident_x
        agree = np.full(val_c.shape, np.nan)
        agree[both_valid] = 0.0  # inconclusivo (padrão dentro do footprint)
        agree[both_confident] = np.where(
            mov_c[both_confident] == mov_x[both_confident], 2.0, 1.0
        )
        cmap_agree = ListedColormap(["#999999", "#d9534f", "#2ca02c"])
        agree_vmax = 2
        agree_handles = [
            Patch(color="#2ca02c", label="Coincidente"),
            Patch(color="#d9534f", label="Não coincidente"),
            Patch(color="#999999", label="Inconclusivo (sem sinal/ambíguo)"),
        ]
        n_conclusive = int(np.sum(both_confident))
        n_footprint = int(np.sum(both_valid))
        pct_agree = (100 * np.sum(agree == 2.0) / n_conclusive) if n_conclusive else float("nan")
        print(f"  [{season_name}] margem={margin}: {n_conclusive}/{n_footprint} pixels "
              f"conclusivos ({100*n_conclusive/n_footprint:.0f}%)  |  "
              f"concordância entre conclusivos: {pct_agree:.0f}%")
    else:
        agree = np.full(val_c.shape, np.nan)
        agree[both_valid] = (mov_c[both_valid] == mov_x[both_valid]).astype(float)
        agree[both_valid & (low_c | low_x)] = 0.0
        cmap_agree = ListedColormap(["#999999", "#2ca02c"])
        agree_vmax = 1
        agree_handles = [Patch(color="#2ca02c", label="Coincidente"),
                         Patch(color="#999999", label="Não Coincidente")]
    cmap_agree.set_bad("white")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    for ax, val, title in ((axes[0, 0], val_c, "CHIRPS"), (axes[0, 1], val_x, "XAVIER")):
        im = ax.imshow(
            np.ma.masked_invalid(val), cmap=cmap_alpha, norm=norm_alpha,
            origin="lower", aspect="auto", interpolation="nearest", extent=extent,
        )
        ax.set_title(f"PREDEP máximo (α) — {title}", fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8, ticks=np.arange(0, 1.1, 0.2)).set_label("α", fontsize=10)

    ax = axes[0, 2]
    if margin is not None:
        nonsig = both_valid & (diff < margin)
        nonsig_layer = np.where(nonsig, 1.0, np.nan)
        ax.imshow(
            np.ma.masked_invalid(nonsig_layer),
            cmap=ListedColormap([NONSIG_COLOR]), vmin=0, vmax=1,
            origin="lower", aspect="auto", interpolation="nearest", extent=extent,
        )
        diff_display = np.where(nonsig, np.nan, diff)
    else:
        diff_display = diff
    im = ax.imshow(
        np.ma.masked_invalid(diff_display), cmap=cmap_diff, vmin=0, vmax=1,
        origin="lower", aspect="auto", interpolation="nearest", extent=extent,
    )
    title_suffix = f"  (cinza = Δ < {margin})" if margin is not None else ""
    ax.set_title(f"Diferença PREDEP máximo (α) (|CHIRPS-XAVIER|){title_suffix}",
                fontsize=12 if margin is not None else 13, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("|Δα|", fontsize=10)

    for ax, mov_idx, low, ambig, title in (
        (axes[1, 0], mov_c, low_c, ambig_c, "CHIRPS"),
        (axes[1, 1], mov_x, low_x, ambig_x, "XAVIER"),
    ):
        gray_layer = np.where(low, 0.75, np.nan)
        ax.imshow(
            np.ma.masked_invalid(gray_layer), cmap="gray", vmin=0, vmax=1,
            origin="lower", aspect="auto", interpolation="nearest", extent=extent,
        )
        if margin is not None:
            ambig_layer = np.where(ambig, 1.0, np.nan)
            ax.imshow(
                np.ma.masked_invalid(ambig_layer),
                cmap=ListedColormap([AMBIG_COLOR]), vmin=0, vmax=1,
                origin="lower", aspect="auto", interpolation="nearest", extent=extent,
            )
        hide = low | ambig
        ax.imshow(
            np.ma.masked_invalid(np.where(hide, np.nan, mov_idx)),
            cmap=cmap_mov_plot, norm=norm_mov,
            origin="lower", aspect="auto", interpolation="nearest", extent=extent,
        )
        ax.set_title(f"MoV responsável — {title}", fontsize=13, fontweight="bold")
        ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(
        np.ma.masked_invalid(agree), cmap=cmap_agree, vmin=0, vmax=agree_vmax,
        origin="lower", aspect="auto", interpolation="nearest", extent=extent,
    )
    ax.set_title("Concordância de MoVs", fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.legend(handles=agree_handles, loc="upper right", fontsize=8, framealpha=0.9)

    mov_handles = [Patch(color=mov_colors[i], label=mov_names[i])
                  for i in range(len(mov_names))]
    if margin is not None:
        mov_handles += [
            Patch(color=AMBIG_COLOR, label=f"Ambíguo (Δ MoV < {margin})"),
        ]
    fig.legend(handles=mov_handles, loc="lower center",
               ncol=min(len(mov_handles), 12), fontsize=9,
               bbox_to_anchor=(0.5, 0.0), frameon=True)

    title_note = f"  —  margem de ruído = {margin}" if margin is not None else ""
    fig.suptitle(
        f"Comparação PREDEP máximo e MoVs Responsáveis — CHIRPS vs XAVIER — {season_name}{title_note}",
        fontsize=16 if margin is not None else 17, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    suffix = "_margin" if margin is not None else ""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"predep_max_comparison_{season_name}{suffix}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")
    return out
