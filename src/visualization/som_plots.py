"""
Plots ADITIVOS de SOM para resultados PREDEP.

Não substituem nada em cluster_plots.py — apenas adicionam uma camada de
síntese espacial: mapa de regimes, U-matrix, mapa de ativação, component
planes por MoV, assinatura dos regimes e tabela regime × MoV.

Reusa colormap de α, contornos de bacia e o idioma de mapa geográfico de
cluster_plots.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pathlib import Path  # noqa: E402

from src.visualization.cluster_plots import (  # noqa: E402
    _make_diff_cmap,
    _draw_basin_outlines,
)


def _regime_mov_matrix(fingerprints):
    """(n_regimes, n_mov) de α médio + anomalia vs média entre regimes."""
    n_regimes = len(fingerprints)
    mat = np.vstack([fingerprints[r]["by_mov"] for r in range(n_regimes)])
    anomaly = mat - mat.mean(axis=0, keepdims=True)
    return mat, anomaly


def _make_regime_cmap(n: int):
    """Colormap categórico para n regimes (tab10 + tab20); 'bad' transparente."""
    base = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    colors = [base[i % len(base)] for i in range(n)]
    cmap = ListedColormap(colors, name="regime_cmap")
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), ncolors=n)
    return cmap, norm, colors


def _scatter_to_grid(values, bundle):
    """Espalha um vetor (n_pixels,) de volta ao grid (n_lat, n_lon); NaN fora."""
    grid = np.full((len(bundle.lats), len(bundle.lons)), np.nan)
    rc = bundle.pixel_rc
    grid[rc[:, 0], rc[:, 1]] = values
    return grid


def _extent(bundle):
    lats, lons = bundle.lats, bundle.lons
    return [float(lons.min()), float(lons.max()),
            float(lats.min()), float(lats.max())]


def _draw_outlines_only(ax, basin_outlines):
    """Só os contornos das bacias (sem números) — para small-multiples."""
    from scipy.ndimage import binary_fill_holes
    for _number, blats, blons, mask in basin_outlines:
        if not mask.any():
            continue
        filled = binary_fill_holes(mask)
        try:
            ax.contour(blons, blats, filled.astype(float), levels=[0.5],
                       colors="#333333", linewidths=0.4)
        except Exception:
            pass


def plot_regime_map(regime_grid, bundle, fingerprints, output_dir, exp_n,
                    basin_outlines=None, ax=None, title=None):
    """Mapa geográfico do Brasil pintado pelo regime SOM de cada pixel.

    Se `ax` for passado, desenha nele (sem salvar/fechar figura — uso em
    small-multiples, ex.: plot_som_comparison) e retorna `ax`; caso
    contrário, cria a própria figura, salva em
    output_dir/som_regime_map_exp{exp_n:02d}.png e retorna o Path.
    """
    n_regimes = len(fingerprints)
    cmap, norm, colors = _make_regime_cmap(n_regimes)
    lats, lons = bundle.lats, bundle.lons
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(np.ma.masked_invalid(regime_grid), cmap=cmap, norm=norm,
              origin="lower", extent=extent, aspect="auto",
              interpolation="nearest")
    if basin_outlines:
        _draw_basin_outlines(ax, basin_outlines)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Regimes de previsibilidade (SOM) — exp{exp_n:02d}")

    # MoV "característico" = maior anomalia vs média entre regimes (ENSO domina
    # o α absoluto em todo lugar, então o argmax bruto não distingue regimes).
    _, anomaly = _regime_mov_matrix(fingerprints)
    handles = []
    for r in range(n_regimes):
        top = int(np.argmax(anomaly[r])) if anomaly.shape[1] else 0
        top_mov = bundle.mov_names[top]
        size = fingerprints[r]["size"]
        handles.append(Patch(facecolor=colors[r], edgecolor="black",
                             label=f"R{r}: {top_mov}↑ (n={size})"))
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9,
              title="MoV característico (anomalia)")

    if not standalone:
        return ax

    out = Path(output_dir) / f"som_regime_map_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_som_comparison(regime_grid_a, bundle_a, fingerprints_a, label_a,
                        regime_grid_b, bundle_b, fingerprints_b, label_b,
                        overlap_grid, top_k, output_dir, out_name,
                        basin_outlines=None, caveat=None):
    """
    Uma imagem 1x3: mapa de regimes SOM de A, de B, e a intensidade de
    concordância local entre eles (nº de MoVs em comum no top-k local de
    cada pixel — ver som_insights.local_mov_importance — normalizado por
    top_k). Regimes SOM não têm identidade compartilhada entre A e B (são
    clusters independentes), então a comparação é feita pelo MoV, não pelo
    id do regime.

    `caveat`: texto opcional exibido no rodapé da figura (ex.: ressalva
    sobre o bootstrap do predep() não ter seed fixa nos dados de entrada).
    """
    extent = _extent(bundle_a)

    fig, axes = plt.subplots(1, 3, figsize=(26, 9))

    plot_regime_map(regime_grid_a, bundle_a, fingerprints_a, output_dir, 0,
                    basin_outlines=basin_outlines, ax=axes[0],
                    title=f"Regimes de previsibilidade (SOM) — {label_a}")
    plot_regime_map(regime_grid_b, bundle_b, fingerprints_b, output_dir, 0,
                    basin_outlines=basin_outlines, ax=axes[1],
                    title=f"Regimes de previsibilidade (SOM) — {label_b}")

    ax = axes[2]
    im = ax.imshow(np.ma.masked_invalid(overlap_grid), cmap="Purples",
                   vmin=0, vmax=top_k, origin="lower", extent=extent,
                   aspect="auto", interpolation="nearest")
    if basin_outlines:
        _draw_basin_outlines(ax, basin_outlines)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Concordância local de MoVs — {label_a} vs {label_b}\n"
                 f"(nº de MoVs em comum no top-{top_k} local de cada pixel)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=range(top_k + 1))
    cbar.set_label(f"MoVs em comum (de {top_k})", fontsize=10)

    fig.suptitle(f"Comparação SOM — {label_a} vs {label_b}", fontsize=17, fontweight="bold")
    bottom_rect = 0.06 if caveat else 0.0
    fig.tight_layout(rect=[0, bottom_rect, 1, 0.95])
    if caveat:
        fig.text(0.5, 0.01, caveat, ha="center", va="bottom", fontsize=10,
                 style="italic", color="#8a4b08", wrap=True)

    out = Path(output_dir) / out_name
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_umatrix(som, output_dir, exp_n):
    """U-matrix (distância média aos vizinhos) — mostra separação entre regimes."""
    um = som.distance_map().T
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(um, cmap="bone_r", origin="lower")
    fig.colorbar(im, ax=ax, label="Distância média aos vizinhos (norm.)")
    ax.set_title(f"U-matrix do SOM — exp{exp_n:02d}")
    ax.set_xlabel("neurônio x")
    ax.set_ylabel("neurônio y")
    out = Path(output_dir) / f"som_umatrix_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_hitmap(som, X, output_dir, exp_n):
    """Mapa de ativação: nº de pixels cujo BMU é cada neurônio."""
    hits = som.activation_response(X).T
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(hits, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax, label="Nº de pixels (BMU)")
    ax.set_title(f"Mapa de ativação (hits) — exp{exp_n:02d}")
    ax.set_xlabel("neurônio x")
    ax.set_ylabel("neurônio y")
    out = Path(output_dir) / f"som_hitmap_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_component_planes(som, bundle, output_dir, exp_n, value_label="α"):
    """Um heatmap por MoV sobre a grade SOM (α médio) — qual regime cada MoV guia."""
    W = som.get_weights()  # (g, g, n_features)
    if bundle.scaler_mean is not None:
        W_alpha = W * bundle.scaler_std + bundle.scaler_mean
    else:
        W_alpha = W

    mov_ids = bundle.feature_index[:, 0]
    n_mov = len(bundle.mov_names)
    ncols = int(np.ceil(np.sqrt(n_mov)))
    nrows = int(np.ceil(n_mov / ncols))

    # α médio (sobre lag×estação) de cada MoV em cada neurônio. Os valores são
    # pequenos (~0.05-0.15); escala sequencial compartilhada dos dados torna as
    # diferenças entre MoVs comparáveis.
    planes = [W_alpha[:, :, np.where(mov_ids == m)[0]].mean(axis=2).T
              for m in range(n_mov)]
    vmin = float(min(p.min() for p in planes))
    vmax = float(max(p.max() for p in planes))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for m in range(n_mov):
        ax = axes[m]
        im = ax.imshow(planes[m], cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_title(bundle.mov_names[m], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n_mov, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Component planes por MoV ({value_label} médio na grade SOM) — exp{exp_n:02d}",
        fontsize=12,
    )
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=value_label)
    out = Path(output_dir) / f"som_component_planes_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_regime_fingerprints(fingerprints, bundle, output_dir, exp_n, top_k=8,
                             value_label="α"):
    """Por regime: α médio por MoV (top-k), por lag e por estação."""
    n_regimes = len(fingerprints)
    _, _, colors = _make_regime_cmap(n_regimes)
    mov_names = bundle.mov_names
    lags = bundle.lags
    seasons = bundle.seasons

    fig, axes = plt.subplots(n_regimes, 3, figsize=(15, 2.6 * n_regimes),
                             squeeze=False)
    for r in range(n_regimes):
        fp = fingerprints[r]
        color = colors[r]

        by_mov = fp["by_mov"]
        order = np.argsort(by_mov)[::-1][:top_k]
        ax = axes[r][0]
        ax.bar(range(len(order)), by_mov[order], color=color)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([mov_names[i] for i in order], rotation=45,
                           ha="right", fontsize=7)
        ax.set_ylabel(f"R{r} (n={fp['size']})\n{value_label} médio", fontsize=8)
        if r == 0:
            ax.set_title(f"Top {top_k} MoVs", fontsize=9)

        ax = axes[r][1]
        ax.bar([str(lg) for lg in lags], fp["by_lag"], color=color)
        if r == 0:
            ax.set_title(f"{value_label} médio por lag", fontsize=9)
        if r == n_regimes - 1:
            ax.set_xlabel("lag (meses)")

        ax = axes[r][2]
        ax.bar(seasons, fp["by_season"], color=color)
        if r == 0:
            ax.set_title(f"{value_label} médio por estação", fontsize=9)

    fig.suptitle(f"Assinatura dos regimes (SOM) — exp{exp_n:02d}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(output_dir) / f"som_regime_fingerprints_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_regime_mov_heatmap(fingerprints, bundle, output_dir, exp_n,
                            value_label="α"):
    """Anomalia de α por regime × MoV (vs média entre regimes)."""
    n_regimes = len(fingerprints)
    n_mov = len(bundle.mov_names)
    _, anomaly = _regime_mov_matrix(fingerprints)
    lim = float(np.nanmax(np.abs(anomaly))) or 1e-6
    cmap, norm = _make_diff_cmap(vmin=-lim, vmax=lim)

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * n_mov), max(4, 0.6 * n_regimes))
    )
    im = ax.imshow(anomaly, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_mov))
    ax.set_xticklabels(bundle.mov_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels([f"R{r}" for r in range(n_regimes)])
    for r in range(n_regimes):
        for m in range(n_mov):
            ax.text(m, r, f"{anomaly[r, m]:+.02f}", ha="center", va="center",
                    fontsize=5, color="black")
    fig.colorbar(im, ax=ax, label=f"{value_label} (anomalia vs média entre regimes)",
                 fraction=0.025, pad=0.02)
    ax.set_title(f"Anomalia de {value_label} por regime × MoV — exp{exp_n:02d}")
    out = Path(output_dir) / f"som_regime_mov_table_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


# --- Versões geográficas (mesma projeção de plot_regime_map) ----------------

def plot_geographic_component_planes(bundle, output_dir, exp_n,
                                     basin_outlines=None, value_label="α"):
    """1 mapa do Brasil por MoV: α médio por pixel (sobre lag×estação)."""
    mov_ids = bundle.feature_index[:, 0]
    n_mov = len(bundle.mov_names)
    extent = _extent(bundle)

    grids = []
    for m in range(n_mov):
        cols = np.nonzero(mov_ids == m)[0]
        grids.append(_scatter_to_grid(bundle.X_raw[:, cols].mean(axis=1), bundle))
    vmin = float(min(np.nanmin(g) for g in grids))
    vmax = float(max(np.nanmax(g) for g in grids))

    ncols = int(np.ceil(np.sqrt(n_mov)))
    nrows = int(np.ceil(n_mov / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for m in range(n_mov):
        ax = axes[m]
        im = ax.imshow(np.ma.masked_invalid(grids[m]), cmap="YlOrRd",
                       vmin=vmin, vmax=vmax, origin="lower", extent=extent,
                       aspect="auto", interpolation="nearest")
        if basin_outlines:
            _draw_outlines_only(ax, basin_outlines)
        ax.set_title(bundle.mov_names[m], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n_mov, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Component planes geográficos — {value_label} médio por MoV — exp{exp_n:02d}",
        fontsize=12,
    )
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02, label=value_label)
    out = Path(output_dir) / f"som_geographic_component_planes_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_atypicality_map(som, regimes, bundle, output_dir, exp_n,
                         basin_outlines=None):
    """Erro de quantização por pixel (distância pixel→BMU) projetado no mapa."""
    W = som.get_weights()
    bmu = regimes.bmu_rc
    w_pix = W[bmu[:, 0], bmu[:, 1]]            # (n_pixels, n_features)
    qe = np.linalg.norm(bundle.X - w_pix, axis=1)
    grid = _scatter_to_grid(qe, bundle)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="magma_r", origin="lower",
                   extent=_extent(bundle), aspect="auto",
                   interpolation="nearest")
    if basin_outlines:
        _draw_basin_outlines(ax, basin_outlines)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Atipicidade (erro de quantização por pixel) — exp{exp_n:02d}")
    fig.colorbar(im, ax=ax, label="distância pixel → BMU (claro=típico)",
                 fraction=0.04, pad=0.02)
    out = Path(output_dir) / f"som_atypicality_map_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_boundary_map(som, regimes, bundle, output_dir, exp_n,
                      basin_outlines=None):
    """U-matrix projetada nos pixels (valor do BMU) — zonas de transição."""
    um = som.distance_map()
    bmu = regimes.bmu_rc
    vals = um[bmu[:, 0], bmu[:, 1]]
    grid = _scatter_to_grid(vals, bundle)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="bone_r", origin="lower",
                   extent=_extent(bundle), aspect="auto",
                   interpolation="nearest")
    if basin_outlines:
        _draw_basin_outlines(ax, basin_outlines)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Fronteiras entre regimes (U-matrix no mapa) — exp{exp_n:02d}")
    fig.colorbar(im, ax=ax, label="distância do BMU aos vizinhos (alto=fronteira)",
                 fraction=0.04, pad=0.02)
    out = Path(output_dir) / f"som_boundary_map_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out


def plot_regime_panels(regime_grid, bundle, fingerprints, output_dir, exp_n,
                       basin_outlines=None):
    """1 mapa do Brasil por regime, destacando sua extensão (resto em cinza)."""
    n_regimes = len(fingerprints)
    _, _, colors = _make_regime_cmap(n_regimes)
    _, anomaly = _regime_mov_matrix(fingerprints)
    extent = _extent(bundle)
    valid = ~np.isnan(regime_grid)

    ncols = int(np.ceil(np.sqrt(n_regimes)))
    nrows = int(np.ceil(n_regimes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for r in range(n_regimes):
        ax = axes[r]
        gray = np.where(valid, 0.0, np.nan)        # base cinza nos pixels válidos
        ax.imshow(np.ma.masked_invalid(gray), cmap=ListedColormap([(0.85, 0.85, 0.85)]),
                  origin="lower", extent=extent, aspect="auto",
                  interpolation="nearest")
        sel = np.where(regime_grid == r, 1.0, np.nan)
        ax.imshow(np.ma.masked_invalid(sel),
                  cmap=ListedColormap([colors[r]]), origin="lower",
                  extent=extent, aspect="auto", interpolation="nearest")
        if basin_outlines:
            _draw_outlines_only(ax, basin_outlines)
        top = int(np.argmax(anomaly[r])) if anomaly.shape[1] else 0
        ax.set_title(f"R{r}: {bundle.mov_names[top]}↑ (n={fingerprints[r]['size']})",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n_regimes, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"Extensão espacial de cada regime — exp{exp_n:02d}", fontsize=12)
    out = Path(output_dir) / f"som_regime_panels_exp{exp_n:02d}.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out.name}")
    return out
