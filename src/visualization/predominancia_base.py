"""
Primitivas compartilhadas pelas figuras da metodologia da predominância.

Existe para que `predominancia_plots` não repita mapa-base, grade e paleta em
cada figura, e para que as duas figuras usem exatamente as mesmas cores — a do
PREDEP e a do SOM são lidas lado a lado.

Os contornos vêm de `data/reference/south_america_110m.geojson`, versionado no
repo e lido com `json` puro: `geopandas.datasets` foi removido no GeoPandas 1.0
e quebra em qualquer ambiente atualizado.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
SA_BORDERS = ROOT / "data/reference/south_america_110m.geojson"

# ── tinta e mapa-base ───────────────────────────────────────────────────────
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
AZUL, RED, VERDE, CINZA = "#2a78d6", "#8a1f1e", "#1f8a70", "#b8b6b0"
BORDA_EIXO = "#d8d7d2"
GRADE = "#e8e7e2"

# Magnitude: rampa de uma matiz só. Diferença e assinatura: divergente centrada
# em zero, para "sem diferença" ler como ausência de cor.
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div_bluered", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f8c9c8", "#ef8b8a", "#d33534", "#8a1f1e"])
for _cm in (CMAP_SEQ, CMAP_DIV):
    _cm.set_bad((0, 0, 0, 0))          # NaN transparente: o mapa-base aparece

# Categóricas. MOV_PALETTE é compartilhada entre as figuras de MoV para que a
# mesma cor signifique o mesmo índice em todas elas.
MOV_PALETTE = ["#2a78d6", "#e0562a", "#1f8a70", "#8e4ec6", "#c9a227", "#d4508f",
               "#3fb0c9", "#8c5a2b", "#6b8f3a", "#b23b3b", "#5b6bbf", "#c77b1e"]
MOV_OUTROS = "#94918a"
REGIME_COLORS = ["#2a78d6", "#e0562a", "#1f8a70", "#8e4ec6", "#c9a227",
                 "#3fb0c9", "#d4508f", "#8c5a2b"]


@lru_cache(maxsize=4)
def load_borders(path: str | None = None) -> list:
    """Anéis dos países da América do Sul (Natural Earth 110m), versionados.

    Devolve lista de features, cada uma lista de arrays (n, 2) em lon/lat.
    Lista vazia se o arquivo não existir — a figura sai sem contorno em vez de
    quebrar a pipeline.
    """
    p = Path(path) if path else SA_BORDERS
    if not p.exists():
        return []
    gj = json.loads(p.read_text())
    out = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        out.append([np.asarray(r) for poly in polys for r in poly])
    return out


def to_grid(values, lat, lon):
    """Vetor por pixel → (grade 2D, extent). NaN onde não há pixel."""
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.full((len(ulat), len(ulon)), np.nan)
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = values
    return g, _extent(ulat, ulon)


def to_grid_rgba(rgba, lat, lon):
    """Cores por pixel → (grade RGBA, extent). Alfa 0 onde não há pixel."""
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.zeros((len(ulat), len(ulon), 4))
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = rgba
    return g, _extent(ulat, ulon)


def to_grid_bool(values, lat, lon):
    """Máscara por pixel → grade booleana (False onde não há pixel)."""
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.zeros((len(ulat), len(ulon)), bool)
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = values
    return g


def _extent(ulat, ulon):
    return [ulon.min() - 0.05, ulon.max() + 0.05,
            ulat.min() - 0.05, ulat.max() + 0.05]


def categorical_rgba(labels, color_of: dict, default: str):
    """Rótulos inteiros → RGBA, usando `color_of` e caindo em `default`."""
    out = np.zeros((len(labels), 4))
    for i, v in enumerate(labels):
        out[i] = matplotlib.colors.to_rgba(color_of.get(int(v), default))
    return out


def draw_basemap(ax, borders, extent, pad_x=1.5, pad_y=1.0):
    """Oceano, terra e fronteiras por trás dos dados, com folga em volta."""
    ax.set_facecolor(OCEAN)
    for rings in borders:
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                 edgecolor=LAND_EDGE, linewidth=0.7, zorder=1))
    ax.set_xlim(extent[0] - pad_x, extent[1] + pad_x)
    ax.set_ylim(extent[2] - pad_y, extent[3] + pad_y)
    ax.set_aspect("equal")


def strip_axes(ax, ticks=False):
    """Tira o excesso de moldura; `ticks=False` deixa o mapa sem coordenadas."""
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDA_EIXO)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def clean_axes(ax, grid_axis="y"):
    """Estilo dos painéis de gráfico (não-mapa): sem topo/direita, grade leve."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRADE)
        ax.set_axisbelow(True)


def caixa(ax, texto, cor=INK, x=0.03, y=0.05, fontsize=8.5, peso="semibold", **kw):
    """Rótulo dentro do mapa, com fundo branco para não sumir sobre o dado.

    `peso="normal"` para números secundários, que não devem competir com a
    métrica principal do painel.
    """
    return ax.text(x, y, texto, transform=ax.transAxes, fontsize=fontsize,
                   color=cor, fontweight=peso, zorder=6,
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                             pad=2), **kw)


def mil(n) -> str:
    """Separador de milhar pt-BR — `f"{n:,}"` produz vírgula, que aqui é decimal."""
    return f"{int(n):,}".replace(",", ".")


def fmt_lon(x, _pos=None):
    return f"{abs(x):.0f}°{'W' if x < 0 else 'E'}"


def fmt_lat(y, _pos=None):
    return f"{abs(y):.0f}°{'S' if y < 0 else 'N'}"


def open_ds(fonte):
    """Aceita um Dataset já aberto ou um caminho — a pipeline passa caminho."""
    import xarray as xr
    if hasattr(fonte, "data_vars"):
        return fonte
    p = Path(fonte)
    if not p.is_absolute():
        p = ROOT / p
    return xr.open_dataset(p)
