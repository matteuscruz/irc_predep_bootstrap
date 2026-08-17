"""
Utility functions for PREDEP paper figure generation.
"""
import numpy as np
import netCDF4 as nc4
from pathlib import Path
import matplotlib as mpl

# Configuração global de fonte para publication-quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'

# Aumentar tamanho das fontes
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABELS = {"DJF": "DJF (Dec–Feb)", "MAM": "MAM (Mar–May)",
                 "JJA": "JJA (Jun–Aug)", "SON": "SON (Sep–Nov)"}
BRAZIL_EXTENT = [-74, -34, -34, 6]

# Basin labels for overlay numbering
BASIN_LABELS = {
    "amazonica": 1, "atlantico_leste": 2, "atlantico_nordeste": 3,
    "atlantico_nordeste_oriental": 4, "atlantico_sudeste": 5,
    "atlantico_sul": 6, "paraguai": 7, "parana": 8,
    "parnaiba": 9, "sao_francisco": 10, "tocantins": 11, "uruguai": 12,
}


def load_mov_data(exp_dir, mov_name):
    """Load merged NC for a single MoV. Returns dict with arrays."""
    nc_file = Path(exp_dir) / mov_name / f"{mov_name}_predep_granular_seasonal.nc"
    if not nc_file.exists():
        return None
    with nc4.Dataset(str(nc_file)) as ds:
        ac = np.ma.filled(ds.variables["alpha_core"][:], np.nan).astype(float)
        r2 = np.ma.filled(ds.variables["r2"][:], np.nan).astype(float)
        lats = np.array(ds.variables["latitude"][:])
        lons = np.array(ds.variables["longitude"][:])
        seasons = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in ds.variables["season"][:]]
        lags = list(map(int, ds.variables["lag"][:]))
    # Reorder seasons to DJF, MAM, JJA, SON
    idx = [seasons.index(s) for s in SEASONS if s in seasons]
    ac, r2 = ac[idx], r2[idx]
    # Ensure lats ascending
    if len(lats) > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        ac = ac[:, :, ::-1, :]
        r2 = r2[:, :, ::-1, :]
    return {"alpha_core": ac, "r2": r2, "lats": lats, "lons": lons,
            "seasons": SEASONS, "lags": lags, "mov": mov_name}


def load_all_movs(exp_dir, mov_list=None):
    """Load all MoVs from experiment directory."""
    exp_dir = Path(exp_dir)
    results = {}
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or d.name == "plots":
            continue
        if mov_list and d.name not in mov_list:
            continue
        data = load_mov_data(exp_dir, d.name)
        if data is not None:
            results[d.name] = data
    return results


def load_basin_outlines(exp_dir, mov_name):
    """Load basin pixel masks from individual basin NCs for contour overlay."""
    from scipy.ndimage import binary_fill_holes
    mov_dir = Path(exp_dir) / mov_name
    outlines = []
    for basin_name, number in BASIN_LABELS.items():
        fname = f"{mov_name}_{basin_name}_predep_granular_seasonal.nc"
        nc_file = mov_dir / fname
        if not nc_file.exists():
            continue
        with nc4.Dataset(str(nc_file)) as ds:
            blats = np.array(ds.variables["latitude"][:])
            blons = np.array(ds.variables["longitude"][:])
            ac = np.ma.filled(ds.variables["alpha_core"][:], np.nan)
        mask = ~np.all(np.isnan(ac), axis=(0, 1))
        if len(blats) > 1 and blats[0] > blats[-1]:
            blats = blats[::-1]
            mask = mask[::-1, :]
        outlines.append((number, basin_name, blats, blons, mask))
    return outlines


def draw_basin_outlines(ax, basin_outlines, fontsize=6):
    """Draw basin contours and labels on an axis."""
    return  # Removido a pedido do usuário (não desenha bacias)
    
    from scipy.ndimage import binary_fill_holes
    for number, name, blats, blons, mask in basin_outlines:
        if not mask.any():
            continue
        filled = binary_fill_holes(mask)
        try:
            ax.contour(blons, blats, filled.astype(float),
                       levels=[0.5], colors="#333333", linewidths=0.5)
        except Exception:
            pass
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        clat = float(blats[int(rows.mean())])
        clon = float(blons[int(cols.mean())])
        ax.text(clon, clat, str(number), ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          alpha=0.7, edgecolor="none"))


def make_alpha_cmap():
    """Publication-quality colormap for α ∈ [0, 1]."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    gray = (0.82, 0.82, 0.82)
    light = (1.0, 0.85, 0.62)
    mid = (0.95, 0.55, 0.15)
    dark = (0.75, 0.15, 0.0)
    n_bins = 10
    colors = [gray]  # bin 0: [0, 0.1]
    for i in range(n_bins - 1):
        t = i / (n_bins - 2)
        if t < 0.5:
            s = t * 2
            c = tuple(light[j] + s * (mid[j] - light[j]) for j in range(3))
        else:
            s = (t - 0.5) * 2
            c = tuple(mid[j] + s * (dark[j] - mid[j]) for j in range(3))
        colors.append(c)
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = ListedColormap(colors, name="alpha_paper")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm, levels


def make_lag_cmap(lags):
    """Colormap for lag values."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    n = len(lags)
    colors = [plt.cm.YlOrBr(0.2 + 0.7 * i / max(n - 1, 1)) for i in range(n)]
    cmap = ListedColormap(colors, name="lag_paper")
    arr = np.array(lags, dtype=float)
    if n == 1:
        boundaries = [arr[0] - 0.5, arr[0] + 0.5]
    else:
        mids = (arr[:-1] + arr[1:]) / 2
        boundaries = np.concatenate([[arr[0] - 0.5], mids, [arr[-1] + 0.5]])
    norm = BoundaryNorm(boundaries, ncolors=n, clip=True)
    return cmap, norm


def make_mov_cmap(mov_names):
    """Categorical colormap for MoV identity maps."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.pyplot as plt
    tab = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    colors = [tab[i % len(tab)] for i in range(len(mov_names))]
    cmap = ListedColormap(colors, name="mov_paper")
    norm = BoundaryNorm(np.arange(-0.5, len(mov_names) + 0.5), ncolors=len(mov_names))
    return cmap, norm, colors


def setup_geo_ax(fig, gs_pos, extent=None):
    """Create a standard axes without Cartopy to avoid segfaults."""
    ax = fig.add_subplot(gs_pos)
    if extent is None:
        extent = BRAZIL_EXTENT
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # Disable ticks for a cleaner map look
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
