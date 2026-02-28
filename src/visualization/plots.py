"""
Funções de visualização PREDEP seguindo especificações plots.md e plot_2.md
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Configurações padrão conforme plots.md
BRAZIL_EXTENT = [-80, -34, -35, 6]
BRAZIL_EXTENT_TIGHT = [-75, -34, -34, 6]  # Conforme plot_2.md
COASTLINE_WIDTH = 0.8
BORDER_WIDTH = 0.8
STATE_WIDTH = 0.5


def setup_brazil_map(ax, extent=BRAZIL_EXTENT):
    """
    Configura mapa do Brasil conforme especificações plots.md.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo com projeção cartopy
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=COASTLINE_WIDTH, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=BORDER_WIDTH, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=STATE_WIDTH, edgecolor='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def get_alpha_colormap():
    """Retorna colormap para α conforme plots.md."""
    return plt.cm.nipy_spectral


def get_lag_colormap():
    """Retorna colormap para lags conforme plots.md."""
    return plt.cm.jet


def get_lag_norm(n_lags=13):
    """
    Retorna BoundaryNorm para lags discretos conforme plots.md.
    
    Parameters
    ----------
    n_lags : int
        Número de lags (padrão: 13 para lags 0-12)
    
    Returns
    -------
    BoundaryNorm
        Normalização com boundaries em [-0.5, 0.5, 1.5, ..., n_lags-0.5]
    """
    boundaries = np.arange(-0.5, n_lags + 0.5, 1.0)
    return BoundaryNorm(boundaries, ncolors=get_lag_colormap().N, clip=True)


def plot_alpha_map(ax, lons, lats, data, title=""):
    """
    Plota mapa de α conforme especificações plots.md.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo com projeção cartopy
    lons : array
        Longitudes
    lats : array
        Latitudes
    data : array
        Valores de α
    title : str
        Título do subplot
    
    Returns
    -------
    mappable
        Objeto para colorbar
    """
    im = ax.pcolormesh(
        lons, lats, data,
        cmap=get_alpha_colormap(),
        vmin=0.0,
        vmax=1.0,
        shading='auto',
        transform=ccrs.PlateCarree()
    )
    
    setup_brazil_map(ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return im


def plot_lag_map(ax, lons, lats, data, title="", n_lags=13):
    """
    Plota mapa de lag conforme especificações plots.md.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo com projeção cartopy
    lons : array
        Longitudes
    lats : array
        Latitudes
    data : array
        Valores de lag (inteiros 0-12)
    title : str
        Título do subplot
    n_lags : int
        Número de lags
    
    Returns
    -------
    mappable
        Objeto para colorbar
    """
    im = ax.pcolormesh(
        lons, lats, data,
        cmap=get_lag_colormap(),
        norm=get_lag_norm(n_lags),
        shading='auto',
        transform=ccrs.PlateCarree()
    )
    
    setup_brazil_map(ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return im


def create_4x4_grid(figsize=(20, 22)):
    """
    Cria grid 4x4 conforme especificações plots.md.
    
    Parameters
    ----------
    figsize : tuple
        Tamanho da figura
    
    Returns
    -------
    fig, gs
        Figura e GridSpec
    """
    from matplotlib.gridspec import GridSpec
    
    # Desabilitar LaTeX, usar mathtext
    plt.rcParams['text.usetex'] = False
    
    fig = plt.figure(figsize=figsize)
    
    # 4 linhas × 5 colunas (coluna extra para colorbars)
    gs = GridSpec(4, 5, figure=fig, 
                  width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.25, wspace=0.15, 
                  left=0.05, right=0.95, 
                  top=0.94, bottom=0.05)
    
    return fig, gs


def add_alpha_colorbar(fig, gs, im, row=0, col=4):
    """
    Adiciona colorbar para α conforme plots.md.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura
    gs : GridSpec
        GridSpec
    im : mappable
        Objeto para colorbar
    row : int
        Linha do GridSpec
    col : int
        Coluna do GridSpec
    """
    cbar_ax = fig.add_subplot(gs[row, col])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0.0, 1.1, 0.1))
    cbar.set_label('$\\alpha$', fontsize=12, fontweight='bold')


def add_lag_colorbar(fig, gs, im, rows=slice(1, 4), col=4, n_lags=13):
    """
    Adiciona colorbar para lags conforme plots.md.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura
    gs : GridSpec
        GridSpec
    im : mappable
        Objeto para colorbar
    rows : slice
        Linhas do GridSpec
    col : int
        Coluna do GridSpec
    n_lags : int
        Número de lags
    """
    cbar_ax = fig.add_subplot(gs[rows, col])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=list(range(n_lags)))
    cbar.set_label('Lag (meses)', fontsize=12, fontweight='bold')


def format_season_title(season, variable_type, rank=None):
    """
    Formata título de subplot conforme plots.md.
    
    Parameters
    ----------
    season : str
        Estação (DJF, MAM, JJA, SON)
    variable_type : str
        'alpha' ou 'lag'
    rank : int, optional
        Ranking (1, 2, 3) para lags
    
    Returns
    -------
    str
        Título formatado
    """
    if variable_type == 'alpha':
        return f'{season} - $\\alpha$ máximo'
    elif variable_type == 'lag' and rank is not None:
        return f'{season} - Lag ${rank}^{{\\circ}}$'
    else:
        return season


def format_main_title(mov_name):
    """
    Formata título principal conforme plots.md.
    
    Parameters
    ----------
    mov_name : str
        Nome do MoV
    
    Returns
    -------
    str
        Título formatado
    """
    return (
        f'Lags Ótimos por Estação: {mov_name}\\n'
        f'Linha 1: $\\alpha$ máximo | '
        f'Linhas 2-4: Lags do $1^{{\\circ}}$, $2^{{\\circ}}$ e $3^{{\\circ}}$ maiores $\\alpha$'
    )


def create_13x4_grid(figsize=(12, 30)):
    """
    Cria grid 13x4 conforme especificações plot_2.md.
    
    Parameters
    ----------
    figsize : tuple
        Tamanho da figura
    
    Returns
    -------
    fig, gs
        Figura e GridSpec
    """
    # Desabilitar LaTeX, usar mathtext
    plt.rcParams['text.usetex'] = False
    
    fig = plt.figure(figsize=figsize)
    
    # 13 linhas × 5 colunas (coluna extra para colorbar)
    gs = GridSpec(13, 5, figure=fig, 
                  width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.05, wspace=0.05,
                  left=0.08, right=0.95, 
                  top=0.96, bottom=0.02)
    
    return fig, gs


def get_discrete_jet_norm():
    """
    Retorna colormap jet e normalização discreta conforme plot_2.md.
    
    Returns
    -------
    cmap, norm
        Colormap e BoundaryNorm com bins de 0.1
    """
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = plt.cm.jet
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm
