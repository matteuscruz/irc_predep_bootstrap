"""
Funções de visualização PREDEP sem cartopy (evita segmentation fault)
Segue especificações de plot_1.md e plot_2.md
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec


def plot_4x4_optimal_lags(ds, mov_index, output_dir, seasons=['DJF', 'MAM', 'JJA', 'SON']):
    """
    Plot 4x4: α máximo + 3 maiores lags.
    Conforme especificações de plot_1.md.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset com predep_alpha[mov, season, lag, lat, lon]
    mov_index : str
        Nome do MoV
    output_dir : Path
        Diretório de saída
    seasons : list
        Lista de estações a plotar
    """
    # Filtrar estações com dados válidos
    valid_seasons = []
    for season_idx, season in enumerate(seasons):
        alpha_data = ds['predep_alpha'].isel(mov=0, season=season_idx)
        if not np.all(np.isnan(alpha_data.values)):
            valid_seasons.append((season_idx, season))
    
    if not valid_seasons:
        print(f"   ⚠️  Nenhuma estação com dados válidos")
        return None
    
    n_cols = len(valid_seasons)
    fig = plt.figure(figsize=(5*n_cols, 22))
    gs = GridSpec(4, n_cols+1, figure=fig, width_ratios=[1]*n_cols + [0.05],
                  hspace=0.25, wspace=0.15, left=0.05, right=0.95, 
                  top=0.94, bottom=0.05)
    
    plt.rcParams['text.usetex'] = False
    
    im_alpha = None
    im_lag = None
    
    for col_idx, (season_idx, season) in enumerate(valid_seasons):
        # Processar dados
        alpha_data = ds['predep_alpha'].isel(mov=0, season=season_idx)
        alpha_values = alpha_data.values
        
        # Manter NaNs
        sorted_indices = np.argsort(np.nan_to_num(alpha_values, nan=-999), axis=0)
        lag_1st = sorted_indices[-1, :, :]
        lag_2nd = sorted_indices[-2, :, :]
        lag_3rd = sorted_indices[-3, :, :]
        
        lat_idx, lon_idx = np.meshgrid(
            np.arange(alpha_values.shape[1]),
            np.arange(alpha_values.shape[2]),
            indexing='ij'
        )
        alpha_max = alpha_values[lag_1st, lat_idx, lon_idx]
        
        # Mascarar onde todos os lags são NaN
        all_nan = np.all(np.isnan(alpha_values), axis=0)
        alpha_max = np.where(all_nan, np.nan, alpha_max)
        lag_1st = np.where(all_nan, np.nan, lag_1st.astype(float))
        lag_2nd = np.where(all_nan, np.nan, lag_2nd.astype(float))
        lag_3rd = np.where(all_nan, np.nan, lag_3rd.astype(float))
        
        # Linha 1: α máximo (jet discreto)
        ax1 = fig.add_subplot(gs[0, col_idx])
        levels_alpha = np.arange(0.0, 1.1, 0.1)
        norm_alpha = BoundaryNorm(levels_alpha, ncolors=plt.cm.jet.N, clip=True)
        im1 = ax1.imshow(alpha_max, cmap='jet', norm=norm_alpha, 
                         aspect='auto', origin='lower', interpolation='nearest')
        ax1.set_title(f'{season} - $\\alpha$ máximo', fontsize=12, fontweight='bold')
        ax1.axis('off')
        if im_alpha is None:
            im_alpha = im1
        
        # Linha 2: Lag 1º (jet discreto com BoundaryNorm)
        ax2 = fig.add_subplot(gs[1, col_idx])
        lag_boundaries = np.arange(-0.5, 13.5, 1.0)
        lag_norm = BoundaryNorm(lag_boundaries, ncolors=plt.cm.jet.N, clip=True)
        im2 = ax2.imshow(lag_1st, cmap='jet', norm=lag_norm, 
                         aspect='auto', origin='lower', interpolation='nearest')
        ax2.set_title(f'{season} - Lag $1^{{\\circ}}$', fontsize=12, fontweight='bold')
        ax2.axis('off')
        if im_lag is None:
            im_lag = im2
        
        # Linha 3: Lag 2º
        ax3 = fig.add_subplot(gs[2, col_idx])
        ax3.imshow(lag_2nd, cmap='jet', norm=lag_norm, 
                   aspect='auto', origin='lower', interpolation='nearest')
        ax3.set_title(f'{season} - Lag $2^{{\\circ}}$', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Linha 4: Lag 3º
        ax4 = fig.add_subplot(gs[3, col_idx])
        ax4.imshow(lag_3rd, cmap='jet', norm=lag_norm, 
                   aspect='auto', origin='lower', interpolation='nearest')
        ax4.set_title(f'{season} - Lag $3^{{\\circ}}$', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    # Colorbars
    cbar_ax1 = fig.add_subplot(gs[0, n_cols])
    cbar1 = fig.colorbar(im_alpha, cax=cbar_ax1, ticks=np.arange(0.0, 1.1, 0.1))
    cbar1.set_label('$\\alpha$', fontsize=12, fontweight='bold')
    
    cbar_ax2 = fig.add_subplot(gs[1:4, n_cols])
    cbar2 = fig.colorbar(im_lag, cax=cbar_ax2, ticks=list(range(13)))
    cbar2.set_label('Lag (meses)', fontsize=12, fontweight='bold')
    
    # Título
    plt.suptitle(
        f'Lags Ótimos por Estação: {mov_index}\\n'
        f'Linha 1: $\\alpha$ máximo | Linhas 2-4: Lags do $1^{{\\circ}}$, $2^{{\\circ}}$ e $3^{{\\circ}}$ maiores $\\alpha$',
        fontsize=18, fontweight='bold'
    )
    
    output_file = output_dir / f"optimal_lag_seasons_{mov_index}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def plot_13x4_all_lags(ds, mov_index, output_dir, seasons=['DJF', 'MAM', 'JJA', 'SON']):
    """
    Plot 13x4: Todos os lags.
    Conforme especificações de plot_2.md.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset com predep_alpha[mov, season, lag, lat, lon]
    mov_index : str
        Nome do MoV
    output_dir : Path
        Diretório de saída
    seasons : list
        Lista de estações a plotar
    """
    # Filtrar estações com dados válidos
    valid_seasons = []
    for season_idx, season in enumerate(seasons):
        alpha_data = ds['predep_alpha'].isel(mov=0, season=season_idx)
        if not np.all(np.isnan(alpha_data.values)):
            valid_seasons.append((season_idx, season))
    
    if not valid_seasons:
        print(f"   ⚠️  Nenhuma estação com dados válidos")
        return None
    
    n_cols = len(valid_seasons)
    fig = plt.figure(figsize=(3*n_cols, 30))
    gs = GridSpec(13, n_cols+1, figure=fig, width_ratios=[1]*n_cols + [0.05],
                  hspace=0.05, wspace=0.05,
                  left=0.08, right=0.95, top=0.96, bottom=0.02)
    
    plt.rcParams['text.usetex'] = False
    
    # Colormap jet discretizado
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = plt.cm.jet
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    n_lags = ds.dims['lag']
    im = None
    for lag in range(n_lags):
        for col_idx, (season_idx, season) in enumerate(valid_seasons):
            ax = fig.add_subplot(gs[lag, col_idx])
            data = ds['predep_alpha'].isel(mov=0, season=season_idx, lag=lag).values
            
            im = ax.imshow(data, cmap=cmap, norm=norm, 
                          aspect='auto', origin='lower', interpolation='nearest')
            ax.axis('off')
            
            # Títulos das colunas apenas na primeira linha
            if lag == 0:
                ax.set_title(season, fontsize=11, fontweight='bold', pad=3)
            
            # Labels das linhas apenas na primeira coluna
            if col_idx == 0:
                ax.text(-0.15, 0.5, f'Lag {lag}', 
                       transform=ax.transAxes,
                       rotation=90, va='center', ha='center',
                       fontsize=10, fontweight='bold')
    
    # Colorbar única à direita
    cbar_ax = fig.add_subplot(gs[:, n_cols])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=levels)
    cbar.set_label(
        '$\\alpha$ (Predictive Dependence)\\n0=Independente, 1=Perfeito',
        fontsize=11, fontweight='bold'
    )
    
    # Título
    plt.suptitle(
        f'PREDEP $\\alpha$: {mov_index}\\n'
        f'Linhas = Lags (0-12 meses) | Colunas = Estações (DJF, MAM, JJA, SON)',
        fontsize=14, fontweight='bold', y=0.99
    )
    
    output_file = output_dir / f"predep_13x4_grid_{mov_index}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file
