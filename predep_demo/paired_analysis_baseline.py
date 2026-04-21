import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Caminhos
NC_PATH = Path("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/data/ERA5_INMET_Paired_ROI_4_Stations.nc")
OUTPUT_DIR = Path("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/05_analysis/plots/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    logger.info(f"Carregando dados de {NC_PATH}...")
    ds = xr.open_dataset(NC_PATH)
    # Converter para dataframe planificado (stacking time e station)
    df = ds.to_dataframe().reset_index()
    return ds, df

def plot_correlation_heatmap(df):
    logger.info("Gerando heatmap de correlação...")
    # Selecionar apenas colunas numéricas de dados (covariáveis + preditor)
    data_cols = [
        'ws_mean', 'ws_max', 'ws_std', 'ws_p90', 'sin_dir_mean', 'cos_dir_mean', 
        'msl_mean', 'msl_min', 'msl_range', 't2m_mean', 't2m_max', 't2m_min', 
        't2m_range', 'rh_mean', 'rh_p10', 'td_dep_mean', 'tp_sum', 'tp_max', 
        'd_msl', 'tp_lag1', 'tp_lag2', 'tp_roll3', 'rajada_max_final_ms'
    ]
    
    # Filtrar colunas que existem no df
    cols = [c for c in data_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Adicionar labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticklabels(cols)
    
    # Adicionar colorbar
    plt.colorbar(im, ax=ax, label='Correlação de Pearson')
    plt.title("Matriz de Correlação: Covariáveis vs Rajada Máxima")
    
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    return corr

def plot_scatter_top_features(df, corr):
    logger.info("Gerando scatter plots para as principais correlações...")
    predictor = 'rajada_max_final_ms'
    # Pegar correlações absolutas com o preditor
    top_corr = corr[predictor].abs().sort_values(ascending=False)
    # Ignorar o próprio preditor e pegar o top 4
    top_features = top_corr.index[1:5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        # Amostrar se o dado for muito grande para o scatter (opcional, aqui deve estar ok)
        sample_df = df.dropna(subset=[feature, predictor])
        if len(sample_df) > 5000:
            sample_df = sample_df.sample(5000)
            
        ax.scatter(sample_df[feature], sample_df[predictor], alpha=0.3, s=10)
        ax.set_xlabel(feature)
        ax.set_ylabel(predictor)
        ax.set_title(f"Correlação: {corr.loc[feature, predictor]:.2f}")
        ax.grid(alpha=0.3)
        
    plt.suptitle("Comportamento das Principais Covariáveis vs Rajada Máxima")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "scatter_top_features.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_station_map(ds):
    logger.info("Gerando mapa com a localização das estações...")
    
    # Coordenadas únicas das estações
    # Assumindo que station_lat e station_lon são coordenadas fixas por estação
    unique_stations = []
    for s in ds.station.values:
        sel = ds.sel(station=s).isel(time=0)
        unique_stations.append({
            'station': s,
            'lat': float(sel.station_lat.values),
            'lon': float(sel.station_lon.values)
        })
    df_geo = pd.DataFrame(unique_stations)
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Adicionar feições geográficas
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle='--', alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    
    # Plotar estações
    ax.scatter(df_geo.lon, df_geo.lat, color='red', s=100, marker='^', 
               edgecolor='black', label='Estações INMET', transform=ccrs.PlateCarree(), zorder=5)
    
    # Adicionar labels para as estações
    for _, row in df_geo.iterrows():
        ax.text(row.lon + 0.1, row.lat + 0.1, str(row.station), 
                transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold')
    
    # Definir limites do mapa com margem
    lat_min, lat_max = df_geo.lat.min() - 2, df_geo.lat.max() + 2
    lon_min, lon_max = df_geo.lon.min() - 2, df_geo.lon.max() + 2
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    plt.title("Localização das 4 Estações do Dataset (ROI)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "map_stations_location.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    try:
        ds, df = load_data()
        
        # 1. Heatmap de correlação
        corr = plot_correlation_heatmap(df)
        
        # 2. Scatter plots das top features
        plot_scatter_top_features(df, corr)
        
        # 3. Mapa das estações
        plot_station_map(ds)
        
        logger.info(f"Análise concluída com sucesso! Gráficos salvos em: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
