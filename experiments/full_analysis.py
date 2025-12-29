"""
Experimento: Análise completa de todos MoVs vs todas estações sazonais.
"""

import pandas as pd
import xarray as xr
import numpy as np
from src.predep_calculation.core import predep
from joblib import Parallel, delayed
import time


def calculate_predep_lag(mov_data, precip_data, mov_index_name, season, lag, n_boot):
    """Calcula PREDEP para um lag específico."""
    try:
        mov_values = mov_data[mov_index_name].dropna().values
        
        season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
        season_idx = season_map.get(season, 1)
        precip_indices = list(range(season_idx, len(precip_data.time), 4))
        precip_values = precip_data.isel(time=precip_indices).mean(dim=['latitude', 'longitude']).pr.values
        
        if lag <= len(mov_values) - len(precip_values):
            mov_lagged = mov_values[lag:lag+len(precip_values)]
            
            if len(np.unique(mov_lagged)) < 3 or len(np.unique(precip_values)) < 3:
                return None
                
            alpha = predep(mov_lagged, precip_values, n_boot=n_boot)
            return {'mov_index': mov_index_name, 'season': season, 'lag': lag, 'predep': alpha}
        else:
            return None
    except Exception:
        return None


def run_full_analysis():
    """Experimento: Análise completa todos MoVs vs todas estações."""
    print("=== EXPERIMENTO: ANÁLISE COMPLETA ===")
    
    # Configurações
    n_boot = 10000
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Carregar dados
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
    # Identificar MoVs
    mov_indices = [col for col in mov_data.columns if col not in ['date', 'time', 'year', 'month']]
    
    print(f"MoVs: {mov_indices}")
    print(f"Estações: {seasons}")
    print(f"Total: {len(mov_indices)} × {len(seasons)} × 13 lags = {len(mov_indices) * len(seasons) * 13}")
    
    # Preparar tarefas
    tasks = []
    for mov_idx in mov_indices:
        for season in seasons:
            for lag in range(13):
                tasks.append((mov_data, precip_data, mov_idx, season, lag, n_boot))
    
    print(f"\nExecutando {len(tasks)} cálculos...")
    start_time = time.time()
    
    # Execução paralela
    all_results = Parallel(n_jobs=-1)(
        delayed(calculate_predep_lag)(*task) for task in tasks
    )
    
    valid_results = [r for r in all_results if r is not None]
    elapsed = time.time() - start_time
    
    print(f"Concluído em {elapsed:.1f}s")
    print(f"Resultados válidos: {len(valid_results)} de {len(tasks)}")
    
    # Salvar resultados
    results_df = pd.DataFrame(valid_results)
    results_df.to_csv("experiments/results/predep_full_results.csv", index=False)
    
    # Criar NetCDF estruturado
    full_ds = xr.Dataset()
    
    for mov_idx in mov_indices:
        for season in seasons:
            subset = results_df[(results_df['mov_index'] == mov_idx) & (results_df['season'] == season)]
            if len(subset) > 0:
                var_name = f'{mov_idx}_{season}'
                full_ds[var_name] = (['lag'], subset.sort_values('lag')['predep'].values)
    
    full_ds['lag'] = (['lag'], range(13))
    full_ds.attrs['n_boot'] = n_boot
    
    full_ds.to_netcdf("experiments/results/predep_full_analysis.nc")
    print("Resultados salvos em experiments/results/")
    
    return results_df


if __name__ == "__main__":
    run_full_analysis()