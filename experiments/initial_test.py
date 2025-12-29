"""
Experimento: Teste inicial ATL3 vs MAM com diferentes valores de n_boot.
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
        # Extrair dados do MoV
        mov_values = mov_data[mov_index_name].dropna().values
        
        # Extrair precipitação sazonal
        season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
        season_idx = season_map.get(season, 1)
        precip_indices = list(range(season_idx, len(precip_data.time), 4))
        precip_values = precip_data.isel(time=precip_indices).mean(dim=['latitude', 'longitude']).pr.values
        
        # Aplicar lag temporal
        if lag <= len(mov_values) - len(precip_values):
            mov_lagged = mov_values[lag:lag+len(precip_values)]
            
            # Verificar variabilidade
            if len(np.unique(mov_lagged)) < 3 or len(np.unique(precip_values)) < 3:
                return None
                
            alpha = predep(mov_lagged, precip_values, n_boot=n_boot)
            return {'mov_index': mov_index_name, 'season': season, 'lag': lag, 'predep': alpha, 'n_boot': n_boot}
        else:
            return None
    except Exception as e:
        print(f"Erro lag {lag}: {str(e)[:50]}...")
        return None


def run_initial_test():
    """Experimento: ATL3 vs MAM com n_boot 10k e 100k."""
    print("=== EXPERIMENTO: TESTE INICIAL ===")
    print("MoV: ATL3, Estação: MAM, Lags: 0-12")
    
    # Configurações
    mov_index_name = "ATL3"
    season = "MAM"
    n_boot_tests = [10000, 100000]
    
    # Carregar dados
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
    results = []
    
    for n_boot in n_boot_tests:
        print(f"\nTestando n_boot={n_boot:,}")
        start_time = time.time()
        
        # Execução paralela
        lag_results = Parallel(n_jobs=-1)(
            delayed(calculate_predep_lag)(mov_data, precip_data, mov_index_name, season, lag, n_boot)
            for lag in range(13)
        )
        
        valid_results = [r for r in lag_results if r is not None]
        results.extend(valid_results)
        
        elapsed = time.time() - start_time
        print(f"Concluído em {elapsed:.1f}s - {len(valid_results)} lags")
        
        for r in valid_results:
            print(f"  Lag {r['lag']}: α={float(r['predep']):.4f}")
    
    # Salvar resultados
    if results:
        results_df = pd.DataFrame(results)
        
        test_ds = xr.Dataset()
        for n_boot in n_boot_tests:
            subset = results_df[results_df['n_boot'] == n_boot]
            if len(subset) > 0:
                var_name = f'predep_{n_boot//1000}k'
                test_ds[var_name] = (['lag'], subset.sort_values('lag')['predep'].values)
        
        test_ds['lag'] = (['lag'], range(13))
        test_ds.attrs['mov_index'] = mov_index_name
        test_ds.attrs['season'] = season
        
        test_ds.to_netcdf("experiments/results/predep_initial_test.nc")
        print(f"\nResultados salvos em experiments/results/predep_initial_test.nc")
        
        return results_df
    else:
        print("\nNenhum resultado válido")
        return pd.DataFrame()


if __name__ == "__main__":
    run_initial_test()