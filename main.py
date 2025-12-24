"""
Script principal para análise PREDEP entre índices MoV e precipitação sazonal.
Implementa teste inicial e estrutura para execução paralela completa.
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
        
        # Extrair precipitação sazonal (assumindo dados trimestrais)
        season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
        season_idx = season_map.get(season, 1)
        precip_indices = list(range(season_idx, len(precip_data.time), 4))
        precip_values = precip_data.isel(time=precip_indices).mean(dim=['latitude', 'longitude']).pr.values
        
        # Aplicar lag temporal
        if lag <= len(mov_values) - len(precip_values):
            mov_lagged = mov_values[lag:lag+len(precip_values)]
            
            # Verificar se há variabilidade suficiente nos dados
            if len(np.unique(mov_lagged)) < 3 or len(np.unique(precip_values)) < 3:
                print(f"Dados insuficientes para lag {lag} - variabilidade baixa")
                return None
                
            alpha = predep(mov_lagged, precip_values, n_boot=n_boot)
            return {'mov_index': mov_index_name, 'season': season, 'lag': lag, 'predep': alpha, 'n_boot': n_boot}
        else:
            return None
    except Exception as e:
        print(f"Erro no cálculo lag {lag}: {str(e)[:100]}...")
        return None


def run_initial_test():
    """Executa teste inicial: ATL3 vs MAM com n_boot 10k e 100k."""
    print("=== TESTE INICIAL ===")
    print("MoV: ATL3, Estação: MAM, Lags: 0-12")
    
    # Configurações do teste inicial
    mov_index_name = "ATL3"
    season = "MAM"
    n_boot_tests = [10000, 100000]
    
    # Carregar dados
    print("Carregando dados...")
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
    results = []
    
    for n_boot in n_boot_tests:
        print(f"\nTestando n_boot={n_boot:,}")
        start_time = time.time()
        
        # Execução paralela dos lags
        lag_results = Parallel(n_jobs=-1)(
            delayed(calculate_predep_lag)(mov_data, precip_data, mov_index_name, season, lag, n_boot)
            for lag in range(13)
        )
        
        # Filtrar resultados válidos
        valid_results = [r for r in lag_results if r is not None]
        results.extend(valid_results)
        
        elapsed = time.time() - start_time
        print(f"Concluído em {elapsed:.1f}s - {len(valid_results)} lags calculados")
        
        # Mostrar resultados
        for r in valid_results:
            print(f"  Lag {r['lag']}: α={float(r['predep']):.4f}")
    
    # Salvar resultados do teste inicial
    if results:
        results_df = pd.DataFrame(results)
        
        # Criar dataset NetCDF
        test_ds = xr.Dataset()
        
        for n_boot in n_boot_tests:
            subset = results_df[results_df['n_boot'] == n_boot]
            if len(subset) > 0:
                var_name = f'predep_{n_boot//1000}k'
                test_ds[var_name] = (['lag'], subset.sort_values('lag')['predep'].values)
        
        test_ds['lag'] = (['lag'], range(13))
        test_ds.attrs['mov_index'] = mov_index_name
        test_ds.attrs['season'] = season
        test_ds.attrs['description'] = f'PREDEP initial test: {mov_index_name} vs {season} precipitation'
        
        test_ds.to_netcdf("predep_initial_test.nc")
        print(f"\nResultados salvos em predep_initial_test.nc")
        
        return results_df
    else:
        print("\nNenhum resultado válido obtido no teste inicial")
        return pd.DataFrame()


def run_full_analysis():
    """Executa análise completa para todos MoVs e estações."""
    print("\n=== ANÁLISE COMPLETA ===")
    
    # Configurações da análise completa
    n_boot = 10000
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Carregar dados
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
    # Identificar colunas MoV (excluir colunas de data)
    mov_indices = [col for col in mov_data.columns if col not in ['date', 'time', 'year', 'month']]
    
    print(f"MoVs encontrados: {mov_indices}")
    print(f"Estações: {seasons}")
    print(f"Total de cálculos: {len(mov_indices)} × {len(seasons)} × 13 lags = {len(mov_indices) * len(seasons) * 13}")
    
    # Preparar lista de tarefas
    tasks = []
    for mov_idx in mov_indices:
        for season in seasons:
            for lag in range(13):
                tasks.append((mov_data, precip_data, mov_idx, season, lag, n_boot))
    
    print(f"\nExecutando {len(tasks)} cálculos em paralelo...")
    start_time = time.time()
    
    # Execução paralela completa
    all_results = Parallel(n_jobs=-1)(
        delayed(calculate_predep_lag)(*task) for task in tasks
    )
    
    # Filtrar resultados válidos
    valid_results = [r for r in all_results if r is not None]
    elapsed = time.time() - start_time
    
    print(f"Análise completa concluída em {elapsed:.1f}s")
    print(f"Resultados válidos: {len(valid_results)} de {len(tasks)}")
    
    # Salvar resultados completos
    results_df = pd.DataFrame(valid_results)
    results_df.to_csv("predep_full_results.csv", index=False)
    
    # Criar dataset NetCDF estruturado
    full_ds = xr.Dataset()
    
    for mov_idx in mov_indices:
        for season in seasons:
            subset = results_df[(results_df['mov_index'] == mov_idx) & (results_df['season'] == season)]
            if len(subset) > 0:
                var_name = f'{mov_idx}_{season}'
                full_ds[var_name] = (['lag'], subset.sort_values('lag')['predep'].values)
    
    full_ds['lag'] = (['lag'], range(13))
    full_ds.attrs['description'] = 'PREDEP analysis: All MoVs vs seasonal precipitation'
    full_ds.attrs['n_boot'] = n_boot
    
    full_ds.to_netcdf("predep_full_analysis.nc")
    print("Resultados completos salvos em predep_full_analysis.nc e predep_full_results.csv")
    
    return results_df


def main():
    """Função principal - executa teste inicial e opcionalmente análise completa."""
    
    # Executar teste inicial
    test_results = run_initial_test()
    
    # Perguntar se deve continuar com análise completa
    print("\n" + "="*50)
    response = input("Executar análise completa para todos MoVs e estações? (s/n): ")
    
    if response.lower() in ['s', 'sim', 'y', 'yes']:
        full_results = run_full_analysis()
        return test_results, full_results
    else:
        print("Análise limitada ao teste inicial.")
        return test_results


if __name__ == "__main__":
    main()