#!/usr/bin/env python3
"""
Análise PREDEP focada no Rio Grande do Sul
AAO vs todas as estações usando dados XAVIER
Cobertura: 46 anos (1980-2025), 552 meses, bootstrap=2000
"""

import pandas as pd
import xarray as xr
import numpy as np
from src.predep_calculation.core import predep
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization import plot_4x4_optimal_lags, plot_13x4_all_lags


def calculate_predep_point(mov_values, precip_point, lag, n_boot, season_offset):
    """Calcula PREDEP para um ponto espacial."""
    try:
        n_years = len(precip_point)
        mov_aligned = []
        
        for year in range(n_years):
            # Índice no array MoV: ano * 12 + offset da estação - lag
            idx = year * 12 + season_offset - lag
            if idx < 0 or idx >= len(mov_values):
                return np.nan
            mov_aligned.append(mov_values[idx])
        
        mov_aligned = np.array(mov_aligned)
        
        # Verificar se há variabilidade suficiente
        if len(np.unique(mov_aligned)) < 3 or len(np.unique(precip_point)) < 3:
            return np.nan
        
        # Remover NaNs
        valid_mask = ~(np.isnan(mov_aligned) | np.isnan(precip_point))
        if valid_mask.sum() < 5:  # Pelo menos 5 pontos válidos
            return np.nan
            
        return predep(mov_aligned[valid_mask], precip_point[valid_mask], n_boot=n_boot)
    except Exception as e:
        return np.nan


def run_rs_analysis():
    """Executa análise para Rio Grande do Sul."""
    # Iniciar medição de tempo
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("="*80)
    print("ANÁLISE PREDEP: RIO GRANDE DO SUL")
    print("AAO vs DJF/MAM/JJA/SON | Bootstrap: 2000")
    print(f"Início: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Carregar dados
    print("\n📥 Carregando dados...")
    mov_data = pd.read_csv("data/raw/mov_indices/AAO_indice data.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
# Filtrar Rio Grande do Sul (aproximadamente)
    rs_lat_min, rs_lat_max = -34, -27
    rs_lon_min, rs_lon_max = -58, -49

    precip_rs = precip_data.sel(
        latitude=slice(rs_lat_min, rs_lat_max),
        longitude=slice(rs_lon_min, rs_lon_max)
    )

    # Identificar período comum
    precip_years = precip_rs.time.dt.year.values
    precip_year_min = precip_years.min()
    precip_year_max = precip_years.max()
    print(f"   Precipitação: {precip_year_min}-{precip_year_max} ({len(precip_rs.time)} estações)")
    
    # Preparar MoV AAO para período comum
    mov_data['time'] = pd.to_datetime(mov_data['time'])
    mov_data['Year'] = mov_data['time'].dt.year
    mov_data['Month'] = mov_data['time'].dt.month
    
    # Filtrar AAO para período da precipitação
    mov_filtered = mov_data[(mov_data['Year'] >= precip_year_min) & (mov_data['Year'] <= precip_year_max)]
    mov_monthly = mov_filtered.groupby(['Year', 'Month'])['value'].mean().reset_index()
    mov_monthly = mov_monthly.sort_values(['Year', 'Month'])
    mov_values = mov_monthly['value'].dropna().values

    lats = precip_rs.latitude.values
    lons = precip_rs.longitude.values

    print(f"   AAO (filtrado): {len(mov_values)} meses ({mov_filtered['Year'].min()}-{mov_filtered['Year'].max()})")
    print(f"   Região RS: {len(lats)} lat × {len(lons)} lon")
    print(f"   Coordenadas: Lat [{rs_lat_min}, {rs_lat_max}], Lon [{rs_lon_min}, {rs_lon_max}]")
    print(f"   Período comum: {precip_year_min}-{precip_year_max}")
    print(f"   Cálculo: {len(mov_values)} meses × {len(lats)*len(lons)} pontos × 13 lags × 2000 bootstrap")
    
    # Configurações
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
    season_offset_map = {'DJF': 0, 'MAM': 3, 'JJA': 6, 'SON': 9}
    n_lags = 13
    n_boot = 2000
    
    # Processar cada estação
    all_results = {}
    season_times = {}
    
    for season in seasons:
        season_start = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSANDO: {season}")
        print(f"{'='*80}")
        
        season_idx = season_map[season]
        season_offset = season_offset_map[season]
        
        precip_seasonal = precip_rs.isel(time=list(range(season_idx, len(precip_rs.time), 4)))
        
        # Criar dataset de resultados
        results = xr.Dataset(
            coords={
                'lag': range(n_lags),
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Calcular para cada lag
        for lag in range(n_lags):
            print(f"🔄 Lag {lag}...", end=" ", flush=True)
            
            alpha_map = np.full((len(lats), len(lons)), np.nan)
            
            for i in range(len(lats)):
                for j in range(len(lons)):
                    precip_point = precip_seasonal.pr.isel(latitude=i, longitude=j).values
                    
                    if not np.isnan(precip_point).all():
                        alpha_map[i, j] = calculate_predep_point(
                            mov_values, precip_point, lag, n_boot, season_offset
                        )
            
            results[f'alpha_lag{lag}'] = (['latitude', 'longitude'], alpha_map)
            valid = np.sum(~np.isnan(alpha_map))
            print(f"✅ {valid} pontos válidos")
        
        season_end = time.time()
        season_duration = season_end - season_start
        season_times[season] = season_duration
        print(f"⏱️  Tempo para {season}: {season_duration/60:.1f} minutos")
        
        all_results[season] = results
    
    # Salvar resultados
    output_dir = Path("results/rs_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SALVANDO RESULTADOS")
    print(f"{'='*80}")
    
    for season, results in all_results.items():
        nc_file = output_dir / f"AAO_{season}_RS.nc"
        results.to_netcdf(nc_file)
        print(f"💾 {nc_file.name}")
    
    # Gerar plots combinados (todas as estações em uma figura)
    print(f"\n{'='*80}")
    print("GERANDO PLOTS COMBINADOS")
    print(f"{'='*80}")
    
    output_4x4 = output_dir / "plots_4x4"
    output_13x4 = output_dir / "plots_13x4"
    output_4x4.mkdir(exist_ok=True)
    output_13x4.mkdir(exist_ok=True)
    
    # Combinar todas as estações em um único dataset
    print("\n📊 Combinando estações...")
    
    # Empilhar dados de todas as estações
    all_alpha_arrays = []
    for season in seasons:
        results = all_results[season]
        alpha_list = [results[f'alpha_lag{lag}'].values for lag in range(13)]
        alpha_stacked = np.stack(alpha_list, axis=0)
        all_alpha_arrays.append(alpha_stacked)
    
    # Shape final: (1, 4, 13, lat, lon)
    combined_alpha = np.stack(all_alpha_arrays, axis=0)[np.newaxis, :, :, :, :]
    
    # Pegar coordenadas de qualquer estação (todas têm as mesmas)
    first_results = all_results[seasons[0]]
    
    ds_combined = xr.Dataset(
        {'predep_alpha': (['mov', 'season', 'lag', 'latitude', 'longitude'], combined_alpha)},
        coords={
            'mov': ['AAO'],
            'season': seasons,
            'lag': list(range(13)),
            'latitude': first_results.latitude,
            'longitude': first_results.longitude
        }
    )
    
    # Gerar plots
    print("   Plot 4x4 (todas as estações)...")
    plot_4x4_optimal_lags(ds_combined, 'AAO_RS', output_4x4, seasons=seasons)
    print("   ✅ Plot 4x4 gerado")
    
    print("   Plot 13x4 (todas as estações)...")
    plot_13x4_all_lags(ds_combined, 'AAO_RS', output_13x4, seasons=seasons)
    print("   ✅ Plot 13x4 gerado")
    
    # Relatório final de tempo
    end_time = time.time()
    total_duration = end_time - start_time
    end_datetime = datetime.now()
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISE CONCLUÍDA!")
    print(f"{'='*80}")
    print(f"📁 Resultados em: {output_dir}")
    print(f"⏱️  RELATÓRIO DE TEMPO:")
    print(f"   Início: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Fim: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Tempo total: {total_duration/3600:.2f} horas ({total_duration/60:.1f} minutos)")
    print(f"   Tempo por estação:")
    for season, duration in season_times.items():
        print(f"      {season}: {duration/60:.1f} min")
    print(f"   Dados processados:")
    print(f"      MoV AAO: {len(mov_values)} meses ({precip_year_min}-{precip_year_max})")
    print(f"      Pontos espaciais: {len(lats)*len(lons)}")
    print(f"      Bootstrap: {n_boot}")
    print(f"      Total cálculos: ~{len(mov_values) * len(lats) * len(lons) * n_lags * n_boot:,}")


if __name__ == "__main__":
    run_rs_analysis()
