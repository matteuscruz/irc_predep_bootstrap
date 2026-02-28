#!/usr/bin/env python3
"""
Análise PREDEP - Rio Grande do Sul com Dados XAVIER
Calcula PREDEP para cada ponto de grade na região do RS usando dados XAVIER
Período completo: 1961-2019 (59 anos), AAO, 13 lags (0-12), bootstrap=500-2000
"""

import matplotlib
matplotlib.use('Agg')  # Backend não-interativo ANTES de importar pyplot

import pandas as pd
import xarray as xr
import numpy as np
from src.predep_calculation.core import predep
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
import sys
import time
from datetime import datetime, timedelta
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Adicionar path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization import plot_4x4_optimal_lags, plot_13x4_all_lags

# Limites geográficos do Rio Grande do Sul (região expandida)
RS_LAT_MIN, RS_LAT_MAX = -36.0, -25.0
RS_LON_MIN, RS_LON_MAX = -60.0, -47.0

def load_xavier_seasonal_data():
    """
    Carrega dados mensais XAVIER pré-processados e calcula médias sazonais.
    """
    print("📥 Carregando dados XAVIER mensais...")
    
    # Carregar arquivo mensal combinado
    monthly_file = "data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc"
    
    try:
        ds = xr.open_dataset(monthly_file)
        print(f"   Período: {ds.time.min().dt.year.item()}-{ds.time.max().dt.year.item()}")
        print(f"   Meses: {len(ds.time)}")
        print(f"   Grade: {len(ds.latitude)} lat × {len(ds.longitude)} lon")
    except FileNotFoundError:
        print(f"   ❌ Arquivo não encontrado: {monthly_file}")
        print(f"   Execute primeiro: python scripts/convert_xavier_daily_to_monthly.py")
        raise
    
    # Definir estações
    season_groups = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5], 
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    seasonal_data = {}
    
    print("   📊 Calculando médias e anomalias sazonais...")
    for season, months in season_groups.items():
        # Filtrar meses da estação
        season_mask = ds.time.dt.month.isin(months)
        season_subset = ds.pr.where(season_mask, drop=True)
        
        # Calcular médias sazonais por ano
        seasonal_means = season_subset.groupby(season_subset.time.dt.year).mean('time')
        
        # Calcular anomalias
        climatology = seasonal_means.mean('year')
        anomalies = seasonal_means - climatology
        
        seasonal_data[season] = anomalies
        
        year_range = f"{anomalies.year.min().item()}-{anomalies.year.max().item()}"
        print(f"      {season}: {len(anomalies.year)} anos de anomalias ({year_range})")
    
    ds.close()
    return seasonal_data


def calculate_predep_point(mov_monthly, precip_years, precip_point, lag, n_boot, season_month):
    """
    Calcula PREDEP para um ponto espacial.
    
    Parameters
    ----------
    mov_monthly : DataFrame
        DataFrame com colunas Year, Month, value do MoV
    precip_years : array
        Anos da precipitação sazonal
    precip_point : array
        Série temporal sazonal da precipitação (anomalias)
    lag : int
        Número de meses de atraso (0-12)
    n_boot : int
        Número de amostras bootstrap
    season_month : int
        Mês central da estação (1=Jan, 4=Abr, 7=Jul, 10=Out)
    """
    try:
        mov_aligned = []
        precip_aligned = []
        
        # Para cada ano de precipitação, buscar MoV correspondente
        for i, year in enumerate(precip_years):
            # Mês do MoV = mês central da estação - lag
            target_month = season_month - lag
            target_year = year
            
            # Ajustar ano se mês ficar negativo
            while target_month <= 0:
                target_month += 12
                target_year -= 1
            
            # Buscar valor do MoV
            mov_row = mov_monthly[
                (mov_monthly['Year'] == target_year) & 
                (mov_monthly['Month'] == target_month)
            ]
            
            if len(mov_row) == 1 and not np.isnan(precip_point[i]):
                mov_aligned.append(mov_row['value'].iloc[0])
                precip_aligned.append(precip_point[i])
        
        if len(mov_aligned) < 10:
            return np.nan
        
        mov_aligned = np.array(mov_aligned)
        precip_aligned = np.array(precip_aligned)
        
        # Verificar variabilidade
        if len(np.unique(mov_aligned)) < 3 or len(np.unique(precip_aligned)) < 3:
            return np.nan
        
        # PREDEP: X=precipitação (preditando), Y=MoV (preditor)
        return predep(precip_aligned, mov_aligned, n_boot=n_boot)
        
    except Exception as e:
        return np.nan


def run_xavier_rs_analysis(mov_index="AAO", seasons=["DJF", "MAM", "JJA", "SON"], n_boot=500, n_jobs=-1):
    """Executa análise PREDEP para o Rio Grande do Sul usando dados XAVIER."""
    # Iniciar medição de tempo
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("="*80)
    print(f"ANÁLISE PREDEP - RIO GRANDE DO SUL (XAVIER) - OTIMIZADO")
    print(f"MoV: {mov_index} | Bootstrap: {n_boot} | Estações: {seasons}")
    print(f"Início: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Carregar dados
    seasonal_data = load_xavier_seasonal_data()
    
    print("\n📥 Carregando índices climáticos...")
    if mov_index == "AAO":
        mov_data = pd.read_csv("data/raw/mov_indices/AAO_indice data.csv")
        mov_data['time'] = pd.to_datetime(mov_data['time'])
        mov_data['Year'] = mov_data['time'].dt.year
        mov_data['Month'] = mov_data['time'].dt.month
        mov_monthly = mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index()
        mov_monthly = mov_monthly.sort_values(['Year', 'Month'])
        mov_values = mov_monthly['value'].dropna().values
    else:
        mov_data = pd.read_csv("data/input/merged_indices.csv", low_memory=False)
        mov_monthly = mov_data.groupby(['Year', 'Month'])[mov_index].mean().reset_index()
        mov_monthly = mov_monthly.sort_values(['Year', 'Month'])
        mov_values = mov_monthly[mov_index].dropna().values
    
    print(f"   {mov_index}: {len(mov_values)} meses de dados")
    
    # Definições das estações (mês central)
    season_month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
    n_lags = 13  # 0 a 12 lags
    
    # Criar diretório de saída
    output_dir = Path("results/xavier_rs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    season_times = {}
    
    # Progress geral para estações
    season_progress = tqdm(seasons, desc="Estações", unit="season")
    
    for season in season_progress:
        season_start = time.time()
        season_progress.set_description(f"Processando {season}")
        
        if season not in seasonal_data:
            print(f"⚠️ Dados sazonais não disponíveis para {season}")
            continue
            
        season_month = season_month_map[season]
        precip_seasonal = seasonal_data[season]
        precip_years = precip_seasonal.year.values
        
        # Usar resolução completa do Rio Grande do Sul
        precip_reduced = precip_seasonal
        
        # Atualizar descrição da barra de progresso com info da estação
        season_progress.write(f"\n📊 {season}: {len(precip_seasonal.year)} anos, "
                             f"Grade {len(precip_reduced.latitude)}x{len(precip_reduced.longitude)} "
                             f"({len(precip_reduced.latitude) * len(precip_reduced.longitude):,} pontos)")
        
        # Criar dataset de resultados (grade reduzida)
        results = xr.Dataset(
            coords={
                'lag': range(n_lags),
                'latitude': precip_reduced.latitude,
                'longitude': precip_reduced.longitude
            }
        )
        
        # Calcular para cada lag
        total_lags = n_lags
        lag_progress = tqdm(range(n_lags), desc=f"Lags {season}", unit="lag")
        
        for lag in lag_progress:
            lag_start_time = time.time()
            
            alpha_map = np.full((len(precip_reduced.latitude), len(precip_reduced.longitude)), np.nan)
            
            # Progress para pontos espaciais
            total_points = len(precip_reduced.latitude) * len(precip_reduced.longitude)
            
            with tqdm(total=total_points, desc=f"Lag {lag}", unit="pt", leave=False) as pbar:
                points_processed = 0
                
                for i in range(len(precip_reduced.latitude)):
                    for j in range(len(precip_reduced.longitude)):
                        precip_point = precip_reduced.isel(latitude=i, longitude=j).values
                        
                        if not np.isnan(precip_point).all():
                            alpha_map[i, j] = calculate_predep_point(
                                mov_monthly, precip_years, precip_point, lag, n_boot, season_month
                            )
                        
                        points_processed += 1
                        pbar.update(1)
                        
                        # Atualizar estimativa de tempo a cada 50 pontos
                        if points_processed % 50 == 0:
                            elapsed = time.time() - lag_start_time
                            if points_processed > 0:
                                rate = points_processed / elapsed
                                eta_seconds = (total_points - points_processed) / rate
                                eta_str = str(timedelta(seconds=int(eta_seconds)))
                                pbar.set_postfix({"ETA": eta_str, "Rate": f"{rate:.1f}pt/s"})
            
            results[f'alpha_lag{lag}'] = (['latitude', 'longitude'], alpha_map)
            valid = np.sum(~np.isnan(alpha_map))
            
            lag_elapsed = time.time() - lag_start_time
            lag_progress.set_postfix({
                "Valid": valid,
                "Time": f"{lag_elapsed/60:.1f}min"
            })
        
        season_end = time.time()
        season_duration = season_end - season_start
        season_times[season] = season_duration
        
        season_progress.set_postfix({
            "Time": f"{season_duration/60:.1f}min",
            "Pontos": f"{len(precip_reduced.latitude) * len(precip_reduced.longitude)}"
        })
        
        # Salvar resultados
        nc_file = output_dir / f"{mov_index}_{season}_xavier_rs.nc"
        results.to_netcdf(nc_file)
        
        all_results[season] = results
    
    # Gerar plots combinados (padrão do projeto)
    print(f"\n{'='*60}")
    print("📊 GERANDO PLOTS COMBINADOS")
    print(f"{'='*60}")
    
    output_4x4 = output_dir / "plots_4x4"
    output_13x4 = output_dir / "plots_13x4"
    output_4x4.mkdir(exist_ok=True)
    output_13x4.mkdir(exist_ok=True)
    
    print("\n📊 Combinando estações...")
    
    # Empilhar dados de todas as estações
    all_alpha_arrays = []
    first_results = all_results[seasons[0]]  # Salvar antes do loop
    
    with tqdm(seasons, desc="Combinando", unit="season") as pbar:
        for season in pbar:
            results = all_results[season]
            alpha_list = []
            for lag in range(n_lags):
                alpha_data = results[f'alpha_lag{lag}'].values
                alpha_list.append(alpha_data)
            alpha_stacked = np.stack(alpha_list, axis=0)
            all_alpha_arrays.append(alpha_stacked)
            pbar.set_postfix({"Season": season})
    
    # Shape final: (1, 4, n_lags, lat, lon) 
    combined_alpha = np.stack(all_alpha_arrays, axis=0)[np.newaxis, :, :, :, :]
    
    ds_combined = xr.Dataset(
        {'predep_alpha': (['mov', 'season', 'lag', 'latitude', 'longitude'], combined_alpha)},
        coords={
            'mov': [mov_index],
            'season': seasons,
            'lag': list(range(n_lags)),
            'latitude': first_results.latitude,
            'longitude': first_results.longitude
        }
    )
    
    # Gerar plots
    print("🎨 Gerando plots combinados...")
    with tqdm(total=2, desc="Plots", unit="tipo") as plot_pbar:
        
        plot_pbar.set_description("Plot 4x4")
        plot_4x4_optimal_lags(ds_combined, f'{mov_index}_XAVIER_RS', output_4x4, seasons=seasons)
        plot_pbar.update(1)
        
        plot_pbar.set_description("Plot 13x4") 
        plot_13x4_all_lags(ds_combined, f'{mov_index}_XAVIER_RS', output_13x4, seasons=seasons)
        plot_pbar.update(1)
        
        plot_pbar.set_description("Concluído")
    
    print("   ✅ Todos os plots gerados")
    
    # Relatório final de tempo
    end_time = time.time()
    total_duration = end_time - start_time
    end_datetime = datetime.now()
    
    print(f"\n{'='*60}")
    print("✅ ANÁLISE CONCLUÍDA!")
    print(f"{'='*60}")
    print(f"📁 Resultados em: {output_dir}")
    print(f"⏱️  RELATÓRIO DE TEMPO:")
    print(f"   Início: {start_datetime.strftime('%H:%M:%S')}")
    print(f"   Fim: {end_datetime.strftime('%H:%M:%S')}")
    print(f"   Tempo total: {total_duration/3600:.2f}h ({total_duration/60:.1f}min)")
    print(f"   Tempo por estação:")
    for season, duration in season_times.items():
        print(f"      {season}: {duration/60:.1f}min")
    print(f"   Dados processados:")
    print(f"      MoV {mov_index}: {len(mov_values)} meses")
    print(f"      Pontos espaciais: {len(first_results.latitude) * len(first_results.longitude)}")
    print(f"      Bootstrap: {n_boot}")
    print(f"      Total cálculos: ~{len(mov_values) * len(first_results.latitude) * len(first_results.longitude) * n_lags * n_boot:,}")
    
    return all_results


def generate_summary_report(all_results, mov_index):
    """Gera relatório sumário dos resultados."""
    print("\n" + "="*80)
    print("📊 RELATÓRIO SUMÁRIO")
    print("="*80)
    
    output_dir = Path("results/xavier_rs")
    
    for season, results in all_results.items():
        print(f"\n{season}:")
        
        for lag in range(len(results.lag)):
            data = results[f'alpha_lag{lag}'].values
            valid_points = np.sum(~np.isnan(data))
            
            if valid_points > 0:
                mean_alpha = np.nanmean(data)
                max_alpha = np.nanmax(data)
                significant_points = np.sum(data > 0.05)
                
                print(f"  Lag {lag:2d}: {valid_points:3d} pontos válidos | "
                      f"α médio: {mean_alpha:.3f} | α máximo: {max_alpha:.3f} | "
                      f">0.05: {significant_points:3d} pontos")
    
    print(f"\n📁 Arquivos gerados em: {output_dir}")
    print(f"   - Dados NetCDF: *_xavier_rs.nc")
    print(f"   - Mapas 4x4: plots_4x4/*_XAVIER_RS_*.png")
    print(f"   - Mapas 13x4: plots_13x4/*_XAVIER_RS_*.png")


if __name__ == "__main__":
    # CONFIGURAÇÃO OTIMIZADA
    FRESH_ANALYSIS = True  # Definir como True para análise totalmente nova
    n_jobs = -1
    n_boot_opt = 500  # Reduzido de 2000 para 500 para teste
    
    if FRESH_ANALYSIS:
        print("🔥 ANÁLISE TOTALMENTE NOVA - SEM CACHE")
        # Limpar arquivos de resultado antigos se existirem
        import shutil
        results_dirs = ["results/xavier_rs", "results/rs_analysis", "results/spatial"]
        for result_dir in results_dirs:
            if Path(result_dir).exists():
                print(f"🗑️ Removendo resultados antigos: {result_dir}")
                shutil.rmtree(result_dir)
        print("✅ Limpeza concluída")
    
    print("🚀 Iniciando análise PREDEP - Rio Grande do Sul (XAVIER)")
    print(f"⚙️  Configurações OTIMIZADAS:")
    print(f"   - Região: RS EXPANDIDA ({RS_LAT_MIN}°S-{RS_LAT_MAX}°S, {RS_LON_MIN}°W-{RS_LON_MAX}°W)")
    print(f"   - Período: 1961-2019 (59 anos - TODOS OS DADOS)")
    print(f"   - MoV: AAO")
    print(f"   - Lags: 0-12 (13 lags)")
    print(f"   - Bootstrap: {n_boot_opt} (otimizado)")
    print(f"   - Resolução: COMPLETA (todos os pontos da grade)")
    print(f"   - Jobs: {n_jobs}")
    print(f"   - Análise nova: {FRESH_ANALYSIS}")
    
    # Executar análise
    all_results = run_xavier_rs_analysis(
        mov_index="AAO",
        seasons=["DJF", "MAM", "JJA", "SON"],
        n_boot=n_boot_opt,
        n_jobs=n_jobs
    )
    
    # Gerar relatório
    generate_summary_report(all_results, "AAO")
    
    print("\n" + "="*80)
    print("✅ ANÁLISE PREDEP CONCLUÍDA (VERSÃO OTIMIZADA)!")
    print("="*80)
    print("📈 Resultados:")
    print("   - Arquivos NetCDF salvos em: results/xavier_rs/")
    print("   - Mapas 4x4 gerados em: results/xavier_rs/plots_4x4/")
    print("   - Mapas 13x4 gerados em: results/xavier_rs/plots_13x4/")
    print("   - Cobertura: Rio Grande do Sul (resolução completa)")
    print("   - Período: 1961-2019 (59 anos - TODOS OS DADOS XAVIER)")
    print("   - Método: Bootstrap PREDEP com dados XAVIER")
    print("\n💡 Nota: Análise com resolução espacial COMPLETA do RS.")
    print("   Tempo de execução: várias horas dependendo do hardware.")