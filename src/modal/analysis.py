#!/usr/bin/env python3
"""
Análise PREDEP - Rio Grande do Sul com Dados XAVIER (Modal Version)
Calcula PREDEP para cada ponto de grade na região do RS usando dados XAVIER
Executado na infraestrutura Modal para alta performance
"""

import modal

# Definir imagem Docker com todas as dependências
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas", 
        "xarray[complete]", 
        "numpy", 
        "matplotlib", 
        "cartopy", 
        "tqdm", 
        "joblib",
        "netcdf4",
        "h5netcdf",
        "scipy",
        "scikit-learn"
    ])
    .apt_install(["libproj-dev", "proj-data", "proj-bin"])
)

# Criar app Modal
app = modal.App("xavier-predep-analysis", image=image)

# Volume para dados (opcional - se você quiser persistir dados)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

@app.function(
    gpu="any",  # Pode usar GPU se disponível
    memory=16384,  # 16GB RAM
    timeout=3600 * 6,  # 6 horas timeout
    volumes={"/data": volume}
)
def run_xavier_analysis():
    """Função principal que roda no Modal."""
    
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo
    
    import pandas as pd
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from pathlib import Path
    import warnings
    import time
    from datetime import datetime, timedelta
    from tqdm import tqdm
    warnings.filterwarnings('ignore')
    
    # Limites geográficos do Rio Grande do Sul (região expandida)
    RS_LAT_MIN, RS_LAT_MAX = -36.0, -25.0
    RS_LON_MIN, RS_LON_MAX = -60.0, -47.0

    def predep(x, y, n_boot=500):
        """
        Implementação básica do PREDEP para Modal
        (você pode copiar a implementação completa do seu módulo)
        """
        from scipy import stats
        
        def _bootstrap_correlation(x, y, n_boot):
            n = len(x)
            correlations = []
            
            for _ in range(n_boot):
                indices = np.random.choice(n, n, replace=True)
                boot_x = x[indices]
                boot_y = y[indices]
                
                if np.var(boot_x) > 0 and np.var(boot_y) > 0:
                    corr, _ = stats.pearsonr(boot_x, boot_y)
                    correlations.append(abs(corr))
            
            return np.array(correlations)
        
        if len(x) < 10 or np.var(x) == 0 or np.var(y) == 0:
            return np.nan
            
        correlations = _bootstrap_correlation(x, y, n_boot)
        
        if len(correlations) == 0:
            return np.nan
            
        # Retorna o percentil 95 das correlações
        return np.percentile(correlations, 95)

    def load_xavier_data_from_url():
        """
        Carrega dados XAVIER de URLs públicas ou do volume Modal.
        Para este exemplo, vamos simular dados - você precisa adaptar 
        para carregar seus dados reais no Modal.
        """
        print("📥 Carregando dados XAVIER no Modal...")
        
        # Simulação de dados para demonstração
        # SUBSTITUA por seu método real de carregamento
        years = list(range(1981, 2020))
        lats = np.linspace(RS_LAT_MIN, RS_LAT_MAX, 20)
        lons = np.linspace(RS_LON_MIN, RS_LON_MAX, 25)
        
        seasonal_data = {}
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            # Simular dados de precipitação
            data = np.random.normal(0, 1, (len(years), len(lats), len(lons)))
            
            seasonal_data[season] = xr.DataArray(
                data,
                coords={
                    'year': years,
                    'latitude': lats,
                    'longitude': lons
                },
                dims=['year', 'latitude', 'longitude']
            )
            print(f"      {season}: {len(years)} anos de dados simulados")
        
        return seasonal_data

    def load_aao_data():
        """Carrega dados AAO - adapte para sua fonte de dados."""
        print("📥 Carregando dados AAO...")
        
        # Simulação para demonstração - SUBSTITUA pelos dados reais
        dates = pd.date_range('1980-01-01', '2025-12-31', freq='M')
        values = np.random.normal(0, 1, len(dates))
        
        mov_data = pd.DataFrame({
            'time': dates,
            'value': values
        })
        
        mov_data['Year'] = mov_data['time'].dt.year
        mov_data['Month'] = mov_data['time'].dt.month
        mov_monthly = mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index()
        mov_monthly = mov_monthly.sort_values(['Year', 'Month'])
        mov_values = mov_monthly['value'].dropna().values
        
        return mov_values

    def calculate_predep_point(mov_values, precip_point, lag, n_boot, season_offset):
        """Calcula PREDEP para um ponto espacial."""
        try:
            n_years = len(precip_point)
            mov_aligned = []
            
            for year in range(n_years):
                idx = year * 12 + season_offset - lag
                if idx < 0 or idx >= len(mov_values):
                    return np.nan
                mov_aligned.append(mov_values[idx])
            
            mov_aligned = np.array(mov_aligned)
            
            if len(np.unique(mov_aligned)) < 3 or len(np.unique(precip_point)) < 3:
                return np.nan
            
            valid_mask = ~(np.isnan(mov_aligned) | np.isnan(precip_point))
            if valid_mask.sum() < 10:
                return np.nan
                
            return predep(mov_aligned[valid_mask], precip_point[valid_mask], n_boot=n_boot)
            
        except Exception:
            return np.nan

    # ====== EXECUÇÃO PRINCIPAL ======
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("="*80)
    print("🚀 ANÁLISE PREDEP - RIO GRANDE DO SUL (MODAL)")
    print("🌐 Executando na infraestrutura Modal")
    print(f"⏰ Início: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Carregar dados
    seasonal_data = load_xavier_data_from_url()
    mov_values = load_aao_data()
    
    print(f"   AAO: {len(mov_values)} meses de dados")
    
    # Configurações
    seasons = ["DJF", "MAM", "JJA", "SON"]
    season_offset_map = {'DJF': 0, 'MAM': 3, 'JJA': 6, 'SON': 9}
    n_lags = 12
    n_boot = 500  # Otimizado para Modal
    
    # Criar diretório de resultados no volume
    output_dir = Path("/data/results/xavier_rs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    season_times = {}
    
    # Progress geral para estações
    season_progress = tqdm(seasons, desc="Estações", unit="season")
    
    for season in season_progress:
        season_start = time.time()
        season_progress.set_description(f"Processando {season}")
        
        season_offset = season_offset_map[season]
        precip_seasonal = seasonal_data[season]
        
        # Info da grade
        season_progress.write(f"\n📊 {season}: {len(precip_seasonal.year)} anos, "
                             f"Grade {len(precip_seasonal.latitude)}x{len(precip_seasonal.longitude)} "
                             f"({len(precip_seasonal.latitude) * len(precip_seasonal.longitude):,} pontos)")
        
        # Dataset de resultados
        results = xr.Dataset(
            coords={
                'lag': range(n_lags),
                'latitude': precip_seasonal.latitude,
                'longitude': precip_seasonal.longitude
            }
        )
        
        # Calcular para cada lag
        lag_progress = tqdm(range(n_lags), desc=f"Lags {season}", unit="lag")
        
        for lag in lag_progress:
            lag_start_time = time.time()
            
            alpha_map = np.full((len(precip_seasonal.latitude), len(precip_seasonal.longitude)), np.nan)
            total_points = len(precip_seasonal.latitude) * len(precip_seasonal.longitude)
            
            with tqdm(total=total_points, desc=f"Lag {lag}", unit="pt", leave=False) as pbar:
                points_processed = 0
                
                for i in range(len(precip_seasonal.latitude)):
                    for j in range(len(precip_seasonal.longitude)):
                        precip_point = precip_seasonal.isel(latitude=i, longitude=j).values
                        
                        if not np.isnan(precip_point).all():
                            alpha_map[i, j] = calculate_predep_point(
                                mov_values, precip_point, lag, n_boot, season_offset
                            )
                        
                        points_processed += 1
                        pbar.update(1)
                        
                        # Atualizar ETA a cada 50 pontos
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
            "Pontos": f"{len(precip_seasonal.latitude) * len(precip_seasonal.longitude)}"
        })
        
        # Salvar resultados
        nc_file = output_dir / f"AAO_{season}_xavier_rs_modal.nc"
        results.to_netcdf(nc_file)
        
        all_results[season] = results
    
    # Relatório final
    end_time = time.time()
    total_duration = end_time - start_time
    end_datetime = datetime.now()
    
    print(f"\n{'='*60}")
    print("✅ ANÁLISE CONCLUÍDA NO MODAL!")
    print(f"{'='*60}")
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"⏱️  RELATÓRIO DE TEMPO:")
    print(f"   Início: {start_datetime.strftime('%H:%M:%S')}")
    print(f"   Fim: {end_datetime.strftime('%H:%M:%S')}")
    print(f"   Tempo total: {total_duration/3600:.2f}h ({total_duration/60:.1f}min)")
    print(f"   Tempo por estação:")
    for season, duration in season_times.items():
        print(f"      {season}: {duration/60:.1f}min")
    
    # Gerar relatório sumário
    print(f"\n📊 RELATÓRIO SUMÁRIO:")
    for season, results in all_results.items():
        print(f"\n{season}:")
        for lag in range(n_lags):
            data = results[f'alpha_lag{lag}'].values
            valid_points = np.sum(~np.isnan(data))
            if valid_points > 0:
                mean_alpha = np.nanmean(data)
                max_alpha = np.nanmax(data)
                significant_points = np.sum(data > 0.05)
                print(f"  Lag {lag:2d}: {valid_points:3d} válidos | "
                      f"α médio: {mean_alpha:.3f} | α máx: {max_alpha:.3f} | "
                      f">0.05: {significant_points:3d}")
    
    print(f"\n🌐 Análise executada com sucesso no Modal!")
    print(f"📦 Arquivos salvos no volume Modal: /data/results/xavier_rs/")
    
    return {
        'total_duration': total_duration,
        'season_times': season_times,
        'output_dir': str(output_dir),
        'valid_results': len(all_results)
    }


@app.local_entrypoint()
def main():
    """Ponto de entrada local para executar no Modal."""
    print("🚀 Iniciando análise PREDEP no Modal...")
    print("📡 Enviando para infraestrutura Modal...")
    
    # Executar a função no Modal
    result = run_xavier_analysis.remote()
    
    print("\n" + "="*60)
    print("✅ EXECUÇÃO NO MODAL CONCLUÍDA!")
    print("="*60)
    print(f"📊 Resultados: {result}")
    print("💾 Para baixar arquivos, use: modal volume get xavier-data")
    
    return result


if __name__ == "__main__":
    # Para executar localmente (vai mandar para Modal)
    # modal run modal_xavier_rs_analysis.py
    pass