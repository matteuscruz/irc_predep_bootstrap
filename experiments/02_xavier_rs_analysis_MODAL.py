#!/usr/bin/env python3
"""
Análise PREDEP - Rio Grande do Sul com Dados XAVIER (Modal Paralelo)
Executa cada trimestre em container isolado para paralelismo real
"""

import modal

# Imagem com dependências
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas", "xarray[complete]", "numpy", "scipy", "astropy",
        "matplotlib", "cartopy", "netcdf4", "h5netcdf", "tqdm"
    ])
    .apt_install(["libproj-dev", "proj-data", "proj-bin"])
)

app = modal.App("xavier-predep-parallel", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# Limites RS
RS_LAT_MIN, RS_LAT_MAX = -36.0, -25.0
RS_LON_MIN, RS_LON_MAX = -60.0, -47.0


@app.function(
    memory=16384,
    timeout=3600 * 8,
    volumes={"/data": volume}
)
def process_season(season: str, n_boot: int = 500):
    """Processa um trimestre em container isolado."""
    import matplotlib
    matplotlib.use('Agg')
    
    import pandas as pd
    import xarray as xr
    import numpy as np
    import scipy.stats as ss
    from astropy.stats import bayesian_blocks
    from pathlib import Path
    import time
    from datetime import datetime, timedelta
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings('ignore')
    
    def ecdf(data):
        sorted_data = np.sort(data)
        n = len(data)
        return lambda x: np.searchsorted(sorted_data, x, side='right') / n
    
    def predep(X, Y, n_boot=500):
        """PREDEP original."""
        try:
            X, Y = np.asarray(X), np.asarray(Y)
            
            BX1 = np.random.choice(X, n_boot)
            BX2 = np.random.choice(X, n_boot)
            DX = BX1 - BX2
            s_x = float(ss.gaussian_kde(DX).pdf(0))
            
            edges_y = bayesian_blocks(Y)
            ecdf_y = ecdf(Y)
            
            s_x_mid_y = 0.0
            if edges_y.shape[0] > 1:
                for i in range(1, edges_y.shape[0]):
                    bg, ed = edges_y[i - 1], edges_y[i]
                    X_mid_Y = X[(Y >= bg) & (Y < ed)]
                    if X_mid_Y.shape[0] == 0:
                        continue
                    
                    BX_mid_Y1 = np.random.choice(X_mid_Y, n_boot)
                    BX_mid_Y2 = np.random.choice(X_mid_Y, n_boot)
                    DX_mid_Y = BX_mid_Y1 - BX_mid_Y2
                    
                    p_range = ecdf_y(ed) - ecdf_y(bg)
                    p_x_mid_y = float(ss.gaussian_kde(DX_mid_Y).pdf(0))
                    s_x_mid_y += p_range * p_x_mid_y
                
                alpha_est = (s_x_mid_y - s_x) / s_x_mid_y
                return float(alpha_est)
            return np.nan
        except:
            return np.nan
    
    def load_xavier_seasonal():
        """Carrega dados XAVIER do volume Modal."""
        monthly_file = "/data/input/pr_monthly_1961_2019_RS_combined.nc"
        ds = xr.open_dataset(monthly_file)
        
        season_months = {
            'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
            'JJA': [6, 7, 8], 'SON': [9, 10, 11]
        }
        
        months = season_months[season]
        season_mask = ds.time.dt.month.isin(months)
        season_subset = ds.pr.where(season_mask, drop=True)
        seasonal_means = season_subset.groupby(season_subset.time.dt.year).mean('time')
        climatology = seasonal_means.mean('year')
        anomalies = seasonal_means - climatology
        
        ds.close()
        return anomalies
    
    def load_aao():
        """Carrega AAO do volume Modal."""
        aao_file = "/data/input/AAO_indice_data.csv"
        mov_data = pd.read_csv(aao_file)
        mov_data['time'] = pd.to_datetime(mov_data['time'])
        mov_data['Year'] = mov_data['time'].dt.year
        mov_data['Month'] = mov_data['time'].dt.month
        mov_monthly = mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index()
        return mov_monthly.sort_values(['Year', 'Month'])
    
    def calculate_predep_point(mov_monthly, precip_years, precip_point, lag, season_month):
        """Calcula PREDEP para um ponto."""
        try:
            mov_aligned, precip_aligned = [], []
            
            for i, year in enumerate(precip_years):
                target_month = season_month - lag
                target_year = year
                
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
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
            
            if len(np.unique(mov_aligned)) < 3 or len(np.unique(precip_aligned)) < 3:
                return np.nan
            
            return predep(precip_aligned, mov_aligned, n_boot=n_boot)
        except:
            return np.nan
    
    # ====== EXECUÇÃO ======
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"🌐 CONTAINER {season} - Iniciando")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Carregar dados
    print(f"📥 Carregando dados para {season}...")
    precip_seasonal = load_xavier_seasonal()
    mov_monthly = load_aao()
    precip_years = precip_seasonal.year.values
    
    print(f"   Anos: {len(precip_years)}")
    print(f"   Grade: {len(precip_seasonal.latitude)}x{len(precip_seasonal.longitude)}")
    
    # Configurações
    season_month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
    season_month = season_month_map[season]
    n_lags = 13
    
    # Dataset resultados
    results = xr.Dataset(
        coords={
            'lag': range(n_lags),
            'latitude': precip_seasonal.latitude,
            'longitude': precip_seasonal.longitude
        }
    )
    
    # Calcular lags
    print(f"🔄 Processando {n_lags} lags...")
    for lag in tqdm(range(n_lags), desc=f"{season} Lags"):
        alpha_map = np.full(
            (len(precip_seasonal.latitude), len(precip_seasonal.longitude)), 
            np.nan
        )
        
        total_points = len(precip_seasonal.latitude) * len(precip_seasonal.longitude)
        
        for i in range(len(precip_seasonal.latitude)):
            for j in range(len(precip_seasonal.longitude)):
                precip_point = precip_seasonal.isel(latitude=i, longitude=j).values
                
                if not np.isnan(precip_point).all():
                    alpha_map[i, j] = calculate_predep_point(
                        mov_monthly, precip_years, precip_point, lag, season_month
                    )
        
        results[f'alpha_lag{lag}'] = (['latitude', 'longitude'], alpha_map)
        valid = np.sum(~np.isnan(alpha_map))
        print(f"   Lag {lag}: {valid} pontos válidos")
    
    # Salvar
    output_dir = Path("/data/results/xavier_rs")
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_file = output_dir / f"AAO_{season}_xavier_rs.nc"
    results.to_netcdf(nc_file)
    
    duration = time.time() - start_time
    
    print(f"\n✅ {season} CONCLUÍDO")
    print(f"⏱️  Tempo: {duration/60:.1f}min")
    print(f"📁 Arquivo: {nc_file}")
    
    return {
        'season': season,
        'duration': duration,
        'file': str(nc_file),
        'n_lags': n_lags,
        'valid_points': int(np.sum(~np.isnan(results['alpha_lag0'].values)))
    }


@app.function(
    memory=4096,
    timeout=3600,
    volumes={"/data": volume}
)
def combine_results():
    """Combina resultados de todas as estações."""
    import xarray as xr
    import numpy as np
    from pathlib import Path
    
    print("\n📊 Combinando resultados...")
    
    output_dir = Path("/data/results/xavier_rs")
    seasons = ["DJF", "MAM", "JJA", "SON"]
    
    all_data = []
    for season in seasons:
        nc_file = output_dir / f"AAO_{season}_xavier_rs.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            all_data.append(ds)
            print(f"   ✓ {season}")
    
    if len(all_data) == 4:
        combined = xr.concat(all_data, dim='season')
        combined_file = output_dir / "AAO_ALL_SEASONS_xavier_rs.nc"
        combined.to_netcdf(combined_file)
        print(f"✅ Arquivo combinado: {combined_file}")
        return str(combined_file)
    
    return None


@app.local_entrypoint()
def main(n_boot: int = 500):
    """Executa análise paralela no Modal."""
    from datetime import datetime
    import time
    
    start_time = time.time()
    
    print("="*80)
    print("🚀 ANÁLISE PREDEP PARALELA - MODAL")
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Bootstrap: {n_boot}")
    print("="*80)
    
    seasons = ["DJF", "MAM", "JJA", "SON"]
    
    print("\n📡 Enviando 4 containers paralelos para Modal...")
    print("   Cada trimestre rodará de forma independente\n")
    
    # Executar em paralelo (cada season em container separado)
    results = []
    for result in process_season.map(seasons, kwargs={"n_boot": n_boot}):
        results.append(result)
        print(f"\n✓ Container finalizado: {result['season']}")
        print(f"  Tempo: {result['duration']/60:.1f}min")
        print(f"  Pontos válidos: {result['valid_points']}")
    
    # Combinar resultados
    print("\n" + "="*60)
    print("📊 Combinando resultados...")
    combined_file = combine_results.remote()
    
    total_duration = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ ANÁLISE PARALELA CONCLUÍDA!")
    print("="*80)
    print(f"⏱️  Tempo total: {total_duration/60:.1f}min ({total_duration/3600:.2f}h)")
    print(f"📊 Resultados por estação:")
    for r in results:
        print(f"   {r['season']}: {r['duration']/60:.1f}min - {r['file']}")
    
    if combined_file:
        print(f"\n📦 Arquivo combinado: {combined_file}")
    
    print("\n💾 Para baixar resultados:")
    print("   modal volume get xavier-data results/xavier_rs ./local_results/")
    
    return {
        'total_duration': total_duration,
        'results': results,
        'combined_file': combined_file
    }
