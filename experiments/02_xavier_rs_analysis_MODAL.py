#!/usr/bin/env python3
"""
Análise PREDEP - Brasil com Dados XAVIER (Modal Paralelo)
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

app = modal.App("tna-predep-brazil", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# Limites Brasil
BRAZIL_LAT_MIN, BRAZIL_LAT_MAX = -34.0, 6.0
BRAZIL_LON_MIN, BRAZIL_LON_MAX = -75.0, -34.0

# Global compartilhado via fork para workers de multiprocessing
_PRECIP_ARR_GLOBAL = None


def _ecdf(data):
    import numpy as np
    s = np.sort(data)
    n = len(data)
    return lambda x: np.searchsorted(s, x, side='right') / n


def _predep_calc(X, Y, n_boot):
    import numpy as np
    import scipy.stats as ss
    from astropy.stats import bayesian_blocks
    try:
        BX1 = np.random.choice(X, n_boot)
        BX2 = np.random.choice(X, n_boot)
        s_x = float(ss.gaussian_kde(BX1 - BX2).pdf(0))
        edges_y = bayesian_blocks(Y)
        ecdf_y = _ecdf(Y)
        s_x_mid_y = 0.0
        if edges_y.shape[0] > 1:
            for k in range(1, edges_y.shape[0]):
                bg, ed = edges_y[k - 1], edges_y[k]
                X_c = X[(Y >= bg) & (Y < ed)]
                if X_c.shape[0] == 0:
                    continue
                dx_c = np.random.choice(X_c, n_boot) - np.random.choice(X_c, n_boot)
                s_x_mid_y += (ecdf_y(ed) - ecdf_y(bg)) * float(ss.gaussian_kde(dx_c).pdf(0))
            return float((s_x_mid_y - s_x) / s_x_mid_y)
        return float('nan')
    except Exception:
        return float('nan')


def _compute_lag_worker(args):
    """Worker: calcula alpha_map completo para um lag. Usa _PRECIP_ARR_GLOBAL via fork."""
    import numpy as np
    import warnings
    import time as _time
    from tqdm import tqdm
    warnings.filterwarnings('ignore')

    lag, mov_aligned, n_lat, n_lon, n_boot = args
    precip_arr = _PRECIP_ARR_GLOBAL  # herdado via fork (Linux copy-on-write)

    t0 = _time.time()
    alpha_map = np.full((n_lat, n_lon), float('nan'))
    for i in tqdm(range(n_lat), desc=f"Lag {lag:2d}", position=lag % 8, leave=True, ncols=80):
        for j in range(n_lon):
            pp = precip_arr[:, i, j]
            mask = ~np.isnan(mov_aligned) & ~np.isnan(pp)
            if mask.sum() < 10:
                continue
            pp_v = pp[mask]
            mv_v = mov_aligned[mask]
            if len(np.unique(mv_v)) < 3 or len(np.unique(pp_v)) < 3:
                continue
            alpha_map[i, j] = _predep_calc(pp_v, mv_v, n_boot)
    elapsed = _time.time() - t0
    valid = int(np.sum(~np.isnan(alpha_map)))
    print(f"  Lag {lag:2d} finalizado em {elapsed/60:.1f}min — {valid} pontos válidos", flush=True)
    return lag, alpha_map


@app.function(
    memory=4096,
    cpu=8,
    timeout=3600 * 12,
    volumes={"/data": volume}
)
def process_season(season: str, n_boot: int = 2000):
    """Processa um trimestre em container isolado com 8 CPUs."""
    import pandas as pd
    import xarray as xr
    import numpy as np
    from pathlib import Path
    import time
    import multiprocessing as mp
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')

    def load_xavier_seasonal():
        monthly_file = "/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc"
        ds = xr.open_dataset(monthly_file)
        season_months = {
            'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
            'JJA': [6, 7, 8], 'SON': [9, 10, 11]
        }
        months = season_months[season]
        season_mask = ds.time.dt.month.isin(months)
        season_subset = ds.pr.where(season_mask, drop=True)
        seasonal_means = season_subset.groupby(season_subset.time.dt.year).mean('time')
        anomalies = seasonal_means - seasonal_means.mean('year')
        ds.close()
        return anomalies

    def load_tna():
        tna_file = "/data/input/TNA_indice_data.csv"
        mov_data = pd.read_csv(tna_file)
        mov_data['time'] = pd.to_datetime(mov_data['time'])
        mov_data['Year'] = mov_data['time'].dt.year
        mov_data['Month'] = mov_data['time'].dt.month
        return mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index().sort_values(['Year', 'Month'])

    def precompute_mov_alignment(mov_monthly, precip_years, lag, season_month):
        """Pré-alinha TNA para todos os anos de precipitação neste lag."""
        aligned = np.full(len(precip_years), np.nan)
        for k, year in enumerate(precip_years):
            tm, ty = season_month - lag, int(year)
            while tm <= 0:
                tm += 12
                ty -= 1
            row = mov_monthly[(mov_monthly['Year'] == ty) & (mov_monthly['Month'] == tm)]
            if len(row) == 1:
                aligned[k] = row['value'].iloc[0]
        return aligned

    # ====== EXECUÇÃO ======
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"CONTAINER {season} - Iniciando (8 CPUs)")
    print(f"{datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    print(f"Carregando dados para {season}...")
    precip_seasonal = load_xavier_seasonal()
    mov_monthly = load_tna()
    precip_years = precip_seasonal.year.values

    n_lat = len(precip_seasonal.latitude)
    n_lon = len(precip_seasonal.longitude)
    print(f"   Anos: {len(precip_years)}")
    print(f"   Grade: {n_lat}x{n_lon}")

    season_month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
    season_month = season_month_map[season]
    n_lags = 12

    # Extrair array numpy e expor como global para fork dos workers
    global _PRECIP_ARR_GLOBAL
    _PRECIP_ARR_GLOBAL = precip_seasonal.values  # shape (n_years, n_lat, n_lon)

    # Pré-alinhar TNA para cada lag (barato: ~59 lookups por lag)
    print(f"Pré-alinhando TNA para {n_lags} lags...")
    lag_args = []
    for lag in range(n_lags):
        mov_aligned = precompute_mov_alignment(mov_monthly, precip_years, lag, season_month)
        lag_args.append((lag, mov_aligned, n_lat, n_lon, n_boot))

    # Processar todos os lags em paralelo (8 workers)
    print(f"Processando {n_lags} lags em paralelo (8 workers)...")
    t_pool = time.time()
    lag_results = []
    with mp.Pool(processes=8) as pool:
        for lag, alpha_map in pool.imap_unordered(_compute_lag_worker, lag_args):
            elapsed = time.time() - t_pool
            valid = int(np.sum(~np.isnan(alpha_map)))
            print(f"  [+{elapsed:5.1f}s] Lag {lag:2d} recebido — {valid} pontos válidos", flush=True)
            lag_results.append((lag, alpha_map))

    # Montar dataset de resultados
    results = xr.Dataset(
        coords={
            'lag': range(n_lags),
            'latitude': precip_seasonal.latitude,
            'longitude': precip_seasonal.longitude
        }
    )
    for lag, alpha_map in sorted(lag_results, key=lambda x: x[0]):
        results[f'alpha_lag{lag}'] = (['latitude', 'longitude'], alpha_map)
        valid = int(np.sum(~np.isnan(alpha_map)))
        print(f"   Lag {lag}: {valid} pontos válidos")

    # Salvar
    output_dir = Path("/data/results/xavier_brazil")
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_file = output_dir / f"TNA_{season}_xavier_brazil.nc"
    results.to_netcdf(nc_file)

    duration = time.time() - start_time
    print(f"\n{season} CONCLUÍDO em {duration/60:.1f}min")
    print(f"Arquivo: {nc_file}")

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
    
    output_dir = Path("/data/results/xavier_brazil")
    seasons = ["DJF", "MAM", "JJA", "SON"]
    
    all_data = []
    for season in seasons:
        nc_file = output_dir / f"TNA_{season}_xavier_brazil.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            all_data.append(ds)
            print(f"   ✓ {season}")
    
    if len(all_data) == 4:
        combined = xr.concat(all_data, dim='season')
        combined_file = output_dir / "TNA_ALL_SEASONS_xavier_brazil.nc"
        combined.to_netcdf(combined_file)
        print(f"✅ Arquivo combinado: {combined_file}")
        return str(combined_file)
    
    return None


LOCAL_XAVIER_FILE = "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
LOCAL_XAVIER_RAW_FILES = [
    "data/raw/xavier_daily/pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.2-002.nc",
    "data/raw/xavier_daily/pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc",
    "data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc",
    "data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-004.nc",
]
LOCAL_TNA_FILE = "data/raw/mov_indices/TNA_indice data.csv"


def generate_brazil_monthly(output_path):
    """Gera arquivo mensal do Brasil a partir dos dados diários Xavier."""
    import xarray as xr
    import numpy as np
    from pathlib import Path

    print("   Gerando arquivo mensal Brasil a partir dos dados diários...")

    available = [f for f in LOCAL_XAVIER_RAW_FILES if Path(f).exists()]
    if not available:
        raise FileNotFoundError(
            f"Nenhum arquivo diário Xavier encontrado em: {LOCAL_XAVIER_RAW_FILES}"
        )

    monthly_datasets = []
    for raw_file in available:
        print(f"   Processando {Path(raw_file).name}...")
        ds = xr.open_dataset(raw_file, chunks={"time": 365})
        monthly = ds.resample(time="1M").sum(dim="time")
        monthly["pr"].attrs["units"] = "mm/month"
        monthly_datasets.append(monthly)
        ds.close()

    combined = xr.concat(monthly_datasets, dim="time").sortby("time")
    _, unique_idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=np.sort(unique_idx))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_path)
    print(f"   Salvo: {output_path}")
    return combined


def upload_input_files():
    """Faz upload dos arquivos de entrada para o volume Modal."""
    import pandas as pd
    import tempfile
    import os
    from pathlib import Path

    print("\n📤 Uploading input files to Modal volume...")

    xavier_path = Path(LOCAL_XAVIER_FILE)
    tna_path = Path(LOCAL_TNA_FILE)

    if not xavier_path.exists():
        print("   Arquivo mensal Brasil não encontrado. Gerando...")
        generate_brazil_monthly(xavier_path)

    if not tna_path.exists():
        raise FileNotFoundError(f"TNA CSV not found: {tna_path}")

    # Preparar TNA CSV no formato esperado pelo container (colunas: time, value)
    df = pd.read_csv(tna_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col = df.columns[1]
    tna_csv = df[[date_col, val_col]].copy()
    tna_csv.columns = ['time', 'value']
    tna_csv['time'] = pd.to_datetime(tna_csv['time'])
    tna_csv = tna_csv.dropna(subset=['value'])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tna_csv.to_csv(f, index=False)
        tmp_tna_path = f.name

    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(xavier_path), "/input/pr_monthly_1961_2019_BRAZIL_combined.nc")
            batch.put_file(tmp_tna_path, "/input/TNA_indice_data.csv")
    finally:
        os.unlink(tmp_tna_path)

    print(f"   ✓ Xavier data: {xavier_path.name} ({xavier_path.stat().st_size / 1e6:.0f} MB)")
    print(f"   ✓ TNA index: {len(tna_csv)} registros "
          f"({tna_csv['time'].dt.year.min()}-{tna_csv['time'].dt.year.max()})")
    print("   Upload concluído!\n")


@app.local_entrypoint()
def main(n_boot: int = 2000):
    """Executa análise paralela no Modal."""
    from datetime import datetime
    import time

    start_time = time.time()

    print("="*80)
    print("🚀 ANÁLISE PREDEP PARALELA - MODAL (BRASIL)")
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Bootstrap: {n_boot}")
    print(f"🌎 Região: Brasil completo")
    print("="*80)

    upload_input_files()

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
    print("   modal volume get xavier-data results/xavier_brazil ./local_results/")
    
    return {
        'total_duration': total_duration,
        'results': results,
        'combined_file': combined_file
    }
