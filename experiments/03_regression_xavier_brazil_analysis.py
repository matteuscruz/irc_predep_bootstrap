#!/usr/bin/env python3
"""
Análise de Regressão Linear - Brasil com Dados XAVIER (Modal Paralelo)
Executa cada trimestre em container isolado para paralelismo real.
Substitui o PREDEP por regressão linear simples (r²) para comparação.

Suporta todos os índices MoV disponíveis em data/raw/mov_indices/.
Processa sequencialmente (um índice por vez), com 4 seasons em paralelo
dentro de cada índice. Faz download dos artefatos ao final de cada índice.
"""

import modal

# Imagem com dependências
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas", "xarray[complete]", "numpy", "scipy",
        "matplotlib", "cartopy", "netcdf4", "h5netcdf", "tqdm"
    ])
    .apt_install(["libproj-dev", "proj-data", "proj-bin"])
    .add_local_python_source("src")
)

app = modal.App("regression-predep-brazil", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# Limites Brasil
BRAZIL_LAT_MIN, BRAZIL_LAT_MAX = -34.0, 6.0
BRAZIL_LON_MIN, BRAZIL_LON_MAX = -75.0, -34.0

# Global compartilhado via fork para workers de multiprocessing
_PRECIP_ARR_GLOBAL = None


def _compute_lag_worker(args):
    """Worker: calcula r2_map para um lag. Usa _PRECIP_ARR_GLOBAL via fork."""
    import numpy as np
    import warnings
    import time as _time
    from tqdm import tqdm
    from src.regression_calculation.core import regression_calc
    warnings.filterwarnings('ignore')

    lag, mov_aligned, n_lat, n_lon = args
    precip_arr = _PRECIP_ARR_GLOBAL  # herdado via fork (Linux copy-on-write)

    t0 = _time.time()
    r2_map = np.full((n_lat, n_lon), float('nan'))
    for i in tqdm(
        range(n_lat), desc=f"Lag {lag:2d}",
        position=lag % 8, leave=True, ncols=80
    ):
        for j in range(n_lon):
            pp = precip_arr[:, i, j]
            mask = ~np.isnan(mov_aligned) & ~np.isnan(pp)
            if mask.sum() < 10:
                continue
            pp_v = pp[mask]
            mv_v = mov_aligned[mask]
            if len(np.unique(mv_v)) < 3 or len(np.unique(pp_v)) < 3:
                continue
            r2_map[i, j] = regression_calc(mv_v, pp_v)
    elapsed = _time.time() - t0
    valid = int(np.sum(~np.isnan(r2_map)))
    print(
        f"  Lag {lag:2d} finalizado em {elapsed/60:.1f}min"
        f" — {valid} pontos válidos",
        flush=True
    )
    return lag, r2_map


@app.function(
    memory=4096,
    cpu=8,
    timeout=3600 * 12,
    volumes={"/data": volume}
)
def process_season(season: str, mov_name: str):
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
        seasonal_means = season_subset.groupby(
            season_subset.time.dt.year
        ).mean('time')
        anomalies = seasonal_means - seasonal_means.mean('year')
        ds.close()
        return anomalies

    def load_mov_index():
        mov_file = f"/data/input/{mov_name}_indice_data.csv"
        mov_data = pd.read_csv(mov_file)
        mov_data['time'] = pd.to_datetime(mov_data['time'])
        mov_data['Year'] = mov_data['time'].dt.year
        mov_data['Month'] = mov_data['time'].dt.month
        return (
            mov_data.groupby(['Year', 'Month'])['value']
            .mean().reset_index()
            .sort_values(['Year', 'Month'])
        )

    def precompute_mov_alignment(mov_monthly, precip_years, lag, season_month):
        """Pré-alinha índice MoV para todos os anos de precipitação neste lag."""
        aligned = np.full(len(precip_years), np.nan)
        for k, year in enumerate(precip_years):
            tm, ty = season_month - lag, int(year)
            while tm <= 0:
                tm += 12
                ty -= 1
            row = mov_monthly[
                (mov_monthly['Year'] == ty) & (mov_monthly['Month'] == tm)
            ]
            if len(row) == 1:
                aligned[k] = row['value'].iloc[0]
        return aligned

    # ====== EXECUÇÃO ======
    volume.reload()  # garante leitura do estado atual do volume
    start_time = time.time()
    sep = '=' * 60
    print(f"\n{sep}")
    print(f"CONTAINER {mov_name}/{season} - Iniciando (8 CPUs)")
    print(datetime.now().strftime('%H:%M:%S'))
    print(sep)

    print(f"Carregando dados para {mov_name}/{season}...")
    precip_seasonal = load_xavier_seasonal()
    mov_monthly = load_mov_index()
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

    # Pré-alinhar índice MoV para cada lag (barato: ~59 lookups por lag)
    print(f"Pré-alinhando {mov_name} para {n_lags} lags...")
    lag_args = []
    for lag in range(n_lags):
        mov_aligned = precompute_mov_alignment(
            mov_monthly, precip_years, lag, season_month
        )
        lag_args.append((lag, mov_aligned, n_lat, n_lon))

    # Processar todos os lags em paralelo (8 workers)
    print(f"Processando {n_lags} lags em paralelo (8 workers)...")
    t_pool = time.time()
    lag_results = []
    with mp.Pool(processes=8) as pool:
        for lag, r2_map in pool.imap_unordered(_compute_lag_worker, lag_args):
            elapsed = time.time() - t_pool
            valid = int(np.sum(~np.isnan(r2_map)))
            print(
                f"  [+{elapsed:5.1f}s] Lag {lag:2d} recebido"
                f" — {valid} pontos válidos",
                flush=True
            )
            lag_results.append((lag, r2_map))

    # Montar dataset de resultados
    results = xr.Dataset(
        coords={
            'lag': range(n_lags),
            'latitude': precip_seasonal.latitude,
            'longitude': precip_seasonal.longitude
        }
    )
    for lag, r2_map in sorted(lag_results, key=lambda x: x[0]):
        results[f'r2_lag{lag}'] = (['latitude', 'longitude'], r2_map)
        valid = int(np.sum(~np.isnan(r2_map)))
        print(f"   Lag {lag}: {valid} pontos válidos")

    # Salvar em subdiretório por índice MoV
    output_dir = Path(f"/data/results/regression_brazil/{mov_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_file = output_dir / f"{mov_name}_{season}_regression_brazil.nc"
    results.to_netcdf(nc_file)

    duration = time.time() - start_time
    print(f"\n{mov_name}/{season} CONCLUÍDO em {duration/60:.1f}min")
    print(f"Arquivo: {nc_file}")

    return {
        'mov_name': mov_name,
        'season': season,
        'duration': duration,
        'file': str(nc_file),
        'n_lags': n_lags,
        'valid_points': int(np.sum(~np.isnan(results['r2_lag0'].values)))
    }


@app.function(
    memory=4096,
    timeout=3600,
    volumes={"/data": volume}
)
def combine_results(mov_name: str):
    """Combina resultados de todas as estações para um índice MoV."""
    import xarray as xr
    from pathlib import Path

    print(f"\nCombinando resultados para {mov_name}...")

    output_dir = Path(f"/data/results/regression_brazil/{mov_name}")
    seasons = ["DJF", "MAM", "JJA", "SON"]

    all_data = []
    for season in seasons:
        nc_file = output_dir / f"{mov_name}_{season}_regression_brazil.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            all_data.append(ds)
            print(f"   + {season}")

    if len(all_data) == 4:
        combined = xr.concat(all_data, dim='season')
        combined_file = (
            output_dir / f"{mov_name}_ALL_SEASONS_regression_combined.nc"
        )
        combined.to_netcdf(combined_file)
        print(f"Arquivo combinado: {combined_file}")
        return str(combined_file)

    return None


# ====== FUNÇÕES LOCAIS ======

LOCAL_XAVIER_FILE = (
    "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
)
LOCAL_XAVIER_RAW_FILES = [
    "data/raw/xavier_daily/"
    "pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.2-002.nc",
    "data/raw/xavier_daily/"
    "pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc",
    "data/raw/xavier_daily/"
    "pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc",
    "data/raw/xavier_daily/"
    "pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-004.nc",
]


def discover_mov_indices():
    """Retorna lista de (mov_name, Path) para todos os índices disponíveis."""
    import re
    from pathlib import Path
    mov_dir = Path("data/raw/mov_indices")
    indices = []
    for f in sorted(mov_dir.glob("*_indice*.csv")):
        match = re.match(r"^(.+?)_indice", f.name)
        if match:
            indices.append((match.group(1), f))
    return indices


def generate_brazil_monthly(output_path):
    """Gera arquivo mensal do Brasil a partir dos dados diários Xavier."""
    import xarray as xr
    import numpy as np
    from pathlib import Path

    print("   Gerando arquivo mensal Brasil a partir dos dados diários...")

    available = [f for f in LOCAL_XAVIER_RAW_FILES if Path(f).exists()]
    if not available:
        raise FileNotFoundError(
            "Nenhum arquivo diário Xavier encontrado em: "
            f"{LOCAL_XAVIER_RAW_FILES}"
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


def upload_xavier_data():
    """Faz upload do arquivo Xavier de precipitação para o volume Modal."""
    from pathlib import Path

    print("\nUploading Xavier precipitation data to Modal volume...")

    xavier_path = Path(LOCAL_XAVIER_FILE)
    if not xavier_path.exists():
        print("   Arquivo mensal Brasil não encontrado. Gerando...")
        generate_brazil_monthly(xavier_path)

    remote_name = "pr_monthly_1961_2019_BRAZIL_combined.nc"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(xavier_path), f"/input/{remote_name}")

    size_mb = xavier_path.stat().st_size / 1e6
    print(f"   Xavier data: {xavier_path.name} ({size_mb:.0f} MB)")
    print("   Upload concluído!\n")


def upload_mov_index(mov_name, local_path):
    """Faz upload do CSV de um índice MoV para o volume Modal."""
    import pandas as pd
    import tempfile
    import os
    from pathlib import Path

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Arquivo MoV não encontrado: {local_path}")

    df = pd.read_csv(local_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col = df.columns[1]
    mov_csv = df[[date_col, val_col]].copy()
    mov_csv.columns = ['time', 'value']
    mov_csv['time'] = pd.to_datetime(mov_csv['time'])
    mov_csv = mov_csv.dropna(subset=['value'])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mov_csv.to_csv(f, index=False)
        tmp_path = f.name

    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(tmp_path, f"/input/{mov_name}_indice_data.csv")
    finally:
        os.unlink(tmp_path)

    yr_min = mov_csv['time'].dt.year.min()
    yr_max = mov_csv['time'].dt.year.max()
    print(f"   {mov_name}: {len(mov_csv)} registros ({yr_min}-{yr_max})")


def download_results(mov_name):
    """Baixa resultados de um índice MoV do volume Modal para o disco local."""
    import subprocess
    from pathlib import Path

    local_dir = Path(f"results/regression/{mov_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando resultados de {mov_name}...")
    result = subprocess.run(
        ["modal", "volume", "get", "xavier-data",
         f"results/regression_brazil/{mov_name}",
         str(local_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"   Erro no download: {result.stderr.strip()}")
    else:
        files = list(local_dir.glob("**/*.nc"))
        print(f"   {len(files)} arquivo(s) em {local_dir}/")


@app.local_entrypoint()
def main():
    """Executa análise de regressão para todos os índices MoV, sequencialmente."""
    from datetime import datetime
    import time

    start_time = time.time()
    seasons = ["DJF", "MAM", "JJA", "SON"]

    mov_list = discover_mov_indices()

    sep = "=" * 80
    print(sep)
    print("ANALISE DE REGRESSAO LINEAR PARALELA - MODAL (BRASIL)")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Metrica: r2 (coeficiente de determinacao)")
    print("Regiao: Brasil completo")
    print(f"Indices MoV: {len(mov_list)} encontrados")
    for mov_name, _ in mov_list:
        print(f"  - {mov_name}")
    print(sep)

    # Upload Xavier uma vez (compartilhado entre todos os índices)
    upload_xavier_data()

    all_results = []

    for idx, (mov_name, mov_path) in enumerate(mov_list, 1):
        mov_start = time.time()
        sep60 = "=" * 60
        print(f"\n{sep60}")
        print(f"[{idx}/{len(mov_list)}] Processando indice: {mov_name}")
        print(sep60)

        # Upload CSV do índice atual
        print(f"Uploading {mov_name}...")
        upload_mov_index(mov_name, mov_path)

        # Executa 4 seasons em paralelo para este índice
        print(f"Enviando 4 containers paralelos para {mov_name}...")
        season_results = list(
            process_season.starmap([(s, mov_name) for s in seasons])
        )

        for r in season_results:
            dur = r['duration'] / 60
            pts = r['valid_points']
            print(f"  {r['mov_name']}/{r['season']}: {dur:.1f}min — {pts} pts")

        # Combina as 4 seasons
        print(f"Combinando estacoes para {mov_name}...")
        combined_file = combine_results.remote(mov_name)

        # Download imediato dos artefatos deste índice
        download_results(mov_name)

        mov_duration = time.time() - mov_start
        print(f"\n{mov_name} concluido em {mov_duration/60:.1f}min")

        all_results.append({
            'mov_name': mov_name,
            'duration': mov_duration,
            'combined_file': combined_file,
            'seasons': season_results
        })

    total_duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("ANALISE DE REGRESSAO CONCLUIDA!")
    print("=" * 80)
    print(f"Tempo total: {total_duration/60:.1f}min ({total_duration/3600:.2f}h)")
    print("Resumo por indice:")
    for r in all_results:
        dur = r['duration'] / 60
        print(f"  {r['mov_name']}: {dur:.1f}min — {r['combined_file']}")

    print("\nResultados locais em: results/regression/")

    return {
        'total_duration': total_duration,
        'results': all_results,
    }
