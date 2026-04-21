#!/usr/bin/env python3
"""
PREDEP Granular por Ponto — Bacia do Uruguai (Exp 06)

Calcula alpha_legacy, alpha_core e r² para cada ponto de grade dentro da
bbox da bacia do Uruguai, sem agregação espacial por cluster.

Objetivo: comparar com exp 05 (clusters) para avaliar se a agregação
espacial amplifica ou atenua o sinal PREDEP dos MoVs.

Uso:
  modal run experiments/06_predep_comparisson_uruguai_granular.py --test
  modal run experiments/06_predep_comparisson_uruguai_granular.py --movs "ONI,NIN04,SOI"
"""

import modal

image = modal.Image.debian_slim().pip_install(
    [
        "pandas",
        "xarray[complete]",
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "netcdf4",
        "h5netcdf",
    ]
)

app    = modal.App("predep-granular-uruguai", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# ── Constantes ────────────────────────────────────────────────────────────────
URU_LAT_MIN, URU_LAT_MAX = -32.3, -25.7
URU_LON_MIN, URU_LON_MAX = -58.1, -48.8

N_LAGS = 12

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SEASON_MONTH_MAP = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}

DEFAULT_MOVS = ["ONI", "NIN04", "SOI"]

# Compartilhado via fork com os workers de multiprocessing
_PRECIP_ARR_GLOBAL = None  # (n_years, n_lat, n_lon)


# ── Funções de cálculo (inlineadas — Modal não acessa src/) ───────────────────

def _ecdf(data):
    import numpy as np
    s = np.sort(data)
    n = len(s)
    return lambda x: np.searchsorted(s, x, side="right") / n


def _predep_legacy(X, Y, n_boot=5000):
    """PREDEP versão original: particiona Y (precip), condiciona X (MoV)."""
    import numpy as np
    import scipy.stats as ss
    from astropy.stats import bayesian_blocks

    try:
        s_x = float(ss.gaussian_kde(
            np.random.choice(X, n_boot) - np.random.choice(X, n_boot)
        ).pdf(0))
        edges_y = bayesian_blocks(Y)
        ecdf_y  = _ecdf(Y)
        s_x_mid_y = 0.0
        if edges_y.shape[0] > 1:
            for k in range(1, edges_y.shape[0]):
                bg, ed = edges_y[k - 1], edges_y[k]
                X_c = X[(Y >= bg) & (Y < ed)]
                if X_c.shape[0] < 2 or len(np.unique(X_c)) < 2:
                    continue
                dx_c = np.random.choice(X_c, n_boot) - np.random.choice(X_c, n_boot)
                if np.std(dx_c) == 0:
                    continue
                s_x_mid_y += (ecdf_y(ed) - ecdf_y(bg)) * float(
                    ss.gaussian_kde(dx_c).pdf(0)
                )
            if s_x_mid_y == 0:
                return float("nan")
            return float((s_x_mid_y - s_x) / s_x_mid_y)
        return float("nan")
    except Exception:
        return float("nan")


def _predep_core(X, Y, n_boot=5000):
    """PREDEP versão modificada: X = preditor (MoV), particiona X, condiciona Y."""
    import numpy as np
    import scipy.stats as ss
    from astropy.stats import bayesian_blocks

    try:
        s_y = float(ss.gaussian_kde(
            np.random.choice(Y, n_boot) - np.random.choice(Y, n_boot)
        ).pdf(0))
        edges_x = bayesian_blocks(X)
        ecdf_x  = _ecdf(X)
        s_y_mid_x = 0.0
        if edges_x.shape[0] > 1:
            for k in range(1, edges_x.shape[0]):
                bg, ed = edges_x[k - 1], edges_x[k]
                Y_c = Y[(X >= bg) & (X < ed)]
                if Y_c.shape[0] < 2 or len(np.unique(Y_c)) < 2:
                    continue
                dy_c = np.random.choice(Y_c, n_boot) - np.random.choice(Y_c, n_boot)
                if np.std(dy_c) == 0:
                    continue
                s_y_mid_x += (ecdf_x(ed) - ecdf_x(bg)) * float(
                    ss.gaussian_kde(dy_c).pdf(0)
                )
            if s_y_mid_x == 0:
                return float("nan")
            return float((s_y_mid_x - s_y) / s_y_mid_x)
        return float("nan")
    except Exception:
        return float("nan")


def _regression(X, Y):
    """r² da regressão linear simples."""
    import numpy as np
    import scipy.stats as ss

    try:
        if len(X) < 3 or len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
            return float("nan")
        return float(ss.linregress(X, Y).rvalue ** 2)
    except Exception:
        return float("nan")


def _precompute_mov_alignment(mov_lookup, years, season_month, lag):
    """Pré-alinha série do MoV para todos os anos dado um lag."""
    import numpy as np

    n = len(years)
    aligned = np.full(n, np.nan)
    for k in range(n):
        tm = season_month - lag
        ty = int(years[k])
        while tm <= 0:
            tm += 12
            ty -= 1
        aligned[k] = mov_lookup.get((ty, tm), np.nan)
    return aligned


# ── Worker ────────────────────────────────────────────────────────────────────

def _compute_lag_worker(args):
    """
    Worker: calcula as 3 métricas em cada pixel para um lag.
    Usa _PRECIP_ARR_GLOBAL (n_years, n_lat, n_lon) herdado via fork.

    args: (lag, mov_aligned, n_boot)
    Retorna (lag, al_map, ac_map, r2_map) — shape (n_lat, n_lon) cada.
    """
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    lag, mov_aligned, n_boot = args
    precip_arr = _PRECIP_ARR_GLOBAL  # (n_years, n_lat, n_lon)
    n_lat = precip_arr.shape[1]
    n_lon = precip_arr.shape[2]

    al_map = np.full((n_lat, n_lon), np.nan)
    ac_map = np.full((n_lat, n_lon), np.nan)
    r2_map = np.full((n_lat, n_lon), np.nan)

    valid_mov = ~np.isnan(mov_aligned)

    for i in range(n_lat):
        for j in range(n_lon):
            pp = precip_arr[:, i, j]
            valid = valid_mov & ~np.isnan(pp)
            if valid.sum() < 10:
                continue
            X = mov_aligned[valid]   # MoV (preditor)
            Y = pp[valid]             # precipitação (resposta)
            if len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
                continue
            al_map[i, j] = _predep_legacy(X, Y, n_boot)
            ac_map[i, j] = _predep_core(X, Y, n_boot)
            r2_map[i, j] = _regression(X, Y)

    valid_count = int(np.sum(~np.isnan(al_map)))
    print(f"  Lag {lag:2d}: {valid_count}/{n_lat * n_lon} pts válidos", flush=True)
    return lag, al_map, ac_map, r2_map


# ── Função Modal ──────────────────────────────────────────────────────────────

@app.function(
    memory=4096,
    cpu=8,
    timeout=3600 * 12,
    volumes={"/data": volume},
)
def process_mov(mov_name: str, n_boot: int = 5000, exp_n: int = 1):
    """
    Container Modal: PREDEP granular ponto-a-ponto no bbox do Uruguai.

    Passos:
      1. Xavier mensal recortado na bbox do Uruguai
      2. MoV mensal → lookup (year, month)
      3. Para cada estação: anomalia sazonal (n_years, n_lat, n_lon)
         → expõe global → pool de 8 workers, um por lag
      4. Salva NC com alpha_legacy, alpha_core, r2 em (season, lag, lat, lon)
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    import multiprocessing as mp
    from pathlib import Path
    import time
    from datetime import datetime
    import warnings

    warnings.filterwarnings("ignore")

    global _PRECIP_ARR_GLOBAL

    volume.reload()
    start = time.time()
    print(f"\n{'='*60}")
    print(f"CONTAINER GRANULAR — MoV: {mov_name}")
    print(datetime.now().strftime("%H:%M:%S"))
    print("=" * 60)

    # ── 1. Xavier mensal recortado ────────────────────────────────────────────
    ds = xr.open_dataset("/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc")
    ds_crop = ds.sel(
        latitude=slice(URU_LAT_MIN, URU_LAT_MAX),
        longitude=slice(URU_LON_MIN, URU_LON_MAX),
    ).load()
    ds.close()

    time_idx    = pd.DatetimeIndex(ds_crop.time.values)
    time_years  = time_idx.year.values
    time_months = time_idx.month.values
    n_lat       = len(ds_crop.latitude)
    n_lon       = len(ds_crop.longitude)
    lats        = ds_crop.latitude.values
    lons        = ds_crop.longitude.values
    pr_values   = ds_crop.pr.values  # (n_time, n_lat, n_lon)

    print(f"  Xavier: {pr_values.shape}  "
          f"({time_idx[0].date()} → {time_idx[-1].date()})")
    print(f"  Grade: {n_lat}×{n_lon} = {n_lat * n_lon} pts")

    # ── 2. MoV mensal ─────────────────────────────────────────────────────────
    mov_df = pd.read_csv(f"/data/input/{mov_name}_indice_data.csv")
    mov_df["time"]  = pd.to_datetime(mov_df["time"])
    mov_df["Year"]  = mov_df["time"].dt.year
    mov_df["Month"] = mov_df["time"].dt.month
    mov_monthly = (
        mov_df.groupby(["Year", "Month"])["value"]
        .mean().reset_index().sort_values(["Year", "Month"])
    )
    mov_lookup = {
        (int(r.Year), int(r.Month)): r.value
        for _, r in mov_monthly.iterrows()
        if not np.isnan(r.value)
    }
    print(f"\n  MoV '{mov_name}': "
          f"{int(mov_monthly['value'].notna().sum())} meses válidos")

    # ── 3. Por estação: anomalia por ponto + PREDEP (paralelo por lag) ─────────
    n_seasons = len(SEASONS)
    al_arr = np.full((n_seasons, N_LAGS, n_lat, n_lon), np.nan)
    ac_arr = np.full((n_seasons, N_LAGS, n_lat, n_lon), np.nan)
    r2_arr = np.full((n_seasons, N_LAGS, n_lat, n_lon), np.nan)

    for s_idx, season in enumerate(SEASONS):
        s_months    = SEASON_MONTHS[season]
        s_mask      = np.isin(time_months, s_months)
        s_years_raw = time_years[s_mask]
        if season == "DJF":
            # Dezembro de ano y pertence ao DJF do ano y+1 (Dec(y-1)+Jan(y)+Feb(y))
            s_years = np.where(time_months[s_mask] == 12, s_years_raw + 1, s_years_raw)
            unique_years = np.unique(s_years)
            counts = np.array([np.sum(s_years == y) for y in unique_years])
            unique_years = unique_years[counts == 3]
        else:
            s_years      = s_years_raw
            unique_years = np.unique(s_years)
        n_years = len(unique_years)

        # Anomalia sazonal por ponto: (n_years, n_lat, n_lon)
        pr_s = pr_values[s_mask]  # (n_s_months, n_lat, n_lon)
        pr_annual = np.array([
            pr_s[s_years == y].mean(axis=0) for y in unique_years
        ])
        anomaly = pr_annual - pr_annual.mean(axis=0, keepdims=True)

        anom_std = float(anomaly.std())
        print(f"\n  {season}: {n_years} anos  anom_std={anom_std:.4e}")

        # Expõe array para os workers via fork
        _PRECIP_ARR_GLOBAL = anomaly

        # Pré-alinha MoV para cada lag
        season_month = SEASON_MONTH_MAP[season]
        lag_args = [
            (
                lag,
                _precompute_mov_alignment(mov_lookup, unique_years, season_month, lag),
                n_boot,
            )
            for lag in range(1, N_LAGS + 1)
        ]

        print(f"  [{season}] 3 métricas × {n_lat * n_lon} pts "
              f"× {N_LAGS} lags  (workers=8)...")
        t_pool = time.time()
        with mp.Pool(processes=8) as pool:
            for lag, al_map, ac_map, r2_map in pool.imap_unordered(
                _compute_lag_worker, lag_args
            ):
                al_arr[s_idx, lag - 1] = al_map
                ac_arr[s_idx, lag - 1] = ac_map
                r2_arr[s_idx, lag - 1] = r2_map
        print(f"  [{season}] Pool finalizado em {time.time() - t_pool:.1f}s")

        for li in range(N_LAGS):
            v = int(np.sum(~np.isnan(al_arr[s_idx, li])))
            print(f"    Lag {li + 1:2d}: {v}/{n_lat * n_lon} pts válidos")

    # ── 4. Salvar NetCDF ──────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
            "alpha_legacy": (
                ["season", "lag", "latitude", "longitude"], al_arr
            ),
            "alpha_core": (
                ["season", "lag", "latitude", "longitude"], ac_arr
            ),
            "r2": (
                ["season", "lag", "latitude", "longitude"], r2_arr
            ),
        },
        coords={
            "season":    SEASONS,
            "lag":       list(range(1, N_LAGS + 1)),
            "latitude":  lats,
            "longitude": lons,
        },
    )
    ds_out.attrs["description"] = (
        f"PREDEP granular ponto-a-ponto — bbox Uruguai vs MoV {mov_name}"
    )
    ds_out["alpha_legacy"].attrs["description"] = (
        "predep(X=MoV, Y=precip) — versão original"
    )
    ds_out["alpha_core"].attrs["description"] = (
        "predep(X=MoV, Y=precip) — versão modificada"
    )
    ds_out["r2"].attrs["description"] = "r² regressão linear — baseline"

    out_dir = Path(f"/data/results/predep_granular_uruguai/exp{exp_n}/{mov_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_file = out_dir / f"{mov_name}_predep_granular_seasonal.nc"
    ds_out.to_netcdf(nc_file)
    volume.commit()

    duration = time.time() - start
    print(f"\n  {mov_name} CONCLUÍDO em {duration / 60:.1f}min")
    print(f"  Arquivo: {nc_file}")

    return {
        "mov_name": mov_name,
        "duration": duration,
        "file": str(nc_file),
        "n_pts": n_lat * n_lon,
    }


# ── Funções locais ────────────────────────────────────────────────────────────

LOCAL_XAVIER_FILE = (
    "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
)


def discover_mov_indices(mov_names=None):
    import re
    from pathlib import Path

    mov_dir = Path("data/raw/mov_indices")
    all_indices = {}
    for f in sorted(mov_dir.glob("*_indice*.csv")):
        match = re.match(r"^(.+?)_indice", f.name)
        if match:
            all_indices[match.group(1)] = f

    if mov_names is None:
        mov_names = DEFAULT_MOVS

    result = []
    for name in mov_names:
        if name in all_indices:
            result.append((name, all_indices[name]))
        else:
            print(f"  [AVISO] MoV '{name}' não encontrado.")
    return result


def upload_xavier_data():
    from pathlib import Path

    print("\nUploading Xavier mensal...")
    xavier_path = Path(LOCAL_XAVIER_FILE)
    if not xavier_path.exists():
        raise FileNotFoundError(f"Não encontrado: {xavier_path}")
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(
                str(xavier_path),
                "input/pr_monthly_1961_2019_BRAZIL_combined.nc",
            )
        print(f"  Xavier: {xavier_path.name} ✓")
    except FileExistsError:
        print("  Xavier: já existe no volume — skip ✓")


def upload_mov_index(mov_name: str, local_path):
    import pandas as pd
    import tempfile
    import os
    from pathlib import Path

    local_path = Path(local_path)
    df = pd.read_csv(local_path)
    df.columns = [c.strip() for c in df.columns]
    mov_csv = df[[df.columns[0], df.columns[1]]].copy()
    mov_csv.columns = ["time", "value"]
    mov_csv["time"] = pd.to_datetime(mov_csv["time"])
    mov_csv = mov_csv.dropna(subset=["value"])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mov_csv.to_csv(f, index=False)
        tmp_path = f.name
    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(tmp_path, f"input/{mov_name}_indice_data.csv")
    finally:
        os.unlink(tmp_path)

    yr_min = mov_csv["time"].dt.year.min()
    yr_max = mov_csv["time"].dt.year.max()
    print(f"  {mov_name}: {len(mov_csv)} registros ({yr_min}–{yr_max})")


def download_results(mov_name: str, exp_n: int = 1):
    import subprocess
    import shutil
    from pathlib import Path

    local_dir = Path(f"results/predep_granular_uruguai/exp{exp_n}/{mov_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando resultados de {mov_name} → exp{exp_n}...")
    result = subprocess.run(
        [
            "modal", "volume", "get", "xavier-data",
            f"results/predep_granular_uruguai/exp{exp_n}/{mov_name}",
            str(local_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Erro no download: {result.stderr.strip()}")
        return

    for f in sorted(local_dir.rglob("*.nc")):
        if f.parent != local_dir:
            shutil.move(str(f), str(local_dir / f.name))
    for d in sorted(local_dir.rglob("*/"), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass

    files = list(local_dir.glob("*.nc"))
    print(f"  {len(files)} arquivo(s) em {local_dir}/")


def get_next_exp_n(base_dir: str) -> int:
    import re
    from pathlib import Path

    base = Path(base_dir)
    if not base.exists():
        return 1
    existing = [
        int(m.group(1))
        for d in base.iterdir()
        if (m := re.match(r"^exp(\d+)$", d.name))
    ]
    return max(existing, default=0) + 1


# ── Entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(movs: str = "ONI,NIN04,SOI", n_boot: int = 5000, test: bool = False):
    """
    PREDEP granular ponto-a-ponto na bacia do Uruguai.

    Flags:
      --movs  "ONI,NIN04,SOI"   MoVs separados por vírgula
      --test                    Roda apenas o primeiro MoV
      --n-boot 1000             Bootstrap por ponto (padrão: 1000)

    Exemplos:
      modal run experiments/06_predep_comparisson_uruguai_granular.py --test
      modal run experiments/06_predep_comparisson_uruguai_granular.py --movs "ONI,SOI"
    """
    from datetime import datetime
    from pathlib import Path
    import time
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.visualization.cluster_plots import generate_plots_granular

    mov_names = [m.strip() for m in movs.split(",")]
    mov_list  = discover_mov_indices(mov_names)

    if test:
        mov_list = mov_list[:1]

    exp_n   = get_next_exp_n("results/predep_granular_uruguai")
    exp_dir = Path(f"results/predep_granular_uruguai/exp{exp_n}")

    print("=" * 80)
    print("PREDEP GRANULAR — BACIA DO URUGUAI (POR PONTO) — MODAL")
    if test:
        print("*** MODO TESTE — apenas 1 MoV ***")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experimento: exp{exp_n}  |  Saída: {exp_dir}/")
    print(f"MoVs: {[m for m, _ in mov_list]}")
    print(f"Estações: {', '.join(SEASONS)}  |  Lags: 1–{N_LAGS}  |  n_boot={n_boot}")
    print("Métricas: alpha_legacy, alpha_core, r2  (por pixel, sem agregação)")
    print("=" * 80)

    upload_xavier_data()

    all_results = []
    start_total = time.time()

    for idx, (mov_name, mov_path) in enumerate(mov_list, 1):
        mov_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(mov_list)}] MoV: {mov_name}")
        print("=" * 60)

        upload_mov_index(mov_name, mov_path)
        result = process_mov.remote(mov_name, n_boot, exp_n)
        download_results(mov_name, exp_n)

        nc_file   = exp_dir / mov_name / f"{mov_name}_predep_granular_seasonal.nc"
        plots_dir = exp_dir / "plots"
        print(f"\n  Gerando plots para {mov_name} (6 figuras)...")
        generate_plots_granular(nc_file, mov_name, plots_dir)

        mov_duration = time.time() - mov_start
        print(f"\n  {mov_name} concluído em {mov_duration / 60:.1f}min")
        all_results.append({"mov_name": mov_name, "duration": mov_duration, **result})

    total = time.time() - start_total
    print("\n" + "=" * 80)
    print("PREDEP GRANULAR URUGUAI — CONCLUÍDO!")
    print("=" * 80)
    print(f"Tempo total: {total / 60:.1f}min ({total / 3600:.2f}h)")
    print(f"Resultados em: {exp_dir}/")
    for r in all_results:
        print(f"  {r['mov_name']}: {r['duration'] / 60:.1f}min  "
              f"({r['n_pts']} pontos)")

    return {"total_duration": total, "results": all_results}
