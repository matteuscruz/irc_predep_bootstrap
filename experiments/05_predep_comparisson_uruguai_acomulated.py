#!/usr/bin/env python3
"""
Comparação PREDEP vs Regressão — Clusters da Bacia do Uruguai

Compara duas métricas de previsibilidade nos 4 clusters da bacia do Uruguai:
  - alpha_core : src/predep_calculation/core.py    — predep(X=MoV, Y=precip)
  - r2         : src/regression_calculation/core.py — r² linear como baseline

Aggregação espacial idêntica ao experimento 04 (integração trapezoidal cos(lat)).
Estrutura Modal idêntica: 1 container por MoV, pool de 8 workers por lag/estação.

Uso:
  modal run experiments/05_predep_comparisson_uruguai_acomulated.py
  modal run experiments/05_predep_comparisson_uruguai_acomulated.py --movs "ONI,SOI"
  modal run experiments/05_predep_comparisson_uruguai_acomulated.py --test
"""

import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
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
    .add_local_python_source("src")
)

app    = modal.App("predep-comparison-uruguai", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# ── Constantes ────────────────────────────────────────────────────────────────
URU_LAT_MIN, URU_LAT_MAX = -32.3, -25.7
URU_LON_MIN, URU_LON_MAX = -58.1, -48.8

N_LAGS = 12  # lags 1–12

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SEASON_MONTH_MAP = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}

DEFAULT_MOVS = ["ONI", "NIN04", "SOI"]




# ── Worker ────────────────────────────────────────────────────────────────────

def _compute_lag_worker(args):
    """
    Worker: para um lag e uma estação, alinha o MoV e calcula as 2 métricas
    para todos os clusters.

    args: (lag, cluster_season_data, mov_lookup, season_month, n_boot)
      cluster_season_data: {cname: (anomaly_array, years_array)}
    Retorna (lag, {cname: (alpha_core, r2)}).
    """
    import numpy as np
    import warnings
    from src.predep_calculation.core import predep
    from src.regression_calculation.core import regression_calc

    warnings.filterwarnings("ignore")

    lag, cluster_season_data, mov_lookup, n_boot = args

    results = {}
    for cname, (anomaly, s_years, s_months) in cluster_season_data.items():
        n_obs = len(s_years)
        mov_aligned = np.full(n_obs, np.nan)
        for k in range(n_obs):
            tm = int(s_months[k]) - lag
            ty = int(s_years[k])
            while tm <= 0:
                tm += 12
                ty -= 1
            mov_aligned[k] = mov_lookup.get((ty, tm), np.nan)

        valid_mask = ~np.isnan(mov_aligned) & ~np.isnan(anomaly)
        if valid_mask.sum() < 10:
            results[cname] = (float("nan"), float("nan"))
            continue

        X = mov_aligned[valid_mask]   # MoV (preditor)
        Y = anomaly[valid_mask]       # precipitação (resposta)

        if len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
            results[cname] = (float("nan"), float("nan"))
            continue

        ac = predep(X, Y, n_boot, method="bayesian_blocks")
        r2 = regression_calc(X, Y)

        results[cname] = (ac, r2)

    return lag, results


# ── Função Modal ──────────────────────────────────────────────────────────────

@app.function(
    memory=4096,
    cpu=8,
    timeout=3600 * 8,
    volumes={"/data": volume},
)
def process_mov(mov_name: str, n_boot: int = 5000, exp_n: int = 1):
    """
    Container Modal: calcula PREDEP e r² sazonais por cluster para um MoV.

    Passos:
      1. Xavier mensal recortado na bbox do Uruguai
      2. Máscaras de cluster pré-computadas
      3. Integração espacial → série mensal acumulada por cluster
      4. Agregação sazonal anual + anomalia (DJF/MAM/JJA/SON)
      5. MoV mensal → lookup (year, month)
      6. Para cada estação × lag: alpha_core, r2 (paralelo)
      7. Salva NetCDF com 2 variáveis (season, cluster, lag)
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

    volume.reload()
    start = time.time()
    print(f"\n{'='*60}")
    print(f"CONTAINER — MoV: {mov_name}")
    print(datetime.now().strftime("%H:%M:%S"))
    print("=" * 60)

    # ── 1. Xavier mensal recortado ────────────────────────────────────────────
    xavier_file = "/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc"
    ds = xr.open_dataset(xavier_file)
    ds_crop = ds.sel(
        latitude=slice(URU_LAT_MIN, URU_LAT_MAX),
        longitude=slice(URU_LON_MIN, URU_LON_MAX),
    ).load()
    ds.close()
    time_idx    = pd.DatetimeIndex(ds_crop.time.values)
    time_years  = time_idx.year.values
    time_months = time_idx.month.values
    n_time      = len(time_idx)
    n_lat       = len(ds_crop.latitude)
    n_lon       = len(ds_crop.longitude)
    print(f"  Xavier: ({n_time}, {n_lat}, {n_lon})  "
          f"({time_idx[0].date()} → {time_idx[-1].date()})")

    # ── 2. Máscaras de cluster ────────────────────────────────────────────────
    npz = np.load("/data/input/uruguai_cluster_masks.npz", allow_pickle=True)
    cluster_names = list(npz["cluster_names"])
    masks_flat    = npz["masks_flat"].astype(bool)
    print(f"  Clusters: {len(cluster_names)}")

    # ── 3. Integração espacial mensal por cluster (Eq.1) ─────────────────────
    R_EARTH = 6_371_000.0
    cos_lat = np.cos(np.deg2rad(ds_crop.latitude))

    cluster_monthly = {}
    for i, cname in enumerate(cluster_names):
        mask = masks_flat[i]
        if mask.sum() == 0:
            print(f"  [AVISO] '{cname}': sem pontos — ignorado.")
            continue
        mask_2d = xr.DataArray(
            mask.reshape(n_lat, n_lon),
            dims=["latitude", "longitude"],
            coords={"latitude": ds_crop.latitude, "longitude": ds_crop.longitude},
        )
        pr_cluster = (
            (ds_crop.pr.where(mask_2d, other=0.0) * cos_lat)
            .integrate("latitude")
            .integrate("longitude")
            .values
        ) * R_EARTH**2 * (np.pi / 180) ** 2
        cluster_monthly[cname] = pr_cluster
        print(f"  {cname}  pts={mask.sum():4d}  pr_mean={pr_cluster.mean():.4e}")

    # ── 4. Anomalia mensal por cluster ────────────────────────────────────────
    cluster_seasonal = {s: {} for s in SEASONS}
    for season in SEASONS:
        s_mask         = np.isin(time_months, SEASON_MONTHS[season])
        s_times_years  = time_years[s_mask]
        s_times_months = time_months[s_mask]
        n_obs          = int(s_mask.sum())
        for cname, pr_monthly_series in cluster_monthly.items():
            # Climatologia mensal do cluster (média de cada mês calendar)
            clim_c = np.array([
                pr_monthly_series[time_months == m].mean() for m in range(1, 13)
            ])
            pr_s    = pr_monthly_series[s_mask]
            anomaly = pr_s - clim_c[s_times_months - 1]
            cluster_seasonal[season][cname] = (anomaly, s_times_years, s_times_months)
        anom_ex = next(iter(cluster_seasonal[season].values()))[0].mean()
        print(f"  {season}: {n_obs} meses  anom_mean={anom_ex:+.4e}")

    # ── 5. MoV mensal ─────────────────────────────────────────────────────────
    mov_df = pd.read_csv(f"/data/input/{mov_name}_indice_data.csv")
    mov_df["time"]  = pd.to_datetime(mov_df["time"])
    mov_df["Year"]  = mov_df["time"].dt.year
    mov_df["Month"] = mov_df["time"].dt.month
    mov_monthly = (
        mov_df.groupby(["Year", "Month"])["value"]
        .mean()
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    mov_lookup = {
        (int(r.Year), int(r.Month)): r.value
        for _, r in mov_monthly.iterrows()
        if not np.isnan(r.value)
    }
    n_valid = int(mov_monthly["value"].notna().sum())
    print(f"\n  MoV '{mov_name}': {n_valid} meses válidos")

    # ── 6. Calcular PREDEP + r² por estação × lag (paralelo) ─────────────────
    n_seasons  = len(SEASONS)
    n_clusters = len(cluster_names)
    alpha_core_arr = np.full((n_seasons, n_clusters, N_LAGS), np.nan)
    r2_arr         = np.full((n_seasons, n_clusters, N_LAGS), np.nan)

    for s_idx, season in enumerate(SEASONS):
        cluster_data = cluster_seasonal[season]

        lag_args = [
            (lag, cluster_data, mov_lookup, n_boot)
            for lag in range(1, N_LAGS + 1)
        ]

        print(f"\n  [{season}] PREDEP × {len(cluster_data)} clusters "
              f"× {N_LAGS} lags  (workers=8)...")
        t_pool = time.time()
        with mp.Pool(processes=8) as pool:
            for lag, cluster_results in pool.imap_unordered(
                _compute_lag_worker, lag_args
            ):
                for c_idx, cname in enumerate(cluster_names):
                    ac, r2 = cluster_results.get(
                        cname, (float("nan"), float("nan"))
                    )
                    alpha_core_arr[s_idx, c_idx, lag - 1] = ac
                    r2_arr[s_idx, c_idx, lag - 1]         = r2
        print(f"  [{season}] Pool finalizado em {time.time() - t_pool:.1f}s")

        for li, lag in enumerate(range(1, N_LAGS + 1)):
            v_core = int(np.sum(~np.isnan(alpha_core_arr[s_idx, :, li])))
            v_r2   = int(np.sum(~np.isnan(r2_arr[s_idx, :, li])))
            print(f"    Lag {lag:2d}: predep={v_core}/{n_clusters}  "
                  f"r2={v_r2}/{n_clusters}")

    # ── 7. Salvar NetCDF ──────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
            "alpha_core": (["season", "cluster", "lag"], alpha_core_arr),
            "r2":         (["season", "cluster", "lag"], r2_arr),
        },
        coords={
            "season":  SEASONS,
            "cluster": cluster_names,
            "lag":     list(range(1, N_LAGS + 1)),
        },
    )
    ds_out.attrs["description"] = (
        f"PREDEP + r² — clusters Uruguai vs MoV {mov_name}"
    )
    ds_out["alpha_core"].attrs["description"] = (
        "predep(X=MoV, Y=precip): alpha_{precip|MoV} — particiona MoV via Ward k=sqrt(N)"
    )
    ds_out["r2"].attrs["description"] = "r² regressão linear — baseline"

    out_dir = Path(f"/data/results/predep_comparison_uruguai/exp{exp_n}/{mov_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_file = out_dir / f"{mov_name}_predep_comparison_seasonal.nc"
    ds_out.to_netcdf(nc_file)
    volume.commit()

    duration = time.time() - start
    print(f"\n  {mov_name} CONCLUÍDO em {duration / 60:.1f}min")
    print(f"  Arquivo: {nc_file}")

    return {
        "mov_name": mov_name,
        "duration": duration,
        "file": str(nc_file),
        "n_clusters": n_clusters,
    }


# ── Funções locais ────────────────────────────────────────────────────────────

LOCAL_XAVIER_FILE = (
    "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
)
LOCAL_SUBBACIAS_SHP = "data/clusters/uruguai/URUGUAI_recortado.shp"


def precompute_cluster_masks():
    """Calcula máscaras de cluster localmente (reutiliza lógica do exp 04)."""
    import numpy as np
    import netCDF4 as nc4
    import shapefile
    from matplotlib.path import Path as MplPath
    from pathlib import Path

    print("Pré-computando máscaras de cluster (local)...")
    with nc4.Dataset(LOCAL_XAVIER_FILE) as nc:
        lats_all = nc.variables["latitude"][:]
        lons_all = nc.variables["longitude"][:]

    lat_mask = (lats_all >= URU_LAT_MIN) & (lats_all <= URU_LAT_MAX)
    lon_mask = (lons_all >= URU_LON_MIN) & (lons_all <= URU_LON_MAX)
    lats = lats_all[lat_mask]
    lons = lons_all[lon_mask]
    lons2d, lats2d = np.meshgrid(lons, lats)
    grid_pts = np.column_stack([lons2d.ravel(), lats2d.ravel()])

    sf = shapefile.Reader(LOCAL_SUBBACIAS_SHP, encoding="utf-8")
    cluster_names, masks_flat = [], []
    for rec, shape in zip(sf.records(), sf.shapes()):
        name = rec["nome"]
        mask = np.zeros(len(grid_pts), dtype=bool)
        parts = list(shape.parts) + [len(shape.points)]
        for i in range(len(shape.parts)):
            pts = np.array(shape.points[parts[i]: parts[i + 1]])
            if len(pts) >= 3:
                mask |= MplPath(pts).contains_points(grid_pts)
        cluster_names.append(name)
        masks_flat.append(mask)
        print(f"  {name}  pts={mask.sum():4d}")

    out_path = Path("data/processed/uruguai_cluster_masks.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        cluster_names=np.array(cluster_names),
        masks_flat=np.array(masks_flat, dtype=bool),
        lats=lats,
        lons=lons,
    )
    print(f"  Salvo: {out_path}")
    return str(out_path)


def discover_mov_indices(mov_names=None):
    """Retorna lista de (mov_name, Path) filtrada pelos nomes pedidos."""
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
            print(f"  [AVISO] MoV '{name}' não encontrado em {mov_dir}/")
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


def upload_cluster_masks(masks_path: str):
    print("\nUploading máscaras de cluster...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(masks_path, "input/uruguai_cluster_masks.npz")
    print("  Máscaras enviadas ✓")


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

    local_dir = Path(f"results/predep_comparison_uruguai/exp{exp_n}/{mov_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando resultados de {mov_name} → exp{exp_n}...")
    result = subprocess.run(
        [
            "modal", "volume", "get", "xavier-data",
            f"results/predep_comparison_uruguai/exp{exp_n}/{mov_name}",
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
    """Retorna o próximo número de experimento (exp1, exp2, ...) em base_dir."""
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
    Compara PREDEP legacy, core e r² nos clusters da bacia do Uruguai.

    Flags:
      --movs  "ONI,NIN04,SOI"   MoVs separados por vírgula (padrão: ONI,NIN04,SOI)
      --test                    Roda apenas o primeiro MoV disponível
      --n-boot 2000             Amostras de bootstrap (padrão: 2000)

    Exemplos:
      modal run experiments/05_predep_comparisson_uruguai_acomulated.py --test
      modal run experiments/05_predep_comparisson_uruguai_acomulated.py --movs "ONI,SOI"
    """
    from datetime import datetime
    from pathlib import Path
    import time
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.visualization.cluster_plots import generate_plots_comparison

    mov_names = [m.strip() for m in movs.split(",")]
    mov_list  = discover_mov_indices(mov_names)

    if test:
        mov_list = mov_list[:1]

    exp_n   = get_next_exp_n("results/predep_comparison_uruguai")
    exp_dir = Path(f"results/predep_comparison_uruguai/exp{exp_n}")

    print("=" * 80)
    print("COMPARAÇÃO PREDEP — BACIA DO URUGUAI POR CLUSTER — MODAL")
    if test:
        print("*** MODO TESTE — apenas 1 MoV ***")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experimento: exp{exp_n}  |  Saída: {exp_dir}/")
    print(f"MoVs: {[m for m, _ in mov_list]}")
    print(f"Estações: {', '.join(SEASONS)}  |  Lags: 1–{N_LAGS}  |  n_boot={n_boot}")
    print("Métricas: alpha_legacy, alpha_core, r2")
    print("=" * 80)

    masks_path = precompute_cluster_masks()
    upload_cluster_masks(masks_path)
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

        nc_file = (
            exp_dir / mov_name
            / f"{mov_name}_predep_comparison_seasonal.nc"
        )
        plots_dir = exp_dir / "plots"
        print(f"\n  Gerando plots para {mov_name} (6 figuras)...")
        generate_plots_comparison(nc_file, mov_name, plots_dir)

        mov_duration = time.time() - mov_start
        print(f"\n  {mov_name} concluído em {mov_duration / 60:.1f}min")
        all_results.append({"mov_name": mov_name, "duration": mov_duration, **result})

    total = time.time() - start_total
    print("\n" + "=" * 80)
    print("COMPARAÇÃO PREDEP URUGUAI — CONCLUÍDA!")
    print("=" * 80)
    print(f"Tempo total: {total / 60:.1f}min ({total / 3600:.2f}h)")
    print(f"Resultados em: {exp_dir}/")
    for r in all_results:
        print(f"  {r['mov_name']}: {r['duration'] / 60:.1f}min")

    return {"total_duration": total, "results": all_results}
