#!/usr/bin/env python3
"""
PREDEP por Cluster da Bacia do Uruguai — Modal

Objetivo (conforme especificação):
  Para cada cluster da bacia do Uruguai, calcula a série de ANOMALIAS MENSAIS
  da precipitação acumulada espacialmente e, a partir disso, estima o PREDEP
  em relação a cada MoV considerando lags t=0 até t=12.

Pipeline por cluster:
  1. Point-in-polygon → pontos de grade do cluster
  2. Acumulado espacial mensal (soma dos pontos do cluster por mês)
  3. Climatologia mensal (média de todos os janeiros, fevereiros, ...)
  4. Anomalia mensal = acumulado mensal − climatologia do mês
  5. PREDEP(anomalia, MoV) para cada lag 0–12

Estrutura Modal (idêntica ao experimento 03):
  - Upload único do Xavier + máscaras de cluster
  - Um container por MoV (sequencial entre MoVs)
  - Dentro de cada container: lags 0–12 em paralelo (8 workers)
  - Download imediato ao final de cada MoV
  - Resultados em results/predep_uruguai/{MOV}/
"""

import modal

# ── Imagem Modal ──────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "pandas",
            "xarray[complete]",
            "numpy",
            "scipy",
            "matplotlib",
            "netcdf4",
            "h5netcdf",
        ]
    )
    .add_local_python_source("src")
)

app = modal.App("predep-uruguai-clusters", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# ── Constantes ────────────────────────────────────────────────────────────────
# Bbox da bacia do Uruguai + margem (recorte do Xavier)
URU_LAT_MIN, URU_LAT_MAX = -32.3, -25.7
URU_LON_MIN, URU_LON_MAX = -58.1, -48.8

N_LAGS = 12  # lags t=1 até t=12 (inclusive)

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
# Mês representativo de cada estação (igual ao experimento 02)
SEASON_MONTH_MAP = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}


def _compute_lag_worker(args):
    """
    Worker: para um lag e uma estação, alinha o MoV com a série de anomalias
    sazonais anuais e calcula PREDEP para todos os clusters.

    args: (lag, cluster_season_data, mov_lookup, season_month, n_boot)
      cluster_season_data: dict {cname: (anomaly_array, years_array)}
      season_month: mês representativo da estação (1=DJF, 4=MAM, 7=JJA, 10=SON)
    Retorna (lag, dict{cluster_name: alpha}).
    """
    import numpy as np
    import warnings
    from src.predep_calculation.core import predep

    warnings.filterwarnings("ignore")

    lag, cluster_season_data, mov_lookup, season_month, n_boot = args

    results = {}
    for cname, (anomaly, years) in cluster_season_data.items():
        n_years = len(years)
        mov_aligned = np.full(n_years, np.nan)
        for k in range(n_years):
            tm = season_month - lag
            ty = int(years[k])
            while tm <= 0:
                tm += 12
                ty -= 1
            mov_aligned[k] = mov_lookup.get((ty, tm), np.nan)

        valid_mask = ~np.isnan(mov_aligned) & ~np.isnan(anomaly)
        if valid_mask.sum() < 10:
            results[cname] = np.nan
            continue
        X = mov_aligned[valid_mask]
        Y = anomaly[valid_mask]
        if len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
            results[cname] = np.nan
            continue
        results[cname] = predep(X, Y, n_boot)

    return lag, results


# ── Função Modal: processa um MoV completo ───────────────────────────────────


@app.function(
    memory=4096,
    cpu=8,
    timeout=3600 * 6,
    volumes={"/data": volume},
)
def process_mov(mov_name: str, n_boot: int = 5000, exp_n: int = 1):
    """
    Container Modal: calcula PREDEP sazonal por cluster para um MoV.

    Passos internos:
      1. Carrega Xavier mensal recortado na bbox do Uruguai
      2. Carrega máscaras de cluster pré-computadas
      3. Para cada cluster: integração espacial → série mensal acumulada
      4. Para cada estação (DJF/MAM/JJA/SON): média sazonal anual → anomalia
      5. Carrega MoV mensal
      6. Para cada estação × lag 0–12: alinha MoV e calcula PREDEP (paralelo)
      7. Salva NetCDF com alpha[season, cluster, lag]
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

    # ── 1. Carregar Xavier mensal recortado ───────────────────────────────────
    xavier_file = "/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc"
    ds = xr.open_dataset(xavier_file)
    ds_crop = ds.sel(
        latitude=slice(URU_LAT_MIN, URU_LAT_MAX),
        longitude=slice(URU_LON_MIN, URU_LON_MAX),
    ).load()  # carrega em memória antes de fechar
    ds.close()
    time_idx = pd.DatetimeIndex(ds_crop.time.values)

    time_years  = time_idx.year.values
    time_months = time_idx.month.values
    n_time = len(time_idx)
    n_lat  = len(ds_crop.latitude)
    n_lon  = len(ds_crop.longitude)
    print(f"  Xavier recortado: ({n_time}, {n_lat}, {n_lon})  "
          f"({time_idx[0].date()} → {time_idx[-1].date()})")

    # ── 2. Carregar máscaras de cluster ───────────────────────────────────────
    npz = np.load("/data/input/uruguai_cluster_masks.npz", allow_pickle=True)
    cluster_names = list(npz["cluster_names"])
    masks_flat    = npz["masks_flat"].astype(bool)
    print(f"  Clusters: {len(cluster_names)}")

    # ── 3. Integração espacial mensal por cluster (Eq.1) ─────────────────────
    #   Acum(t) = R² × (π/180)² × ∫∫ P(t,φ,θ)·cos(φ) dφ dθ
    R_EARTH = 6_371_000.0
    cos_lat = np.cos(np.deg2rad(ds_crop.latitude))

    cluster_monthly = {}   # cname → (n_time,) mensal
    for i, cname in enumerate(cluster_names):
        mask = masks_flat[i]
        if mask.sum() == 0:
            print(f"  [AVISO] '{cname}': sem pontos de grade — ignorado.")
            continue

        mask_2d = xr.DataArray(
            mask.reshape(n_lat, n_lon),
            dims=["latitude", "longitude"],
            coords={"latitude": ds_crop.latitude, "longitude": ds_crop.longitude},
        )
        # Fora do cluster → 0.0 para não propagar NaN na integração trapezoidal
        pr_cluster = (
            (ds_crop.pr.where(mask_2d, other=0.0) * cos_lat)
            .integrate("latitude")
            .integrate("longitude")
            .values
        ) * R_EARTH**2 * (np.pi / 180) ** 2
        cluster_monthly[cname] = pr_cluster
        print(f"  {cname}  pts={mask.sum():4d}  pr_mean={pr_cluster.mean():.4e}")

    # ── 4. Agregação sazonal anual + anomalia por estação ────────────────────
    #   Para cada estação S: filtrar meses de S, média por ano, anomalia anual.
    #   Idêntico ao experimento 02 (load_xavier_seasonal).
    cluster_seasonal = {s: {} for s in SEASONS}  # season → {cname: (anomaly, years)}

    for season in SEASONS:
        s_months = SEASON_MONTHS[season]
        s_mask   = np.isin(time_months, s_months)
        s_years_raw = time_years[s_mask]
        if season == "DJF":
            # Dezembro de ano y pertence ao DJF do ano y+1 (Dec(y-1)+Jan(y)+Feb(y))
            s_years = np.where(time_months[s_mask] == 12, s_years_raw + 1, s_years_raw)
            unique_years = np.unique(s_years)
            counts = np.array([np.sum(s_years == y) for y in unique_years])
            unique_years = unique_years[counts == 3]
        else:
            s_years = s_years_raw
            unique_years = np.unique(s_years)

        for cname, pr_monthly in cluster_monthly.items():
            pr_s = pr_monthly[s_mask]
            pr_annual = np.array(
                [pr_s[s_years == y].mean() for y in unique_years]
            )
            anomaly = pr_annual - pr_annual.mean()
            cluster_seasonal[season][cname] = (anomaly, unique_years)

        anom_ex = next(iter(cluster_seasonal[season].values()))[0].mean()
        print(f"  {season}: {len(unique_years)} anos  anom_mean={anom_ex:+.4e}")

    print(f"\n  Séries sazonais prontas: {len(SEASONS)} estações × {len(cluster_monthly)} clusters")

    # ── 5. Carregar MoV mensal ────────────────────────────────────────────────
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
    n_valid_mov = int(mov_monthly["value"].notna().sum())
    print(f"\n  MoV '{mov_name}': {n_valid_mov} meses válidos")

    # ── 6. PREDEP por estação (paralelo por lag dentro de cada estação) ───────
    n_seasons  = len(SEASONS)
    n_clusters = len(cluster_names)
    alpha_array = np.full((n_seasons, n_clusters, N_LAGS), np.nan)

    for s_idx, season in enumerate(SEASONS):
        season_month = SEASON_MONTH_MAP[season]
        cluster_data = cluster_seasonal[season]  # {cname: (anomaly, years)}

        lag_args = [
            (lag, cluster_data, mov_lookup, season_month, n_boot)
            for lag in range(1, N_LAGS + 1)
        ]

        print(f"\n  [{season}] PREDEP: {len(cluster_data)} clusters "
              f"× {N_LAGS} lags  (workers=8)...")
        t_pool = time.time()
        with mp.Pool(processes=8) as pool:
            for lag, cluster_alphas in pool.imap_unordered(_compute_lag_worker, lag_args):
                for c_idx, cname in enumerate(cluster_names):
                    alpha_array[s_idx, c_idx, lag - 1] = cluster_alphas.get(cname, np.nan)
        print(f"  [{season}] Pool finalizado em {time.time() - t_pool:.1f}s")

        for li, lag in enumerate(range(1, N_LAGS + 1)):
            valid = int(np.sum(~np.isnan(alpha_array[s_idx, :, li])))
            print(f"    Lag {lag:2d}: {valid}/{n_clusters} clusters válidos")

    # ── 7. Salvar NetCDF ──────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {"alpha": (["season", "cluster", "lag"], alpha_array)},
        coords={
            "season":  SEASONS,
            "cluster": cluster_names,
            "lag":     list(range(1, N_LAGS + 1)),
        },
    )
    ds_out.attrs["description"] = (
        f"PREDEP alpha — anomalia sazonal de precipitação acumulada "
        f"por cluster vs MoV {mov_name}"
    )
    ds_out.attrs["lags"] = (
        "0 = mesma estação, 1 = MoV 1 mês antes do mês representativo, ..."
    )

    out_dir = Path(f"/data/results/predep_uruguai/exp{exp_n}/{mov_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_file = out_dir / f"{mov_name}_predep_clusters_seasonal.nc"
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
    """
    Calcula, localmente, quais pontos de grade Xavier pertencem a cada
    cluster (sub-bacia do Uruguai). Salva em data/processed/ para upload.
    """
    import numpy as np
    import netCDF4 as nc4
    import shapefile
    from matplotlib.path import Path as MplPath
    from pathlib import Path

    print("Pré-computando máscaras de cluster (local)...")

    # Ler grade lat/lon do Xavier com netCDF4 (evita incompatibilidade xarray/NumPy 2.0)
    with nc4.Dataset(LOCAL_XAVIER_FILE) as nc:
        lats_all = nc.variables["latitude"][:]
        lons_all = nc.variables["longitude"][:]
    lat_mask = (lats_all >= URU_LAT_MIN) & (lats_all <= URU_LAT_MAX)
    lon_mask = (lons_all >= URU_LON_MIN) & (lons_all <= URU_LON_MAX)
    lats = lats_all[lat_mask]
    lons = lons_all[lon_mask]
    lons2d, lats2d = np.meshgrid(lons, lats)
    lons_flat = lons2d.ravel()
    lats_flat = lats2d.ravel()
    grid_pts = np.column_stack([lons_flat, lats_flat])

    sf = shapefile.Reader(LOCAL_SUBBACIAS_SHP, encoding="utf-8")
    cluster_names = []
    masks_flat = []

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


def discover_mov_indices():
    """Retorna lista de (mov_name, Path) para todos os CSVs disponíveis."""
    import re
    from pathlib import Path

    mov_dir = Path("data/raw/mov_indices")
    indices = []
    for f in sorted(mov_dir.glob("*_indice*.csv")):
        match = re.match(r"^(.+?)_indice", f.name)
        if match:
            indices.append((match.group(1), f))
    return indices


def upload_xavier_data():
    """Upload do Xavier mensal para o volume Modal."""
    from pathlib import Path

    print("\nUploading Xavier mensal...")
    xavier_path = Path(LOCAL_XAVIER_FILE)
    if not xavier_path.exists():
        raise FileNotFoundError(f"Não encontrado: {xavier_path}")
    remote_name = "pr_monthly_1961_2019_BRAZIL_combined.nc"
    size_mb = xavier_path.stat().st_size / 1e6
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(str(xavier_path), f"/input/{remote_name}")
        print(f"  Xavier: {xavier_path.name} ({size_mb:.0f} MB) ✓")
    except FileExistsError:
        print(f"  Xavier: {remote_name} já existe no volume — skip ✓")


def upload_cluster_masks(masks_path: str):
    """Upload das máscaras de cluster pré-computadas."""
    print("\nUploading máscaras de cluster...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(masks_path, "/input/uruguai_cluster_masks.npz")
    print("  Máscaras enviadas ✓")


def upload_mov_index(mov_name: str, local_path):
    """Upload do CSV de um MoV para o volume Modal (formato: time, value)."""
    import pandas as pd
    import tempfile
    import os
    from pathlib import Path

    local_path = Path(local_path)
    df = pd.read_csv(local_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col = df.columns[1]
    mov_csv = df[[date_col, val_col]].copy()
    mov_csv.columns = ["time", "value"]
    mov_csv["time"] = pd.to_datetime(mov_csv["time"])
    mov_csv = mov_csv.dropna(subset=["value"])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        mov_csv.to_csv(f, index=False)
        tmp_path = f.name

    try:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(tmp_path, f"/input/{mov_name}_indice_data.csv")
    finally:
        os.unlink(tmp_path)

    yr_min = mov_csv["time"].dt.year.min()
    yr_max = mov_csv["time"].dt.year.max()
    print(f"  {mov_name}: {len(mov_csv)} registros ({yr_min}–{yr_max})")


def download_results(mov_name: str, exp_n: int = 1):
    """Baixa resultados de um MoV do volume Modal para results/predep_uruguai/exp{n}/."""
    import subprocess
    import shutil
    from pathlib import Path

    local_dir = Path(f"results/predep_uruguai/exp{exp_n}/{mov_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando resultados de {mov_name} → exp{exp_n}...")
    result = subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "xavier-data",
            f"results/predep_uruguai/exp{exp_n}/{mov_name}",
            str(local_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Erro no download: {result.stderr.strip()}")
        return

    # `modal volume get` baixa em local_dir/{mov_name}/*.nc — achatar um nível
    nested = local_dir / mov_name
    if nested.is_dir():
        for f in nested.iterdir():
            shutil.move(str(f), str(local_dir / f.name))
        nested.rmdir()

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


# ── Entrypoint local ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(n_boot: int = 5000, test: bool = False):
    """
    Executa PREDEP mensal por cluster para todos os MoVs disponíveis.
    Um container Modal por MoV, sequencial entre MoVs.

    Flags:
      --test   Roda apenas o primeiro MoV disponível (smoke test rápido).
               Uso: modal run experiments/04_predep_uruguai_clusters.py --test
    """
    from datetime import datetime
    from pathlib import Path
    import time
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.visualization.cluster_plots import generate_plots_predep

    mov_list = discover_mov_indices()
    if test:
        mov_list = mov_list[:1]

    exp_n = get_next_exp_n("results/predep_uruguai")
    exp_dir = Path(f"results/predep_uruguai/exp{exp_n}")

    print("=" * 80)
    print("PREDEP — BACIA DO URUGUAI POR CLUSTER (ANOMALIA SAZONAL) — MODAL")
    if test:
        print("*** MODO TESTE — apenas 1 MoV ***")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experimento: exp{exp_n}  |  Saída: {exp_dir}/")
    print(f"MoVs: {len(mov_list)}  |  Clusters: 4 (URUGUAI_recortado)")
    print(f"Estações: {', '.join(SEASONS)}  |  Lags: 1–{N_LAGS}  |  n_boot={n_boot}")
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

        nc_file = exp_dir / mov_name / f"{mov_name}_predep_clusters_seasonal.nc"
        plots_dir = exp_dir / "plots"
        print(f"\n  Gerando plots para {mov_name}...")
        generate_plots_predep(nc_file, mov_name, plots_dir)

        mov_duration = time.time() - mov_start
        print(f"\n  {mov_name} concluído em {mov_duration / 60:.1f}min")
        all_results.append(
            {"mov_name": mov_name, "duration": mov_duration, **result}
        )

    total = time.time() - start_total
    print("\n" + "=" * 80)
    print("PREDEP URUGUAI CLUSTERS — CONCLUÍDO!")
    print("=" * 80)
    print(f"Tempo total: {total / 60:.1f}min ({total / 3600:.2f}h)")
    print(f"Resultados em: {exp_dir}/")
    for r in all_results:
        print(f"  {r['mov_name']}: {r['duration'] / 60:.1f}min")

    return {"total_duration": total, "results": all_results}
