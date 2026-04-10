#!/usr/bin/env python3
"""
PREDEP por Cluster da Bacia do Uruguai

Pipeline:
1. Carrega sub-bacias do Uruguai (SNIRH_SUbBaciasDNAEE, DNS_DNB_CD=7)
2. Identifica pontos de grade Xavier dentro de cada sub-bacia (point-in-polygon)
3. Calcula acumulado espacial mensal de precipitação por cluster
4. Calcula climatologia mensal e anomalia
5. Calcula PREDEP(anomalia, MoV) para todos os clusters, MoVs e lags 0–12
"""

import sys
import warnings
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import shapefile
from matplotlib.path import Path as MplPath
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SUBBACIAS_SHP  = ROOT / "data/clusters/bacias/SNIRH_SUbBaciasDNAEE.shp"
XAVIER_NC      = ROOT / "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
MOV_CSV        = ROOT / "data/input/merged_indices.csv"
OUTPUT_DIR     = ROOT / "results/predep_uruguai_clusters"

sys.path.insert(0, str(ROOT / "src"))
from predep_calculation.core import predep  # noqa: E402

# ── Configuração ──────────────────────────────────────────────────────────────
# Mapeamento: nome do MoV (config) → coluna no merged_indices.csv
MOV_COL_MAP = {
    "ATL3": "ATL3",
    "NAO":  "NAO_Indice",
    "PDO":  "PDO_Indice",
    "AMO":  "AMO_Indice",
    "ENSO": "NIÑO_3.4_Indice",
    "IOD":  "IOD_Indice",
    "SAM":  "SAM_Indice",
    "TNA":  "TNA_Indice",
    "TSA":  "TSA_Indice",
}

LAG_RANGE = list(range(13))   # 0 … 12
N_BOOT    = 2000
N_WORKERS = min(8, mp.cpu_count())

# DNS_DNB_CD=7 corresponde ao Rio Uruguai no SNIRH
URUGUAI_BACIA_CD = 7


# ── Funções auxiliares ────────────────────────────────────────────────────────

def load_clusters():
    """Carrega as sub-bacias do Uruguai (DNS_DNB_CD=7) do shapefile SNIRH."""
    sf = shapefile.Reader(str(SUBBACIAS_SHP), encoding="utf-8")
    clusters = []
    for rec, shape in zip(sf.records(), sf.shapes()):
        if rec["DNS_DNB_CD"] == URUGUAI_BACIA_CD:
            clusters.append({
                "name": rec["DNS_NM"],
                "sub_id": int(rec["DNS_NU_SUB"]),
                "shape": shape,
            })
    print(f"  {len(clusters)} clusters encontrados (DNS_DNB_CD={URUGUAI_BACIA_CD}):")
    for c in clusters:
        print(f"    sub={c['sub_id']:3.0f}  {c['name']}")
    return clusters


def get_cluster_mask(shape, lons_flat, lats_flat, grid_shape):
    """
    Máscara 2D (n_lat, n_lon) com True nos pontos de grade dentro do polígono.
    Suporta polígonos multi-part (buracos são ignorados — seleção positiva).
    """
    grid_pts = np.column_stack([lons_flat, lats_flat])
    mask = np.zeros(len(grid_pts), dtype=bool)
    parts = list(shape.parts) + [len(shape.points)]
    for i in range(len(shape.parts)):
        pts = np.array(shape.points[parts[i]: parts[i + 1]])
        if len(pts) >= 3:
            mask |= MplPath(pts).contains_points(grid_pts)
    return mask.reshape(grid_shape)


def load_xavier(clusters):
    """
    Carrega Xavier mensal, recortado na bbox da união de todos os clusters.
    Retorna (ds_crop, pr_arr, time_index, grid_shape, lons_flat, lats_flat).
    """
    # Bbox da união dos clusters + margem
    all_pts = []
    for c in clusters:
        all_pts.extend(c["shape"].points)
    all_pts = np.array(all_pts)
    lon_min, lon_max = all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5
    lat_min, lat_max = all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5

    ds = xr.open_dataset(str(XAVIER_NC))
    ds_crop = ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    pr_arr = ds_crop.pr.values  # (n_time, n_lat, n_lon)
    time_index = pd.DatetimeIndex(ds_crop.time.values)
    grid_shape = (len(ds_crop.latitude), len(ds_crop.longitude))
    lons, lats = np.meshgrid(ds_crop.longitude.values, ds_crop.latitude.values)
    ds.close()

    print(f"\n  Xavier mensal recortado: {pr_arr.shape}  "
          f"(lat {lat_min:.1f}–{lat_max:.1f}, lon {lon_min:.1f}–{lon_max:.1f})")
    return pr_arr, time_index, grid_shape, lons.ravel(), lats.ravel()


def compute_anomaly(pr_arr, mask2d, time_index):
    """
    Para um cluster (mask2d booleana):
    1. Soma espacial → série temporal mensal
    2. Climatologia mensal (média de todos os janeiros, fevereiros, ...)
    3. Anomalia = acumulado - climatologia
    Retorna (pr_cluster, climatologia, anomalia) todos shape (n_time,).
    """
    # Aplanar máscara para indexar array (n_time, n_lat, n_lon)
    mask_flat = mask2d.ravel()
    n_pts = mask_flat.sum()
    if n_pts == 0:
        return None, None, None

    # Acumulado espacial: soma simples dos pontos do cluster
    pr_spatial = pr_arr.reshape(len(time_index), -1)[:, mask_flat]  # (n_time, n_pts)
    pr_cluster = pr_spatial.sum(axis=1)  # (n_time,) — mm acumulados no cluster

    # Climatologia mensal
    months = time_index.month
    clim = np.array([pr_cluster[months == m].mean() for m in range(1, 13)])

    # Anomalia
    anomaly = pr_cluster - clim[months - 1]

    return pr_cluster, clim, anomaly


def load_mov():
    """
    Carrega merged_indices.csv e agrega para mensal.
    Retorna MultiIndex DataFrame (Year, Month) × colunas dos MoVs.
    """
    df = pd.read_csv(str(MOV_CSV), low_memory=False)
    cols_needed = ["Year", "Month"] + list(MOV_COL_MAP.values())
    # Manter apenas colunas que existem
    cols_present = [c for c in cols_needed if c in df.columns]
    df = df[cols_present].copy()
    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Year", "Month"])
    df["Year"]  = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    mov_monthly = (
        df.groupby(["Year", "Month"])[
            [c for c in MOV_COL_MAP.values() if c in df.columns]
        ]
        .mean()
    )
    print(f"\n  MoVs mensais carregados: {mov_monthly.shape[0]} meses  "
          f"({mov_monthly.index[0]} → {mov_monthly.index[-1]})")
    return mov_monthly


def align_mov(mov_series, time_index, lag):
    """
    Para cada mês de time_index, pega o valor do MoV com lag meses de antecedência.
    lag=0 → mesmo mês; lag=1 → mês anterior; lag=12 → 12 meses antes.
    """
    aligned = np.full(len(time_index), np.nan)
    for k, dt in enumerate(time_index):
        year  = dt.year
        month = dt.month - lag
        while month <= 0:
            month += 12
            year  -= 1
        try:
            aligned[k] = mov_series.loc[(year, month)]
        except KeyError:
            pass
    return aligned


# ── Worker para paralelismo por lag ──────────────────────────────────────────

def _worker_lag(args):
    """
    Calcula PREDEP(anomaly, mov_aligned) para um lag específico.
    Retorna (lag, alpha).
    """
    lag, anomaly, mov_series_values, mov_series_index, time_index, n_boot = args

    # Recriar Series a partir dos valores (evitar problemas de pickle)
    mov_series = pd.Series(mov_series_values, index=pd.MultiIndex.from_tuples(mov_series_index))
    mov_aligned = align_mov(mov_series, time_index, lag)

    valid = ~np.isnan(anomaly) & ~np.isnan(mov_aligned)
    if valid.sum() < 10:
        return lag, np.nan

    X = anomaly[valid]
    Y = mov_aligned[valid]
    if len(np.unique(X)) < 3 or len(np.unique(Y)) < 3:
        return lag, np.nan

    alpha = predep(X, Y, n_boot=n_boot)
    return lag, alpha


# ── Pipeline principal ────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREDEP — Bacia do Uruguai por Cluster")
    print("=" * 70)

    # 1. Clusters
    print("\n[1/5] Carregando clusters (sub-bacias do Uruguai)...")
    clusters = load_clusters()

    # 2. Xavier mensal (recortado na bbox dos clusters)
    print("\n[2/5] Carregando precipitação Xavier mensal...")
    pr_arr, time_index, grid_shape, lons_flat, lats_flat = load_xavier(clusters)

    # 3. Anomalias por cluster
    print("\n[3/5] Calculando acumulado espacial, climatologia e anomalia...")
    cluster_anomalies = {}
    for c in clusters:
        mask2d = get_cluster_mask(c["shape"], lons_flat, lats_flat, grid_shape)
        n_pts = mask2d.sum()
        pr_cluster, clim, anomaly = compute_anomaly(pr_arr, mask2d, time_index)
        if pr_cluster is None:
            print(f"    [AVISO] Cluster '{c['name']}' sem pontos de grade — ignorado.")
            continue
        cluster_anomalies[c["name"]] = {
            "anomaly": anomaly,
            "clim": clim,
            "pr_cluster": pr_cluster,
            "n_pts": int(n_pts),
        }
        # Verificação: média da anomalia deve ser ≈ 0
        anom_mean = np.nanmean(anomaly)
        print(f"    sub={c['sub_id']:3.0f}  {c['name']:<45}  "
              f"pts={n_pts:4d}  anom_mean={anom_mean:.4f}")

    # 4. MoVs mensais
    print("\n[4/5] Carregando índices climáticos (MoVs)...")
    mov_monthly = load_mov()

    # 5. PREDEP para todos os (cluster × MoV × lag)
    print("\n[5/5] Calculando PREDEP...")
    n_clusters = len(cluster_anomalies)
    n_movs     = len(MOV_COL_MAP)
    n_lags     = len(LAG_RANGE)
    total      = n_clusters * n_movs * n_lags
    print(f"  {n_clusters} clusters × {n_movs} MoVs × {n_lags} lags = {total} cálculos")
    print(f"  n_boot={N_BOOT}, workers={N_WORKERS}\n")

    all_results = []

    for cluster_name, cdata in cluster_anomalies.items():
        anomaly = cdata["anomaly"]
        n_pts   = cdata["n_pts"]

        for mov_code, col_name in MOV_COL_MAP.items():
            if col_name not in mov_monthly.columns:
                print(f"  [AVISO] Coluna '{col_name}' não encontrada em merged_indices.csv")
                continue

            mov_series = mov_monthly[col_name].dropna()
            # Serializar para workers
            mov_values = mov_series.values
            mov_index  = list(mov_series.index)

            lag_args = [
                (lag, anomaly, mov_values, mov_index, time_index, N_BOOT)
                for lag in LAG_RANGE
            ]

            with mp.Pool(processes=N_WORKERS) as pool:
                lag_results = pool.map(_worker_lag, lag_args)

            for lag, alpha in lag_results:
                all_results.append({
                    "cluster":  cluster_name,
                    "mov":      mov_code,
                    "lag":      lag,
                    "alpha":    alpha,
                    "n_pts":    n_pts,
                    "n_valid":  int((~np.isnan(anomaly) &
                                    ~np.isnan(align_mov(mov_monthly[col_name],
                                                        time_index, lag))).sum()),
                })

        print(f"  ✓ {cluster_name}")

    # Salvar CSV
    df_out = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "predep_results.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Resultados salvos: {csv_path}")
    print(f"  Linhas: {len(df_out)}  (esperado: {total})")

    # Salvar NetCDF estruturado (clusters × movs × lags)
    cluster_names = list(cluster_anomalies.keys())
    mov_names     = list(MOV_COL_MAP.keys())
    alpha_arr = np.full((n_clusters, n_movs, n_lags), np.nan)
    for row in all_results:
        ci = cluster_names.index(row["cluster"])
        mi = mov_names.index(row["mov"])
        li = row["lag"]
        alpha_arr[ci, mi, li] = row["alpha"]  # NaN já é preservado como float

    ds_out = xr.Dataset(
        {"alpha": (["cluster", "mov", "lag"], alpha_arr)},
        coords={
            "cluster": cluster_names,
            "mov":     mov_names,
            "lag":     LAG_RANGE,
        },
    )
    ds_out["alpha"].attrs["long_name"] = "PREDEP alpha (precipitação acumulada do cluster vs MoV)"
    ds_out["alpha"].attrs["valid_range"] = [0.0, 1.0]
    nc_path = OUTPUT_DIR / "predep_results.nc"
    ds_out.to_netcdf(nc_path)
    print(f"  NetCDF salvo: {nc_path}")

    # Resumo: top valores
    df_valid = df_out.dropna(subset=["alpha"])
    if len(df_valid) > 0:
        print(f"\n  Valores alpha válidos: {len(df_valid)} / {len(df_out)}")
        print(f"  alpha: min={df_valid['alpha'].min():.3f}  "
              f"max={df_valid['alpha'].max():.3f}  "
              f"média={df_valid['alpha'].mean():.3f}")
        print("\n  Top 10 combinações (maior alpha):")
        top = df_valid.nlargest(10, "alpha")[["cluster", "mov", "lag", "alpha"]]
        print(top.to_string(index=False))

    print("\n" + "=" * 70)
    print("CONCLUÍDO")
    print("=" * 70)

    return df_out


if __name__ == "__main__":
    run()
