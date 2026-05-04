#!/usr/bin/env python3
"""
PREDEP Granular por Ponto — Brasil por Bacia (Exp 06)

Calcula alpha_core e r² ponto-a-ponto para cada bacia hidrográfica do Brasil.
Cada bacia roda em um container Modal separado; ao final, os resultados são
mesclados em um único NetCDF Brasil-todo.

Uso:
  modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI" --basins "uruguai"
  modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI" --basins "all"
  modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI,NIN04" --basins "all"
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
            "statsmodels",
            "matplotlib",
            "netcdf4",
            "h5netcdf",
            "tqdm",
        ]
    )
    .add_local_python_source("src")
)

app    = modal.App("predep-granular-brazil", image=image)
volume = modal.Volume.from_name("xavier-data", create_if_missing=True)

# ── Constantes ────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -34.0, 6.0
LON_MIN, LON_MAX = -75.0, -34.0

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

# Bacias hidrográficas do Brasil — diretórios locais dos shapefiles
BASINS = {
    "amazonica":          "data/clusters/amazonica",
    "atlantico_leste":    "data/clusters/atlantico_leste",
    "atlantico_nordeste": "data/clusters/atlantico_nordeste",
    "atlantico_sudeste":  "data/clusters/atlantico_sudeste",
    "atlantico_sul":      "data/clusters/atlantico_sul",
    "paraguai":           "data/clusters/paraguai",
    "parnaiba":           "data/clusters/paranaiba",
    "sao_francisco":      "data/clusters/sao_francisco",
    "tocantins":          "data/clusters/tocantins",
    "uruguai":            "data/clusters/uruguai",
}

# Compartilhado via fork com os workers de multiprocessing
_PRECIP_ARR_GLOBAL = None  # (n_obs, n_lat, n_lon)




def _precompute_mov_alignment(mov_lookup, s_years, s_months, lag):
    """Alinha o MoV para cada mês da estação dado um lag."""
    import numpy as np

    n = len(s_years)
    aligned = np.full(n, np.nan)
    for k in range(n):
        tm = int(s_months[k]) - lag
        ty = int(s_years[k])
        while tm <= 0:
            tm += 12
            ty -= 1
        aligned[k] = mov_lookup.get((ty, tm), np.nan)
    return aligned


# ── Worker ────────────────────────────────────────────────────────────────────

def _compute_lag_worker(args):
    """
    Worker: calcula PREDEP + r² em cada pixel para um lag.
    Usa _PRECIP_ARR_GLOBAL (n_obs, n_lat, n_lon) herdado via fork.

    args: (lag, mov_aligned, n_boot, debug, basin_mask_flat)
      basin_mask_flat: bool array (n_lat*n_lon,) ou None (sem máscara)
    Retorna (lag, ac_map, r2_map) — shape (n_lat, n_lon) cada.
    """
    import numpy as np
    import warnings
    import time as _time
    from src.predep_calculation.core import predep
    from src.regression_calculation.core import regression_calc

    warnings.filterwarnings("ignore")

    lag, mov_aligned, n_boot, debug, basin_mask_flat = args
    precip_arr = _PRECIP_ARR_GLOBAL  # (n_obs, n_lat, n_lon)
    n_lat = precip_arr.shape[1]
    n_lon = precip_arr.shape[2]
    n_total = n_lat * n_lon

    ac_map = np.full((n_lat, n_lon), np.nan)
    r2_map = np.full((n_lat, n_lon), np.nan)
    valid_mov = ~np.isnan(mov_aligned)
    t0 = _time.time()

    if debug:
        from tqdm import tqdm
        pixel_iter = tqdm(
            range(n_total),
            desc=f"  Lag {lag:2d}",
            position=lag % 8,
            leave=True,
            ncols=90,
            unit="px",
        )

    report_every = max(1, n_total // 10)  # print a cada ~10%
    done = 0

    for i in range(n_lat):
        for j in range(n_lon):
            if basin_mask_flat is not None and not basin_mask_flat[i * n_lon + j]:
                done += 1
                if debug:
                    pixel_iter.update(1)
                continue

            pp = precip_arr[:, i, j]
            valid = valid_mov & ~np.isnan(pp)
            if valid.sum() >= 10:
                X = mov_aligned[valid]
                Y = pp[valid]
                if len(np.unique(X)) >= 3 and len(np.unique(Y)) >= 3:
                    ac_map[i, j] = predep(X, Y, n_boot)
                    r2_map[i, j] = regression_calc(X, Y)

            done += 1
            if debug:
                pixel_iter.update(1)
            elif done % report_every == 0:
                elapsed = _time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate / 60 if rate > 0 else float("inf")
                print(
                    f"  Lag {lag:2d}: {done/n_total*100:3.0f}%"
                    f"  {elapsed/60:.1f}min  ETA ~{eta:.0f}min",
                    flush=True,
                )

    elapsed = _time.time() - t0
    valid_count = int(np.sum(~np.isnan(ac_map)))
    if debug:
        pixel_iter.close()
    print(
        f"  Lag {lag:2d} CONCLUÍDO em {elapsed/60:.1f}min"
        f" — {valid_count}/{n_total} pts válidos",
        flush=True,
    )

    return lag, ac_map, r2_map


# ── Função Modal ──────────────────────────────────────────────────────────────

@app.function(
    memory=4096,
    cpu=8,
    timeout=86400,  # 24h — máximo do Modal
    volumes={"/data": volume},
)
def process_mov(
    mov_name: str,
    n_boot: int = 5000,
    exp_n: int = 1,
    debug: bool = False,
    basin_name: str = "brazil",
    use_anomaly: bool = True,
):
    """
    Container Modal: PREDEP granular ponto-a-ponto por bacia.

    basin_name: chave de BASINS ou "brazil" (sem máscara, bbox completo).
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    import multiprocessing as mp
    from pathlib import Path
    import time
    from datetime import datetime
    import warnings
    from tqdm import tqdm

    warnings.filterwarnings("ignore")

    global _PRECIP_ARR_GLOBAL

    volume.reload()
    start = time.time()
    print(f"\n{'='*60}")
    print(f"CONTAINER GRANULAR — MoV: {mov_name}  bacia: {basin_name}")
    print(datetime.now().strftime("%H:%M:%S"))
    print("=" * 60)

    # ── 1. Máscara da bacia (se aplicável) ───────────────────────────────────
    basin_mask_flat = None
    if basin_name != "brazil":
        npz = np.load(f"/data/input/basin_masks/{basin_name}.npz")
        basin_lats = npz["lats"]
        basin_lons = npz["lons"]
        basin_mask_2d = npz["mask"]          # (n_lat_bbox, n_lon_bbox)
        basin_mask_flat = basin_mask_2d.ravel()
        lat_sel = (slice(float(basin_lats.min()), float(basin_lats.max())))
        lon_sel = (slice(float(basin_lons.min()), float(basin_lons.max())))
        n_basin = int(basin_mask_flat.sum())
        print(f"  Máscara: {n_basin} pixels dentro da bacia")
    else:
        lat_sel = slice(LAT_MIN, LAT_MAX)
        lon_sel = slice(LON_MIN, LON_MAX)

    # ── 2. Xavier mensal recortado ────────────────────────────────────────────
    ds = xr.open_dataset("/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc")
    ds_crop = ds.sel(latitude=lat_sel, longitude=lon_sel).load()
    ds.close()

    time_idx    = pd.DatetimeIndex(ds_crop.time.values)
    time_years  = time_idx.year.values
    time_months = time_idx.month.values
    n_lat       = len(ds_crop.latitude)
    n_lon       = len(ds_crop.longitude)
    lats        = ds_crop.latitude.values
    lons        = ds_crop.longitude.values
    pr_values   = ds_crop.pr.values  # (n_time, n_lat, n_lon)

    # Climatologia mensal: média de cada mês calendar em todos os anos
    if use_anomaly:
        monthly_clim = np.stack([
            pr_values[time_months == m].mean(axis=0) for m in range(1, 13)
        ], axis=0)  # (12, n_lat, n_lon)

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

    # ── 3. Por estação: precipitação por ponto + PREDEP (paralelo por lag) ──────
    n_seasons = len(SEASONS)
    ac_arr = np.full((n_seasons, N_LAGS, n_lat, n_lon), np.nan)
    r2_arr = np.full((n_seasons, N_LAGS, n_lat, n_lon), np.nan)

    mov_bar = None if debug else tqdm(
        total=N_LAGS * n_seasons,
        desc=mov_name,
        unit="lag",
        ncols=100,
    )

    for s_idx, season in enumerate(SEASONS):
        s_mask         = np.isin(time_months, SEASON_MONTHS[season])
        s_times_years  = time_years[s_mask]
        s_times_months = time_months[s_mask]
        n_obs          = int(s_mask.sum())

        pr_s = pr_values[s_mask]                        # (n_obs, n_lat, n_lon)
        if use_anomaly:
            clim_s  = monthly_clim[s_times_months - 1] # (n_obs, n_lat, n_lon)
            pr_s    = pr_s - clim_s

        if debug:
            print(f"\n  {season}: {n_obs} meses  std={float(pr_s.std()):.4e}")

        # Expõe array para os workers via fork
        _PRECIP_ARR_GLOBAL = pr_s

        # Pré-alinha MoV para cada mês × lag
        lag_args = [
            (
                lag,
                _precompute_mov_alignment(
                    mov_lookup, s_times_years, s_times_months, lag
                ),
                n_boot,
                debug,
                basin_mask_flat,
            )
            for lag in range(1, N_LAGS + 1)
        ]

        if debug:
            print(f"  [{season}] PREDEP × {n_lat * n_lon} pts "
                  f"× {N_LAGS} lags  (workers=8)...")
        t_pool = time.time()
        with mp.Pool(processes=8) as pool:
            for lag, ac_map, r2_map in pool.imap_unordered(
                _compute_lag_worker, lag_args
            ):
                ac_arr[s_idx, lag - 1] = ac_map
                r2_arr[s_idx, lag - 1] = r2_map
                if mov_bar is not None:
                    mov_bar.update(1)

        if debug:
            print(f"  [{season}] Pool finalizado em {time.time() - t_pool:.1f}s")
            for li in range(N_LAGS):
                v = int(np.sum(~np.isnan(ac_arr[s_idx, li])))
                print(f"    Lag {li + 1:2d}: {v}/{n_lat * n_lon} pts válidos")

    if mov_bar is not None:
        mov_bar.close()

    # ── 4. Salvar NetCDF ──────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        {
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
    mode_label = "anomaly" if use_anomaly else "raw_monthly"
    ds_out.attrs["precip_mode"] = mode_label
    ds_out.attrs["description"] = (
        f"PREDEP granular ponto-a-ponto — bacia={basin_name} MoV={mov_name} "
        f"precip_mode={mode_label}"
    )
    ds_out["alpha_core"].attrs["description"] = (
        "alpha_{precip|MoV} — Bayesian Blocks, dados mensais"
    )
    ds_out["r2"].attrs["description"] = "r² regressão linear — baseline"

    out_dir = Path(f"/data/results/predep_granular_brazil/exp{exp_n}/{mov_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_file = out_dir / f"{mov_name}_{basin_name}_predep_granular_seasonal.nc"
    ds_out.to_netcdf(nc_file)
    volume.commit()

    duration = time.time() - start
    print(f"\n  {mov_name}/{basin_name} CONCLUÍDO em {duration / 60:.1f}min")
    print(f"  Arquivo: {nc_file}")

    return {
        "mov_name":   mov_name,
        "basin_name": basin_name,
        "duration":   duration,
        "file":       str(nc_file),
        "n_pts":      n_lat * n_lon,
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

    local_dir = Path(f"results/predep_granular_brazil/exp{exp_n}/{mov_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando resultados de {mov_name} → exp{exp_n}...")
    result = subprocess.run(
        [
            "modal", "volume", "get", "xavier-data",
            f"results/predep_granular_brazil/exp{exp_n}/{mov_name}",
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


def get_xavier_grid():
    """Retorna (lats, lons) do grid Xavier dentro da bbox Brasil."""
    import xarray as xr

    ds = xr.open_dataset(LOCAL_XAVIER_FILE)
    lats = ds.sel(latitude=slice(LAT_MIN, LAT_MAX)).latitude.values
    lons = ds.sel(longitude=slice(LON_MIN, LON_MAX)).longitude.values
    ds.close()
    return lats, lons


def compute_basin_mask(basin_dir: str, lats_brazil, lons_brazil):
    """
    Carrega shapefile da bacia, faz point-in-polygon no grid Xavier e retorna
    (mask_2d, lats_crop, lons_crop) onde lats/lons são o bbox do shapefile.
    """
    import shapefile
    from matplotlib.path import Path as MplPath
    from pathlib import Path
    import numpy as np

    shp_files = list(Path(basin_dir).glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"Nenhum .shp em {basin_dir}")
    sf = shapefile.Reader(str(shp_files[0]), encoding="utf-8")

    bbox = sf.bbox  # [lon_min, lat_min, lon_max, lat_max]
    pad  = 0.2
    lat_min = max(float(lats_brazil.min()), bbox[1] - pad)
    lat_max = min(float(lats_brazil.max()), bbox[3] + pad)
    lon_min = max(float(lons_brazil.min()), bbox[0] - pad)
    lon_max = min(float(lons_brazil.max()), bbox[2] + pad)

    lats_crop = lats_brazil[(lats_brazil >= lat_min) & (lats_brazil <= lat_max)]
    lons_crop = lons_brazil[(lons_brazil >= lon_min) & (lons_brazil <= lon_max)]

    grid_lons, grid_lats = np.meshgrid(lons_crop, lats_crop)
    grid_pts = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])

    mask_flat = np.zeros(len(grid_pts), dtype=bool)
    for shape in sf.shapes():
        parts = list(shape.parts) + [len(shape.points)]
        for k in range(len(shape.parts)):
            pts = np.array(shape.points[parts[k]: parts[k + 1]])
            if len(pts) >= 3:
                mask_flat |= MplPath(pts).contains_points(grid_pts)

    mask_2d = mask_flat.reshape(len(lats_crop), len(lons_crop))
    return mask_2d, lats_crop, lons_crop


def upload_basin_mask(basin_name: str, mask_2d, lats, lons):
    """Salva máscara da bacia no volume Modal como NPZ."""
    import numpy as np
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        np.savez(tmp_path, mask=mask_2d, lats=lats, lons=lons)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(tmp_path, f"input/basin_masks/{basin_name}.npz")
    finally:
        os.unlink(tmp_path)

    n_pixels = int(mask_2d.sum())
    print(f"  Máscara {basin_name}: {n_pixels} pixels  ({len(lats)}×{len(lons)} bbox) ✓")


def merge_basin_results(mov_name: str, basin_names: list, exp_n: int):
    """
    Mescla os NCs por bacia em um único arquivo Brasil-todo.
    Usa xr.merge com join='outer'; bacias não se sobrepõem.
    """
    import xarray as xr
    from pathlib import Path

    mov_dir = Path(f"results/predep_granular_brazil/exp{exp_n}/{mov_name}")
    dsets = []
    for bname in basin_names:
        nc_file = mov_dir / f"{mov_name}_{bname}_predep_granular_seasonal.nc"
        if nc_file.exists():
            dsets.append(xr.open_dataset(nc_file))
        else:
            print(f"  [AVISO] Não encontrado: {nc_file.name}")

    if not dsets:
        print("  Nenhum arquivo para mesclar.")
        return None

    merged = xr.merge(dsets, join="outer", fill_value=float("nan"))
    merged.attrs["description"] = (
        f"PREDEP granular Brasil-todo — MoV={mov_name} "
        f"bacias={','.join(basin_names)}"
    )

    out_file = mov_dir / f"{mov_name}_predep_granular_seasonal.nc"
    merged.to_netcdf(out_file)
    for ds in dsets:
        ds.close()

    print(f"  Mesclado: {len(dsets)} bacias → {out_file.name}")
    return out_file


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
def main(
    movs: str = "ONI",
    n_boot: int = 5000,
    basins: str = "all",
    debug: bool = False,
    test: bool = False,
    use_anomaly: bool = True,
):
    """
    PREDEP granular ponto-a-ponto no Brasil, por bacia.

    Flags:
      --movs    "ONI,NIN04"     MoVs separados por vírgula
      --basins  "all"           "all" ou nomes separados por vírgula
                                ex: "uruguai,paraguai,sao_francisco"
      --n-boot  1000            Bootstrap por pixel (padrão: 1000)
      --debug                   Prints detalhados por lag
      --test                    Apenas a primeira bacia do primeiro MoV

    Exemplos:
      modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI" --basins "uruguai"
      modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI" --basins "all"
      modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI,NIN04" --basins "all"
    """
    from datetime import datetime
    from pathlib import Path
    import time
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.visualization.cluster_plots import generate_plots_granular

    mov_names  = [m.strip() for m in movs.split(",")]
    mov_list   = discover_mov_indices(mov_names)
    basin_list = (
        list(BASINS.keys()) if basins.strip().lower() == "all"
        else [b.strip() for b in basins.split(",")]
    )

    if test:
        mov_list   = mov_list[:1]
        basin_list = basin_list[:1]

    exp_n   = get_next_exp_n("results/predep_granular_brazil")
    exp_dir = Path(f"results/predep_granular_brazil/exp{exp_n}")

    print("=" * 80)
    print("PREDEP GRANULAR — BRASIL POR BACIA — MODAL")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experimento: exp{exp_n}  |  Saída: {exp_dir}/")
    print(f"MoVs:   {[m for m, _ in mov_list]}")
    print(f"Bacias: {basin_list}")
    print(f"Estações: {', '.join(SEASONS)}  |  Lags: 1-{N_LAGS}  |  n_boot={n_boot}")
    print("=" * 80)

    upload_xavier_data()

    # Pré-calcula e sobe máscaras de todas as bacias
    print("\nCalculando máscaras das bacias...")
    xavier_lats, xavier_lons = get_xavier_grid()
    for bname in basin_list:
        if bname not in BASINS:
            print(f"  [AVISO] Bacia '{bname}' não encontrada em BASINS.")
            continue
        mask_2d, lats_c, lons_c = compute_basin_mask(BASINS[bname], xavier_lats, xavier_lons)
        upload_basin_mask(bname, mask_2d, lats_c, lons_c)

    all_results = []
    start_total = time.time()

    for idx, (mov_name, mov_path) in enumerate(mov_list, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(mov_list)}] MoV: {mov_name}")
        print("=" * 60)
        upload_mov_index(mov_name, mov_path)

        mov_start = time.time()
        for b_idx, bname in enumerate(basin_list, 1):
            if bname not in BASINS:
                continue
            print(f"  [{b_idx}/{len(basin_list)}] bacia: {bname}")
            result = process_mov.remote(mov_name, n_boot, exp_n, debug, bname, use_anomaly)
            download_results(mov_name, exp_n)
            all_results.append(result)

            basin_nc = exp_dir / mov_name / f"{mov_name}_{bname}_predep_granular_seasonal.nc"
            if basin_nc.exists():
                base_plots = exp_dir / "plots" / mov_name
                print(f"  Gerando plots para bacia {bname}...")
                generate_plots_granular(basin_nc, mov_name, base_plots, basin_name=bname)

        # Mescla todas as bacias num único NC
        print(f"\n  Mesclando {len(basin_list)} bacias...")
        nc_merged = merge_basin_results(mov_name, basin_list, exp_n)

        if nc_merged and nc_merged.exists():
            base_plots = exp_dir / "plots" / mov_name
            print(f"  Gerando plots Brasil para {mov_name}...")
            generate_plots_granular(nc_merged, mov_name, base_plots, basin_name="Brasil")

        print(f"  {mov_name} concluído em {(time.time()-mov_start)/60:.1f}min")

    total = time.time() - start_total
    print("\n" + "=" * 80)
    print("PREDEP GRANULAR BRASIL — CONCLUÍDO!")
    print(f"Tempo total: {total/60:.1f}min  ({total/3600:.2f}h)")
    print(f"Resultados em: {exp_dir}/")
    print("=" * 80)
    return {"total_duration": total, "results": all_results}
