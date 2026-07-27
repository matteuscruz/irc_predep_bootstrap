#!/usr/bin/env python3
"""
Converte o NC brasil-todo de um MoV para parquets por season com labels de bacia,
usando as grades de um MoV de referência (padrão: ONI) para identificar pixels
por bacia.

Uso:
    python src/data_processing/export/brasil_nc_to_parquet.py --movs AMO,AO --exp exp_brasil
    python src/data_processing/export/brasil_nc_to_parquet.py --movs AMO --exp exp_brasil --ref-mov ONI
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "predep_granular_brazil"
DEFAULT_OUT = ROOT.parent / "predep_visualization" / "results" / "predep_granular_brazil"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
_NC_SUFFIX = "_predep_granular_seasonal"


def build_basin_pixel_index(ref_mov_dir: Path, ref_mov: str) -> dict:
    """
    Lê os NCs por bacia do MoV de referência e retorna um dict:
      { (lat_rounded, lon_rounded) → basin_name }
    """
    index = {}
    for nc in sorted(ref_mov_dir.glob(f"{ref_mov}_*{_NC_SUFFIX}.nc")):
        basin = nc.stem.replace(f"{ref_mov}_", "").replace(_NC_SUFFIX, "")
        if not basin:
            continue
        ds = xr.open_dataset(nc)
        lats = ds.latitude.values.astype("float32")
        lons = ds.longitude.values.astype("float32")
        # usa os dados de um lag/season qualquer para detectar pixels válidos
        valid_mask = np.any(~np.isnan(ds["alpha_core"].values[0]), axis=0)  # (lat, lon)
        ii, jj = np.where(valid_mask)
        for i, j in zip(ii, jj):
            key = (round(float(lats[i]), 3), round(float(lons[j]), 3))
            index[key] = basin
        ds.close()
    return index


def convert_brasil_nc(mov: str, exp: str, ref_mov: str, out_dir: Path,
                      lags_keep: set | None) -> None:
    exp_dir = RESULTS_DIR / exp
    brasil_nc = exp_dir / mov / f"{mov}{_NC_SUFFIX}.nc"
    if not brasil_nc.exists():
        raise FileNotFoundError(f"NC brasil-todo não encontrado: {brasil_nc}")

    ref_mov_dir = exp_dir / ref_mov
    if not ref_mov_dir.is_dir():
        raise FileNotFoundError(f"MoV de referência não encontrado: {ref_mov_dir}")

    print(f"\n{'='*60}")
    print(f"{mov} — brasil-todo NC → parquet  (ref={ref_mov})")
    print(f"{'='*60}")

    print(f"  Construindo índice de bacias a partir de {ref_mov}...")
    basin_index = build_basin_pixel_index(ref_mov_dir, ref_mov)
    print(f"  {len(basin_index):,} pixels indexados em {len(set(basin_index.values()))} bacias")

    ds = xr.open_dataset(brasil_nc)
    lats = ds.latitude.values.astype("float32")
    lons = ds.longitude.values.astype("float32")
    lags_all = [int(x) for x in ds.lag.values]
    seasons_nc = list(ds.season.values)
    r2_full = ds["r2"].values         # (season, lag, lat, lon)
    al_full = ds["alpha_core"].values
    ds.close()

    lag_idx = [
        li for li, lag in enumerate(lags_all)
        if lags_keep is None or lag in lags_keep
    ]
    lags_used = [lags_all[li] for li in lag_idx]
    print(f"  Lags: {lags_used}")

    season_frames: dict[str, list] = {s: [] for s in SEASONS}

    for si, season in enumerate(SEASONS):
        if season not in seasons_nc:
            continue
        s_idx = seasons_nc.index(season)
        for li in lag_idx:
            lag = lags_all[li]
            r2_2d = r2_full[s_idx, li]    # (lat, lon)
            al_2d = al_full[s_idx, li]

            # Itera só nos pixels que têm dados e pertencem a alguma bacia
            rows = []
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    if np.isnan(r2_2d[i, j]) and np.isnan(al_2d[i, j]):
                        continue
                    key = (round(float(lat), 3), round(float(lon), 3))
                    basin = basin_index.get(key)
                    if basin is None:
                        continue
                    rows.append((basin, float(lat), float(lon), lag,
                                 float(r2_2d[i, j]), float(al_2d[i, j])))

            if rows:
                df = pd.DataFrame(rows,
                                  columns=["basin", "latitude", "longitude",
                                           "lag", "r2", "alpha"])
                df["basin"] = pd.Categorical(df["basin"])
                df["latitude"] = df["latitude"].astype("float32")
                df["longitude"] = df["longitude"].astype("float32")
                df["lag"] = df["lag"].astype("int8")
                df["r2"] = df["r2"].astype("float32")
                df["alpha"] = df["alpha"].astype("float32")
                season_frames[season].append(df)

    out_mov = out_dir / exp / mov
    out_mov.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for season in SEASONS:
        frames = season_frames[season]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        bacias = sorted(df.basin.unique())
        out_path = out_mov / f"{season}.parquet"
        df.to_parquet(out_path, compression="zstd", index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        total_bytes += out_path.stat().st_size
        print(f"  {season}: {len(df):>9,} linhas | {size_mb:.1f} MB | "
              f"{len(bacias)} bacias")

    total_mb = total_bytes / 1024 / 1024
    print(f"  Total {mov}: {total_mb:.1f} MB → {out_mov}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--movs", required=True,
                        help="MoVs a converter (vírgula), ex: AMO,AO")
    parser.add_argument("--exp", default="exp_brasil",
                        help="Nome do experimento (padrão: exp_brasil)")
    parser.add_argument("--ref-mov", default="ONI",
                        help="MoV de referência para mapa de bacias (padrão: ONI)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"Diretório de saída (padrão: {DEFAULT_OUT})")
    parser.add_argument("--lags", default="0,1,3,6,9,12",
                        help="Lags a manter (padrão: 0,1,3,6,9,12)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lags_keep = {int(x) for x in args.lags.split(",") if x.strip()}
    movs = [m.strip() for m in args.movs.split(",") if m.strip()]

    print(f"Convertendo: {movs}  |  exp={args.exp}  |  ref={args.ref_mov}")
    print(f"Lags: {sorted(lags_keep)}  |  Saída: {out_dir}")

    for mov in movs:
        convert_brasil_nc(mov, args.exp, args.ref_mov, out_dir, lags_keep)

    total = sum(f.stat().st_size for f in out_dir.rglob("*.parquet"))
    print(f"\nTotal em {out_dir}: {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
