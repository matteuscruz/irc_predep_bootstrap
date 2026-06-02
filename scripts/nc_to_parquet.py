#!/usr/bin/env python3
"""
Converte resultados NetCDF para Parquet comprimido (zstd) para armazenamento
no GitHub.

Schema (formato longo): uma linha por (pixel, lag) dentro de cada bacia.
Tudo (máximo/média sobre lags, lag ótimo, diferença) é derivado no viewer.
Variáveis armazenadas por season:
    basin        category   nome da bacia
    latitude     float32    graus decimais
    longitude    float32    graus decimais
    lag          int8       lag (meses)
    r2           float32    R² naquele lag
    alpha        float32    α (PREDEP) naquele lag

Saída: {out_dir}/{exp}/{MoV}/{SEASON}.parquet

Uso:
    python scripts/nc_to_parquet.py --exp exp01
    python scripts/nc_to_parquet.py --exp exp01 --out-dir /caminho/para/results
    python scripts/nc_to_parquet.py --all
    python scripts/nc_to_parquet.py --all --lags 0,1,3,6,9,12
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "predep_granular_brazil"
DEFAULT_OUT = (
    ROOT.parent / "predep_visualization" / "results"
    / "predep_granular_brazil"
)

SEASONS = ["DJF", "MAM", "JJA", "SON"]
_NC_SUFFIX = "_predep_granular_seasonal"
_GITHUB_LIMIT_MB = 90


def _basin_from_stem(mov: str, stem: str) -> str:
    """Extrai bacia do stem do NC. Retorna '' para Brasil-wide."""
    after = stem[len(mov) + 1:]
    if not after.endswith(_NC_SUFFIX):
        return ""
    basin = after[: -len(_NC_SUFFIX)]
    return basin


def convert_experiment(exp: str, out_dir: Path, lags_keep=None) -> None:
    exp_dir = RESULTS_DIR / exp
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experimento não encontrado: {exp_dir}")

    print(f"\n{'='*60}")
    print(f"Experimento: {exp}")
    print(f"Saída:       {out_dir / exp}")
    print(f"{'='*60}")

    for mov_dir in sorted(exp_dir.iterdir()):
        if not mov_dir.is_dir() or mov_dir.name == "plots":
            continue
        mov = mov_dir.name

        # Descobrir arquivos de bacias (excluir Brasil-wide)
        basin_files = {}
        for nc in sorted(mov_dir.glob(f"*{_NC_SUFFIX}.nc")):
            basin = _basin_from_stem(mov, nc.stem)
            if basin:
                basin_files[basin] = nc

        if not basin_files:
            continue

        print(f"\n  {mov}  ({len(basin_files)} bacia(s))")

        # Acumular dados por season
        season_frames: dict[str, list] = {s: [] for s in SEASONS}

        for basin, nc_path in sorted(basin_files.items()):
            ds = xr.open_dataset(nc_path)
            lats = ds.coords["latitude"].values.astype("float32")
            lons = ds.coords["longitude"].values.astype("float32")
            seasons_nc = list(ds.coords["season"].values)
            lags = [int(x) for x in ds.coords["lag"].values]
            r2_full = ds["r2"].values           # (season, lag, lat, lon)
            al_full = ds["alpha_core"].values
            ds.close()

            # Índices dos lags a manter (todos, ou subconjunto pedido)
            lag_idx = [
                li for li, lag in enumerate(lags)
                if lags_keep is None or lag in lags_keep
            ]

            for season in SEASONS:
                if season not in seasons_nc:
                    continue
                si = seasons_nc.index(season)
                for li in lag_idx:
                    lag = lags[li]
                    r2_2d = r2_full[si, li]
                    al_2d = al_full[si, li]
                    valid = ~(np.isnan(r2_2d) & np.isnan(al_2d))
                    if not valid.any():
                        continue
                    ii, jj = np.where(valid)
                    season_frames[season].append(pd.DataFrame({
                        "basin":     pd.Categorical([basin] * len(ii)),
                        "latitude":  lats[ii],
                        "longitude": lons[jj],
                        "lag":       np.full(len(ii), lag, dtype=np.int8),
                        "r2":        r2_2d[ii, jj].astype("float32"),
                        "alpha":     al_2d[ii, jj].astype("float32"),
                    }))

        # Salvar um Parquet por season
        out_mov = out_dir / exp / mov
        out_mov.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        for season in SEASONS:
            frames = season_frames[season]
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            out_path = out_mov / f"{season}.parquet"
            df.to_parquet(out_path, compression="zstd", index=False)
            size_mb = out_path.stat().st_size / 1024 / 1024
            total_bytes += out_path.stat().st_size
            flag = " ⚠ > 90 MB" if size_mb > _GITHUB_LIMIT_MB else ""
            n_rows = f"{len(df):>9,}"
            print(f"    {season}: {n_rows} linhas | {size_mb:.1f} MB{flag}")

        total_mb = total_bytes / 1024 / 1024
        print(f"    {'—'*40}")
        print(f"    Total {mov}: {total_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--exp", metavar="EXP",
                     help="Nome do experimento (ex: exp01)")
    grp.add_argument("--all", action="store_true",
                     help="Converte todos os experimentos disponíveis")
    parser.add_argument(
        "--out-dir", metavar="DIR", default=str(DEFAULT_OUT),
        help=f"Diretório de saída (padrão: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--lags", metavar="L", default="",
        help="Subconjunto de lags a manter, ex: 0,1,3,6,9,12 "
             "(padrão: todos os lags presentes no NC)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lags_keep = None
    if args.lags:
        lags_keep = {int(x) for x in args.lags.split(",") if x.strip()}
        print(f"Mantendo apenas os lags: {sorted(lags_keep)}")

    if args.all:
        exps = sorted(
            d.name for d in RESULTS_DIR.iterdir()
            if d.is_dir() and re.match(r"^exp\d+$", d.name)
        )
        print(f"Convertendo {len(exps)} experimento(s): {exps}")
        for exp in exps:
            convert_experiment(exp, out_dir, lags_keep)
    else:
        convert_experiment(args.exp, out_dir, lags_keep)

    # Resumo final
    total = sum(f.stat().st_size for f in out_dir.rglob("*.parquet"))
    n_files = sum(1 for _ in out_dir.rglob("*.parquet"))
    print(f"\n{'='*60}")
    print(f"Total: {n_files} arquivos | {total/1024/1024:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
