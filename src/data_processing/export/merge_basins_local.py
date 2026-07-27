#!/usr/bin/env python3
"""
Re-cria o brasil-todo NC de um MoV a partir de todos os NCs de bacia
presentes localmente.

Uso:
    python src/data_processing/export/merge_basins_local.py --movs SAODI,SASDI --exp exp_brasil
"""
import argparse
from pathlib import Path
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "predep_granular_brazil"

BASIN_SUFFIX = "_predep_granular_seasonal.nc"


def merge_mov(mov: str, exp_dir: Path) -> None:
    mov_dir = exp_dir / mov
    basin_ncs = sorted([
        f for f in mov_dir.glob(f"{mov}_*{BASIN_SUFFIX}")
        if f.stem != f"{mov}_predep_granular_seasonal"  # exclui o brasil-todo
    ])

    if not basin_ncs:
        print(f"  {mov}: nenhum NC de bacia encontrado em {mov_dir}")
        return

    print(f"\n{mov}: mergeando {len(basin_ncs)} bacias...")
    for nc in basin_ncs:
        print(f"  + {nc.name}")

    dsets = [xr.open_dataset(nc) for nc in basin_ncs]
    merged = xr.merge(dsets, join="outer", fill_value=float("nan"))
    merged.attrs["description"] = (
        f"PREDEP granular Brasil-todo — MoV={mov} "
        f"bacias={','.join(nc.stem.replace(f'{mov}_','').replace('_predep_granular_seasonal','') for nc in basin_ncs)}"
    )

    out = mov_dir / f"{mov}{BASIN_SUFFIX}"
    merged.to_netcdf(out)
    for ds in dsets:
        ds.close()
    merged.close()

    import numpy as np
    ds_check = xr.open_dataset(out)
    valid = (~np.isnan(ds_check["alpha_core"].values)).sum()
    total = ds_check["alpha_core"].values.size
    print(f"  → {out.name}: {valid:,}/{total:,} valores válidos ({100*valid/total:.1f}%)")
    ds_check.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--movs", required=True, help="MoVs separados por vírgula")
    parser.add_argument("--exp", default="exp_brasil")
    args = parser.parse_args()

    exp_dir = RESULTS / args.exp
    movs = [m.strip() for m in args.movs.split(",") if m.strip()]

    for mov in movs:
        merge_mov(mov, exp_dir)


if __name__ == "__main__":
    main()
