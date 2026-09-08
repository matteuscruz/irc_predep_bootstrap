#!/usr/bin/env python3
"""
Converte CHIRPS diário (0.05°) em acumulado mensal regrid para a grade Xavier (0.1°).

O grid CHIRPS está "nested" dentro do grid Xavier: cada célula Xavier (0.1°)
corresponde a exatamente 2×2 células CHIRPS (0.05°), centradas em
xavier_center ± 0.025°. A primeira linha/coluna do CHIRPS (índice 0) é um
"órfão" — sua célula-par (xavier_center - 0.025°) fica fora do domínio do
CHIRPS — então ela é descartada antes do coarsen para os pares baterem
exatamente com os centros da grade Xavier (verificado: 113×123 pontos,
0 mismatches).

Uso:
  python tools/convert_chirps_daily_to_monthly.py
"""

import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_processing.transform.chirps_pr_accumulated import (
    calculate_monthly_accumulation,
)
from src.data_processing.transform.chirps_regrid import regrid_to_xavier_grid

CHIRPS_FILE = "data/raw/chirps_daily/chirps_v2.nc"
XAVIER_FILE = "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
OUTPUT_FILE = "data/processed/chirps_monthly/pr_monthly_1981_2026_CHIRPS_0p1deg.nc"


def main():
    print("=" * 70)
    print("CONVERSÃO CHIRPS: DIÁRIO (0.05°) → MENSAL REGRID (0.1°, grade Xavier)")
    print("=" * 70)

    print(f"\n📂 Lendo: {CHIRPS_FILE}")
    ds = xr.open_dataset(CHIRPS_FILE, chunks={"time": 3650})
    print(f"   Grade original: {len(ds.latitude)} lat × {len(ds.longitude)} lon")
    print(f"   Período: {ds.time.min().dt.year.item()}-{ds.time.max().dt.year.item()}")

    print("\n🔄 Calculando acumulado mensal...")
    monthly = calculate_monthly_accumulation(ds)
    monthly = monthly.compute()
    ds.close()
    print(f"   Meses gerados: {len(monthly.time)}")

    print(f"\n📐 Lendo grade Xavier de referência: {XAVIER_FILE}")
    xavier_ds = xr.open_dataset(XAVIER_FILE)
    xavier_lats = xavier_ds.latitude.values
    xavier_lons = xavier_ds.longitude.values
    xavier_ds.close()

    print("🔄 Regrid CHIRPS → grade Xavier (coarsen 2×2, média)...")
    regridded = regrid_to_xavier_grid(monthly, xavier_lats, xavier_lons)
    print(f"   Grade final: {len(regridded.latitude)} lat × {len(regridded.longitude)} lon")

    regridded.attrs["source"] = f"Regridded from {Path(CHIRPS_FILE).name}"
    regridded.attrs["regrid_method"] = "coarsen 2x2 mean (0.05deg -> 0.1deg), snapped to Xavier grid"

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regridded.to_netcdf(output_path)
    print(f"\n✅ Salvo: {output_path}")
    print(f"   📅 Período: {regridded.time.min().dt.year.item()}-{regridded.time.max().dt.year.item()}")
    print(f"   📊 Total de meses: {len(regridded.time)}")


if __name__ == "__main__":
    main()
