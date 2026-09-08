#!/usr/bin/env python3
"""
Converte o CHIRPS diário do Brasil inteiro (zarr v3, 0.05°) em acumulado
mensal regrid para a grade Xavier (0.1°).

Diferença para `tools/convert_chirps_daily_to_monthly.py`
--------------------------------------------------------
Aquele script lê `data/raw/chirps_daily/chirps_v2.nc`, que é um recorte da
bacia do Paraná (113x123 células Xavier, 1981-2026). Este lê o store zarr do
Brasil inteiro (`chirps_v2_brazil.zarr`, 780x783 células CHIRPS, 1996-2026),
que não pode ser aberto por `xr.open_zarr` neste ambiente — ver o docstring de
`tools/zarr3_read.py`. Por isso a agregação mensal é feita em streaming com
numpy puro, chunk de tempo por chunk de tempo, em vez de `resample("MS")`.

Alinhamento de grade
--------------------
Mesma regra do regrid do Paraná: a primeira linha/coluna do CHIRPS é órfã
(o par 2x2 que a acompanharia cai fora do domínio), então é descartada antes
do coarsen. Depois disso os 389x391 centros batem exatamente com a grade
Xavier (verificado, 0 mismatches). Como no regrid do Paraná, uma célula
Xavier só recebe valor onde os 4 sub-pixels CHIRPS são válidos;
`coverage_count` (0-4) é exposto para transparência.

Requer `zstandard`, que não está no venv do projeto:

    python3 -m venv /tmp/zarrenv
    /tmp/zarrenv/bin/pip install "numpy==1.24.3" "pandas==2.0.3" \
        "xarray==2023.6.0" netCDF4 zstandard
    /tmp/zarrenv/bin/python tools/convert_chirps_brazil_zarr_to_monthly.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.zarr3_read import iter_time_blocks, meta, read_array  # noqa: E402

STORE = "data/raw/chirps_daily/chirps_v2_brazil.zarr"
XAVIER_FILE = "data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc"
OUTPUT_FILE = "data/processed/chirps_monthly/pr_monthly_1996_2026_CHIRPS_BRAZIL_0p1deg.nc"

TIME_EPOCH = datetime(1996, 1, 1)  # de time/zarr.json: "days since 1996-01-01"
FILL = -9999.0


def coarsen2x2_mean(plane: np.ndarray) -> np.ndarray:
    """Média de blocos 2x2 ignorando NaN, sobre um plano já trimado."""
    n_lat = (plane.shape[0] // 2) * 2
    n_lon = (plane.shape[1] // 2) * 2
    blocks = plane[:n_lat, :n_lon].reshape(n_lat // 2, 2, n_lon // 2, 2)
    return np.nanmean(blocks, axis=(1, 3))


def main():
    print("=" * 70)
    print("CHIRPS BRASIL: DIÁRIO zarr (0.05°) → MENSAL REGRID (0.1°, grade Xavier)")
    print("=" * 70)

    lat = read_array(STORE, "latitude").astype("float64")
    lon = read_array(STORE, "longitude").astype("float64")
    days = read_array(STORE, "time")
    times = pd.DatetimeIndex([TIME_EPOCH + timedelta(days=int(d)) for d in days])
    n_time = meta(STORE, "precip")["shape"][0]
    print(f"\n📂 {STORE}")
    print(f"   Grade original: {len(lat)} lat × {len(lon)} lon")
    print(f"   Período: {times[0].date()} → {times[-1].date()} ({n_time} dias)")

    months = pd.PeriodIndex(times, freq="M")
    uniq_months = months.unique()
    month_of = {p: k for k, p in enumerate(uniq_months)}
    print(f"   Meses: {len(uniq_months)}")

    # Grade Xavier de referência
    with xr.open_dataset(XAVIER_FILE) as xds:
        xavier_lats = xds.latitude.values.astype("float64")
        xavier_lons = xds.longitude.values.astype("float64")

    # Alvo: centros dos blocos 2x2 depois de descartar a linha/coluna órfã
    trim_lat, trim_lon = lat[1:], lon[1:]
    n_out_lat = len(trim_lat) // 2
    n_out_lon = len(trim_lon) // 2
    centers_lat = (trim_lat[0 : 2 * n_out_lat : 2] + trim_lat[1 : 2 * n_out_lat : 2]) / 2
    centers_lon = (trim_lon[0 : 2 * n_out_lon : 2] + trim_lon[1 : 2 * n_out_lon : 2]) / 2

    lat_match = np.isclose(centers_lat[:, None], xavier_lats[None, :], atol=1e-3)
    lon_match = np.isclose(centers_lon[:, None], xavier_lons[None, :], atol=1e-3)
    if not lat_match.any(axis=1).all() or not lon_match.any(axis=1).all():
        raise ValueError("Regrid CHIRPS Brasil não bateu com a grade Xavier.")
    out_lats = xavier_lats[lat_match.argmax(axis=1)]
    out_lons = xavier_lons[lon_match.argmax(axis=1)]
    print(f"   Grade final: {n_out_lat} lat × {n_out_lon} lon (grade Xavier)")

    # Acumulação em streaming: um único acumulador de mês no 0.05° é mantido
    # em memória e descarregado (coarsen 2x2 → 0.1°) assim que o mês fecha —
    # o cubo mensal inteiro no 0.05° seria ~0.9 GB, o no 0.1° é ~0.2 GB.
    n_months = len(uniq_months)
    out = np.full((n_months, n_out_lat, n_out_lon), np.nan, dtype=np.float32)
    msum = np.zeros((len(lat), len(lon)), dtype=np.float32)
    mcnt = np.zeros((len(lat), len(lon)), dtype=np.int16)
    cur = -1
    cov = None

    def flush(month_idx):
        """Fecha o mês acumulado: min_count=1, trim da linha/coluna órfã, coarsen."""
        month = np.where(mcnt > 0, msum, np.nan)
        with np.errstate(invalid="ignore"):
            out[month_idx] = coarsen2x2_mean(month[1:, 1:])

    print("\n🔄 Acumulando mensal em streaming...")
    for t0, t1, plane in iter_time_blocks(STORE, "precip"):
        plane = plane.astype(np.float32)
        plane[plane <= FILL + 1] = np.nan
        valid = ~np.isnan(plane)
        if cov is None:
            # A máscara de validade do CHIRPS é estática no tempo, então o
            # coverage_count pode ser derivado do primeiro dia.
            v = valid[0, 1:, 1:].astype(np.int16)
            cov = (
                v[0 : 2 * n_out_lat : 2, 0 : 2 * n_out_lon : 2]
                + v[0 : 2 * n_out_lat : 2, 1 : 2 * n_out_lon : 2]
                + v[1 : 2 * n_out_lat : 2, 0 : 2 * n_out_lon : 2]
                + v[1 : 2 * n_out_lat : 2, 1 : 2 * n_out_lon : 2]
            )
        for k in range(t1 - t0):
            m = month_of[months[t0 + k]]
            if m != cur:
                if cur >= 0:
                    flush(cur)
                msum[:] = 0.0
                mcnt[:] = 0
                cur = m
            msum += np.where(valid[k], plane[k], 0.0)
            mcnt += valid[k]
        print(f"   dias {t0:>6}–{t1:<6} ({times[t0].date()})", end="\r")
    if cur >= 0:
        flush(cur)
    del msum, mcnt
    print("\n   ok")

    out[:, cov != 4] = np.nan

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds = xr.Dataset(
        {
            "pr": (("time", "latitude", "longitude"), out),
            "coverage_count": (("latitude", "longitude"), cov),
        },
        coords={
            "time": uniq_months.to_timestamp(),
            "latitude": out_lats.astype("float32"),
            "longitude": out_lons.astype("float32"),
        },
    )
    ds["pr"].attrs.update(
        {
            "long_name": "Monthly accumulated precipitation",
            "units": "mm/month",
            "description": "Monthly accumulated precipitation from daily CHIRPS (Brazil zarr)",
            "original_units": "mm/day",
            "processing_method": "sum_of_daily_values",
            "processing_date": now,
        }
    )
    ds["coverage_count"].attrs["description"] = (
        "Número de sub-pixels CHIRPS (0.05°) válidos dentro da célula Xavier "
        "(0.1°) — 4 = cobertura completa, usado para mascarar pr."
    )
    ds.attrs.update(
        {
            "title": "Monthly Accumulated Precipitation - CHIRPS v2 (Brazil)",
            "source": f"Aggregated from {STORE}",
            "regrid_method": "coarsen 2x2 mean (0.05deg -> 0.1deg), snapped to Xavier grid",
            "original_temporal_resolution": "daily",
            "new_temporal_resolution": "monthly",
            "processing_date": now,
        }
    )

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding={"pr": {"zlib": True, "complevel": 4}})
    print(f"\n✅ Salvo: {out_path}")
    print(f"   📅 {ds.time.min().dt.strftime('%Y-%m').item()} → "
          f"{ds.time.max().dt.strftime('%Y-%m').item()}  ({n_months} meses)")
    print(f"   📊 pixels com cobertura completa: {int((cov == 4).sum())}")


if __name__ == "__main__":
    main()
