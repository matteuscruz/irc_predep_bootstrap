#!/usr/bin/env python3
"""
Gera os acumulados mensais do Xavier e do CHIRPS a partir do RAW, no mesmo formato.

Motivação
---------
Os dois mensais que o projeto usava não eram intercambiáveis:

* `pr_monthly_1961_2019_BRAZIL_combined.nc` (Xavier) parava em 2019 embora o raw
  diário chegue a **2024-12** (o arquivo `...-004.nc` tem nome de 2001-2019 mas
  contém 2020-2024), gravava **0.0 em vez de NaN** fora do território brasileiro —
  artefato do merge, ausente na origem — e usava float64 sem compressão (870 MB
  para um conteúdo que cabe em ~140 MB).
* O mensal do CHIRPS carimbava o **início** do mês; o do Xavier, o **fim**. Cruzar
  os dois exigia converter para `PeriodIndex` em toda análise.

Este script produz os dois pela mesma rotina e com o mesmo contrato de formato:

    variável   `pr`, mm/mês, float32, zlib nível 4 + shuffle
    tempo      datetime64, carimbo no PRIMEIRO dia do mês
    grade      Xavier 0.1° (latitude/longitude float32, mesmos centros)
    ausência   NaN — nunca 0.0
    agregação  soma dos dias com `min_count=1`

O arquivo antigo do Xavier NÃO é sobrescrito: `experiments/02..08`, o Kedro e
`src/analysis/som_to_parquet.py` ainda apontam para ele. O novo entra ao lado.

Uso:
  python tools/build_monthly_from_raw.py            # ambas as fontes
  python tools/build_monthly_from_raw.py --only xavier
  python tools/build_monthly_from_raw.py --only chirps

O CHIRPS depende de `zstandard` (store zarr v3 — ver tools/zarr3_read.py).
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

XAVIER_RAW_DIR = ROOT / "data/raw/xavier_daily"
CHIRPS_ZARR = ROOT / "data/raw/chirps_daily/chirps_v2_brazil.zarr"
XAVIER_OUT = ROOT / "data/processed/xavier_monthly/pr_monthly_1961_2024_BRAZIL_XAVIER.nc"
CHIRPS_OUT_DIR = ROOT / "data/processed/chirps_monthly"

# A época do eixo de tempo é LIDA de time/zarr.json, nunca fixada aqui. Ela já
# mudou uma vez: o store baixado em 2026-09-14 começa em 1981 ("days since
# 1981-01-01"), o anterior em 1996. Com a época fixa em 1996, o store novo teria
# todas as datas deslocadas 15 anos para a frente, sem erro nenhum.
CHIRPS_FILL = -9999.0

ENCODING = {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}


# ── contrato de formato ──────────────────────────────────────────────────────


def finalize(ds: xr.Dataset, *, title: str, source: str, extra: dict) -> xr.Dataset:
    """Aplica o contrato comum de formato a um dataset mensal já montado."""
    ds = ds.assign_coords(
        latitude=ds.latitude.astype("float32"),
        longitude=ds.longitude.astype("float32"),
    )
    ds["pr"] = ds["pr"].astype("float32")
    ds["pr"].attrs.update({
        "long_name": "Monthly accumulated precipitation",
        "units": "mm/month",
        "cell_methods": "time: sum (min_count=1)",
        "original_temporal_resolution": "daily",
    })
    ds.attrs.update({
        "title": title,
        "source": source,
        "temporal_resolution": "monthly",
        "time_convention": "timestamp on the FIRST day of the month",
        "missing_data": "NaN (never 0.0)",
        "grid": "Xavier BR-DWGD 0.1 deg",
        "format_contract": "tools/build_monthly_from_raw.py",
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    })
    return ds


def write(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: dict(ENCODING) for v in ds.data_vars if ds[v].dtype.kind == "f"}
    ds.to_netcdf(path, encoding=enc)
    n_valid = int(np.isfinite(ds["pr"].isel(time=0).values).sum())
    print(f"\n✅ {path.relative_to(ROOT)}")
    print(f"   {ds.sizes['time']} meses  "
          f"{pd.Timestamp(ds.time.values[0]):%Y-%m} → {pd.Timestamp(ds.time.values[-1]):%Y-%m}")
    print(f"   grade {ds.sizes['latitude']}×{ds.sizes['longitude']}  "
          f"pixels válidos: {n_valid:,}")
    print(f"   {path.stat().st_size / 1e6:.0f} MB")


# ── Xavier ───────────────────────────────────────────────────────────────────


def build_xavier() -> xr.Dataset:
    """Soma mensal dos 4 NetCDFs diários do Xavier (1961-2024).

    Cada arquivo começa em 1º de janeiro e termina em 31 de dezembro, então
    nenhum mês fica dividido entre arquivos e o resample pode ser feito por
    arquivo, sem materializar os ~11,6 GB de uma vez. O `-004` traz 8 variáveis
    (ETo, RH, Rs, Tmax, Tmin, pr, u2, interpolated_mask) — só `pr` interessa.
    """
    # Seleção explícita pelo padrão de nome do BR-DWGD: um glob "*.nc" cru já
    # engoliu uma cópia do PRÓPRIO mensal combinado que alguém deixou nesta
    # pasta, misturando dado mensal com diário sem nenhum erro visível.
    files = sorted(XAVIER_RAW_DIR.glob("pr_*_BR-DWGD_*.nc"))
    if not files:
        raise FileNotFoundError(f"Nenhum NetCDF BR-DWGD em {XAVIER_RAW_DIR}")

    parts, land = [], None
    for f in files:
        ds = xr.open_dataset(f, chunks={"time": 366})
        t = pd.DatetimeIndex(ds.time.values)
        step = pd.Series(t).diff().dropna().median()
        if step != pd.Timedelta(days=1):
            raise ValueError(f"{f.name} não é diário (passo mediano {step}) — "
                             "só entra NetCDF diário do BR-DWGD aqui.")
        print(f"   {f.name[:42]:<44} {t[0]:%Y-%m-%d} → {t[-1]:%Y-%m-%d}  "
              f"({len(t)} dias, {len(ds.data_vars)} var)")
        if land is None:
            land = np.isfinite(ds["pr"].isel(time=0).values)
        monthly = ds["pr"].resample(time="MS").sum(dim="time", skipna=True, min_count=1)
        parts.append(monthly.compute().astype("float32"))
        ds.close()

    pr = xr.concat(parts, dim="time").sortby("time")
    pr = pr.isel(time=~pd.Index(pr.time.values).duplicated())

    # A máscara de território é estática de 1961-01-01 até 2024-03-20; a partir
    # de 2024-03-21 o arquivo `-004` para de aplicá-la e a interpolação ADW vaza
    # para o oceano e países vizinhos (até 33 mm/dia onde não há estação alguma).
    # Reimpor a máscara mantém o produto uniforme no registro inteiro.
    leaked = int((np.isfinite(pr.values) & ~land[None, :, :]).any(axis=(1, 2)).sum())
    if leaked:
        print(f"   ⚠ {leaked} meses com dado fora da máscara de território "
              f"(vazamento do -004 a partir de 2024-03-21) — remascarados")
    pr = pr.where(xr.DataArray(land, dims=("latitude", "longitude")))
    return xr.Dataset({"pr": pr})


# ── CHIRPS ───────────────────────────────────────────────────────────────────


def chirps_epoch(store) -> date:
    """Época do eixo de tempo, a partir de `units` em time/zarr.json."""
    import re
    from tools.zarr3_read import meta
    units = meta(store, "time")["attributes"]["units"]
    m = re.match(r"days since (\d{4})-(\d{2})-(\d{2})", units)
    if not m:
        raise ValueError(f"Unidade de tempo inesperada no CHIRPS: {units!r}")
    return date(int(m[1]), int(m[2]), int(m[3]))


def chirps_out(ds) -> Path:
    """Nome pelo período de fato coberto — um período novo nunca sobrescreve o antigo."""
    t = pd.DatetimeIndex(ds.time.values)
    return CHIRPS_OUT_DIR / f"pr_monthly_{t[0].year}_{t[-1].year}_BRAZIL_CHIRPS.nc"


def build_chirps(xavier_lats, xavier_lons) -> xr.Dataset:
    """Soma mensal do CHIRPS diário (zarr v3) + regrid 0.05°→0.1° na grade Xavier.

    Streaming: um acumulador de mês no 0.05° é descarregado (coarsen 2×2) assim
    que o mês fecha. A primeira linha/coluna do CHIRPS é órfã (o par 2×2 que a
    acompanharia cai fora do domínio) e é descartada antes do coarsen; depois
    disso os centros batem exatamente com a grade Xavier. Uma célula Xavier só
    recebe valor onde os 4 sub-pixels CHIRPS são válidos.
    """
    from tools.zarr3_read import iter_time_blocks, meta, read_array

    store = str(CHIRPS_ZARR)
    lat = read_array(store, "latitude").astype("float64")
    lon = read_array(store, "longitude").astype("float64")
    days = read_array(store, "time")
    epoca = chirps_epoch(store)
    times = pd.DatetimeIndex([epoca + timedelta(days=int(d)) for d in days])
    if not (np.diff(days) == 1).all():
        raise ValueError("Eixo de tempo do CHIRPS não é diário e contínuo.")
    print(f"   {CHIRPS_ZARR.name:<44} {times[0]:%Y-%m-%d} → {times[-1]:%Y-%m-%d}  "
          f"({meta(store, 'precip')['shape'][0]} dias)")

    months = pd.PeriodIndex(times, freq="M")
    uniq = months.unique()
    month_of = {p: k for k, p in enumerate(uniq)}

    trim_lat, trim_lon = lat[1:], lon[1:]
    n_lat, n_lon = len(trim_lat) // 2, len(trim_lon) // 2
    centers_lat = (trim_lat[0:2 * n_lat:2] + trim_lat[1:2 * n_lat:2]) / 2
    centers_lon = (trim_lon[0:2 * n_lon:2] + trim_lon[1:2 * n_lon:2]) / 2
    m_lat = np.isclose(centers_lat[:, None], xavier_lats[None, :], atol=1e-3)
    m_lon = np.isclose(centers_lon[:, None], xavier_lons[None, :], atol=1e-3)
    if not m_lat.any(axis=1).all() or not m_lon.any(axis=1).all():
        raise ValueError("Regrid CHIRPS não bateu com a grade Xavier.")
    out_lats = xavier_lats[m_lat.argmax(axis=1)]
    out_lons = xavier_lons[m_lon.argmax(axis=1)]

    out = np.full((len(uniq), n_lat, n_lon), np.nan, dtype=np.float32)
    msum = np.zeros((len(lat), len(lon)), dtype=np.float32)
    mcnt = np.zeros((len(lat), len(lon)), dtype=np.int16)
    cur, cov = -1, None

    def flush(k):
        month = np.where(mcnt > 0, msum, np.nan)[1:, 1:]
        blocks = month[:2 * n_lat, :2 * n_lon].reshape(n_lat, 2, n_lon, 2)
        with np.errstate(invalid="ignore"):
            out[k] = np.nanmean(blocks, axis=(1, 3))

    for t0, t1, plane in iter_time_blocks(store, "precip"):
        plane = plane.astype(np.float32)
        plane[plane <= CHIRPS_FILL + 1] = np.nan
        valid = ~np.isnan(plane)
        if cov is None:
            v = valid[0, 1:, 1:].astype(np.int16)   # máscara estática no tempo
            cov = (v[0:2 * n_lat:2, 0:2 * n_lon:2] + v[0:2 * n_lat:2, 1:2 * n_lon:2]
                   + v[1:2 * n_lat:2, 0:2 * n_lon:2] + v[1:2 * n_lat:2, 1:2 * n_lon:2])
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
        print(f"      dias {t0:>6}–{t1:<6}", end="\r")
    if cur >= 0:
        flush(cur)
    out[:, cov != 4] = np.nan
    print(" " * 30, end="\r")

    ds = xr.Dataset(
        {"pr": (("time", "latitude", "longitude"), out),
         "coverage_count": (("latitude", "longitude"), cov.astype("int8"))},
        coords={"time": uniq.to_timestamp(), "latitude": out_lats, "longitude": out_lons},
    )
    ds["coverage_count"].attrs["description"] = (
        "Sub-pixels CHIRPS (0.05°) válidos dentro da célula Xavier (0.1°); "
        "pr só é preenchido onde vale 4.")
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["xavier", "chirps"], help="gerar só uma fonte")
    args = ap.parse_args()

    print("=" * 74)
    print("MENSAL A PARTIR DO RAW — Xavier e CHIRPS no mesmo contrato de formato")
    print("=" * 74)

    if args.only != "chirps":
        print(f"\n📂 Xavier — {XAVIER_RAW_DIR.relative_to(ROOT)}")
        ds = build_xavier()
        ds = finalize(
            ds,
            title="Monthly Accumulated Precipitation — Xavier BR-DWGD v3.2.2",
            source="Aggregated from data/raw/xavier_daily/*.nc",
            extra={"institution": "UFES; UTEXAS",
                   "note": "Substitui pr_monthly_1961_2019_BRAZIL_combined.nc: "
                           "cobre até 2024 e preserva NaN fora do território.",
                   "land_mask": "Máscara estática do território tomada do primeiro "
                                "passo do registro e reimposta a todos os meses; sem "
                                "isso o arquivo -004 vaza chuva interpolada para fora "
                                "do Brasil a partir de 2024-03-21."},
        )
        write(ds, XAVIER_OUT)
        xavier_lats = ds.latitude.values.astype("float64")
        xavier_lons = ds.longitude.values.astype("float64")
        ds.close()
    else:
        with xr.open_dataset(XAVIER_OUT) as x:
            xavier_lats = x.latitude.values.astype("float64")
            xavier_lons = x.longitude.values.astype("float64")

    if args.only != "xavier":
        print(f"\n📂 CHIRPS — {CHIRPS_ZARR.relative_to(ROOT)}")
        ds = build_chirps(xavier_lats, xavier_lons)
        ds = finalize(
            ds,
            title="Monthly Accumulated Precipitation — CHIRPS v2 (Brazil)",
            source="Aggregated from data/raw/chirps_daily/chirps_v2_brazil.zarr",
            extra={"regrid_method": "coarsen 2x2 mean (0.05deg -> 0.1deg), "
                                    "snapped to Xavier grid; requires 4/4 sub-pixels"},
        )
        write(ds, chirps_out(ds))


if __name__ == "__main__":
    main()
