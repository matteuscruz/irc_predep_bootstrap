"""
Regrid do CHIRPS mensal (0.05°) para a grade do Xavier (0.1°).

Extraído de `tools/convert_chirps_daily_to_monthly.py` para poder ser
reusado fora do script CLI (ex.: pela pipeline Kedro).
"""

import numpy as np
import xarray as xr


def regrid_to_xavier_grid(monthly_ds: xr.Dataset, xavier_lats, xavier_lons) -> xr.Dataset:
    """Coarsen CHIRPS 0.05°→0.1° (média de blocos 2×2) e alinha às coordenadas Xavier.

    Cada célula Xavier (0.1°) é formada por 4 células CHIRPS (0.05°). A máscara
    de validade do CHIRPS bruto é estática no tempo (verificado), então uma
    célula Xavier pode ser suportada por 0-4 sub-pixels válidos dependendo de
    quão perto está da borda da área de cobertura do CHIRPS. Para evitar médias
    ruidosas formadas por só 1-3 sub-pixels (artefato de borda de bacia),
    mantemos `pr` apenas onde os 4 sub-pixels são válidos, e expomos
    `coverage_count` (0-4) para transparência.

    A primeira linha/coluna do CHIRPS (índice 0) é um "órfão" — sua célula-par
    (xavier_center − 0.025°) fica fora do domínio do CHIRPS — então é
    descartada antes do coarsen para os pares baterem exatamente com os
    centros da grade Xavier.
    """
    trimmed = monthly_ds.isel(latitude=slice(1, None), longitude=slice(1, None))

    coarse = trimmed.coarsen(latitude=2, longitude=2, boundary="trim").mean()

    valid_mask = trimmed["pr"].isel(time=0).notnull().astype(int)
    coverage_count = valid_mask.coarsen(latitude=2, longitude=2, boundary="trim").sum()
    coverage_count = coverage_count.rename("coverage_count")

    coarse["pr"] = coarse["pr"].where(coverage_count == 4)
    coarse["coverage_count"] = coverage_count
    coarse["coverage_count"].attrs["description"] = (
        "Número de sub-pixels CHIRPS (0.05°) válidos dentro da célula Xavier "
        "(0.1°) — 4 = cobertura completa, usado para mascarar pr."
    )

    lat_match = np.isclose(coarse.latitude.values[:, None], xavier_lats[None, :], atol=1e-6)
    lon_match = np.isclose(coarse.longitude.values[:, None], xavier_lons[None, :], atol=1e-6)
    if not lat_match.any(axis=1).all() or not lon_match.any(axis=1).all():
        raise ValueError(
            "Regrid CHIRPS não bateu exatamente com a grade Xavier — verificar alinhamento."
        )

    coarse_lats = xavier_lats[lat_match.argmax(axis=1)]
    coarse_lons = xavier_lons[lon_match.argmax(axis=1)]
    return coarse.assign_coords(latitude=coarse_lats, longitude=coarse_lons)
