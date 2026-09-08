"""
Acumulação mensal dos dados de precipitação CHIRPS.

Espelha a lógica de `xavier_pr_accumulated.py:calculate_monthly_accumulation`,
adaptada para a variável `precip` do CHIRPS (em vez de `pr`).
"""

from datetime import datetime

import xarray as xr


def calculate_monthly_accumulation(ds: xr.Dataset) -> xr.Dataset:
    """
    Calcula o acumulado mensal de precipitação a partir de dados diários CHIRPS.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset com a variável `precip` (mm/dia) e dimensão `time` diária.

    Returns
    -------
    xr.Dataset
        Dataset com a variável `pr` (mm/mês), acumulada mensalmente.
    """
    if "precip" not in ds.data_vars:
        raise ValueError("Variável 'precip' não encontrada no dataset")

    # min_count=1 garante que um mês sem nenhum dia válido vire NaN em vez de
    # 0.0 (o comportamento default do sum com skipna=True trata all-NaN como
    # zero, o que corromperia pixels fora da área de cobertura do CHIRPS).
    monthly_ds = ds.resample(time="MS").sum(dim="time", skipna=True, min_count=1)
    monthly_ds = monthly_ds.rename({"precip": "pr"})

    monthly_ds["pr"].attrs.update(
        {
            "long_name": "Monthly accumulated precipitation",
            "units": "mm/month",
            "description": "Monthly accumulated precipitation from daily CHIRPS dataset",
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_units": "mm/day",
            "processing_method": "sum_of_daily_values",
        }
    )
    monthly_ds.attrs.update(
        {
            "title": "Monthly Accumulated Precipitation - CHIRPS v2",
            "processing_info": "Converted from daily to monthly accumulated values",
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_temporal_resolution": "daily",
            "new_temporal_resolution": "monthly",
        }
    )

    return monthly_ds
