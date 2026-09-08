"""
Stage 0/1 — preparação dos dados: raw → NetCDF mensal na grade comum →
máscara de pixels da bacia do Paraná.

Todos os nós chamam as funções REAIS do repo principal (nada é
reimplementado aqui): `src/data_processing/transform/chirps_pr_accumulated.py`,
`src/data_processing/transform/chirps_regrid.py` e
`src/predep_calculation/parana_mask.py`.
"""

import sys
from pathlib import Path

import xarray as xr

# kedro/src/predep_viz/pipelines/data_prep/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def load_xavier_monthly(xavier_path: str) -> xr.Dataset:
    """
    **ENTRADA EXTERNA — sem produtor reproduzível no repo.**

    Abre `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc`,
    o NetCDF mensal do Xavier que é a base de TODO o resto (todos os
    experimentos 02-08, esta pipeline e a cadeia SOM leem esse arquivo).

    Dívida técnica conhecida (mapeada em 2026-09-01, fora de escopo por
    decisão do usuário): **nenhum script do repo produz esse arquivo**.
    - `src/data_processing/transform/xavier_pr_accumulated.py` é legado
      (dict de arquivos hardcoded com um NC inexistente, `input()`
      bloqueante, e nomes de saída que ninguém lê).
    - `tools/convert_xavier_daily_to_monthly.py` gera uma versão recortada
      só do Rio Grande do Sul (`..._RS_combined.nc`), não a do Brasil.
    Ou seja: o arquivo foi gerado ad-hoc fora do repo e hoje não é
    reproduzível a partir dos NCs diários brutos (`data/raw/xavier_daily/`).
    """
    return xr.open_dataset(ROOT / xavier_path)


def convert_chirps_daily_to_monthly(chirps_raw_path: str, xavier: xr.Dataset) -> xr.Dataset:
    """
    CHIRPS diário (0.05°) → acumulado mensal → regrid para a grade do
    Xavier (0.1°).

    Duas funções reais encadeadas:
    - `chirps_pr_accumulated.calculate_monthly_accumulation`: `resample('MS').sum`
      com `min_count=1` (evita que um mês todo-NaN vire 0.0) e renomeia
      `precip` → `pr`.
    - `chirps_regrid.regrid_to_xavier_grid`: coarsen 2×2 (média), descartando
      a primeira linha/coluna "órfã" para os blocos baterem exatamente com
      os centros da grade Xavier, e expõe `coverage_count` (0-4) — `pr` fica
      mascarado onde a cobertura não é completa (4/4), evitando médias
      ruidosas de borda.
    """
    from src.data_processing.transform.chirps_pr_accumulated import (
        calculate_monthly_accumulation,
    )
    from src.data_processing.transform.chirps_regrid import regrid_to_xavier_grid

    ds = xr.open_dataset(ROOT / chirps_raw_path, chunks={"time": 3650})
    monthly = calculate_monthly_accumulation(ds).compute()
    ds.close()

    regridded = regrid_to_xavier_grid(
        monthly, xavier.latitude.values, xavier.longitude.values
    )
    regridded.attrs["source"] = f"Regridded from {Path(chirps_raw_path).name}"
    regridded.attrs["regrid_method"] = (
        "coarsen 2x2 mean (0.05deg -> 0.1deg), snapped to Xavier grid"
    )
    return regridded


def save_chirps_monthly(chirps_monthly: xr.Dataset, chirps_monthly_path: str) -> str:
    """
    Persiste o CHIRPS mensal regridado em
    `data/processed/chirps_monthly/pr_monthly_1981_2026_CHIRPS_0p1deg.nc` —
    o mesmo caminho que a produção (`experiments/08`) e a cadeia SOM leem.

    Retorna o caminho, para o nó da máscara depender explicitamente deste
    passo no grafo (e não de um efeito colateral invisível).
    """
    out = ROOT / chirps_monthly_path
    out.parent.mkdir(parents=True, exist_ok=True)
    chirps_monthly.to_netcdf(out)
    return str(out)


def build_parana_combined_mask(
    chirps_monthly_path_saved: str,
    xavier_path: str,
    basin_dir: str,
    mask_path: str,
) -> dict:
    """
    Máscara de pixels usada no cálculo: polígono da bacia do Paraná
    (shapefile, point-in-polygon sobre a grade Xavier) ∩ CHIRPS com
    cobertura completa (`coverage_count == 4`).

    A MESMA máscara é usada nos dois lados (Xavier e CHIRPS) na produção,
    garantindo footprint idêntico entre as fontes — 7503 pixels.

    Chama `src/predep_calculation/parana_mask.py:build_combined_mask`
    (extraída de `experiments/08_predep_parana_recompute.py`, que passou a
    importar de lá). Reaproveita o `.npz` se já existir.
    """
    from src.predep_calculation.parana_mask import build_combined_mask

    mask, lats, lons = build_combined_mask(
        xavier_file=ROOT / xavier_path,
        chirps_file=chirps_monthly_path_saved,
        basin_dir=ROOT / basin_dir,
        out_npz=ROOT / mask_path,
    )
    return {
        "n_pixels": int(mask.sum()),
        "shape": list(mask.shape),
        "lat_range": [float(lats.min()), float(lats.max())],
        "lon_range": [float(lons.min()), float(lons.max())],
        "path": str(ROOT / mask_path),
    }
