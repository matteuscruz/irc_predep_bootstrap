"""
Construção da máscara de pixels usada no cálculo granular do PREDEP na
bacia do Paraná: polígono da bacia (shapefile) ∩ cobertura CHIRPS completa.

Extraído de `experiments/08_predep_parana_recompute.py` para poder ser
reusado fora do script Modal (ex.: pela pipeline Kedro) sem depender de
importar um módulo cujo nome começa com dígito.
"""

from pathlib import Path

import numpy as np
import xarray as xr


def compute_basin_mask(basin_dir, lats_brazil, lons_brazil):
    """
    Point-in-polygon do shapefile da bacia sobre o grid Xavier.

    Lê o primeiro `.shp` de `basin_dir`, recorta o grid à bbox do shapefile
    (com 0.2° de padding) e marca os pontos dentro de qualquer anel do
    polígono. Mesma lógica de
    `experiments/06_predep_comparisson_brazil_granular.py:compute_basin_mask`.

    Retorna `(mask_2d, lats_crop, lons_crop)`.
    """
    import shapefile
    from matplotlib.path import Path as MplPath

    shp_files = list(Path(basin_dir).glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"Nenhum .shp em {basin_dir}")
    sf = shapefile.Reader(str(shp_files[0]), encoding="utf-8")

    bbox = sf.bbox  # [lon_min, lat_min, lon_max, lat_max]
    pad = 0.2
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


def build_combined_mask(xavier_file, chirps_file, basin_dir, out_npz, verbose=True):
    """
    Máscara combinada: polígono do Paraná ∩ CHIRPS `coverage_count == 4`,
    sobre o grid do Xavier. Salva em `out_npz` (`mask`, `lats`, `lons`) e
    reaproveita o arquivo se ele já existir.

    O domínio bruto do CHIRPS é mais estreito que a bbox (com padding) do
    polígono do Paraná, então as coordenadas são intersecionadas em vez de
    assumir que o CHIRPS cobre toda a faixa recortada.

    Retorna `(mask, lats, lons)`.
    """
    out_npz = Path(out_npz)
    if out_npz.exists():
        if verbose:
            print(f"Máscara combinada já existe: {out_npz} — pulando recomputo")
        npz = np.load(out_npz)
        return npz["mask"], npz["lats"], npz["lons"]

    xavier_ds = xr.open_dataset(xavier_file)
    lats_full = xavier_ds.latitude.values
    lons_full = xavier_ds.longitude.values
    xavier_ds.close()

    parana_mask, lats_crop, lons_crop = compute_basin_mask(basin_dir, lats_full, lons_full)
    if verbose:
        print(f"Paraná (polígono): {int(parana_mask.sum())} pixels na bbox {parana_mask.shape}")

    chirps_ds = xr.open_dataset(chirps_file).sel(
        latitude=slice(float(lats_crop.min()), float(lats_crop.max())),
        longitude=slice(float(lons_crop.min()), float(lons_crop.max())),
    )
    chirps_lats = chirps_ds.latitude.values
    chirps_lons = chirps_ds.longitude.values

    lat_keep = np.isin(np.round(lats_crop, 6), np.round(chirps_lats, 6))
    lon_keep = np.isin(np.round(lons_crop, 6), np.round(chirps_lons, 6))
    lats_final = lats_crop[lat_keep]
    lons_final = lons_crop[lon_keep]
    parana_mask_final = parana_mask[np.ix_(lat_keep, lon_keep)]
    if verbose:
        print(
            f"Interseção com domínio CHIRPS: bbox {parana_mask.shape} -> "
            f"{parana_mask_final.shape} ({int(parana_mask_final.sum())} pixels Paraná)"
        )

    chirps_valid = (
        chirps_ds["coverage_count"].sel(latitude=lats_final, longitude=lons_final) == 4
    ).values
    chirps_ds.close()

    combined = parana_mask_final & chirps_valid
    if verbose:
        print(f"Máscara combinada (Paraná ∩ CHIRPS completo): {int(combined.sum())} pixels")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, mask=combined, lats=lats_final, lons=lons_final)
    return combined, lats_final, lons_final
