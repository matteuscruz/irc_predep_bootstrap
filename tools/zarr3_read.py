"""Leitor mínimo de zarr v3, para o store do CHIRPS Brasil.

Por que existe
--------------
O CHIRPS do Brasil foi baixado como zarr **formato 3**. Ler com ``xr.open_zarr`` não
funciona neste ambiente por dois motivos independentes:

1. o pacote ``zarr`` não está instalado — ``xarray.backends.list_engines()`` devolve
   apenas ``['netcdf4', 'h5netcdf', 'scipy', 'store']``;
2. ``zarr>=3`` (necessário para ler o formato 3) exige Python ≥ 3.11, e o ambiente tem
   3.10.12. Instalar arrastaria também um upgrade do ``xarray==2023.6.0`` fixado.

O store, porém, é trivial: float32/int64 little-endian, comprimido com zstd, chunks
nomeados ``<array>/c/<i>/<j>/<k>``. Este módulo decodifica direto, sem dependência de
``zarr``. Precisa apenas de ``numpy`` e ``zstandard``.

Validação
---------
Confrontado com ``data/raw/chirps_daily/chirps_v2.nc`` (extrato do Paraná) em
2000-01-15: 30.637 pixels comparados, diferença absoluta máxima **0,0**.

Uso
---
``zstandard`` não está no venv do projeto; rodar num venv separado::

    python3 -m venv /tmp/zarrenv && /tmp/zarrenv/bin/pip install numpy zstandard
    /tmp/zarrenv/bin/python -c "import tools.zarr3_read as z; ..."
"""
import json, os
import numpy as np
import zstandard as zstd

DEC = zstd.ZstdDecompressor()
DTYPE = {"float32": np.float32, "float64": np.float64,
         "int64": np.int64, "int32": np.int32}


def meta(store, array):
    with open(os.path.join(store, array, "zarr.json")) as f:
        return json.load(f)


def _chunk(store, array, idx, shape, dt):
    path = os.path.join(store, array, "c", *map(str, idx))
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = DEC.decompress(f.read(), max_output_size=int(np.prod(shape)) * dt.itemsize)
    return np.frombuffer(raw, dtype=dt).reshape(shape)


def read_array(store, array):
    """Lê um array inteiro (só usar em arrays pequenos: coordenadas)."""
    m = meta(store, array)
    shape = tuple(m["shape"])
    cs = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
    dt = np.dtype(DTYPE[m["data_type"]])
    out = np.full(shape, m.get("fill_value") or 0, dtype=dt)
    grid = [int(np.ceil(s / c)) for s, c in zip(shape, cs)]
    for idx in np.ndindex(*grid):
        blk = _chunk(store, array, idx, cs, dt)
        if blk is None:
            continue
        sl = tuple(slice(i * c, min(i * c + c, s))
                   for i, c, s in zip(idx, cs, shape))
        out[sl] = blk[tuple(slice(0, s.stop - s.start) for s in sl)]
    return out


def iter_time_blocks(store, array="precip"):
    """Gera (t0, t1, bloco) percorrendo os chunks no eixo do tempo.

    Monta o plano espacial completo para cada faixa de tempo, para que a agregação
    mensal possa ser feita em streaming sem materializar os 27 GB do cubo.
    """
    m = meta(store, array)
    shape = tuple(m["shape"])
    cs = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
    dt = np.dtype(DTYPE[m["data_type"]])
    fill = m.get("fill_value", np.nan)
    grid = [int(np.ceil(s / c)) for s, c in zip(shape, cs)]

    for ti in range(grid[0]):
        t0, t1 = ti * cs[0], min(ti * cs[0] + cs[0], shape[0])
        plane = np.full((t1 - t0, shape[1], shape[2]), fill, dtype=dt)
        for yi in range(grid[1]):
            for xi in range(grid[2]):
                blk = _chunk(store, array, (ti, yi, xi), cs, dt)
                if blk is None:
                    continue
                y0, y1 = yi * cs[1], min(yi * cs[1] + cs[1], shape[1])
                x0, x1 = xi * cs[2], min(xi * cs[2] + cs[2], shape[2])
                plane[:, y0:y1, x0:x1] = blk[: t1 - t0, : y1 - y0, : x1 - x0]
        yield t0, t1, plane
