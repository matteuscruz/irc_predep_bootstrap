"""
Registro das bacias hidrográficas e a máscara de pixels de cada uma.

Generaliza `src/predep_calculation/parana_mask.py`, que só sabia da Paraná, para
as doze bacias em `data/clusters/`. A lógica de point-in-polygon é a do
`experiments/14_modal_todas_bacias_ci.py`: percorre TODOS os `*_recortado.shp`
da pasta (algumas bacias vêm partidas em vários arquivos), e recorta pela bbox
de cada anel antes do teste caro.

Lê com `pyshp` e não com `geopandas`: os shapefiles têm nomes acentuados e
encoding inconsistente (`ATLAÌNTICO LESTE_recortado.shp`), e `encodingErrors=
"replace"` resolve sem depender de fiona.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

# numpy NÃO entra aqui de propósito: `listar_bacias`, `resolver_bacias` e
# `resolver_movs` são registro puro e são chamados pelo caminho de despacho ao
# Modal, que precisa rodar sem a pilha científica local (ver pipeline/config.py).
ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = ROOT / "data/clusters"

# `bacias` é uma pasta agregadora, não uma bacia — fica de fora do "all".
NAO_E_BACIA = {"bacias"}


def listar_bacias(base: Path | None = None) -> list[str]:
    """Nomes das bacias que têm shapefile, em ordem alfabética."""
    base = Path(base) if base else CLUSTERS
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name not in NAO_E_BACIA and any(d.glob("*.shp")))


def resolver_bacias(pedido: str | list[str] | None, base: Path | None = None) -> list[str]:
    """Traduz o argumento da linha de comando em nomes de bacia.

    Aceita "all", uma lista separada por vírgula, ou uma lista Python. Erra
    cedo e com a lista completa quando o nome não existe — é o tipo de engano
    que só apareceria depois de minutos de processamento.
    """
    disponiveis = listar_bacias(base)
    if pedido is None or (isinstance(pedido, str) and pedido.strip().lower() == "all"):
        return disponiveis
    nomes = ([p.strip() for p in pedido.split(",")] if isinstance(pedido, str)
             else [str(p).strip() for p in pedido])
    nomes = [n for n in nomes if n]
    faltando = [n for n in nomes if n not in disponiveis]
    if faltando:
        raise ValueError(
            f"bacia(s) desconhecida(s): {', '.join(faltando)}\n"
            f"disponíveis: {', '.join(disponiveis)}")
    return nomes


def resolver_movs(pedido: str | list[str] | None, disponiveis: list[str]) -> list[int]:
    """Índices dos MoVs pedidos dentro de `disponiveis`, preservando essa ordem.

    Devolve índices e não nomes porque é assim que os cubos de α são
    indexados; manter a ordem do dataset evita ter de reordenar depois.
    """
    if pedido is None or (isinstance(pedido, str) and pedido.strip().lower() == "all"):
        return list(range(len(disponiveis)))
    nomes = ([p.strip() for p in pedido.split(",")] if isinstance(pedido, str)
             else [str(p).strip() for p in pedido])
    nomes = [n for n in nomes if n]
    idx = {m.upper(): i for i, m in enumerate(disponiveis)}
    faltando = [n for n in nomes if n.upper() not in idx]
    if faltando:
        raise ValueError(
            f"MoV(s) desconhecido(s): {', '.join(faltando)}\n"
            f"disponíveis: {', '.join(disponiveis)}")
    return sorted(idx[n.upper()] for n in nomes)


@lru_cache(maxsize=32)
def _aneis(bacia: str, base: str | None = None) -> tuple:
    """Anéis de polígono da bacia, com a bbox de cada um pré-calculada."""
    import numpy as np
    import shapefile

    pasta = Path(base) if base else CLUSTERS / bacia
    if base:
        pasta = Path(base) / bacia
    shps = sorted(glob.glob(str(pasta / "*_recortado.shp"))) or \
        sorted(glob.glob(str(pasta / "*.shp")))
    if not shps:
        raise FileNotFoundError(f"shapefile da bacia {bacia!r} não encontrado em {pasta}")
    out = []
    for shp in shps:
        sf = shapefile.Reader(shp, encodingErrors="replace")
        for sr in sf.shapeRecords():
            p = np.asarray(sr.shape.points)
            partes = list(sr.shape.parts) + [len(p)]
            for i in range(len(partes) - 1):
                anel = p[partes[i]:partes[i + 1]]
                if len(anel) >= 3:
                    out.append((anel, anel.min(axis=0), anel.max(axis=0)))
    return tuple(out)


def pixels_da_bacia(bacia: str, lat, lon, base: Path | None = None):
    """Índices dos pixels de (lat, lon) que caem dentro da bacia.

    A bbox de cada anel filtra antes do `contains_points`, que é o custo real —
    sem isso o teste roda sobre os 70 mil pixels do Brasil para cada anel.
    """
    import numpy as np
    from matplotlib.path import Path as MPath

    pts = np.column_stack([np.asarray(lon), np.asarray(lat)])
    dentro = np.zeros(len(pts), bool)
    for anel, (x0, y0), (x1, y1) in _aneis(bacia, str(base) if base else None):
        sel = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
               & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & ~dentro)
        if sel.any():
            dentro[sel] |= MPath(anel).contains_points(pts[sel])
    return np.where(dentro)[0]


def resumo_bacias(lat, lon, bacias: list[str] | None = None) -> "list[dict]":
    """Quantos pixels cada bacia tem na grade dada — para estimar custo antes de rodar."""
    out = []
    for b in (bacias or listar_bacias()):
        try:
            n = len(pixels_da_bacia(b, lat, lon))
        except FileNotFoundError as e:
            out.append({"bacia": b, "pixels": 0, "erro": str(e)})
            continue
        out.append({"bacia": b, "pixels": n})
    return out
