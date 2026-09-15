"""
Junta os NetCDFs por bacia num único arquivo para o Brasil.

Cada bacia sai da pipeline em `results/bacias/<janela>/<bacia>.nc`. Aqui eles viram
`brasil.nc`, e a junção separa duas naturezas de grandeza:

**Por pixel — viram um mapa nacional.** `alpha`, `r2`, `alpha_escolhido`
(observado), `regime`, `erro_bmu` e `mascara`. O índice `pixel` é global na janela
comum e as bacias não compartilham pixels (verificado: 70.189 pixels distintos,
sobreposição zero), então concatenar é exato. `bacia_do_pixel` diz de onde cada
pixel veio.

**Por bacia — ganham a dimensão `bacia`.** A escolha (`sel_lag`, `sel_mov`), a
`concordancia`, o `consenso` com os surrogates, o `ari_fontes` e a `assinatura`
são DECIDIDOS dentro de cada bacia e não viram um mapa único.

Duas consequências que o arquivo não esconde:

- `alpha_escolhido` de pixels de bacias diferentes é o α de CANDIDATOS diferentes
  (cada bacia escolhe o seu MoV e lag). O mapa nacional é um mosaico de escolhas.
- `regime` é rótulo LOCAL: o regime 1 do Uruguai não é o regime 1 do Paraná. Só o
  par (`bacia_do_pixel`, `regime`) identifica um regime.

Recusa juntar arquivos incompatíveis — janela, período, precipitação (anomalias ou
totais), MoVs, lags, deslocamentos ou nº de regimes diferentes. Misturar um arquivo
com anomalias e outro com totais produziria um mapa sem erro nenhum e sem sentido.

Uso:
  python -m src.pipeline.juntar results/bacias/predep_janela_1982_2024_anom
"""
from __future__ import annotations

import argparse
from pathlib import Path

POR_PIXEL = ["alpha", "r2", "alpha_escolhido", "regime", "erro_bmu", "mascara"]
POR_BACIA = ["sel_lag", "sel_mov", "concordancia", "consenso", "ari_fontes", "assinatura"]
ATTRS_IGUAIS = ["janela", "periodo", "precipitacao"]
COORDS_IGUAIS = ["source", "season", "lag", "mov_name", "shift", "regime_id"]


def juntar(arquivos, saida=None, esperadas=None) -> Path:
    """Junta os `.nc` por bacia. Devolve o caminho do arquivo nacional.

    `esperadas` é a lista de bacias que deveriam estar presentes; as ausentes são
    registradas em `attrs["bacias_ausentes"]` e impressas, sem impedir a junção.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    arquivos = sorted(Path(a) for a in arquivos if Path(a).name != "brasil.nc")
    if not arquivos:
        raise FileNotFoundError("nenhum .nc de bacia para juntar")
    dss = [xr.open_dataset(a) for a in arquivos]
    nomes = [str(d.attrs.get("bacia", a.stem)) for d, a in zip(dss, arquivos)]
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"bacia repetida entre os arquivos: {nomes}")

    ref = dss[0]
    for d, n in zip(dss[1:], nomes[1:]):
        for k in ATTRS_IGUAIS:
            if d.attrs.get(k) != ref.attrs.get(k):
                raise ValueError(f"{n}: atributo {k!r} = {d.attrs.get(k)!r}, mas "
                                 f"{nomes[0]} tem {ref.attrs.get(k)!r} — não junto")
        for c in COORDS_IGUAIS:
            a_ref = ref[c].values if c in ref.coords else None
            a_d = d[c].values if c in d.coords else None
            if (a_ref is None) != (a_d is None) or (
                    a_ref is not None and not np.array_equal(a_ref, a_d)):
                raise ValueError(f"{n}: coordenada {c!r} difere de {nomes[0]} — não junto")

    todos_px = np.concatenate([d.pixel.values for d in dss])
    if len(np.unique(todos_px)) != len(todos_px):
        vals, cont = np.unique(todos_px, return_counts=True)
        raise ValueError(f"{int((cont > 1).sum())} pixels aparecem em mais de uma bacia")

    partes = []
    for i, d in enumerate(dss):
        v = [x for x in POR_PIXEL if x in d]
        p = d[v]
        if "alpha_escolhido" in p and "shift" in p["alpha_escolhido"].dims:
            p["alpha_escolhido"] = p["alpha_escolhido"].isel(shift=0, drop=True)
        p = p.drop_vars([c for c in ("shift",) if c in p.coords and "shift" not in p.dims])
        p["bacia_do_pixel"] = ("pixel", np.full(d.sizes["pixel"], i, dtype=np.int16))
        partes.append(p)
    pix = xr.concat(partes, dim="pixel").sortby("pixel")

    porb = []
    for d in dss:
        v = [x for x in POR_BACIA if x in d]
        q = d[v].drop_vars([c for c in ("pixel", "latitude", "longitude") if c in d[v].coords])
        porb.append(q)
    bac = xr.concat(porb, dim=pd.Index(nomes, name="bacia"))

    out = xr.merge([pix, bac], combine_attrs="drop")
    out["bacia_do_pixel"].attrs["description"] = "índice na coordenada `bacia`"
    out["regime"].attrs["description"] = ("rótulo LOCAL à bacia: só o par "
                                          "(bacia_do_pixel, regime) identifica um regime")
    out["alpha_escolhido"].attrs["description"] = (
        "α no (lag, MoV) escolhido pela PRÓPRIA bacia do pixel — bacias diferentes "
        "escolhem candidatos diferentes")
    ausentes = sorted(set(esperadas or []) - set(nomes))
    out.attrs.update({
        **{k: str(ref.attrs.get(k, "")) for k in ATTRS_IGUAIS},
        "title": "PREDEP e r² por pixel — todas as bacias disponíveis juntas",
        "bacias": ", ".join(nomes),
        "bacias_ausentes": ", ".join(ausentes) or "nenhuma",
        "pixels": int(out.sizes["pixel"]),
        "criterio": str(ref.attrs.get("criterio", "")),
        "predep": str(ref.attrs.get("predep", "")),
        "som": str(ref.attrs.get("som", "")),
        "juntado_por": "src/pipeline/juntar.py",
    })

    saida = Path(saida) if saida else arquivos[0].parent / "brasil.nc"
    enc = {v: {"zlib": True, "complevel": 4}
           for v in out.data_vars if out[v].dtype.kind == "f"}
    out.to_netcdf(saida, encoding=enc)
    print(f"✅ {saida}  ({saida.stat().st_size / 1e6:.0f} MB)")
    print(f"   {len(nomes)} bacia(s) · {out.sizes['pixel']:,} pixels")
    if ausentes:
        print(f"   ⚠ ausentes: {', '.join(ausentes)}")
    return saida


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", help="pasta com os <bacia>.nc de UMA janela/namespace")
    ap.add_argument("--saida", default=None, help="padrão: <pasta>/brasil.nc")
    a = ap.parse_args()
    pasta = Path(a.pasta)
    try:
        from src.data_processing.bacias import listar_bacias
        esperadas = listar_bacias()
    except Exception:
        esperadas = None
    juntar(sorted(pasta.glob("*.nc")), a.saida, esperadas)


if __name__ == "__main__":
    main()
