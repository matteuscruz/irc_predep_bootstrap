"""
A cadeia completa de uma bacia: PREDEP + r² → escolha por predominância → SOM.

Ordem e por quê:

1. **α e r² por pixel** para cada (estação, lag, MoV) e cada fonte, com as
   mesmas amostras e a mesma semente nas duas — só assim a diferença entre
   Xavier e CHIRPS é diferença de dado, e não de sorteio.
2. **A escolha por predominância**: UM (lag, MoV) por estação para a bacia
   inteira, pelo `argmax` da média espacial de `min(α_xavier, α_chirps)`.
   Escolher o máximo em cada pixel infla α por construção — o máximo de 203
   candidatos é sempre alto — e faz as duas fontes escolherem candidatos
   diferentes, misturando diferença de valor com diferença de escolha.
3. **O SOM**, treinado sob a máscara de concordância e projetado em todos os
   pixels.

Cada bacia produz um NetCDF em `results/bacias/<bacia>.nc`, com o α, o r², a
escolha e os regimes juntos — para as etapas a jusante não precisarem refazer
nada e para a comparação entre bacias ler um arquivo por bacia.

O custo é dominado pelo passo 1: ~13 ms por chamada de `predep`, e são
`pixels × MoVs × lags × estações × 2` chamadas. Uruguai (1.570 px, 29 MoVs, 7
lags) leva ~70 min com 8 núcleos; a Amazônica (31.041 px) levaria ~23 h. Para
bacias grandes ou para `all`, use o Modal.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.grid_predep import (SEASONS, calcular_grid,  # noqa: E402
                                      escolher_por_predominancia)
from src.analysis.som_regimes import regimes_por_estacao  # noqa: E402
from src.data_processing.bacias import (pixels_da_bacia,  # noqa: E402
                                        resolver_bacias, resolver_movs)
from src.pipeline.config import (JANELA_COMUM, MS_POR_CHAMADA,  # noqa: E402,F401
                                 SAIDA_PADRAO, estimar_custo)


def executar_bacia(bacia, *, movs="all", lags=range(7), seasons=SEASONS,
                   nc_janela=JANELA_COMUM, out_dir=SAIDA_PADRAO, n_regimes=4,
                   tol=0.10, seed=20260908, n_proc=None, iters=50,
                   pular_som=False, sobrescrever=False):
    """Roda a cadeia inteira para uma bacia e grava o NetCDF. Devolve o `Path`."""
    import xarray as xr

    destino = Path(out_dir)
    if not destino.is_absolute():
        destino = ROOT / destino
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{bacia}.nc"
    if caminho.exists() and not sobrescrever:
        print(f"  [=] {caminho.name} já existe — pulando (use --sobrescrever)")
        return caminho

    t0 = time.time()
    ds = xr.open_dataset(ROOT / nc_janela if not Path(nc_janela).is_absolute()
                         else nc_janela)
    movs_nomes = [str(m) for m in ds.mov_name.values]
    movs_idx = resolver_movs(movs, movs_nomes)
    lat, lon = ds.latitude.values, ds.longitude.values
    px = pixels_da_bacia(bacia, lat, lon)
    if len(px) == 0:
        print(f"  [!] {bacia}: nenhum pixel na janela comum — pulando")
        return None

    lags = list(lags)
    print(f"  [1/3] PREDEP + r² — {bacia}", flush=True)
    alpha, r2 = calcular_grid(ds, px, movs_idx, lags=lags, seasons=seasons,
                              seed=seed, n_proc=n_proc)

    print(f"  [2/3] escolha por predominância", flush=True)
    sel_lag, sel_mov, concord = escolher_por_predominancia(alpha)
    for si, sn in enumerate(seasons):
        print(f"      {sn}: {movs_nomes[movs_idx[sel_mov[si]]]} · "
              f"lag {lags[sel_lag[si]]}", flush=True)

    # α das duas fontes NO MESMO candidato — a base da máscara e do SOM
    n_se = len(seasons)
    alpha_sel = np.stack([[alpha[s, si, sel_lag[si], sel_mov[si]]
                           for si in range(n_se)] for s in range(2)])
    features = np.stack([[alpha[s, si, sel_lag[si]] for si in range(n_se)]
                         for s in range(2)])          # (fonte, est, MoV, pixel)

    som = {}
    if not pular_som:
        print(f"  [3/3] SOM · k={n_regimes}", flush=True)
        som = regimes_por_estacao(features, alpha_sel, n_regimes=n_regimes,
                                  tol=tol, iters=iters)
    else:
        print(f"  [3/3] SOM pulado")

    dados = {
        "alpha": (("source", "season", "lag", "mov_name", "pixel"), alpha),
        "r2": (("source", "season", "lag", "mov_name", "pixel"), r2),
        "concordancia": (("season", "lag", "mov_name"), concord.astype(np.float32)),
        "sel_lag": (("season",), sel_lag.astype(np.int16)),
        "sel_mov": (("season",), sel_mov.astype(np.int16)),
        "alpha_escolhido": (("source", "season", "pixel"), alpha_sel.astype(np.float32)),
    }
    if som:
        dados.update({
            "regime": (("source", "season", "pixel"), som["regime"]),
            "erro_bmu": (("source", "season", "pixel"), som["erro_bmu"]),
            "assinatura": (("source", "season", "regime_id", "mov_name"),
                           som["assinatura"]),
            "mascara": (("season", "pixel"), som["mascara"]),
            "ari_fontes": (("season",), som["ari_fontes"]),
        })
    coords = {"source": ["xavier", "chirps"], "season": list(seasons),
              "lag": lags, "mov_name": [movs_nomes[i] for i in movs_idx],
              "pixel": px, "latitude": ("pixel", lat[px]),
              "longitude": ("pixel", lon[px])}
    if som:
        coords["regime_id"] = list(range(n_regimes))

    out = xr.Dataset(dados, coords=coords)
    out.attrs.update({
        "bacia": bacia,
        "pixels": len(px),
        "criterio": "um (lag, MoV) por estação — argmax da média espacial de "
                    "min(α_xavier, α_chirps)",
        "predep": "estimador do artigo (i==j permitidos), partição ward k=√N, "
                  "n_boot = ceil(n log n)",
        "mascara": f"|α_chirps − α_xavier| < {tol} no candidato escolhido",
        "som": f"treinado na máscara, projetado em todos os pixels, k={n_regimes}",
        "seed": seed,
        "fonte_dados": str(nc_janela),
    })
    out.to_netcdf(caminho)
    print(f"  ✅ {caminho}  ({time.time()-t0:.0f} s)", flush=True)
    return caminho


def executar_varias(bacias="all", *, movs="all", **kw):
    """A cadeia para várias bacias, da menor para a maior.

    A ordem importa na prática: as bacias pequenas terminam cedo e dão o
    resultado (e o erro, se houver) antes de comprometer horas na Amazônica.
    """
    import xarray as xr

    nomes = resolver_bacias(bacias)
    nc = kw.get("nc_janela", JANELA_COMUM)
    ds = xr.open_dataset(ROOT / nc if not Path(nc).is_absolute() else nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    n_mov = len(resolver_movs(movs, [str(m) for m in ds.mov_name.values]))
    n_lag = len(list(kw.get("lags", range(7))))

    tamanhos = [(b, len(pixels_da_bacia(b, lat, lon))) for b in nomes]
    tamanhos.sort(key=lambda t: t[1])
    total_s = sum(estimar_custo(n, n_mov, n_lag, n_proc=kw.get("n_proc") or 8)
                  for _, n in tamanhos)
    print(f"\n{len(tamanhos)} bacia(s), {sum(n for _, n in tamanhos):,} pixels · "
          f"custo estimado do PREDEP: {total_s/3600:.1f} h\n")

    saidas = []
    for i, (b, n) in enumerate(tamanhos, 1):
        est = estimar_custo(n, n_mov, n_lag, n_proc=kw.get("n_proc") or 8)
        print(f"[{i}/{len(tamanhos)}] {b} — {n:,} px (~{est/60:.0f} min)", flush=True)
        try:
            p = executar_bacia(b, movs=movs, **kw)
            if p:
                saidas.append(p)
        except Exception as e:                     # uma bacia não derruba as outras
            print(f"  ⚠ {b}: {type(e).__name__}: {e}", flush=True)
    return saidas
