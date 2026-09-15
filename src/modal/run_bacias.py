#!/usr/bin/env python3
"""
A cadeia por bacia rodando no Modal — PREDEP + r² → predominância → SOM.

É o executor padrão do `main.py`. Local a mesma conta leva ~70 min para o
Uruguai e mais de 50 h para as doze bacias; aqui cada (bacia, estação, lag) vai
para o seu container e o conjunto sai em minutos.

Divisão do trabalho:

  `calcular_fatia`  um container por (bacia, estação, lag). Calcula α e r² de
                    todos os MoVs e pixels daquela fatia e grava um `.npz` no
                    volume. É onde está todo o custo.
  `montar_bacia`    junta as fatias da bacia, aplica a escolha por
                    predominância, roda os SOMs e grava o NetCDF final.

As fatias ficam em `/fatias/<janela>[_anom]/<bacia>/s<est>_l<lag>_d<desloc>.npz`, e
os resultados em `/results/<janela>[_anom]/`. O prefixo da janela é obrigatório:
reexecutar depois de uma falha pula o que já saiu, e sem ele uma janela nova
reaproveitaria em silêncio as fatias calculadas sobre outra.

O volume `predep-common-window` já traz as janelas (`predep_janela_1982_2024.nc`,
e a antiga `predep_common_window.nc`) e os
shapefiles em `/clusters` — nada é enviado a cada corrida.

`local_entrypoint` não importa numpy nem xarray de propósito: o ambiente local
do usuário tem numpy 2.x incompatível com o xarray instalado, e qualquer import
desses aqui quebraria antes de chegar ao Modal.

Uso:
  modal run src/modal/run_bacias.py --bacias uruguai --movs all
  modal run src/modal/run_bacias.py --bacias all --movs all
  modal run src/modal/run_bacias.py --bacias uruguai --janela predep_janela_1982_2024.nc --anomalias
"""
import modal

image = (
    modal.Image.debian_slim()
    .pip_install(["pandas", "xarray[complete]", "numpy", "scipy", "astropy",
                  "netcdf4", "h5netcdf", "pyshp", "matplotlib", "minisom",
                  "scikit-learn"])
    .add_local_python_source("src")
)
app = modal.App("predep-bacias", image=image)
volume = modal.Volume.from_name("predep-common-window", create_if_missing=False)

DATA = "/data"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
# 0 = observado; os demais são surrogates por deslocamento anual. Sem eles o α
# e o consenso não têm contra o que ser lidos: em n=87 o estimador do artigo põe
# o α do NULO em +0,34, e o consenso entre fontes chega a 64% sem sinal nenhum.
SHIFTS_PADRAO = [0, 3, 5, 7, 9, 11, 13]
MESES = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
JANELA_PADRAO = "predep_janela_1982_2024.nc"   # mesma de JANELA_MODAL_PADRAO em config.py


def _tag(janela: str, anomalias: bool) -> str:
    """Namespace das fatias e dos resultados. MESMA regra de `tag_janela` em
    `src/pipeline/config.py` — o `main.py` baixa os arquivos por ela."""
    base = janela.rsplit("/", 1)[-1]
    base = base[:-3] if base.endswith(".nc") else base
    return base + ("_anom" if anomalias else "")


def _pixels(bacia, janela=JANELA_PADRAO):
    """Point-in-polygon dos shapefiles da bacia sobre a grade da janela comum."""
    import glob

    import numpy as np
    import shapefile
    import xarray as xr
    from matplotlib.path import Path as MPath

    ds = xr.open_dataset(f"{DATA}/{janela}")
    pts = np.column_stack([ds.longitude.values, ds.latitude.values])
    dentro = np.zeros(len(pts), bool)
    shps = sorted(glob.glob(f"{DATA}/clusters/{bacia}/*_recortado.shp")) or \
        sorted(glob.glob(f"{DATA}/clusters/{bacia}/*.shp"))
    if not shps:
        raise FileNotFoundError(f"shapefile da bacia {bacia!r} não encontrado")
    for shp in shps:
        sf = shapefile.Reader(shp, encodingErrors="replace")
        for sr in sf.shapeRecords():
            p = np.asarray(sr.shape.points)
            partes = list(sr.shape.parts) + [len(p)]
            for i in range(len(partes) - 1):
                anel = p[partes[i]:partes[i + 1]]
                if len(anel) < 3:
                    continue
                (x0, y0), (x1, y1) = anel.min(axis=0), anel.max(axis=0)
                sel = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                       & (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & ~dentro)
                if sel.any():
                    dentro[sel] |= MPath(anel).contains_points(pts[sel])
    return np.where(dentro)[0]


def _movs_idx(pedido, nomes):
    if not pedido or pedido.strip().lower() == "all":
        return list(range(len(nomes)))
    idx = {m.upper(): i for i, m in enumerate(nomes)}
    pedidos = [p.strip().upper() for p in pedido.split(",") if p.strip()]
    faltando = [p for p in pedidos if p not in idx]
    if faltando:
        raise ValueError(f"MoV(s) desconhecido(s): {', '.join(faltando)}")
    return sorted(idx[p] for p in pedidos)


# Os processos do Pool são fixados neste número, e não em os.cpu_count(): num
# container o cpu_count enxerga os núcleos do host, não a reserva.
CPUS_FATIA = 8


@app.function(cpu=CPUS_FATIA, memory=32768, timeout=4 * 3600, volumes={DATA: volume})
def calcular_fatia(bacia: str, si: int, lag: int, shift: int, movs: str, seed: int,
                   janela: str = JANELA_PADRAO, anomalias: bool = False):
    """α e r² de todos os MoVs e pixels, para uma (bacia, estação, lag, deslocamento).

    `shift = 0` é o observado; os demais deslocam a série do MoV em N anos e
    produzem a distribuição nula pelo MESMO procedimento.
    """
    import os
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.analysis.grid_predep import deslocar_anos, fatia_alfa_r2

    destino = Path(f"{DATA}/fatias/{_tag(janela, anomalias)}/{bacia}")
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"s{si}_l{lag}_d{shift}.npz"
    if arquivo.exists():
        print(f"[{bacia}] s{si} l{lag} d{shift} já existe — pulando", flush=True)
        return str(arquivo)

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

    t0 = time.time()
    ds = xr.open_dataset(f"{DATA}/{janela}")
    nomes = [str(m) for m in ds.mov_name.values]
    mi_list = _movs_idx(movs, nomes)
    px = _pixels(bacia, janela)
    meses = pd.DatetimeIndex(ds.time.values).month.values
    mov = ds["mov"].values.astype(float)
    if shift:
        mov = np.column_stack([deslocar_anos(mov[:, c], shift)
                               for c in range(mov.shape[1])])
    pr = np.stack([ds["pr_xavier"].values[:, px],
                   ds["pr_chirps"].values[:, px]]).astype(np.float32)
    if anomalias:
        # Climatologia mensal por pixel e por fonte, sobre a janela inteira. Sem
        # isto, dentro de MAM e SON ~45% da variância da chuva é o ciclo sazonal,
        # e um MoV com estrutura mensal residual "prediz" o calendário. É a CHUVA
        # que perde o ciclo; os surrogates seguem deslocando só o MoV.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for m in range(1, 13):
                sel = meses == m
                if sel.any():
                    pr[:, sel, :] -= np.nanmean(pr[:, sel, :], axis=1, keepdims=True)

    def progresso(k, total, mi):
        print(f"[{bacia}] s{si} l{lag} d{shift} · {nomes[mi]} ({k}/{total}) "
              f"{time.time()-t0:.0f}s", flush=True)

    alpha, r2 = fatia_alfa_r2(pr, mov, meses, si, lag, mi_list, seed, shift=shift,
                              n_proc=CPUS_FATIA, ao_terminar=progresso)

    np.savez_compressed(arquivo, alpha=alpha, r2=r2, px=px,
                        movs=np.array([nomes[i] for i in mi_list]))
    volume.commit()
    print(f"[{bacia}] s{si} l{lag} d{shift} ✅ {time.time()-t0:.0f}s", flush=True)
    return str(arquivo)


@app.function(cpu=8, memory=32768, timeout=7200, volumes={DATA: volume})
def montar_bacia(bacia: str, lags: list, shifts: list, n_regimes: int,
                 tol: float, iters: int, janela: str = JANELA_PADRAO,
                 anomalias: bool = False):
    """Junta as fatias, escolhe por predominância em CADA deslocamento e monta o NetCDF.

    O cubo completo de α só é guardado para o observado: com 7 deslocamentos ×
    29 MoVs × 7 lags, guardar tudo daria 2 GB na Amazônica. Dos surrogates
    ficam as grandezas derivadas — a escolha, o α nela e o consenso — que é o
    que serve de piso para ler o observado.
    """
    from pathlib import Path

    import numpy as np
    import xarray as xr

    from src.analysis.grid_predep import escolher_por_predominancia
    from src.analysis.som_regimes import regimes_por_estacao

    volume.reload()
    tag = _tag(janela, anomalias)
    ds0 = xr.open_dataset(f"{DATA}/{janela}")
    n_se, n_sh, n_lag = len(SEASONS), len(shifts), len(lags)

    alpha0 = r20 = None
    movs = px = None
    sel_lag = np.zeros((n_sh, n_se), dtype=np.int16)
    sel_mov = np.zeros((n_sh, n_se), dtype=np.int16)
    concord = None
    alpha_sel = None
    consenso = np.full((n_sh, n_se), np.nan, dtype=np.float32)

    for di, sh in enumerate(shifts):
        cubo = cubo_r2 = None
        for si in range(n_se):
            for li, lag in enumerate(lags):
                f = Path(f"{DATA}/fatias/{tag}/{bacia}/s{si}_l{lag}_d{sh}.npz")
                if not f.exists():
                    raise FileNotFoundError(f"fatia ausente: {f}")
                z = np.load(f, allow_pickle=True)
                if cubo is None:
                    if px is None:
                        px = z["px"]
                        movs = [str(m) for m in z["movs"]]
                        concord = np.full((n_sh, n_se, n_lag, len(movs)),
                                          np.nan, dtype=np.float32)
                        alpha_sel = np.full((2, n_sh, n_se, len(px)),
                                            np.nan, dtype=np.float32)
                    forma = (2, n_se, n_lag, len(movs), len(px))
                    cubo = np.full(forma, np.nan, dtype=np.float32)
                    cubo_r2 = np.full(forma, np.nan, dtype=np.float32)
                cubo[:, si, li] = z["alpha"]
                cubo_r2[:, si, li] = z["r2"]

        sl, sm, cc = escolher_por_predominancia(cubo)
        sel_lag[di], sel_mov[di], concord[di] = sl, sm, cc
        for si in range(n_se):
            alpha_sel[:, di, si] = cubo[:, si, sl[si], sm[si]]
        d = alpha_sel[1, di] - alpha_sel[0, di]
        consenso[di] = np.nanmean(np.abs(d) < tol, axis=-1)
        rot = " · ".join(f"{SEASONS[si]} {movs[sm[si]]}/lag{lags[sl[si]]}"
                         for si in range(n_se))
        # `rotulo`, não `tag`: `tag` é o namespace da janela, usado no caminho das
        # fatias nesta mesma função — reusar o nome o sobrescrevia a partir do
        # segundo deslocamento e a montagem procurava em /fatias/observado/.
        rotulo = "observado" if sh == 0 else f"surrogate {sh}a"
        print(f"[{bacia}] {rotulo:<14} {rot}", flush=True)
        if sh == 0:
            alpha0, r20 = cubo, cubo_r2

    # SOM só no observado: o nulo entra pelo consenso, e treinar 7× os SOMs
    # multiplicaria o custo sem responder nada que o consenso já não responda.
    features = np.stack([[alpha0[s, si, sel_lag[0, si]] for si in range(n_se)]
                         for s in range(2)])
    som = regimes_por_estacao(features, alpha_sel[:, 0], n_regimes=n_regimes,
                              tol=tol, iters=iters)

    out = xr.Dataset(
        {"alpha": (("source", "season", "lag", "mov_name", "pixel"), alpha0),
         "r2": (("source", "season", "lag", "mov_name", "pixel"), r20),
         "concordancia": (("shift", "season", "lag", "mov_name"), concord),
         "sel_lag": (("shift", "season"), sel_lag),
         "sel_mov": (("shift", "season"), sel_mov),
         "alpha_escolhido": (("source", "shift", "season", "pixel"), alpha_sel),
         "consenso": (("shift", "season"), consenso),
         "regime": (("source", "season", "pixel"), som["regime"]),
         "erro_bmu": (("source", "season", "pixel"), som["erro_bmu"]),
         "assinatura": (("source", "season", "regime_id", "mov_name"), som["assinatura"]),
         "mascara": (("season", "pixel"), som["mascara"]),
         "ari_fontes": (("season",), som["ari_fontes"])},
        coords={"source": ["xavier", "chirps"], "shift": shifts, "season": SEASONS,
                "lag": lags, "mov_name": movs, "regime_id": list(range(n_regimes)),
                "pixel": px,
                "latitude": ("pixel", ds0.latitude.values[px]),
                "longitude": ("pixel", ds0.longitude.values[px])})
    out.attrs.update({
        "bacia": bacia, "pixels": len(px),
        "criterio": "um (lag, MoV) por estação — argmax da média espacial de "
                    "min(α_xavier, α_chirps)",
        "predep": "estimador do artigo (i==j permitidos), ward k=√N, "
                  "n_boot = ceil(n log n)",
        "mascara": f"|α_chirps − α_xavier| < {tol} no candidato escolhido",
        "som": f"treinado na máscara, projetado em todos os pixels, k={n_regimes}",
        "nulo": "shift != 0 são surrogates por deslocamento anual; cada um "
                "escolhe o SEU candidato pelo mesmo critério. α e consenso do "
                "observado devem ser lidos contra eles, nunca contra zero.",
        "alpha_completo": "só o observado — os surrogates guardam a escolha, o "
                          "α nela e o consenso",
        "executor": "modal",
        "janela": janela,
        "periodo": str(ds0.attrs.get("time_window", "")),
        "precipitacao": ("anomalias (climatologia mensal da janela removida)"
                         if anomalias else "totais mensais brutos"),
    })
    Path(f"{DATA}/results/{tag}/bacias").mkdir(parents=True, exist_ok=True)
    caminho = f"{DATA}/results/{tag}/bacias/{bacia}.nc"
    out.to_netcdf(caminho)
    volume.commit()
    print(f"[{bacia}] ✅ {caminho}", flush=True)
    return caminho


@app.function(cpu=4, memory=16384, timeout=1800, volumes={DATA: volume})
def figuras_bacia(bacia: str, tag: str):
    """As duas figuras da bacia, geradas no Modal.

    Ficam aqui e não na máquina do usuário porque o ambiente local tem numpy
    2.x com scipy antigo — matplotlib e xarray quebram lá. O container tem a
    pilha certa, e o `main.py` só baixa os PNGs.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    from src.visualization.bacia_plots import gerar_figuras_bacia

    volume.reload()
    nc = f"{DATA}/results/{tag}/bacias/{bacia}.nc"
    if not Path(nc).exists():
        raise FileNotFoundError(f"{nc} não existe — rode o cálculo primeiro")
    destino = f"{DATA}/results/{tag}/figuras"
    saidas = gerar_figuras_bacia(nc, destino)
    volume.commit()
    return [str(s) for s in saidas]


@app.function(cpu=2, memory=32768, timeout=3600, volumes={DATA: volume})
def juntar_bacias(tag: str, esperadas: list):
    """Junta os `<bacia>.nc` do namespace num `brasil.nc`, onde os arquivos já estão."""
    import glob

    from src.pipeline.juntar import juntar

    volume.reload()
    arquivos = sorted(glob.glob(f"{DATA}/results/{tag}/bacias/*.nc"))
    if not arquivos:
        raise FileNotFoundError(f"nenhuma bacia calculada em /results/{tag}/bacias")
    saida = juntar(arquivos, f"{DATA}/results/{tag}/brasil.nc", esperadas)
    volume.commit()
    return str(saida)


@app.local_entrypoint()
def main(bacias: str = "all", movs: str = "all", lags: str = "0,1,2,3,4,5,6",
         shifts: str = "0,3,5,7,9,11,13", n_regimes: int = 4, tol: float = 0.10,
         iters: int = 50, seed: int = 20260908, so_figuras: bool = False,
         janela: str = JANELA_PADRAO, anomalias: bool = False, juntar: bool = False):
    """Espalha as fatias, monta cada bacia e desenha as figuras — ou só junta."""
    tag = _tag(janela, anomalias)
    print(f"janela {janela} · precipitação: "
          f"{'anomalias' if anomalias else 'totais brutos'} · namespace {tag}\n")
    todas = ["amazonica", "atlantico_leste", "atlantico_nordeste",
             "atlantico_nordeste_oriental", "atlantico_sudeste", "atlantico_sul",
             "paraguai", "parana", "paranaiba", "sao_francisco", "tocantins",
             "uruguai"]
    alvo = (todas if bacias.strip().lower() == "all"
            else [b.strip() for b in bacias.split(",") if b.strip()])
    desconhecidas = [b for b in alvo if b not in todas]
    if desconhecidas:
        raise SystemExit(f"bacia(s) desconhecida(s): {', '.join(desconhecidas)}\n"
                         f"disponíveis: {', '.join(todas)}")
    if juntar:
        saida = juntar_bacias.remote(tag, todas)
        print(f"\n✅ {saida}\n\n  modal volume get predep-common-window "
              f"/results/{tag}/brasil.nc results/bacias/{tag}/brasil.nc")
        return
    lag_list = [int(x) for x in lags.split(",") if x.strip() != ""]
    shift_list = [int(x) for x in shifts.split(",") if x.strip() != ""]
    if shift_list[0] != 0:
        shift_list = [0] + shift_list           # 0 é o observado, sempre presente

    if not so_figuras:
        tarefas = [(b, si, lag, sh, movs, seed, janela, anomalias)
                   for b in alvo for si in range(len(SEASONS))
                   for lag in lag_list for sh in shift_list]
        print(f"{len(alvo)} bacia(s) · {len(tarefas)} fatias "
              f"({len(SEASONS)} estações × {len(lag_list)} lags × "
              f"{len(shift_list)} deslocamentos, sendo {len(shift_list)-1} "
              f"surrogates)\n")
        for _ in calcular_fatia.starmap(tarefas):
            pass

        print("\nmontando as bacias...\n")
        list(montar_bacia.starmap([(b, lag_list, shift_list, n_regimes, tol, iters,
                                    janela, anomalias)
                                   for b in alvo]))

    print("\ndesenhando as figuras...\n")
    figuras = []
    for lote in figuras_bacia.starmap([(b, tag) for b in alvo]):
        figuras.extend(lote)
    for f in figuras:
        print(f"  {f}")

    print("\n" + "=" * 62)
    print(f"{len(alvo)} bacia(s) prontas no volume. Para baixar:\n")
    for b in alvo:
        print(f"  modal volume get predep-common-window "
              f"/results/{tag}/bacias/{b}.nc results/bacias/{tag}/{b}.nc")
        for suf in ("predep", "som"):
            print(f"  modal volume get predep-common-window "
                  f"/results/{tag}/figuras/{b}_{suf}.png plots/bacias/{tag}/{b}_{suf}.png")
