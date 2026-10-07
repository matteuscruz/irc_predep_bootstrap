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


def _tag(janela: str, anomalias: bool, n_boot: int = 0) -> str:
    """Namespace das fatias e dos resultados. MESMA regra de `tag_janela` em
    `src/pipeline/config.py` — o `main.py` baixa os arquivos por ela."""
    base = janela.rsplit("/", 1)[-1]
    base = base[:-3] if base.endswith(".nc") else base
    return base + ("_anom" if anomalias else "") + (f"_nb{n_boot}" if n_boot else "")


def _pixels(bacia, janela=JANELA_PADRAO):
    """Point-in-polygon dos shapefiles da bacia sobre a grade da janela comum."""
    import glob

    import numpy as np
    import shapefile
    import xarray as xr
    from matplotlib.path import Path as MPath

    # Tudo que se abre aqui é fechado: `volume.reload()` recusa rodar com arquivo
    # do volume aberto, e a busca do SOM chama esta função num container que
    # depois recarrega o volume.
    with xr.open_dataset(f"{DATA}/{janela}") as ds:
        pts = np.column_stack([ds.longitude.values, ds.latitude.values])
    dentro = np.zeros(len(pts), bool)
    shps = sorted(glob.glob(f"{DATA}/clusters/{bacia}/*_recortado.shp")) or \
        sorted(glob.glob(f"{DATA}/clusters/{bacia}/*.shp"))
    if not shps:
        raise FileNotFoundError(f"shapefile da bacia {bacia!r} não encontrado")
    for shp in shps:
        with shapefile.Reader(shp, encodingErrors="replace") as sf:
            registros = sf.shapeRecords()
        for sr in registros:
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
# 32 e não 8: os 29 MoVs de uma fatia (e os 12–18 SOMs de uma busca) cabem numa
# leva só. O resultado não depende disto — a paralelização por MoV é idêntica
# bit a bit à sequencial —, só o tempo de parede.
CPUS_FATIA = 32


@app.function(cpu=CPUS_FATIA, memory=65536, timeout=4 * 3600, volumes={DATA: volume})
def calcular_fatia(bacia: str, si: int, lag: int, shift: int, movs: str, seed: int,
                   janela: str = JANELA_PADRAO, anomalias: bool = False,
                   n_boot: int = 0):
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

    destino = Path(f"{DATA}/fatias/{_tag(janela, anomalias, n_boot)}/{bacia}")
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"s{si}_l{lag}_d{shift}.npz"
    if arquivo.exists():
        print(f"[{bacia}] s{si} l{lag} d{shift} já existe — pulando", flush=True)
        return str(arquivo)

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

    t0 = time.time()
    px = _pixels(bacia, janela)
    with xr.open_dataset(f"{DATA}/{janela}") as ds:
        nomes = [str(m) for m in ds.mov_name.values]
        meses = pd.DatetimeIndex(ds.time.values).month.values
        mov = ds["mov"].values.astype(float)
        pr = np.stack([ds["pr_xavier"].values[:, px],
                       ds["pr_chirps"].values[:, px]]).astype(np.float32)
    mi_list = _movs_idx(movs, nomes)
    if shift:
        mov = np.column_stack([deslocar_anos(mov[:, c], shift)
                               for c in range(mov.shape[1])])
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
                              n_proc=CPUS_FATIA, ao_terminar=progresso,
                              n_boot=n_boot or None)

    np.savez_compressed(arquivo, alpha=alpha, r2=r2, px=px,
                        movs=np.array([nomes[i] for i in mi_list]))
    volume.commit()
    print(f"[{bacia}] s{si} l{lag} d{shift} ✅ {time.time()-t0:.0f}s", flush=True)
    return str(arquivo)


def _montar_e_gravar(nome, bacias, pasta, lags, shifts, n_regimes, tol, iters,
                     janela, anomalias, n_boot=0, aceitar_incompleto=False):
    """Monta uma bacia ou região com `src.pipeline.montagem` e grava no volume."""
    from pathlib import Path

    import xarray as xr

    from src.pipeline.montagem import montar

    volume.reload()
    tag = _tag(janela, anomalias, n_boot)
    ds0 = xr.open_dataset(f"{DATA}/{janela}")
    dirs = {b: f"{DATA}/fatias/{tag}/{b}" for b in bacias}
    out = montar(nome, dirs, ds0, lags, shifts, n_regimes=n_regimes, tol=tol,
                 iters=iters, janela=janela, anomalias=anomalias,
                 log=lambda m: print(m, flush=True), n_boot=n_boot,
                 aceitar_incompleto=aceitar_incompleto)
    destino = Path(f"{DATA}/results/{tag}/{pasta}")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.nc"
    out.to_netcdf(caminho, encoding={v: {"zlib": True, "complevel": 4}
                                     for v in ("alpha", "r2")})
    volume.commit()
    print(f"[{nome}] ✅ {caminho}", flush=True)
    return str(caminho)


@app.function(cpu=8, memory=32768, timeout=7200, volumes={DATA: volume})
def montar_bacia(bacia: str, lags: list, shifts: list, n_regimes: int,
                 tol: float, iters: int, janela: str = JANELA_PADRAO,
                 anomalias: bool = False, n_boot: int = 0,
                 aceitar_incompleto: bool = False):
    """Junta as fatias da bacia, escolhe por predominância, roda o SOM e grava."""
    return _montar_e_gravar(bacia, [bacia], "bacias", lags, shifts, n_regimes, tol,
                            iters, janela, anomalias, n_boot, aceitar_incompleto)


# Memória e tempo folgados: o Brasil inteiro são 70 mil pixels — o cubo de α do
# observado sozinho tem 456 MB, e os 16 SOMs treinam sobre dezenas de milhares.
@app.function(cpu=8, memory=65536, timeout=6 * 3600, volumes={DATA: volume})
def montar_regiao(nome: str, bacias: list, lags: list, shifts: list, n_regimes: int,
                  tol: float, iters: int, janela: str = JANELA_PADRAO,
                  anomalias: bool = False):
    """Várias bacias como uma unidade só: refaz a escolha (com surrogates) e o SOM
    sobre a união. Não recalcula PREDEP — lê as fatias das bacias."""
    return _montar_e_gravar(nome, bacias, "regioes", lags, shifts, n_regimes, tol,
                            iters, janela, anomalias)


# Só o SOM: lê os <bacia>.nc prontos. Memória folgada para o Brasil todo (o cubo de
# α do observado tem ~456 MB).
# 4 CPUs: uma estação por processo. Memória para os cubos de α e r² do Brasil.
@app.function(cpu=4, memory=49152, timeout=4 * 3600, volumes={DATA: volume})
def som_regiao(nome: str, bacias: list, tag: str, n_regimes: int, tol: float,
               iters: int, metrica: str = "alpha"):
    """SOM de uma região a partir dos `<bacia>.nc` já calculados.

    Não lê fatias, não recalcula PREDEP nem surrogates: usa o cubo de α do observado
    que cada arquivo de bacia já guarda.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    from src.pipeline.montagem import som_de_bacias
    from src.visualization.bacia_plots import plot_bacia_som

    volume.reload()
    arquivos = [Path(f"{DATA}/results/{tag}/bacias/{b}.nc") for b in bacias]
    faltam = [b for b, a in zip(bacias, arquivos) if not a.exists()]
    if faltam:
        raise FileNotFoundError("bacias ainda não calculadas: " + ", ".join(faltam))
    out = som_de_bacias(nome, arquivos, n_regimes=n_regimes, tol=tol, iters=iters,
                        log=lambda m: print(m, flush=True), metrica=metrica, n_proc=4)
    if metrica != "alpha":
        nome = f"{nome}_{metrica}"
    destino = Path(f"{DATA}/results/{tag}/som")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.nc"
    out.to_netcdf(caminho)
    fig = plot_bacia_som(out, f"{DATA}/results/{tag}/figuras_som", filename=f"{nome}_som.png")
    volume.commit()
    return [str(caminho), str(fig)]


@app.function(cpu=4, memory=32768, timeout=3600, volumes={DATA: volume})
def figuras_bacia(bacia: str, tag: str, pasta: str = "bacias"):
    """As duas figuras da bacia, geradas no Modal.

    Ficam aqui e não na máquina do usuário porque o ambiente local tem numpy
    2.x com scipy antigo — matplotlib e xarray quebram lá. O container tem a
    pilha certa, e o `main.py` só baixa os PNGs.
    """
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    from src.visualization.bacia_plots import extrair_som, gerar_figuras_bacia

    volume.reload()
    nc = f"{DATA}/results/{tag}/{pasta}/{bacia}.nc"
    if not Path(nc).exists():
        raise FileNotFoundError(f"{nc} não existe — rode o cálculo primeiro")
    destino = (f"{DATA}/results/{tag}/figuras" if pasta == "bacias"
               else f"{DATA}/results/{tag}/figuras_{pasta}")
    saidas = gerar_figuras_bacia(nc, destino)
    # o SOM da bacia num .nc próprio: é o que o main.py baixa para artifacts/<>/som/
    try:
        saidas.append(extrair_som(nc, f"{DATA}/results/{tag}/som_{pasta}/{bacia}.nc"))
    except (KeyError, ValueError, OSError) as e:
        print(f"  ⚠ SOM da bacia {bacia}: {type(e).__name__}: {e}", flush=True)
    volume.commit()
    return [str(s) for s in saidas]


@app.function(cpu=CPUS_FATIA, memory=65536, timeout=2 * 3600, volumes={DATA: volume})
def mapa_nboot(bacia: str, si: int, lag: int, n_boot: int, seed: int,
               janela: str = JANELA_PADRAO, anomalias: bool = False):
    """α de todos os MoVs e pixels de uma (estação, lag), com `n_boot` e semente dados.

    Serve ao diagnóstico do ruído: o mesmo cálculo com duas sementes mostra quanto
    da identidade do MoV vencedor em cada pixel é sorteio, e não dado. `n_boot=0`
    usa o padrão do Algoritmo 1 (⌈n ln n⌉), que é o da pipeline.
    """
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import xarray as xr

    from src.analysis.grid_predep import fatia_alfa_r2

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    volume.reload()
    tag = _tag(janela, anomalias)
    destino = Path(f"{DATA}/results/{tag}/nboot")
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"{bacia}_s{si}_l{lag}_nb{n_boot}_sem{seed}.npz"
    if arquivo.exists():
        return str(arquivo)

    px = _pixels(bacia, janela)
    with xr.open_dataset(f"{DATA}/{janela}") as ds:
        nomes = [str(m) for m in ds.mov_name.values]
        meses = pd.DatetimeIndex(ds.time.values).month.values
        mov = ds["mov"].values.astype(float)
        pr = np.stack([ds["pr_xavier"].values[:, px],
                       ds["pr_chirps"].values[:, px]]).astype(np.float32)
    if anomalias:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for m in range(1, 13):
                sel = meses == m
                if sel.any():
                    pr[:, sel, :] -= np.nanmean(pr[:, sel, :], axis=1, keepdims=True)

    alpha, _ = fatia_alfa_r2(pr, mov, meses, si, lag, range(len(nomes)), seed,
                             n_proc=CPUS_FATIA, n_boot=n_boot or None)
    np.savez_compressed(arquivo, alpha=alpha, px=px, movs=np.array(nomes),
                        n_boot=n_boot, seed=seed)
    volume.commit()
    print(f"[{bacia}] s{si} l{lag} n_boot={n_boot or 'padrão'} semente={seed} ✅",
          flush=True)
    return str(arquivo)


@app.function(cpu=4, memory=49152, timeout=4 * 3600, volumes={DATA: volume})
def som_nboot(bacias: list, nome: str, tag: str, n_boot: int, seed: int, si_lista: list,
              lag: int = 0, n_regimes: int = 4, iters: int = 50, tol_rel: float = 1 / 3):
    """SOM da região sobre os α do diagnóstico de `n_boot` — mesma receita de sempre.

    As features são o α dos 29 MoVs na defasagem `lag` (o diagnóstico só calculou
    essa), e o par é escolhido pela predominância com concordância restrita a ela.
    A tolerância da máscara é RELATIVA (`tol_rel` do α mediano no par escolhido):
    com n_boot maior o α sobe de nível, e uma tolerância fixa apertaria a máscara
    de um lado só, estragando a comparação entre as duas configurações.
    """
    from pathlib import Path

    import matplotlib
    import numpy as np
    import xarray as xr
    matplotlib.use("Agg")

    from src.analysis.som_regimes import regimes_por_estacao
    from src.visualization.bacia_plots import plot_bacia_som

    volume.reload()
    base = Path(f"{DATA}/results/{tag}/nboot")
    alphas, pxs, lats, lons = [], [], [], []
    for b in bacias:
        with xr.open_dataset(f"{DATA}/results/{tag}/bacias/{b}.nc") as d:
            lats.append(d.latitude.values)
            lons.append(d.longitude.values)
        por_estacao = []
        for si in si_lista:
            with np.load(base / f"{b}_s{si}_l{lag}_nb{n_boot}_sem{seed}.npz",
                         allow_pickle=True) as z:
                por_estacao.append(z["alpha"])
                px, movs = z["px"], [str(m) for m in z["movs"]]
        alphas.append(np.stack(por_estacao, axis=1))      # (fonte, estação, MoV, px)
        pxs.append(px)

    px = np.concatenate(pxs)
    ordem = np.argsort(px, kind="stable")
    px = px[ordem]
    features = np.concatenate(alphas, axis=-1)[..., ordem]
    lat = np.concatenate(lats)[ordem]
    lon = np.concatenate(lons)[ordem]

    n_se = len(si_lista)
    conc = np.nanmean(np.minimum(features[0], features[1]), axis=-1)   # (estação, MoV)
    sel_mov = np.nanargmax(np.where(np.isfinite(conc), conc, -np.inf), axis=1)
    alpha_sel = np.stack([[features[s, i, sel_mov[i]] for i in range(n_se)]
                          for s in range(2)])
    tol = tol_rel * float(np.nanmedian(alpha_sel))
    print(f"[{nome}] n_boot={n_boot or 'padrão'} · escolha: "
          + " · ".join(f"{SEASONS[si_lista[i]]} {movs[sel_mov[i]]}" for i in range(n_se))
          + f" · tolerância {tol:.3f}", flush=True)

    som = regimes_por_estacao(features, alpha_sel, n_regimes=n_regimes, tol=tol,
                              iters=iters)
    out = xr.Dataset(
        {"regime": (("source", "season", "pixel"), som["regime"]),
         "erro_bmu": (("source", "season", "pixel"), som["erro_bmu"]),
         "assinatura": (("source", "season", "regime_id", "mov_name"), som["assinatura"]),
         "mascara": (("season", "pixel"), som["mascara"]),
         "ari_fontes": (("season",), som["ari_fontes"]),
         "ari_teto": (("season", "par_sementes"), som["ari_teto"]),
         "sel_mov": (("season",), sel_mov.astype(np.int16)),
         "alpha_escolhido": (("source", "season", "pixel"), alpha_sel.astype(np.float32))},
        coords={"source": ["xavier", "chirps"],
                "season": [SEASONS[i] for i in si_lista],
                "mov_name": movs, "regime_id": list(range(n_regimes)), "pixel": px,
                "latitude": ("pixel", lat), "longitude": ("pixel", lon)})
    out.attrs.update({
        "bacia": nome, "bacias": ", ".join(bacias), "tipo": "som_nboot",
        "n_boot": n_boot or "padrão (ceil(n ln n) = 627)", "semente": seed, "lag": lag,
        "mascara": f"|α_chirps − α_xavier| < {tol:.4f} ({tol_rel:.0%} do α mediano "
                   "no par escolhido)",
        "criterio": f"um MoV por estação na defasagem {lag} — argmax da média "
                    "espacial de min(α_xavier, α_chirps)",
        "origem": "α do diagnóstico de n_boot; sem surrogates, sem calibração nula",
    })
    destino = Path(f"{DATA}/results/{tag}/som")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}_nb{n_boot}.nc"
    out.to_netcdf(caminho)
    fig = plot_bacia_som(out, f"{DATA}/results/{tag}/figuras_som",
                         filename=f"{nome}_nb{n_boot}_som.png")
    volume.commit()
    ari = [round(float(v), 3) for v in som["ari_fontes"]]
    teto = [round(float(np.nanmean(v)), 3) for v in som["ari_teto"]]
    print(f"[{nome}] n_boot={n_boot} · ARI {ari} · teto {teto}", flush=True)
    return {"nc": str(caminho), "fig": str(fig), "ari": ari, "teto": teto}


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


# ── Busca do SOM concordante, calibrada — sempre sobre o Brasil inteiro ────────
# A unidade do SOM é o país numa passada só: por bacia isolada a partição fica
# distorcida. As fatias continuam por bacia (é onde está o custo do PREDEP); a
# escolha do par, a máscara e o SOM são refeitos sobre a união.

TODAS_BACIAS = ["amazonica", "atlantico_leste", "atlantico_nordeste",
                "atlantico_nordeste_oriental", "atlantico_sudeste", "atlantico_sul",
                "paraguai", "parana", "paranaiba", "sao_francisco", "tocantins",
                "uruguai"]


def _pasta_busca(tag, nome):
    return f"{DATA}/results/{tag}/som_busca/{nome}"


@app.function(cpu=1, memory=1024, timeout=600, volumes={DATA: volume})
def fatias_ausentes(tarefas: list, tag: str):
    """Só as tarefas de `calcular_fatia` cuja fatia ainda não existe no volume.

    Despachar uma fatia pronta abre um container inteiro só para ela descobrir que
    já existe; com centenas de fatias prontas, isso é custo à toa.
    """
    from pathlib import Path

    volume.reload()
    return [t for t in tarefas
            if not Path(f"{DATA}/fatias/{tag}/{t[0]}/s{t[1]}_l{t[2]}_d{t[3]}.npz").exists()]


@app.function(cpu=2, memory=32768, timeout=3600, volumes={DATA: volume})
def escolha_observada(bacias: list, tag: str):
    """O par (lag, MoV) do observado em cada estação, escolhido sobre a união.

    Devolve listas de inteiros — o entrypoint local não tem numpy.
    """
    from src.pipeline.montagem import abrir_bacias, entradas_busca

    volume.reload()
    u = abrir_bacias([f"{DATA}/results/{tag}/bacias/{b}.nc" for b in bacias])
    _, e = next(entradas_busca(u, {}, [0], log=lambda m: print(m, flush=True)))
    return {"sel_lag": [int(v) for v in e["sel_lag"]],
            "lags": [int(v) for v in u["lags"]],
            "sel_mov": [int(v) for v in e["sel_mov"]]}


@app.function(cpu=4, memory=65536, timeout=4 * 3600, volumes={DATA: volume})
def preparar_busca(nome: str, bacias: list, tag: str, shifts: list,
                   par_fixo: bool = False):
    """Escolha da região e features do SOM de cada deslocamento, gravadas no volume.

    Uma vez só: as 28 buscas (estação × deslocamento) leem daqui, em vez de cada
    uma remontar o cubo de 70 mil pixels.
    """
    from pathlib import Path

    import numpy as np

    from src.pipeline.montagem import abrir_bacias, entradas_busca

    volume.reload()
    pasta = Path(_pasta_busca(tag, nome))
    pasta.mkdir(parents=True, exist_ok=True)
    u = abrir_bacias([f"{DATA}/results/{tag}/bacias/{b}.nc" for b in bacias])
    np.savez(pasta / "comum.npz", px=u["px"], lat=u["lat"], lon=u["lon"],
             origem=u["origem"], movs=np.array(u["movs"]), lags=np.array(u["lags"]),
             bacias=np.array(u["bacias"]),
             attrs=np.array([f"{k}={v}" for k, v in u["attrs"].items()
                             if k in ("janela", "periodo", "precipitacao")]))
    dirs = {b: f"{DATA}/fatias/{tag}/{b}" for b in bacias}
    log = lambda m: print(m, flush=True)  # noqa: E731
    # Incremental: a parada antecipada chama isto um deslocamento por vez.
    if not (pasta / "entrada_d0.npz").exists():
        for _, e in entradas_busca(u, dirs, [0], log=log):
            np.savez(pasta / "entrada_d0.npz", **e)
    fixo = None
    if par_fixo:
        with np.load(pasta / "entrada_d0.npz") as z:
            fixo = (z["sel_lag"], z["sel_mov"])
    pendentes = [s for s in shifts
                 if s != 0 and not (pasta / f"entrada_d{s}.npz").exists()]
    for sh, e in entradas_busca(u, dirs, pendentes, par_fixo=fixo, log=log):
        np.savez(pasta / f"entrada_d{sh}.npz", **e)
    volume.commit()
    return str(pasta)


# A busca do Brasil treina 12 SOMs (2 grades × 3 sementes × 2 fontes): 16 CPUs os
# rodam numa leva só sem pagar núcleo ocioso.
CPUS_BUSCA = 16


@app.function(cpu=CPUS_BUSCA, memory=65536, timeout=6 * 3600, volumes={DATA: volume})
def busca_som_fatia(nome: str, si: int, shift: int, tag: str, iters: int, tol: float):
    """A busca do SOM (grade × K) para uma (estação, deslocamento) da unidade."""
    from pathlib import Path

    import numpy as np

    from src.analysis.som_busca import KS_BUSCA, assinaturas, buscar

    volume.reload()
    pasta = Path(_pasta_busca(tag, nome))
    parte = pasta / f"partes/s{si}_d{shift}.npz"
    if parte.exists():
        with np.load(parte) as z:
            ari = z["ari"]
        return {"si": si, "shift": shift, "reaproveitada": True,
                "ari_max": float(np.nanmax(ari)) if np.isfinite(ari).any()
                else float("nan")}

    with np.load(pasta / f"entrada_d{shift}.npz") as z:
        feat = z["features"][:, si]                  # (fonte, MoV, pixel)
        alpha_sel = z["alpha_sel"][:, si]            # (fonte, pixel)
        lag, mov = int(z["sel_lag"][si]), int(z["sel_mov"][si])

    print(f"[{nome}] s{si} d{shift} · {feat.shape[-1]:,} pixels · buscando...", flush=True)
    r = buscar(feat, alpha_sel, tol=tol, iters=iters, n_proc=CPUS_BUSCA)
    extra = {}
    if shift == 0 and r["melhor"][0] >= 0:
        extra["assinatura"] = assinaturas(feat, r["regime"], max(KS_BUSCA))
    parte.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(parte, ari=r["ari"], teto=r["teto"],
                        grades=np.array(r["grades"]), ks=np.array(r["ks"]),
                        melhor=np.array(r["melhor"]), regime=r["regime"],
                        erro_bmu=r["erro_bmu"], mascara=r["mascara"],
                        lag=lag, mov=mov, **extra)
    volume.commit()
    ari_max = float(np.nanmax(r["ari"])) if np.isfinite(r["ari"]).any() else float("nan")
    print(f"[{nome}] s{si} d{shift} ✅ melhor ARI {ari_max:+.3f}", flush=True)
    return {"si": si, "shift": shift, "ari_max": ari_max, "reaproveitada": False}


@app.function(cpu=2, memory=32768, timeout=3600, volumes={DATA: volume})
def gravar_busca(nome: str, tag: str, shifts: list, par_fixo: bool = False):
    """Junta as partes numa `.nc` e desenha a figura do SOM vencedor."""
    from pathlib import Path

    import matplotlib
    import numpy as np
    import xarray as xr
    matplotlib.use("Agg")

    from src.analysis.som_busca import KS_BUSCA, veredito
    from src.visualization.bacia_plots import plot_bacia_som

    volume.reload()
    pasta = Path(_pasta_busca(tag, nome))
    with np.load(pasta / "comum.npz") as z:
        comum = {k: z[k] for k in z.files}
    # Com parada antecipada, uma estação decidida não tem as partes dos
    # deslocamentos seguintes: fica NaN, e o veredito usa os que rodaram.
    partes = {}
    for si in range(len(SEASONS)):
        for sh in shifts:
            f = pasta / f"partes/s{si}_d{sh}.npz"
            if f.exists():
                with np.load(f) as z:
                    partes[si, sh] = {k: z[k] for k in z.files}
    movs = [str(m) for m in comum["movs"]]
    lags = [int(v) for v in comum["lags"]]

    ks = list(KS_BUSCA)
    k_max = max(ks)
    n_g = max(len(p["grades"]) for p in partes.values())
    n_se, n_sh, n_px = len(SEASONS), len(shifts), len(comum["px"])
    ari = np.full((n_sh, n_se, n_g, len(ks)), np.nan, dtype=np.float32)
    teto = np.full_like(ari, np.nan)
    grade = np.zeros((n_sh, n_se, n_g), dtype=np.int16)
    sel_lag = np.zeros((n_sh, n_se), dtype=np.int16)
    sel_mov = np.zeros((n_sh, n_se), dtype=np.int16)
    for (si, sh), p in partes.items():
        di, g = shifts.index(sh), len(p["grades"])
        ari[di, si, :g], teto[di, si, :g] = p["ari"], p["teto"]
        grade[di, si, :g] = p["grades"]
        sel_lag[di, si], sel_mov[di, si] = int(p["lag"]), int(p["mov"])

    regime = np.full((2, n_se, n_px), -1, dtype=np.int16)
    erro = np.full((2, n_se, n_px), np.nan, dtype=np.float32)
    assin = np.full((2, n_se, k_max, len(movs)), np.nan, dtype=np.float32)
    mascara = np.zeros((n_se, n_px), dtype=bool)
    ari_fontes, teto_melhor, surr_max = (np.full(n_se, np.nan, dtype=np.float32)
                                         for _ in range(3))
    k_esc, g_esc = np.zeros(n_se, dtype=np.int16), np.zeros(n_se, dtype=np.int16)
    supera = np.zeros(n_se, dtype=bool)
    n_surr = np.zeros(n_se, dtype=np.int16)
    linhas = []
    for si in range(n_se):
        p = partes[si, 0]
        mascara[si] = p["mascara"]
        feitos = [sh for sh in shifts if sh != 0 and (si, sh) in partes]
        v = veredito(p["ari"], [partes[si, sh]["ari"] for sh in feitos])
        surr_max[si], supera[si] = v["surr_max"], v["supera"]
        gi, ki = (int(x) for x in p["melhor"])
        if gi >= 0:
            regime[:, si], erro[:, si] = p["regime"], p["erro_bmu"]
            assin[:, si] = p["assinatura"]
            ari_fontes[si], teto_melhor[si] = p["ari"][gi, ki], p["teto"][gi, ki]
            k_esc[si], g_esc[si] = ks[ki], p["grades"][gi]
        ranking = sorted([v["obs"], *v["surr"]], reverse=True).index(v["obs"]) + 1
        linhas.append(f"  {SEASONS[si]} {movs[sel_mov[0, si]]}/lag{lags[sel_lag[0, si]]}: "
                      f"observado {v['obs']:+.3f} (grade {g_esc[si]}, K={k_esc[si]}, "
                      f"teto {teto_melhor[si]:+.3f}) · melhor surrogate "
                      f"{v['surr_max']:+.3f} · posição {ranking}/{1 + len(v['surr'])} "
                      f"({len(feitos)} surrogate(s)) "
                      f"→ {'SUPERA' if v['supera'] else 'não supera'}")
        n_surr[si] = len(feitos)

    attrs = dict(a.split("=", 1) for a in comum["attrs"])
    out = xr.Dataset(
        {"regime": (("source", "season", "pixel"), regime),
         "erro_bmu": (("source", "season", "pixel"), erro),
         "assinatura": (("source", "season", "regime_id", "mov_name"), assin),
         "mascara": (("season", "pixel"), mascara),
         "ari_fontes": (("season",), ari_fontes),
         "ari_teto": (("season",), teto_melhor),
         "k_escolhido": (("season",), k_esc),
         "grade_escolhida": (("season",), g_esc),
         "ari_surrogate_max": (("season",), surr_max),
         "supera_nulo": (("season",), supera),
         "n_surrogates": (("season",), n_surr),
         "sel_lag": (("shift", "season"), sel_lag),
         "sel_mov": (("shift", "season"), sel_mov),
         "ari_busca": (("shift", "season", "grade_idx", "k"), ari),
         "teto_busca": (("shift", "season", "grade_idx", "k"), teto),
         "grade": (("shift", "season", "grade_idx"), grade),
         "bacia_do_pixel": (("pixel",), comum["origem"])},
        coords={"source": ["xavier", "chirps"], "season": SEASONS, "shift": shifts,
                "k": ks, "lag": lags, "regime_id": list(range(k_max)),
                "mov_name": movs, "bacia": [str(b) for b in comum["bacias"]],
                "pixel": comum["px"], "latitude": ("pixel", comum["lat"]),
                "longitude": ("pixel", comum["lon"])})
    out.attrs.update({
        **attrs, "bacia": nome, "tipo": "som_busca",
        "unidade": "as bacias como UMA unidade — escolha do par, máscara e SOM "
                   "sobre a união",
        "busca": "grade × K, a mesma nas duas fontes; sementes consolidadas pela "
                 "partição medoide de cada fonte; vencedor = maior ARI entre fontes",
        "nulo": ("surrogates com o PAR DO OBSERVADO (não refazem a escolha)" if par_fixo
                 else "cada surrogate refaz a escolha do par sobre a unidade") +
                "; a MESMA busca em cada um, com a sua máscara; supera_nulo = melhor "
                "ARI observado > melhor ARI de todos os surrogates (acaso 1/7)",
    })
    destino = Path(f"{DATA}/results/{tag}/som_busca/{nome}.nc")
    out.to_netcdf(destino)
    fig = plot_bacia_som(out, f"{DATA}/results/{tag}/figuras_som_busca",
                         filename=f"{nome}_som_busca.png")
    volume.commit()
    return {"nc": str(destino), "fig": str(fig), "linhas": linhas,
            "supera": int(supera.sum())}


@app.local_entrypoint()
def main(bacias: str = "all", movs: str = "all", lags: str = "0,1,2,3,4,5,6",
         shifts: str = "0,3,5,7,9,11,13", n_regimes: int = 4, tol: float = 0.10,
         iters: int = 50, seed: int = 20260908, so_figuras: bool = False,
         janela: str = JANELA_PADRAO, anomalias: bool = False, juntar: bool = False,
         regiao: str = "", som: bool = False, busca_som: bool = False,
         busca_par_fixo: bool = False, busca_todos: bool = False,
         metrica: str = "alpha", diagnostico_nboot: bool = False,
         som_nboot_flag: bool = False, nome: str = "",
         nboots: str = "0,10000", sementes: str = "20260908,777", lag: int = 0,
         estacao: int = 0, n_boot: int = 0, usar_fatias_existentes: bool = False):
    """Espalha as fatias, monta cada bacia e desenha as figuras — ou só junta."""
    tag = _tag(janela, anomalias, n_boot)
    print(f"janela {janela} · precipitação: "
          f"{'anomalias' if anomalias else 'totais brutos'} · "
          f"n_boot {n_boot or 'padrão (⌈n ln n⌉)'} · namespace {tag}\n")
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

    if som_nboot_flag:
        rotulo = nome or "_".join(alvo)
        resultados = []
        for nb in nboots.split(","):
            r = som_nboot.remote(alvo, rotulo, tag, int(nb), int(sementes.split(",")[0]),
                                 list(range(len(SEASONS))), lag, n_regimes, iters)
            resultados.append((nb, r))
            print(f"\n✅ {r['nc']}\n✅ {r['fig']}", flush=True)
        print("\nARI entre fontes por configuração:")
        for nb, r in resultados:
            print(f"  n_boot {nb or 'padrão':>6}: ARI {r['ari']} · teto {r['teto']}")
        return

    if diagnostico_nboot:
        combos = [(b, estacao, lag, int(nb), int(sm), janela, anomalias)
                  for b in alvo for nb in nboots.split(",")
                  for sm in sementes.split(",")]
        print(f"diagnóstico do ruído: {len(combos)} mapas de α "
              f"(estação {SEASONS[estacao]}, lag {lag}, n_boot {nboots}, "
              f"sementes {sementes})\n", flush=True)
        for caminho in mapa_nboot.starmap(combos):
            print(f"  ✅ {caminho}", flush=True)
        print(f"\n  modal volume get predep-common-window /results/{tag}/nboot "
              f"results/nboot/{tag}")
        return

    if busca_som:
        # Sempre o Brasil inteiro: por bacia isolada a partição fica distorcida.
        if sorted(alvo) != sorted(TODAS_BACIAS):
            raise SystemExit("--busca-som roda sempre sobre o Brasil inteiro "
                             "(--bacias all)")
        # Pastas separadas por modo: as partes dos surrogates têm os mesmos nomes
        # nos dois, e uma corrida reaproveitaria em silêncio as da outra.
        nome = "brasil_par_fixo" if busca_par_fixo else "brasil"
        surr = [sh for sh in shift_list if sh != 0]
        if busca_par_fixo:
            esc = escolha_observada.remote(alvo, tag)
            lags_por_estacao = [esc["lags"][li] for li in esc["sel_lag"]]
            fatias = [(b, si, lags_por_estacao[si], sh, movs, seed, janela, anomalias)
                      for b in alvo for si in range(len(SEASONS)) for sh in surr]
            modo = "surrogates com o par do observado"
        else:
            fatias = [(b, si, lag, sh, movs, seed, janela, anomalias)
                      for b in alvo for si in range(len(SEASONS))
                      for lag in lag_list for sh in surr]
            modo = "cada surrogate refaz a escolha do par sobre o Brasil"
        print(f"busca do SOM concordante — Brasil numa passada ({modo})")
        print("parada antecipada: " + ("desligada — todos os surrogates" if busca_todos
              else "cada estação para no 1º surrogate que empatar ou superar o "
                   "observado"), flush=True)

        def por_deslocamento(sh):
            return [f for f in fatias if f[3] == sh]

        print("1/3  observado: escolha, features e busca", flush=True)
        preparar_busca.remote(nome, alvo, tag, [0], busca_par_fixo)
        obs = {}
        for r in busca_som_fatia.starmap([(nome, si, 0, tag, iters, tol)
                                          for si in range(len(SEASONS))]):
            obs[r["si"]] = r["ari_max"]
            print(f"  observado {SEASONS[r['si']]}: melhor ARI {r['ari_max']:+.3f}",
                  flush=True)
        decidida = {si: False for si in range(len(SEASONS))}
        for sh in surr:
            pend = [si for si in range(len(SEASONS)) if busca_todos or not decidida[si]]
            if not pend:
                print(f"2/3  todas as estações decididas — {sh}a e seguintes não rodam",
                      flush=True)
                break
            faltam = fatias_ausentes.remote(por_deslocamento(sh), tag)
            print(f"2/3  surrogate {sh}a: {len(faltam)} fatia(s) a calcular · estações "
                  + ", ".join(SEASONS[si] for si in pend), flush=True)
            if faltam:
                for _ in calcular_fatia.starmap(faltam):
                    pass
            preparar_busca.remote(nome, alvo, tag, [sh], busca_par_fixo)
            for r in busca_som_fatia.starmap([(nome, si, sh, tag, iters, tol)
                                              for si in pend]):
                si = r["si"]
                if r["ari_max"] >= obs[si]:
                    decidida[si] = True
                print(f"  surrogate {sh}a {SEASONS[si]}: {r['ari_max']:+.3f} vs observado "
                      f"{obs[si]:+.3f}" + (" → decidida (não supera)"
                                           if decidida[si] else ""), flush=True)
        print("3/3  veredito\n", flush=True)
        r = gravar_busca.remote(nome, tag, shift_list, busca_par_fixo)
        print(f"brasil: supera o nulo em {r['supera']}/{len(SEASONS)} estações")
        for linha in r["linhas"]:
            print(linha)
        print(f"\n✅ {r['nc']}\n✅ {r['fig']}")
        return

    if som or regiao:
        import re
        # MESMA regra de `nome_unidade` em main.py, que baixa os arquivos por ela
        nome = regiao or ("brasil" if sorted(alvo) == sorted(todas) else "_".join(alvo))
        if not re.fullmatch(r"[a-z0-9_]+", nome):
            raise SystemExit(f"nome inválido: {nome!r} (use a-z, 0-9 e _)")
        if som:
            print(f"SOM '{nome}': {len(alvo)} bacia(s) — {', '.join(alvo)}")
            print("lê os <bacia>.nc já calculados — sem fatias, sem PREDEP, sem surrogates\n")
            nc, fig = som_regiao.remote(nome, alvo, tag, n_regimes, tol, iters, metrica)
            saida = nome if metrica == "alpha" else f"{nome}_{metrica}"
            print(f"\n✅ {nc}\n✅ {fig}\n\n  modal volume get predep-common-window "
                  f"/results/{tag}/som/{saida}.nc results/som/{tag}/{saida}.nc")
            return
        print(f"região {nome}: {len(alvo)} bacia(s) — {', '.join(alvo)}")
        print("nenhum PREDEP é recalculado; as fatias dessas bacias precisam existir\n")
        caminho = montar_regiao.remote(nome, alvo, lag_list, shift_list, n_regimes,
                                       tol, iters, janela, anomalias)
        print("\ndesenhando as figuras...\n")
        for f in figuras_bacia.remote(nome, tag, "regioes"):
            print(f"  {f}")
        print(f"\n✅ {caminho}\n\n  modal volume get predep-common-window "
              f"/results/{tag}/regioes/{nome}.nc results/regioes/{tag}/{nome}.nc")
        return

    if usar_fatias_existentes:
        # Não calcula nada: monta cada bacia com as fatias que já estão no volume.
        # Combinações (estação, lag) com fatia ausente saem da escolha de TODAS as
        # séries — ver `montar(aceitar_incompleto=True)`.
        print("montando só com as fatias existentes (nenhuma fatia é calculada)\n")
        list(montar_bacia.starmap([(b, lag_list, shift_list, n_regimes, tol, iters,
                                    janela, anomalias, n_boot, True)
                                   for b in alvo]))
    elif not so_figuras:
        tarefas = [(b, si, lag, sh, movs, seed, janela, anomalias, n_boot)
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
                                    janela, anomalias, n_boot)
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
