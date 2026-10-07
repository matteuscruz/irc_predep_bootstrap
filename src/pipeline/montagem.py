"""
Montagem de uma unidade de análise — uma bacia ou uma região de várias bacias — a
partir das fatias de α e r² já calculadas.

Uma região NÃO é a junção dos resultados por bacia (isso é `juntar.py`). Cada bacia
escolhe o seu (lag, MoV), e o vetor de atributos do SOM é o α dos 29 MoVs no lag
ESCOLHIDO: com lags diferentes, pixels de bacias diferentes não estariam no mesmo
espaço de atributos. Por isso a região refaz, sobre a união dos pixels, a escolha por
predominância (no observado e em cada surrogate), a máscara de concordância e o SOM.

As fatias — onde está todo o custo do PREDEP — são as mesmas da bacia. Montar uma
região não recalcula PREDEP nenhum.

Uma bacia é uma região de uma bacia só: o mesmo código monta as duas, para não
divergirem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEASONS = ["DJF", "MAM", "JJA", "SON"]


def nome_fatia(si, lag, shift) -> str:
    return f"s{si}_l{lag}_d{shift}.npz"


def fatias_faltando(dirs: dict, lags, shifts) -> dict:
    """{bacia: nº de fatias ausentes}, só para as bacias incompletas."""
    total = len(SEASONS) * len(lags) * len(shifts)
    out = {}
    for b, d in dirs.items():
        n = sum((Path(d) / nome_fatia(si, lag, sh)).exists()
                for si in range(len(SEASONS)) for lag in lags for sh in shifts)
        if n < total:
            out[b] = total - n
    return out


def combinacoes_ausentes(dirs: dict, lags, shifts) -> list:
    """(estação, lag) em que falta a fatia de algum deslocamento em alguma bacia."""
    return sorted({(si, lag) for d in dirs.values() for si in range(len(SEASONS))
                   for lag in lags for sh in shifts
                   if not (Path(d) / nome_fatia(si, lag, sh)).exists()})


def montar(nome, dirs: dict, ds_janela, lags, shifts, *, n_regimes=4, tol=0.10,
           iters=50, janela="", anomalias=False, log=print, n_boot=0,
           aceitar_incompleto=False):
    """Monta o Dataset da unidade `nome` a partir das fatias de `dirs` ({bacia: pasta}).

    Com `aceitar_incompleto`, fatias ausentes não interrompem a montagem: cada
    combinação (estação, lag) em que falte QUALQUER deslocamento sai da escolha do
    par de TODAS as séries, observado inclusive. Sem isso, a série observada
    escolheria entre mais candidatos que os substitutos e o teste contra o nulo
    ficaria enviesado a favor dela. Os valores que existem continuam gravados; só
    não entram na seleção.

    Devolve o Dataset; quem chama decide onde gravar.
    """
    import xarray as xr

    from src.analysis.grid_predep import escolher_por_predominancia
    from src.analysis.som_regimes import regimes_por_estacao

    bacias = list(dirs)
    lags, shifts = list(lags), list(shifts)
    falta = fatias_faltando(dirs, lags, shifts)
    excluidas = []
    if falta and not aceitar_incompleto:
        total = len(SEASONS) * len(lags) * len(shifts)
        raise FileNotFoundError(
            "fatias ausentes — calcule estas bacias antes de montar: "
            + "; ".join(f"{b} (faltam {n} de {total})" for b, n in falta.items()))
    if falta:
        excluidas = combinacoes_ausentes(dirs, lags, shifts)
        log(f"[{nome}] montagem com fatias ausentes — fora da escolha de todas as "
            "séries: " + ", ".join(f"{SEASONS[si]} lag {lag}" for si, lag in excluidas))

    # pixels e MoVs a partir da primeira fatia de cada bacia; conferidos em todas
    ref = [np.load(Path(dirs[b]) / nome_fatia(0, lags[0], shifts[0]), allow_pickle=True)
           for b in bacias]
    px_bac = [r["px"] for r in ref]
    movs = [str(m) for m in ref[0]["movs"]]
    for b, r in zip(bacias, ref):
        if [str(m) for m in r["movs"]] != movs:
            raise ValueError(f"{b}: MoVs das fatias diferem de {bacias[0]}")
    px_cat = np.concatenate(px_bac)
    if len(np.unique(px_cat)) != len(px_cat):
        raise ValueError("há pixels em mais de uma bacia — região inválida")
    ordem = np.argsort(px_cat, kind="stable")
    px = px_cat[ordem]
    origem = np.concatenate([np.full(len(p), i, dtype=np.int16)
                             for i, p in enumerate(px_bac)])[ordem]

    n_se, n_sh, n_lag, n_mov, n_px = len(SEASONS), len(shifts), len(lags), len(movs), len(px)
    log(f"[{nome}] {len(bacias)} bacia(s) · {n_px:,} pixels · {n_mov} MoVs · "
        f"{n_lag} lags · {n_sh} deslocamentos")

    sel_lag = np.zeros((n_sh, n_se), dtype=np.int16)
    sel_mov = np.zeros((n_sh, n_se), dtype=np.int16)
    concord = np.full((n_sh, n_se, n_lag, n_mov), np.nan, dtype=np.float32)
    alpha_sel = np.full((2, n_sh, n_se, n_px), np.nan, dtype=np.float32)
    consenso = np.full((n_sh, n_se), np.nan, dtype=np.float32)
    alpha0 = r20 = None

    for di, sh in enumerate(shifts):
        forma = (2, n_se, n_lag, n_mov, n_px)
        cubo = np.full(forma, np.nan, dtype=np.float32)
        cubo_r2 = np.full(forma, np.nan, dtype=np.float32) if sh == 0 else None
        for si in range(n_se):
            for li, lag in enumerate(lags):
                a_l, r_l = [], []
                if any(not (Path(dirs[b]) / nome_fatia(si, lag, sh)).exists()
                       for b in bacias):
                    continue                     # fica NaN; e (si, lag) está em `excluidas`
                for b, p_ref in zip(bacias, px_bac):
                    z = np.load(Path(dirs[b]) / nome_fatia(si, lag, sh), allow_pickle=True)
                    if not np.array_equal(z["px"], p_ref):
                        raise ValueError(f"{b}: pixels da fatia s{si} l{lag} d{sh} "
                                         "diferem das demais fatias da bacia")
                    a_l.append(z["alpha"])
                    if cubo_r2 is not None:
                        r_l.append(z["r2"])
                cubo[:, si, li] = np.concatenate(a_l, axis=-1)[..., ordem]
                if cubo_r2 is not None:
                    cubo_r2[:, si, li] = np.concatenate(r_l, axis=-1)[..., ordem]

        if excluidas:
            escolha = cubo.copy()
            for si, lag in excluidas:
                escolha[:, si, lags.index(lag)] = np.nan
        else:
            escolha = cubo
        sl, sm, cc = escolher_por_predominancia(escolha)
        sel_lag[di], sel_mov[di], concord[di] = sl, sm, cc
        for si in range(n_se):
            alpha_sel[:, di, si] = cubo[:, si, sl[si], sm[si]]
        d = alpha_sel[1, di] - alpha_sel[0, di]
        consenso[di] = np.nanmean(np.abs(d) < tol, axis=-1)
        rotulo = "observado" if sh == 0 else f"surrogate {sh}a"
        log(f"[{nome}] {rotulo:<14} " + " · ".join(
            f"{SEASONS[si]} {movs[sm[si]]}/lag{lags[sl[si]]}" for si in range(n_se)))
        if sh == 0:
            alpha0, r20 = cubo, cubo_r2
        else:
            del cubo                      # o cubo de um surrogate não é guardado

    if alpha0 is None:
        raise ValueError("shifts precisa incluir 0 (o observado)")

    # SOM só no observado: o nulo entra pelo consenso
    features = np.stack([[alpha0[s, si, sel_lag[0, si]] for si in range(n_se)]
                         for s in range(2)])
    som = regimes_por_estacao(features, alpha_sel[:, 0], n_regimes=n_regimes,
                              tol=tol, iters=iters)

    lat = ds_janela.latitude.values[px]
    lon = ds_janela.longitude.values[px]
    dados = {
        "alpha": (("source", "season", "lag", "mov_name", "pixel"), alpha0),
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
        "ari_fontes": (("season",), som["ari_fontes"]),
    }
    coords = {"source": ["xavier", "chirps"], "shift": shifts, "season": SEASONS,
              "lag": lags, "mov_name": movs, "regime_id": list(range(n_regimes)),
              "pixel": px, "latitude": ("pixel", lat), "longitude": ("pixel", lon)}
    if len(bacias) > 1:
        dados["bacia_do_pixel"] = (("pixel",), origem)
        coords["bacia"] = bacias

    out = xr.Dataset(dados, coords=coords)
    out.attrs.update({
        "bacia": nome,
        "bacias": ", ".join(bacias),
        "tipo": "regiao" if len(bacias) > 1 else "bacia",
        "pixels": n_px,
        "criterio": "um (lag, MoV) por estação — argmax da média espacial de "
                    "min(α_xavier, α_chirps) sobre TODOS os pixels da unidade",
        "predep": "estimador do artigo (i==j permitidos), ward k=√N, "
                  + (f"n_boot = {n_boot}" if n_boot else "n_boot = ceil(n log n)"),
        "n_boot": int(n_boot),
        "lags_fora_da_escolha": (", ".join(f"{SEASONS[si]}:{lag}" for si, lag in excluidas)
                                 or "nenhum"),
        "mascara": f"|α_chirps − α_xavier| < {tol} no candidato escolhido",
        "som": f"treinado na máscara, projetado em todos os pixels, k={n_regimes}",
        "nulo": "shift != 0 são surrogates por deslocamento anual; cada um "
                "escolhe o SEU candidato pelo mesmo critério. α e consenso do "
                "observado devem ser lidos contra eles, nunca contra zero.",
        "alpha_completo": "só o observado — os surrogates guardam a escolha, o "
                          "α nela e o consenso",
        "executor": "modal",
        "janela": janela,
        "periodo": str(ds_janela.attrs.get("time_window", "")),
        "precipitacao": ("anomalias (climatologia mensal da janela removida)"
                         if anomalias else "totais mensais brutos"),
    })
    return out


# ── SOM a partir das bacias JÁ CALCULADAS ────────────────────────────────────

ATTRS_COMPATIVEIS = ["janela", "periodo", "precipitacao"]


def abrir_bacias(arquivos, com_r2=False):
    """Os `<bacia>.nc` prontos como UMA unidade, na ordem global de pixels.

    Confere que todas vêm da mesma corrida (janela, período, tratamento da chuva) e
    têm as mesmas coordenadas. Devolve um dicionário com o cubo de α do observado
    `(fonte, estação, lag, MoV, pixel)` e o que é preciso para reordenar fatias.
    """
    import xarray as xr

    arquivos = sorted(Path(a) for a in arquivos)
    dss = [xr.open_dataset(a) for a in arquivos]
    try:
        bacias = [str(d.attrs.get("bacia", a.stem)) for d, a in zip(dss, arquivos)]
        ref = dss[0]
        for d, b in zip(dss[1:], bacias[1:]):
            for k in ATTRS_COMPATIVEIS:
                if d.attrs.get(k) != ref.attrs.get(k):
                    raise ValueError(f"{b}: {k!r} = {d.attrs.get(k)!r}, mas {bacias[0]} "
                                     f"tem {ref.attrs.get(k)!r} — bacias de corridas "
                                     "diferentes")
            for c in ("source", "season", "lag", "mov_name"):
                if not np.array_equal(d[c].values, ref[c].values):
                    raise ValueError(f"{b}: coordenada {c!r} difere de {bacias[0]}")

        px_bac = [d.pixel.values for d in dss]
        px_cat = np.concatenate(px_bac)
        if len(np.unique(px_cat)) != len(px_cat):
            raise ValueError("há pixels em mais de uma bacia")
        ordem = np.argsort(px_cat, kind="stable")
        return {
            "bacias": bacias, "px_bac": px_bac, "ordem": ordem, "px": px_cat[ordem],
            "origem": np.concatenate([np.full(len(p), i, dtype=np.int16)
                                      for i, p in enumerate(px_bac)])[ordem],
            "alpha": np.concatenate([d["alpha"].values for d in dss], axis=-1)[..., ordem],
            "r2": (np.concatenate([d["r2"].values for d in dss], axis=-1)[..., ordem]
                   if com_r2 else None),
            "lat": np.concatenate([d.latitude.values for d in dss])[ordem],
            "lon": np.concatenate([d.longitude.values for d in dss])[ordem],
            "movs": [str(m) for m in ref.mov_name.values],
            "lags": [int(v) for v in ref.lag.values],
            "attrs": dict(ref.attrs),
        }
    finally:
        for d in dss:
            d.close()


def entradas_busca(unidade, dirs_fatias, shifts, *, par_fixo=None, log=print):
    """A escolha da região e as features do SOM, deslocamento a deslocamento.

    `unidade` é o que `abrir_bacias` devolve. O observado (`shift = 0`) sai do cubo
    que os `.nc` já guardam; cada surrogate lê as SUAS fatias (todos os lags e
    MoVs) e refaz a escolha por predominância sobre a região inteira — o mesmo
    procedimento do observado, que é o que torna o nulo comparável.

    `par_fixo=(sel_lag, sel_mov)` é o modo barato: os surrogates NÃO refazem a
    escolha e usam o par do observado, então só a fatia desse lag é lida. O nulo
    continua destruindo o pareamento índice–chuva, mas deixa de incluir a etapa de
    seleção do par.

    Gera `(shift, {"sel_lag", "sel_mov", "alpha_sel", "features"})`, com
    `alpha_sel` em `(fonte, estação, pixel)` e `features` em
    `(fonte, estação, MoV, pixel)`.
    """
    from src.analysis.grid_predep import escolher_por_predominancia

    bacias, px_bac, ordem = unidade["bacias"], unidade["px_bac"], unidade["ordem"]
    movs, lags = unidade["movs"], unidade["lags"]
    n_se, n_px = len(SEASONS), len(unidade["px"])
    for sh in shifts:
        if sh == 0:
            cubo = unidade["alpha"]
        else:
            cubo = np.full((2, n_se, len(lags), len(movs), n_px), np.nan, dtype=np.float32)
            for si in range(n_se):
                for li, lag in enumerate(lags):
                    if par_fixo is not None and li != par_fixo[0][si]:
                        continue
                    partes = []
                    for b, p_ref in zip(bacias, px_bac):
                        with np.load(Path(dirs_fatias[b]) / nome_fatia(si, lag, sh),
                                     allow_pickle=True) as z:
                            if [str(m) for m in z["movs"]] != movs:
                                raise ValueError(f"{b}: MoVs da fatia diferem do .nc")
                            o = np.argsort(z["px"])
                            if not np.array_equal(z["px"][o], p_ref):
                                raise ValueError(f"{b}: pixels da fatia s{si} l{lag} "
                                                 f"d{sh} diferem do .nc")
                            partes.append(z["alpha"][..., o])
                    cubo[:, si, li] = np.concatenate(partes, axis=-1)[..., ordem]
        if sh != 0 and par_fixo is not None:
            sl, sm = (np.asarray(v, dtype=int) for v in par_fixo)
        else:
            sl, sm, _ = escolher_por_predominancia(cubo)
        rotulo = "observado" if sh == 0 else f"surrogate {sh}a"
        log(f"  {rotulo:<14} " + " · ".join(
            f"{SEASONS[si]} {movs[sm[si]]}/lag{lags[sl[si]]}" for si in range(n_se)))
        yield sh, {
            "sel_lag": sl, "sel_mov": sm,
            "alpha_sel": np.stack([[cubo[s, si, sl[si], sm[si]] for si in range(n_se)]
                                   for s in range(2)]).astype(np.float32),
            "features": np.stack([[cubo[s, si, sl[si]] for si in range(n_se)]
                                  for s in range(2)]).astype(np.float32),
        }
        del cubo


def _regimes_uma_estacao(args):
    """`regimes_por_estacao` para uma estação só — função de módulo para o Pool."""
    from src.analysis.som_regimes import regimes_por_estacao

    features, sel, n_regimes, tol, iters, suave, tau = args
    return regimes_por_estacao(features, sel, n_regimes=n_regimes, tol=tol, iters=iters,
                               suave=suave, tau=tau)


# Fração do valor mediano usada como tolerância do r². Com 627 reamostragens,
# a tolerância fixa de 0,1 do α equivale a ~1/3 do α mediano; o r² usa essa MESMA
# fração do SEU valor mediano. Não depende do α: com mais reamostragens o α sobe de
# nível, o r² não muda, e a tolerância do r² também não deve mudar.
TOL_REL_R2 = 1 / 3


def tolerancia_da_metrica(alpha, cubo, tol_alpha):
    """A tolerância de concordância na escala da métrica.

    Para o α devolve `tol_alpha` (0,1 — a convenção de toda a pipeline, igual para
    observado e substitutos). Para o r², que é uma ordem de grandeza menor, uma
    tolerância absoluta de 0,1 deixaria passar ~100% dos pixels: usa-se
    `TOL_REL_R2` do r² mediano nos pares escolhidos pelo próprio r².
    """
    if cubo is alpha:
        return tol_alpha
    from src.analysis.grid_predep import escolher_por_predominancia

    sl, sm, _ = escolher_por_predominancia(cubo)
    return TOL_REL_R2 * float(np.nanmedian([cubo[:, si, sl[si], sm[si]]
                                            for si in range(len(sl))]))


def som_de_bacias(nome, arquivos, *, n_regimes=4, tol=0.10, iters=50, log=print,
                  metrica="alpha", n_proc=1, lag_fixo=None, suave=False, tau=0.10):
    """SOM de uma região a partir dos `<bacia>.nc` prontos — sem fatias, sem PREDEP.

    Usa só o que o arquivo de cada bacia já guarda: o cubo de α do observado. Como
    cada bacia escolheu o seu lag, a região refaz a escolha por predominância sobre a
    união dos pixels (uma conta de média, instantânea) para os vetores do SOM ficarem
    no mesmo espaço. Os surrogates NÃO são refeitos: o arquivo de saída não traz
    consenso contra o nulo — só a escolha, a máscara e o SOM.

    Devolve o Dataset; quem chama decide onde gravar.
    """
    import xarray as xr

    from src.analysis.grid_predep import escolher_por_predominancia

    if metrica not in ("alpha", "r2"):
        raise ValueError("metrica deve ser 'alpha' ou 'r2'")
    u = abrir_bacias(arquivos, com_r2=metrica == "r2")
    bacias, px, origem = u["bacias"], u["px"], u["origem"]
    # `alpha` abaixo é o cubo da MÉTRICA do SOM; o de α só entra na tolerância.
    alpha = u["alpha"] if metrica == "alpha" else u["r2"]
    tol = tolerancia_da_metrica(u["alpha"], alpha, tol)
    if metrica == "r2":
        log(f"[{nome}] SOM sobre r² · tolerância de concordância {tol:.4f} "
            "(a mesma regra relativa do α)")
    lat, lon, movs, lags, attrs = u["lat"], u["lon"], u["movs"], u["lags"], u["attrs"]
    n_se = len(SEASONS)
    log(f"[{nome}] {len(bacias)} bacia(s) · {len(px):,} pixels — lendo os resultados, "
        "sem recalcular PREDEP")

    sel_lag, sel_mov, concord = escolher_por_predominancia(alpha)
    if lag_fixo is not None:
        # SOM por defasagem: a feature é o vetor dos 29 MoVs NESTA defasagem, e o par
        # da máscara é o de maior predominância restrita a ela — mesmo critério.
        li = lags.index(int(lag_fixo))
        plano = np.where(np.isfinite(concord[:, li]), concord[:, li], -np.inf)
        sel_lag = np.full(n_se, li, dtype=int)
        sel_mov = plano.argmax(axis=1)
    log(f"[{nome}] escolha da região: " + " · ".join(
        f"{SEASONS[si]} {movs[sel_mov[si]]}/lag{lags[sel_lag[si]]}" for si in range(n_se)))
    alpha_sel = np.stack([[alpha[s, si, sel_lag[si], sel_mov[si]] for si in range(n_se)]
                          for s in range(2)])
    features = np.stack([[alpha[s, si, sel_lag[si]] for si in range(n_se)]
                         for s in range(2)])
    # Uma estação por processo: são independentes, e o resultado é o mesmo da
    # versão sequencial — só o tempo de parede cai.
    tarefas = [(features[:, si:si + 1], alpha_sel[:, si:si + 1], n_regimes, tol, iters,
                suave, tau) for si in range(n_se)]
    if n_proc > 1:
        from multiprocessing import Pool
        with Pool(min(n_proc, n_se)) as pool:
            partes = pool.map(_regimes_uma_estacao, tarefas)
    else:
        partes = [_regimes_uma_estacao(t) for t in tarefas]
    som = {"regime": np.concatenate([p["regime"] for p in partes], axis=1),
           "erro_bmu": np.concatenate([p["erro_bmu"] for p in partes], axis=1),
           "assinatura": np.concatenate([p["assinatura"] for p in partes], axis=1),
           "mascara": np.concatenate([p["mascara"] for p in partes], axis=0),
           "ari_fontes": np.concatenate([p["ari_fontes"] for p in partes]),
           "ari_teto": np.concatenate([p["ari_teto"] for p in partes], axis=0),
           "ari_todos": np.concatenate([p["ari_todos"] for p in partes])}
    for si in range(n_se):
        log(f"[{nome}] {SEASONS[si]}: {som['mascara'][si].sum():,} px na máscara · "
            f"ARI {som['ari_fontes'][si]:+.3f} · teto {np.nanmean(som['ari_teto'][si]):+.3f}")

    out = xr.Dataset(
        {"regime": (("source", "season", "pixel"), som["regime"]),
         "erro_bmu": (("source", "season", "pixel"), som["erro_bmu"]),
         "assinatura": (("source", "season", "regime_id", "mov_name"), som["assinatura"]),
         "mascara": (("season", "pixel"), som["mascara"]),
         "ari_fontes": (("season",), som["ari_fontes"]),
         # ARI entre sementes na mesma fonte: sem ele o ari_fontes não tem escala
         "ari_teto": (("season", "par_sementes"), som["ari_teto"]),
         "ari_todos": (("season",), som["ari_todos"]),
         "sel_lag": (("season",), sel_lag.astype(np.int16)),
         "sel_mov": (("season",), sel_mov.astype(np.int16)),
         "concordancia": (("season", "lag", "mov_name"), concord.astype(np.float32)),
         ("alpha_escolhido" if metrica == "alpha" else "r2_escolhido"):
             (("source", "season", "pixel"), alpha_sel.astype(np.float32)),
         "bacia_do_pixel": (("pixel",), origem)},
        coords={"source": ["xavier", "chirps"], "season": SEASONS, "lag": lags,
                "mov_name": movs, "regime_id": list(range(n_regimes)), "bacia": bacias,
                "pixel": px, "latitude": ("pixel", lat), "longitude": ("pixel", lon)})
    out.attrs.update({
        **{k: str(attrs.get(k, "")) for k in ATTRS_COMPATIVEIS},
        "bacia": nome,
        "bacias": ", ".join(bacias),
        "tipo": "som_regiao",
        "lag_fixo": "nenhum" if lag_fixo is None else int(lag_fixo),
        "treino": (f"máscara suave: todos os pixels, peso exp(-(|Δ|/{tau})²), reamostrados"
                   if suave else "máscara dura: só pixels com |Δ| < tol"),
        "metrica": metrica,
        "pixels": len(px),
        "origem": "resultados por bacia já calculados; PREDEP e surrogates não refeitos",
        "criterio": "um (lag, MoV) por estação — argmax da média espacial de "
                    "min(α_xavier, α_chirps) sobre TODOS os pixels da região",
        "mascara": f"|chirps − xavier| < {tol:.4g} ({metrica}) no candidato escolhido",
        "som": f"treinado na máscara, projetado em todos os pixels, k={n_regimes}",
    })
    return out
