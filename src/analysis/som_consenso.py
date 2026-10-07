"""
SOM único sobre o consenso das duas bases (plano A).

Em vez de dois SOMs — um por fonte — comparados pixel a pixel pelo ARI, treina-se
UM SOM sobre o α médio de Xavier e CHIRPS, em todos os pixels. A concordância entre
as bases deixa de ser filtro e passa a ser diagnóstico, por regime:

* `assinatura` de cada base calculada separadamente sobre os MESMOS pixels do regime;
* `concord_assinatura` — correlação de Spearman entre os contrastes das duas bases;
* `delta_medio` — |α_chirps − α_xavier| médio no regime, no par escolhido.

O par (lag, MoV) continua escolhido pela predominância com concordância — o mínimo
das duas fontes —, uma vez por estação para a unidade inteira. Estabilidade: ARI
entre sementes (`ari_teto`). Sem calibração nula aqui: o mesmo procedimento nos
substitutos é o passo seguinte.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np

SEEDS = (42, 777, 1234)


def votar(particoes, n_regimes):
    """Partição de consenso: rótulos alinhados à 1ª semente (Hungarian) e voto por pixel.

    Devolve `(rotulos, confianca)`; confiança = fração das sementes que concordam com
    o regime vencedor no pixel. Evita a matriz pixel × pixel, que não cabe em 70 mil.
    """
    from src.analysis.som_regimes import alinhar_rotulos

    ref = particoes[0]
    alinhadas = np.stack([ref] + [alinhar_rotulos(ref, p, n_regimes) for p in particoes[1:]])
    votos = np.stack([(alinhadas == r).sum(axis=0) for r in range(n_regimes)])
    return votos.argmax(axis=0).astype(np.int16), (votos.max(axis=0) / len(particoes)
                                                   ).astype(np.float32)


def _uma_estacao(args):
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score

    from src.analysis.som_regimes import (alinhar_rotulos, assinatura, padronizar,
                                          treinar_e_projetar)

    feat_src, alpha_sel, n_regimes, iters, sementes = args
    xm, xa = entrada_som(feat_src)
    parts, erros = [], []
    for sd in sementes:
        r, e = treinar_e_projetar(xm, xa, n_regimes, sd, iters)
        parts.append(r)
        erros.append(e)
    return consolidar(feat_src, alpha_sel, parts, erros, n_regimes)


def entrada_som(feat_src):
    """(X de treino, X de projeção) do SOM: o consenso das bases, padronizado."""
    from src.analysis.som_regimes import padronizar

    feat = np.nanmean(feat_src, axis=0)              # consenso das bases (MoV, px)
    return padronizar(feat.T, np.ones(feat.shape[1], bool))


def consolidar(feat_src, alpha_sel, parts, erros, n_regimes):
    """Voto entre as sementes, estabilidade e assinaturas por base."""
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score

    from src.analysis.som_regimes import assinatura

    feat = np.nanmean(feat_src, axis=0)
    reg, conf = votar(parts, n_regimes)
    # estabilidade do CONSENSO: dois consensos independentes (sementes pares × ímpares)
    if len(parts) >= 4:
        a, _ = votar(parts[0::2], n_regimes)
        b, _ = votar(parts[1::2], n_regimes)
        estab = adjusted_rand_score(a, b)
    else:
        estab = np.nan
    uma = float(np.mean([adjusted_rand_score(parts[0], p) for p in parts[1:]]))

    ass = np.stack([assinatura(feat.T, reg, n_regimes),
                    assinatura(feat_src[0].T, reg, n_regimes),
                    assinatura(feat_src[1].T, reg, n_regimes)])   # (consenso, xav, chi)
    cx = ass[1] - np.nanmean(ass[1], axis=0)
    cc = ass[2] - np.nanmean(ass[2], axis=0)
    concord = np.full(n_regimes, np.nan, np.float32)
    delta = np.full(n_regimes, np.nan, np.float32)
    dif = np.abs(alpha_sel[1] - alpha_sel[0])
    for r in range(n_regimes):
        sel = reg == r
        if sel.any() and np.isfinite(cx[r]).any():
            concord[r] = spearmanr(cx[r], cc[r], nan_policy="omit").correlation
            delta[r] = np.nanmean(dif[sel])
    return {"particoes": np.stack(parts).astype(np.int16), "regime": reg, "confianca": conf, "erro_bmu": erros[0], "assinatura": ass,
            "estab_consenso": np.float32(estab), "estab_uma_semente": np.float32(uma),
            "concord": concord, "delta": delta}


def som_consenso(features, alpha_sel, *, n_regimes=4, iters=50, n_proc=4,
                 sementes=SEEDS):
    """`features` (fonte, estação, MoV, px); `alpha_sel` (fonte, estação, px)."""
    n_se = features.shape[1]
    tarefas = [(features[:, si], alpha_sel[:, si], n_regimes, iters, list(sementes))
               for si in range(n_se)]
    if n_proc > 1:
        with Pool(min(n_proc, n_se)) as pool:
            partes = pool.map(_uma_estacao, tarefas)
    else:
        partes = [_uma_estacao(t) for t in tarefas]
    return {k: np.stack([p[k] for p in partes]) for k in partes[0]}
