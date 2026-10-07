"""
Busca do SOM mais concordante entre Xavier e CHIRPS — calibrada contra o nulo.

**Por que a calibração não é opcional.** Escolher, entre várias configurações, a
que dá o maior ARI entre as fontes é selecionar pela própria estatística de teste:
o ARI do vencedor sai inflado mesmo sem sinal nenhum. É o mesmo viés que a
predominância com concordância foi feita para evitar na escolha do (lag, MoV).
Por isso a busca é uma função pura, aplicada IDÊNTICA ao observado e a cada
surrogate — mesmo espaço de busca, mesmas sementes, a máscara de cada um. O
observado só conta se o seu melhor ARI superar o melhor ARI de todos os
surrogates (nível de acaso 1/7 com seis deles).

**O que é buscado.** Tamanho da grade × número de regimes K, sempre a MESMA
configuração nas duas fontes. As sementes não entram na busca: para cada fonte, as
partições das várias sementes são consolidadas na partição medoide — a de maior
ARI médio contra as demais. Ela é escolhida olhando só aquela fonte, então não
seleciona pela concordância; e é mais estável que uma semente qualquer, o que
importa porque o teto (ARI entre sementes) limita tudo o que a comparação entre
fontes pode alcançar.

**O que NÃO muda em relação a `som_regimes`.** A feature (α dos 29 MoVs no lag
escolhido), a máscara (|α_chirps − α_xavier| < tol no par escolhido), a
padronização pela máscara e a regra "treina na máscara, projeta em todos".
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np

from src.analysis.som_regimes import (alinhar_rotulos, assinatura,
                                      mascara_concordancia, padronizar)

SEEDS_BUSCA = (42, 777, 1234)
KS_BUSCA = (3, 4, 5, 6)
PASSO_GRADE = 4


def grade_heuristica(n):
    """A grade da pipeline: g = ⌈√(5√n)⌉, entre 8 e 30."""
    return int(np.clip(np.ceil(np.sqrt(5.0 * np.sqrt(n))), 8, 30))


def grades_candidatas(n, passo=PASSO_GRADE):
    """A grade da pipeline e as vizinhas, uma menor e uma maior."""
    g0 = grade_heuristica(n)
    return sorted({int(np.clip(g, 6, 30)) for g in (g0 - passo, g0, g0 + passo)})


def _bmu_e_erro(pesos, x, bloco=2000):
    """Neurônio vencedor e distância a ele, em blocos (a matriz inteira não cabe)."""
    idx = np.empty(len(x), dtype=np.int32)
    err = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), bloco):
        d = np.linalg.norm(x[i:i + bloco, None, :] - pesos[None, :, :], axis=2)
        idx[i:i + bloco] = d.argmin(axis=1)
        err[i:i + bloco] = d.min(axis=1)
    return idx, err


def _treinar(args):
    """Um SOM (grade, semente) numa fonte → rótulos de cada K em todos os pixels.

    Função de módulo para o Pool conseguir serializá-la.
    """
    from sklearn.cluster import KMeans

    from src.analysis.som_insights import train_som

    x_mask, x_all, g, seed, ks, iters = args
    som, _ = train_som(x_mask, som_size=g, seed=seed, num_iteration=iters,
                       use_epochs=True)
    pesos = som.get_weights().reshape(-1, x_all.shape[1])
    bmu, erro = _bmu_e_erro(pesos, x_all)
    rotulos = np.empty((len(ks), len(x_all)), dtype=np.int16)
    for ki, k in enumerate(ks):
        km = KMeans(n_clusters=min(k, len(pesos)), random_state=seed, n_init=10)
        rotulos[ki] = km.fit(pesos).labels_[bmu]
    return rotulos, erro


def medoide(particoes, mascara):
    """Partição de maior ARI médio contra as demais, e o teto (ARI médio entre elas).

    `particoes` é `(semente, pixel)`. O ARI é medido na máscara, como no resto.
    """
    from sklearn.metrics import adjusted_rand_score

    n = len(particoes)
    if n == 1:
        return 0, np.nan
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = adjusted_rand_score(particoes[i][mascara],
                                                    particoes[j][mascara])
    media = (m.sum(axis=1) - 1) / (n - 1)
    fora_diag = m[~np.eye(n, dtype=bool)]
    return int(np.argmax(media)), float(fora_diag.mean())


def buscar(features, alpha_sel, *, tol=0.10, grades=None, ks=KS_BUSCA,
           seeds=SEEDS_BUSCA, iters=50, n_proc=None, min_mascara=None):
    """A busca completa para UMA (unidade, estação, deslocamento).

    `features` é `(fonte, MoV, pixel)` — o α de cada MoV no lag escolhido.
    `alpha_sel` é `(fonte, pixel)` — o α no par escolhido, de onde sai a máscara.

    Devolve um dicionário com:
      ari      (grade, K)  ARI entre as partições medoides das duas fontes
      teto     (grade, K)  ARI médio entre sementes, média das duas fontes
      melhor   (gi, ki)    a configuração de maior `ari`
      regime   (fonte, pixel)  as partições medoides da melhor configuração,
                               com o CHIRPS renumerado para casar com o Xavier
      erro_bmu (fonte, pixel)  distância ao neurônio vencedor, nessas partições
      mascara  (pixel,)
    Se a máscara tiver pixels de menos para o maior K, devolve `ari` todo NaN.
    """
    from sklearn.metrics import adjusted_rand_score

    ks = tuple(ks)
    mascara = mascara_concordancia(alpha_sel, tol)
    n_mask = int(mascara.sum())
    grades = list(grades or grades_candidatas(n_mask))
    n_px = features.shape[-1]
    vazio = {"ari": np.full((len(grades), len(ks)), np.nan),
             "teto": np.full((len(grades), len(ks)), np.nan),
             "melhor": (-1, -1), "grades": grades, "ks": ks,
             "regime": np.full((2, n_px), -1, dtype=np.int16),
             "erro_bmu": np.full((2, n_px), np.nan, dtype=np.float32),
             "mascara": mascara, "n_mascara": n_mask}
    if n_mask < (min_mascara or max(ks) * 4):
        return vazio

    x = [padronizar(features[s].T, mascara) for s in range(2)]
    tarefas = [(x[s][0], x[s][1], g, sd, ks, iters)
               for g in grades for sd in seeds for s in range(2)]
    n_proc = n_proc or min(len(tarefas), os.cpu_count() or 1)
    if n_proc > 1:
        with Pool(n_proc) as pool:
            saidas = pool.map(_treinar, tarefas)
    else:
        saidas = [_treinar(t) for t in tarefas]

    # (grade, semente, fonte) → rótulos (K, pixel) e erro (pixel)
    it = iter(saidas)
    rot = np.empty((len(grades), len(seeds), 2, len(ks), n_px), dtype=np.int16)
    err = np.empty((len(grades), len(seeds), 2, n_px), dtype=np.float32)
    for gi in range(len(grades)):
        for si in range(len(seeds)):
            for s in range(2):
                rot[gi, si, s], err[gi, si, s] = next(it)

    ari = np.full((len(grades), len(ks)), np.nan)
    teto = np.full((len(grades), len(ks)), np.nan)
    escolha = {}
    for gi in range(len(grades)):
        for ki in range(len(ks)):
            m0, t0 = medoide(rot[gi, :, 0, ki], mascara)
            m1, t1 = medoide(rot[gi, :, 1, ki], mascara)
            ari[gi, ki] = adjusted_rand_score(rot[gi, m0, 0, ki][mascara],
                                              rot[gi, m1, 1, ki][mascara])
            teto[gi, ki] = np.nanmean([t0, t1])
            escolha[gi, ki] = (m0, m1)

    gi, ki = np.unravel_index(int(np.nanargmax(ari)), ari.shape)
    m0, m1 = escolha[gi, ki]
    xav = rot[gi, m0, 0, ki].astype(np.int16)
    chi = alinhar_rotulos(xav, rot[gi, m1, 1, ki], ks[ki]).astype(np.int16)
    return {**vazio, "ari": ari, "teto": teto, "melhor": (int(gi), int(ki)),
            "regime": np.stack([xav, chi]),
            "erro_bmu": np.stack([err[gi, m0, 0], err[gi, m1, 1]])}


def assinaturas(features, regime, k_max):
    """Assinatura `(fonte, regime, MoV)` das partições da melhor configuração."""
    return np.stack([assinatura(features[s].T, regime[s], k_max) for s in range(2)])


def veredito(ari_obs, ari_surr):
    """O melhor ARI do observado supera o melhor de todos os surrogates?

    `ari_obs` é `(grade, K)`; `ari_surr` é `(surrogate, grade, K)`. O melhor de
    cada um é tomado sobre o MESMO espaço de busca — é isso que torna a
    comparação justa.
    """
    obs = float(np.nanmax(ari_obs)) if np.isfinite(ari_obs).any() else np.nan
    surr = np.array([np.nanmax(a) if np.isfinite(a).any() else np.nan
                     for a in ari_surr])
    return {"obs": obs, "surr_max": float(np.nanmax(surr)) if np.isfinite(surr).any()
            else np.nan, "surr": surr,
            "supera": bool(np.isfinite(obs) and np.isfinite(surr).any()
                           and obs > np.nanmax(surr))}
