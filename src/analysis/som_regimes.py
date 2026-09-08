"""
SOM por trimestre e por fonte, sob a máscara de concordância.

É a lógica do `experiments/26_som_completo_assinaturas.py` promovida a módulo.
Três decisões de desenho que vieram de erros medidos e não devem ser desfeitas
sem refazer os testes que as motivaram:

**A feature não seleciona nada.** O vetor do pixel é o α de TODOS os MoVs no
lag que a predominância escolheu para aquela estação. A versão antiga usava
`nanmax` sobre estação e lag, levando para dentro do SOM o mesmo viés de
seleção que a metodologia da predominância havia removido.

**A máscara governa o treino, não a atribuição.** O SOM aprende só onde as duas
fontes concordam (|α_chirps − α_xavier| < tol) e depois projeta TODO pixel no
neurônio mais próximo. Recusar a atribuição fora da máscara jogaria fora um
terço a dois terços da bacia sem necessidade. A padronização usa média e desvio
DA MÁSCARA — com as estatísticas globais, o espaço de features muda entre
treino e projeção e o BMU deixa de significar o que significava.

**O erro de quantização é guardado.** Distância ao neurônio vencedor, por
pixel: é a camada de confiança do mapa preenchido, maior onde a atribuição é
extrapolação.

Nada aqui interpreta o resultado. O ARI entre fontes só tem sentido lido contra
um piso empírico (surrogates) e contra o teto da instabilidade do próprio SOM —
ver `METODOLOGIA_PREDOMINANCIA.md`, seção 6b.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.som_insights import assign_regimes, train_som  # noqa: E402

SEEDS_PADRAO = [42, 777, 1234]


def mascara_concordancia(alpha_sel, tol=0.10):
    """Pixels em que as fontes concordam no candidato escolhido.

    `alpha_sel` é `(fonte, pixel)` — o α das duas fontes no MESMO candidato.
    """
    return np.abs(alpha_sel[1] - alpha_sel[0]) < tol


def padronizar(x, mascara):
    """(X da máscara, X de todos) na MESMA escala — a da máscara.

    `x` é `(pixel, feature)`. Devolver os dois juntos é proposital: garante que
    treino e projeção nunca sejam padronizados por estatísticas diferentes.
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x[mascara], axis=0)
    sd = np.nanstd(x[mascara], axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    z = np.nan_to_num((x - mu) / sd, nan=0.0)
    return z[mascara], z


def treinar_e_projetar(x_mascara, x_todos, n_regimes, seed=42, iters=50):
    """Treina na máscara, atribui em todos. Devolve `(regimes, erro_bmu)`."""
    som, _ = train_som(x_mascara, seed=seed, num_iteration=iters, use_epochs=True)
    res = assign_regimes(som, x_todos, n_regimes=n_regimes, seed=seed)
    pesos = som.get_weights().reshape(-1, x_todos.shape[1])
    d = np.linalg.norm(x_todos[:, None, :] - pesos[None, :, :], axis=2)
    return res.pixel_regime, d.min(axis=1)


def alinhar_rotulos(referencia, outro, n_regimes):
    """Renumera `outro` para casar com `referencia` (Hungarian).

    Numeração de SOM é arbitrária: sem isto, dois mapas idênticos podem ter
    cores completamente diferentes e o ARI, que ignora rótulos, esconderia
    esse fato de quem olha a figura.
    """
    from scipy.optimize import linear_sum_assignment

    M = np.zeros((n_regimes, n_regimes), dtype=int)
    for i in range(n_regimes):
        for j in range(n_regimes):
            M[i, j] = np.sum((referencia == i) & (outro == j))
    linhas, colunas = linear_sum_assignment(-M)
    mapa = {int(c): int(r) for r, c in zip(linhas, colunas)}
    return np.array([mapa.get(int(v), int(v)) for v in outro])


def assinatura(alpha_nativo, rotulos, n_regimes):
    """α médio de cada MoV nos pixels de cada regime, em escala nativa.

    `alpha_nativo` é `(pixel, MoV)`. Devolve `(regime, MoV)`. É o que responde
    "quais MoVs definem este regime" — e a resposta é um perfil sobre todos os
    índices, nunca um índice vencedor.
    """
    out = np.full((n_regimes, alpha_nativo.shape[1]), np.nan, dtype=np.float32)
    for r in range(n_regimes):
        sel = rotulos == r
        if sel.any():
            out[r] = np.nanmean(alpha_nativo[sel], axis=0)
    return out


def regimes_por_estacao(features, alpha_sel, *, n_regimes=4, tol=0.10,
                        seeds=None, iters=50, progresso=True):
    """Roda o SOM de todas as estações e das duas fontes.

    `features` é `(fonte, estação, MoV, pixel)` — o α de cada MoV no lag
    escolhido. `alpha_sel` é `(fonte, estação, pixel)`, o α no candidato
    escolhido, de onde sai a máscara.

    Devolve um dicionário com `regime`, `erro_bmu`, `assinatura`, `mascara` e
    `ari_teto` (o ARI entre sementes na mesma base — o limite do que qualquer
    comparação pode alcançar).
    """
    from sklearn.metrics import adjusted_rand_score

    seeds = list(seeds or SEEDS_PADRAO)
    n_src, n_se, n_mov, n_px = features.shape
    reg = np.full((n_src, n_se, n_px), -1, dtype=np.int16)
    qerr = np.full((n_src, n_se, n_px), np.nan, dtype=np.float32)
    assin = np.full((n_src, n_se, n_regimes, n_mov), np.nan, dtype=np.float32)
    masc = np.zeros((n_se, n_px), dtype=bool)
    teto = np.full((n_se, max(1, len(seeds) - 1)), np.nan, dtype=np.float32)
    ari = np.full(n_se, np.nan, dtype=np.float32)

    for si in range(n_se):
        m = mascara_concordancia(alpha_sel[:, si], tol)
        masc[si] = m
        if m.sum() < n_regimes * 4:
            if progresso:
                print(f"      estação {si}: só {m.sum()} pixels na máscara — pulando")
            continue
        rotulos_ref = None
        for s in range(n_src):
            xm, xa = padronizar(features[s, si].T, m)
            if s == 0:
                variantes = [treinar_e_projetar(xm, xa, n_regimes, sd, iters)
                             for sd in seeds]
                rotulos, qerr[s, si] = variantes[0]
                teto[si] = [adjusted_rand_score(variantes[0][0], v[0])
                            for v in variantes[1:]] or [np.nan]
                rotulos_ref = rotulos
            else:
                rotulos, qerr[s, si] = treinar_e_projetar(xm, xa, n_regimes,
                                                          seeds[0], iters)
                rotulos = alinhar_rotulos(rotulos_ref, rotulos, n_regimes)
                ari[si] = adjusted_rand_score(rotulos_ref[m], rotulos[m])
            reg[s, si] = rotulos
            assin[s, si] = assinatura(features[s, si].T, rotulos, n_regimes)
        if progresso:
            print(f"      estação {si}: {m.sum():,} px na máscara · "
                  f"ARI {ari[si]:+.3f} · teto {np.nanmean(teto[si]):+.3f}", flush=True)

    return {"regime": reg, "erro_bmu": qerr, "assinatura": assin,
            "mascara": masc, "ari_fontes": ari, "ari_teto": teto}


def tabela_assinaturas(fonte, fonte_idx=0, top=3):
    """Quais MoVs distinguem cada regime — uma linha por (estação, regime).

    O SOM **não seleciona** MoVs: ele agrupa pixels usando os 29 de uma vez. O
    que esta tabela mostra é onde cada regime se afasta da média da estação —
    quais índices o *distinguem* dos outros regimes. Um índice com α alto em
    todos os regimes é forte na bacia e mesmo assim não aparece aqui, porque
    não separa nada.

    Devolve um DataFrame com o tamanho do regime, o α médio e os `top` índices
    acima e abaixo da média, com os valores.
    """
    import pandas as pd
    import xarray as xr

    ds = fonte if hasattr(fonte, "data_vars") else xr.open_dataset(fonte)
    movs = [str(m) for m in ds.mov_name.values]
    seasons = [str(s) for s in ds.season.values]
    fp = ds["assinatura"].values[fonte_idx]          # (estação, regime, MoV)
    reg = ds["regime"].values[fonte_idx]

    linhas = []
    for si, sn in enumerate(seasons):
        base = np.nanmean(fp[si], axis=0)
        for r in range(fp.shape[1]):
            anom = fp[si, r] - base
            pos = np.argsort(anom)[::-1][:top]
            neg = np.argsort(anom)[:top]
            linhas.append({
                "estacao": sn,
                "regime": f"R{r+1}",
                "pixels": int((reg[si] == r).sum()),
                "alpha_medio": round(float(np.nanmean(fp[si, r])), 4),
                "acima": ", ".join(f"{movs[m]} {anom[m]:+.3f}" for m in pos),
                "abaixo": ", ".join(f"{movs[m]} {anom[m]:+.3f}" for m in neg),
            })
    return pd.DataFrame(linhas)
