import math

import numpy as np
import scipy.stats as ss

_VALID_METHODS = ("bayesian_blocks", "ward")


def _paired_diff(rng, v, n_boot, exclude_self_pairs):
    """Amostra W = v[i] - v[j] para o truque "densidade da diferença em zero".

    `exclude_self_pairs=True` força j != i. Isso importa muito: sorteando i e j de
    forma independente COM reposição, com probabilidade 1/len(v) os dois caem no
    MESMO ponto e W = 0 exatamente — um átomo de massa 1/n em zero, justamente
    onde a KDE é avaliada. Como a massa do átomo é 1/n_bin, ela é maior em bins
    pequenos, o que inflava α proporcionalmente ao número de bins e fazia o α do
    NULO subir para ~0.095 em vez de ~0. Excluir i == j é o estimador padrão
    (U-estatística) de ∫f².

    `exclude_self_pairs=False` reproduz bit-a-bit o comportamento anterior (e o das
    implementações de referência dos autores), para reanálise de resultados antigos.
    """
    if not exclude_self_pairs:
        return rng.choice(v, n_boot) - rng.choice(v, n_boot)
    m = len(v)
    if m < 2:
        return np.zeros(0, dtype=float)
    i = rng.integers(0, m, n_boot)
    j = (i + rng.integers(1, m, n_boot)) % m   # uniforme entre os m-1 outros
    return v[i] - v[j]


def _bins_bayesian_blocks(X):
    """Partição de X por Bayesian Blocks (astropy).

    NÃO é o que o artigo especifica — veio de `references/predep.py`, um script
    auxiliar dos autores. Produz um número de bins VARIÁVEL por série (1-4 nos
    nossos MoVs), o que torna α incomparável entre preditores.
    """
    from astropy.stats import bayesian_blocks

    edges = bayesian_blocks(X)
    if len(edges) < 2:
        return None
    x_sorted = np.sort(X)
    n = len(X)

    def ecdf(v):
        return np.searchsorted(x_sorted, v, side="right") / n

    out = []
    for i in range(1, len(edges)):
        bg, ed = edges[i - 1], edges[i]
        mask = (X >= bg) & (X < ed)
        if i == len(edges) - 1:
            mask = mask | (X == ed)      # último bin fechado à direita
        out.append((mask, ecdf(ed) - ecdf(bg)))
    return out


def _bins_ward(X):
    """Partição de X por clustering hierárquico Ward com k=√N.

    É o que o Algorithm 1 do artigo especifica: "We define bins by a Hierarchical
    Clustering on X using k = √n clusters". O número de bins passa a depender só
    de N, ficando igual para todos os preditores — o que torna α comparável entre
    eles. O peso de cada bin é a proporção de pontos (linha 17 do Algorithm 1).
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(X)
    k = max(2, int(np.sqrt(n)))
    labels = fcluster(linkage(X.reshape(-1, 1), method="ward"), k,
                      criterion="maxclust") - 1
    out = []
    for c in range(k):
        mask = labels == c
        if mask.any():
            out.append((mask, mask.sum() / n))
    return out or None


def predep(X, Y, n_boot=None, seed=None, method="ward",
           exclude_self_pairs=False):
    """
    PREDEP α_{Y|X}: quanto X (MoV) prediz Y (precipitação).

    α = (S_{Y|X} - S_Y) / S_{Y|X}
    - 0: X e Y independentes
    - 1: X prediz Y perfeitamente

    Parâmetros
    ----------
    n_boot : int | None
        Amostras de Monte Carlo para estimar as densidades. NÃO é o bootstrap de
        intervalo de confiança — é o estimador de ∫f². `None` usa o default do
        Algorithm 1, **⌈n·log n⌉** (389 para n=87; 6908 para n=1000). As três
        implementações de referência discordam entre si aqui: o pseudocódigo diz
        ⌈n log n⌉, `references/predep.py` fixa 10000, e `references/predep_paper.py`
        recalcula por bin (~21 para bins de 9 pontos).
    seed : int | None
        Torna o Monte Carlo determinístico. Passar a MESMA seed para duas fontes
        de precipitação no mesmo pixel/lag faz o ruído deixar de ser uma fonte de
        diferença entre elas. `None` preserva o comportamento não-reprodutível.
    method : {'ward', 'bayesian_blocks'}
        Como particionar X. **`'ward'` (k=√N) é o padrão e é o que o Algorithm 1
        do artigo especifica.** Medido em n=87 com a bivariada normal ρ=0.8
        (α verdadeiro 0.400): ward devolve 0.402, `bayesian_blocks` devolve 0.192
        — este último subestima pela metade em amostra pequena, além de produzir
        número de bins variável por série, o que torna α incomparável entre
        preditores. `'bayesian_blocks'` fica disponível para reproduzir
        resultados anteriores a esta correção.
    exclude_self_pairs : bool
        `False` (default) segue o artigo e as duas implementações de referência,
        que sorteiam i e j de forma independente com reposição ("selected
        independently and with equal probability"), permitindo i == j. É o
        estimador V-statistic, com o termo diagonal.

        `True` exclui i == j (U-statistic, não-enviesado). **Consequência medida
        em n=87 com ward**: o α sob o NULO é **+0,336** com o default do artigo e
        **−0,019** com a exclusão. Em n=87 os bins têm ~9 pontos, e o átomo de
        massa 1/9 em zero domina; o artigo opera em N=1000, com bins de ~32
        pontos, onde o efeito é modesto. **Com o default, a significância NÃO
        pode ser medida contra zero** — tem de ser contra o nulo empírico
        (`null_calibration.py`). Ver `results/conformidade_final.md`.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method deve ser um de {_VALID_METHODS}: {method!r}")
    part = build_partition(X, method)
    return predep_with_partition(part, Y, n_boot=n_boot, seed=seed,
                                 exclude_self_pairs=exclude_self_pairs)


def build_partition(X, method="ward"):
    """Parte do PREDEP que depende só de X — a partição em bins.

    Existe como API pública porque num varrimento por pixel o X é o MESMO em
    todos os pixels de um dado (estação, lag, MoV): recalcular a partição por
    pixel desperdiça a maior parte do tempo (no `bayesian_blocks`, 83% do custo).
    Os experimentos 09/10/15 mantinham cópias inline disto — que divergiram do
    `core.py` quando o estimador mudou. Importar daqui evita a divergência.

    Devolve lista de `(máscara booleana, peso)` ou `None` se X não for
    particionável.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method deve ser um de {_VALID_METHODS}: {method!r}")
    X = np.asarray(X, dtype=float)
    return _bins_bayesian_blocks(X) if method == "bayesian_blocks" else _bins_ward(X)


def predep_with_partition(part, Y, n_boot=None, seed=None,
                          exclude_self_pairs=False):
    """α com a partição de X já calculada por `build_partition`.

    Equivale a `predep(X, Y, ...)` bit-a-bit quando `part = build_partition(X)`:
    a partição não consome o RNG, então hoistá-la não desloca o fluxo aleatório.
    """
    try:
        if part is None:
            return np.nan
        rng = np.random.default_rng(seed)
        Y = np.asarray(Y, dtype=float)
        if n_boot is None:                       # default do Algorithm 1
            n_boot = max(2, math.ceil(len(Y) * math.log(len(Y)))) if len(Y) > 1 else 2

        # S_Y: densidade marginal de Y
        diff_y = _paired_diff(rng, Y, n_boot, exclude_self_pairs)
        if len(diff_y) == 0 or np.std(diff_y) == 0:
            return np.nan
        s_y = float(np.squeeze(ss.gaussian_kde(diff_y).pdf(0)))

        s_y_mid_x = 0.0
        peso = 0.0
        for mask, p_range in part:
            y_bin = Y[mask]
            if len(y_bin) < 2 or len(np.unique(y_bin)) < 2:
                continue          # bin degenerado: Y constante, KDE indefinida
            dy_c = _paired_diff(rng, y_bin, n_boot, exclude_self_pairs)
            if len(dy_c) == 0 or np.std(dy_c) == 0:
                continue
            peso += p_range
            s_y_mid_x += p_range * float(np.squeeze(ss.gaussian_kde(dy_c).pdf(0)))

        if s_y_mid_x == 0 or peso == 0:
            return np.nan
        # Renormaliza pelos bins efetivamente estimados. Sem isto, um bin pulado
        # some da soma dos pesos enquanto S_Y continua sendo o total, e S_{Y|X}
        # fica subestimado por construção. No Piecewise da Table 1 do artigo,
        # 14 de 31 bins são degenerados (Y satura), os pesos somam 0,544 e α
        # despenca para -3,7; com a renormalização dá 0,853 contra 0,88 do artigo.
        # O Algorithm 1 não prevê pular bins, então isto é correção nossa.
        s_y_mid_x /= peso
        return float((s_y_mid_x - s_y) / s_y_mid_x)
    except Exception:
        return np.nan


def predep_ci(X, Y, n_boot=None, n_resample=200, alpha_level=0.05, seed=None,
              method="ward", exclude_self_pairs=False, blocks=None, m_frac=0.7):
    """Intervalo de confiança para o α, como o artigo propõe mas não implementa.

    O artigo diz: *"To address uncertainty, we propose Bootstrap Confidence
    Intervals (CI). PREDEP is an expected value."* Isto é a reamostragem dos
    PARES (X_i, Y_i) recalculando α — não confundir com o `n_boot` interno, que
    é o Monte Carlo do estimador de ∫f².

    **Subamostragem SEM reposição**, não bootstrap clássico. Motivo medido: com
    reposição ~37% das observações viram duplicatas, e valores duplicados
    recriam o átomo em zero na distribuição das diferenças (a exclusão de pares
    em `_paired_diff` é por índice, não por valor). O IC resultante fica ~3×
    estreito demais — cobertura medida de 55-60% contra os 95% nominais. Com
    subamostragem sem reposição a cobertura sobe para 85-100%.

    `blocks` : rótulos (ex.: ano da temporada) do mesmo tamanho de X. Quando
        dado, sorteia BLOCOS inteiros em vez de observações soltas — obrigatório
        em série temporal autocorrelacionada, onde o n efetivo é ~30 de 87
        nominais (ONI tem ICC 0,98 dentro de uma temporada).

    `m_frac` : fração do n usada em cada subamostra; o desvio é reescalado por
        sqrt(m/n) para valer no tamanho cheio.

    Devolve (alpha, lo, hi, n_validas).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    point = predep(X, Y, n_boot, seed=seed, method=method,
                   exclude_self_pairs=exclude_self_pairs)
    if not np.isfinite(point):
        return point, np.nan, np.nan, 0

    rng = np.random.default_rng(seed)
    if blocks is None:
        units = [np.array([i]) for i in range(len(X))]
    else:
        blocks = np.asarray(blocks)
        units = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    m = max(2, int(round(m_frac * len(units))))

    reps = np.empty(n_resample)
    for r in range(n_resample):
        pick = rng.permutation(len(units))[:m]          # sem reposição
        idx = np.concatenate([units[g] for g in pick])
        reps[r] = predep(X[idx], Y[idx], n_boot, seed=int(rng.integers(0, 2**31)),
                         method=method, exclude_self_pairs=exclude_self_pairs)

    ok = np.isfinite(reps)
    if ok.sum() < max(20, n_resample // 10):
        return point, np.nan, np.nan, int(ok.sum())
    z = 1.959963984540054                                # normal 95%
    sd = reps[ok].std() * np.sqrt(m / len(units))        # escala para o n cheio
    half = z * sd if alpha_level == 0.05 else \
        abs(np.percentile(reps[ok], 100 * (1 - alpha_level / 2))
            - np.percentile(reps[ok], 100 * alpha_level / 2)) / 2
    return point, float(point - half), float(point + half), int(ok.sum())
