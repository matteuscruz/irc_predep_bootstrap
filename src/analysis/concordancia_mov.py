"""
Concordância de IDENTIDADE do MoV, calibrada pixel a pixel.

A pergunta é outra, mais exigente que a dos mapas anteriores. Lá, um par (lag, MoV)
único para o Brasil era imposto e as fontes eram comparadas pelo VALOR de α nele.
Aqui cada fonte escolhe, em cada pixel, o SEU MoV de maior α, e o pixel só entra no
mapa quando as duas escolhem o MESMO índice e cada uma supera os próprios
substitutos naquele pixel.

**Por que o nulo é obrigatório.** O máximo de 203 estimativas ruidosas é alto mesmo
sem dependência nenhuma, e duas fontes que medem a mesma chuva escolhem o mesmo
índice com frequência bem acima de 1/29. Nada disso é evidência. Por isso o mesmo
argmax é calculado nos 6 substitutos, no mesmo pixel, e o observado precisa vencê-los.

**Teste 7-way simétrico.** Cada uma das 7 séries (observado + 6 substitutos) é
testada contra o MÁXIMO das outras 6. Sob a hipótese nula as sete são
intercambiáveis, então a chance de o observado ser o maior é 1/7 ≈ 14,3%. A
alternativa ingênua — observado contra 6, cada substituto contra 5 — daria ao
observado uma comparação a mais e inflaria o resultado.

**Defasagens.** Onde faltam fatias de algum deslocamento, o conjunto de defasagens é
a interseção sobre TODOS os deslocamentos daquela bacia e estação. Observado e
substitutos disputam sempre o mesmo número de candidatos; menos candidatos tornam o
teste mais fraco, nunca enviesado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SHIFTS_NULO = [3, 5, 7, 9, 11, 13]


def nome_fatia(si, lag, shift) -> str:
    """MESMO nome de `src.pipeline.montagem` — as fatias são as mesmas."""
    return f"s{si}_l{lag}_d{shift}.npz"


def lags_comuns(dir_fatias, si, shifts=SHIFTS_NULO, lags=range(7)):
    """As defasagens presentes em todos os deslocamentos daquela estação."""
    d = Path(dir_fatias)
    return [l for l in lags
            if all((d / nome_fatia(si, l, sh)).exists() for sh in shifts)]


def cubo_da_bacia(alpha_obs, dir_fatias, si, lags, px_nc, movs_nc,
                  shifts=SHIFTS_NULO):
    """Cubo `(série, fonte, lag, MoV, pixel)` com o observado na posição 0.

    `alpha_obs` é o cubo do `.nc` da bacia, `(fonte, estação, lag, MoV, pixel)`, já
    na ordem de pixels do arquivo. As fatias são reordenadas para essa mesma ordem.
    """
    d = Path(dir_fatias)
    n_px, n_mov = len(px_nc), len(movs_nc)
    cubo = np.full((1 + len(shifts), 2, len(lags), n_mov, n_px), np.nan, np.float32)
    for li, lag in enumerate(lags):
        cubo[0, :, li] = alpha_obs[:, si, lag]
        for di, sh in enumerate(shifts, start=1):
            with np.load(d / nome_fatia(si, lag, sh), allow_pickle=True) as z:
                movs, px, a = z["movs"], z["px"], z["alpha"]
            if [str(m) for m in movs] != list(movs_nc):
                raise ValueError(f"{d.name}: MoVs da fatia diferem do .nc")
            ordem = np.argsort(px)
            if not np.array_equal(px[ordem], px_nc):
                raise ValueError(f"{d.name}: pixels da fatia s{si} l{lag} d{sh} "
                                 "diferem do .nc")
            cubo[di, :, li] = a[..., ordem]
    return cubo


def melhor_por_pixel(cubo):
    """Para cada série e fonte: o maior α do pixel e o MoV que o atingiu.

    `cubo` é `(série, fonte, lag, MoV, pixel)`. Devolve `(valor, mov)`, ambos
    `(série, fonte, pixel)`. O argmax é sobre (lag, MoV) juntos; o MoV é o resto da
    divisão pelo número de índices, como em `plot_bacia_predep`.
    """
    n_ser, n_src, n_lag, n_mov, n_px = cubo.shape
    plano = cubo.reshape(n_ser, n_src, n_lag * n_mov, n_px)
    finito = np.where(np.isfinite(plano), plano, -np.inf)
    k = finito.argmax(axis=2)
    valor = np.take_along_axis(plano, k[:, :, None], axis=2)[:, :, 0]
    return valor, (k % n_mov).astype(np.int16)


def testar(valor, mov):
    """O teste 7-way, simétrico, e a seleção conjunta.

    `valor` e `mov` são `(série, fonte, pixel)`. Devolve:
      `significativo` (série, fonte, pixel) — a série supera o máximo das outras 6;
      `mesmo_mov`     (série, pixel)        — as duas fontes escolheram o mesmo MoV;
      `selecionado`   (série, pixel)        — significativo nas duas E mesmo MoV.
    A série 0 é o observado; as demais são os substitutos, e as máscaras delas são a
    distribuição nula da fração selecionada.
    """
    n_ser = valor.shape[0]
    significativo = np.zeros(valor.shape, bool)
    for d in range(n_ser):
        outras = np.delete(valor, d, axis=0)
        significativo[d] = valor[d] > np.nanmax(outras, axis=0)
    mesmo_mov = mov[:, 0] == mov[:, 1]
    return {"significativo": significativo, "mesmo_mov": mesmo_mov,
            "selecionado": significativo.all(axis=1) & mesmo_mov}


def piores_substitutos(valor):
    """O maior α entre os substitutos, por fonte e pixel — o piso do observado."""
    return np.nanmax(valor[1:], axis=0)


def resumo(sel, mov, movs_nc, n_top=8):
    """Fração selecionada (observado × substitutos) e composição por MoV."""
    frac = sel.mean(axis=1)
    escolhidos = mov[0, 0][sel[0]]
    cont = np.bincount(escolhidos, minlength=len(movs_nc)) if escolhidos.size else \
        np.zeros(len(movs_nc), int)
    ordem = np.argsort(cont)[::-1][:n_top]
    return {
        "frac_obs": float(frac[0]),
        "frac_surr": frac[1:].astype(float),
        "frac_surr_max": float(frac[1:].max()) if len(frac) > 1 else float("nan"),
        "supera": bool(len(frac) > 1 and frac[0] > frac[1:].max()),
        "top_movs": [(str(movs_nc[m]), int(cont[m])) for m in ordem if cont[m] > 0],
    }
