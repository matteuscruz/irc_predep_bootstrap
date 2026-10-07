"""
Constantes e estimativas de custo — **só biblioteca padrão**.

Existe para o caminho de despacho ao Modal não precisar de numpy, scipy ou
xarray. O `main.py` importava `src.pipeline.bacia` no topo, que puxa
`predep_calculation.core` → `scipy`, e isso quebrava antes mesmo de chegar ao
`modal run` em ambientes com numpy 2.x e scipy antigo convivendo. O despacho
não pode depender da pilha científica local: ele só monta uma linha de comando.

Qualquer coisa que importe numpy vai em outro módulo.
"""
from __future__ import annotations

JANELA_COMUM = "data/processed/common_window/predep_janela_1982_2024.nc"
SAIDA_PADRAO = "results/bacias"
APP_MODAL = "src/modal/run_bacias.py"
VOLUME_MODAL = "predep-common-window"
# Nome do arquivo da janela DENTRO do volume Modal. Padrão: 1982–2024, o período
# em que Xavier, CHIRPS e os 29 MoVs coexistem — as duas fontes e as duas métricas
# (α e r²) usam exatamente os mesmos meses. A janela antiga (1996–2024) continua
# disponível com --janela predep_common_window.nc.
JANELA_MODAL_PADRAO = "predep_janela_1982_2024.nc"


def tag_janela(janela: str, anomalias: bool, n_boot: int = 0) -> str:
    """Namespace das fatias e resultados. MESMA regra de `_tag` em
    `src/modal/run_bacias.py`: se as duas divergirem, o download não acha nada.

    `n_boot` entra no nome quando não é o padrão (0 = ⌈n ln n⌉ do Algoritmo 1): as
    fatias com outro número de reamostragens têm outro nível de α e outro ruído, e
    misturá-las com as do padrão produziria um cubo sem sentido — sem erro nenhum.
    """
    base = janela.rsplit("/", 1)[-1]
    base = base[:-3] if base.endswith(".nc") else base
    return base + ("_anom" if anomalias else "") + (f"_nb{n_boot}" if n_boot else "")

# Medido nesta máquina com n=87 e n_boot=⌈n·log n⌉ (389). Serve só para avisar
# a ordem de grandeza antes de uma corrida local; o Modal não usa isto.
MS_POR_CHAMADA = 0.0134

SEASONS = ["DJF", "MAM", "JJA", "SON"]
MESES = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
         "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def fator_nboot(n_boot: int = 0) -> float:
    """Custo relativo ao padrão. Medido: 15 ms por estimativa com 627 reamostragens
    e 27 ms com 10.000 (n=129) — 1,8×, não 16×, porque o custo fixo domina."""
    import math
    if not n_boot or n_boot <= 627:
        return 1.0
    return 1.0 + 0.8 * math.log(n_boot / 627) / math.log(10000 / 627)


def estimar_custo(n_px, n_mov, n_lag, n_se=4, n_proc=8, n_boot=0):
    """Segundos estimados do cálculo de α e r² — `pixels × MoVs × lags × est × 2`."""
    return (n_px * n_mov * n_lag * n_se * 2 * MS_POR_CHAMADA * fator_nboot(n_boot)
            / max(1, n_proc))
