"""Figura do estimador: ruído de Monte Carlo × número de sorteios (n_b).

    python plots_artigo/fig_nboot.py

Saída: <SAIDA>/estimador/fig_nboot.png. Os valores medidos estão fixos em
tools/diagnostico_ruido_predep.py (MEDIDO); não lê nenhum .nc.
"""
from comum import SAIDA

from tools.diagnostico_ruido_predep import figura

if __name__ == "__main__":
    f = figura(SAIDA / "estimador")
    f.rename(f.with_name("fig_nboot.png"))
    print(f"  → {f.with_name('fig_nboot.png')}")
