"""Regimes do SOM de consenso (4 estações + ano) e os MoVs de cada regime.

    python plots_artigo/fig_som.py --metrica alpha
    python plots_artigo/fig_som.py --metrica r2
    python plots_artigo/fig_som.py --metrica alpha --intersecao

Entrada: <DADOS>/som/som_{predep,r2}.nc, gerado por tools/som_consenso_local.py.
Saída: <SAIDA>/som/som_{predep,r2}[_intersecao].png. Com --intersecao, cada regime
lista só os MoVs que se destacam na assinatura de Xavier E na de CHIRPS.
"""
import argparse

from comum import DADOS, SAIDA, exigir

from tools.som_consenso_local import figura

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrica", choices=["alpha", "r2"], default="alpha")
    ap.add_argument("--intersecao", action="store_true")
    a = ap.parse_args()
    nome = "som_predep" if a.metrica == "alpha" else "som_r2"
    figura(exigir(DADOS / "som" / f"{nome}.nc"), SAIDA / "som",
           nome + ("_intersecao" if a.intersecao else ""), intersecao=a.intersecao)
