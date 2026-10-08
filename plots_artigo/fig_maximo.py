"""PREDEP (ou r²) máximo por estação e defasagem, e o MoV que o produz.

    python plots_artigo/fig_maximo.py --metrica alpha
    python plots_artigo/fig_maximo.py --metrica r2

Entrada: <DADOS>/max_lags/{predep,r2}_max_lags.nc (variáveis valor_max, mov_max,
dif_fontes), gerado por tools/fig_predep_max_lags.py.
Saída: <SAIDA>/{predep,r2}_max/{predep,r2}_max_<estação>.png — uma figura por estação,
colunas = defasagens; acima o valor máximo, abaixo o MoV vencedor. Opacidade plena
onde Xavier e CHIRPS concordam no MoV vencedor (|Δ| < tol), caindo até 35% em 3·tol;
tol = 0,1 no α e 1/3 da mediana do máximo no r².
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from comum import DADOS, SAIDA, SEASONS, exigir
from src.visualization.bacia_plots import _salvar
from src.visualization.predominancia_base import (CMAP_SEQ, INK, INK_2, MOV_OUTROS,
                                                  MOV_PALETTE, draw_basemap,
                                                  load_borders, strip_axes,
                                                  to_grid_rgba)


def opacidade(dif, tol, piso=0.35):
    t = np.clip((np.abs(dif) - tol) / (2 * tol), 0, 1)
    return np.where(np.isfinite(dif), 1 - (1 - piso) * t, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrica", choices=["alpha", "r2"], default="alpha")
    a = ap.parse_args()
    nome = "predep" if a.metrica == "alpha" else "r2"
    rot = "PREDEP α" if a.metrica == "alpha" else "r²"

    with xr.open_dataset(exigir(DADOS / "max_lags" / f"{nome}_max_lags.nc")) as d:
        vmax, venc, dif = d["valor_max"].values, d["mov_max"].values, d["dif_fontes"].values
        lags = [int(v) for v in d.lag.values]
        lat, lon = d.latitude.values, d.longitude.values
        movs = [str(m) for m in d.mov_name.values]

    tol = 0.10 if a.metrica == "alpha" else float(np.nanmedian(vmax)) / 3
    norm = matplotlib.colors.Normalize(*(float(np.nanpercentile(vmax, q)) for q in (2, 98)))
    cmap = matplotlib.colormaps[CMAP_SEQ] if isinstance(CMAP_SEQ, str) else CMAP_SEQ
    bd = load_borders()
    nl = len(lags)

    for si, sn in enumerate(SEASONS):
        cnt = np.bincount(venc[si].ravel(), minlength=len(movs))
        top = [int(m) for m in np.argsort(cnt)[::-1][:len(MOV_PALETTE)] if cnt[m] > 0]
        cor_de = {m: MOV_PALETTE[i] for i, m in enumerate(top)}
        tab = np.array([matplotlib.colors.to_rgba(cor_de.get(m, MOV_OUTROS))
                        for m in range(len(movs))])

        fig = plt.figure(figsize=(3.6 * nl, 8.8))
        gs = fig.add_gridspec(2, nl, wspace=0.03, hspace=0.06, left=0.03, right=0.99,
                              top=0.93, bottom=0.20)
        for li, lag in enumerate(lags):
            alfa = opacidade(dif[si, li], tol)
            for linha, rgba in ((0, cmap(norm(vmax[si, li]))), (1, tab[venc[si, li]].copy())):
                ax = fig.add_subplot(gs[linha, li])
                rgba[:, 3] = alfa
                g, ext = to_grid_rgba(rgba, lat, lon)
                draw_basemap(ax, bd, ext)
                ax.imshow(g, origin="lower", extent=ext, interpolation="nearest", zorder=2)
                strip_axes(ax)
                if linha == 0:
                    ax.set_title(f"lag {lag}", fontsize=13, fontweight="semibold",
                                 color=INK, pad=6)
                if li == 0:
                    ax.text(-0.06, 0.5, f"{rot} máximo" if linha == 0 else "MoV do máximo",
                            transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=12, fontweight="semibold", color=INK)

        area = cnt / cnt.sum()
        handles = [Patch(facecolor=cor_de[m], label=f"{movs[m]} {area[m]:.0%}") for m in top]
        handles.append(Patch(facecolor=MOV_OUTROS, label="demais"))
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                   frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.105))
        cax = fig.add_axes([0.35, 0.075, 0.30, 0.012])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal")
        cb.set_label(f"{rot} máximo (média de Xavier e CHIRPS)", fontsize=9.5, color=INK_2)
        fig.suptitle(sn, fontsize=15, fontweight="semibold", color=INK, y=0.985)
        _salvar(fig, SAIDA / f"{nome}_max", f"{nome}_max_{sn.lower()}.png", 140)


if __name__ == "__main__":
    main()
