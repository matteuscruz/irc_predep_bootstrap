#!/usr/bin/env python3
"""
Exp 25 — os oito SOMs (4 trimestres × 2 fontes) e a validação do ARI.

As duas linhas de mapas são os regimes que cada fonte encontrou, com os rótulos
do CHIRPS alinhados aos do Xavier por Hungarian — sem isso as cores não seriam
comparáveis, porque numeração de SOM é arbitrária. Cinza claro é fora da
máscara de concordância.

O painel de baixo é o que decide. Xavier e CHIRPS medem a MESMA chuva, então
seus SOMs coincidem bem acima de zero mesmo sem sinal: o piso do ARI não é 0,
é o que os surrogates alcançam. A faixa cinza é o teto imposto pela
instabilidade do próprio SOM.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch, Polygon

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
AZUL, RED, CINZA, VERDE = "#2a78d6", "#8a1f1e", "#b8b6b0", "#1f8a70"
FORA = "#cfcdc7"
REGIME = ["#2a78d6", "#e0562a", "#1f8a70", "#8e4ec6", "#c9a227",
          "#3fb0c9", "#d4508f", "#8c5a2b"]
SA = ROOT / "data/reference/south_america_110m.geojson"


def borders():
    if not SA.exists():
        return []
    gj = json.loads(SA.read_text())
    return [[np.asarray(r) for p in (f["geometry"]["coordinates"]
             if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]])
             for r in p] for f in gj["features"]]


def to_grid_rgba(v, lat, lon):
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.zeros((len(ulat), len(ulon), 4))
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = v
    return g, [ulon.min() - .05, ulon.max() + .05, ulat.min() - .05, ulat.max() + .05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp25_som_trimestre.nc")
    ap.add_argument("--out", default="plots/exp25_som_trimestre.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    ks = [int(k) for k in ds.k.values]
    reg = ds["regime"].values                 # (source, season, pixel)
    obs = ds["ari_observado"].values          # (k, season)
    nul = ds["ari_nulo"].values               # (k, season, surrogate)
    teto = ds["ari_teto"].values              # (k, season, par)
    k_mapa = int(ds.attrs.get("regime", "k=5").split("k=")[1].split(";")[0])
    n_reg = int(reg.max()) + 1
    bd = borders()

    def rgba(lbl):
        out = np.zeros((len(lbl), 4))
        for i, v in enumerate(lbl):
            out[i] = matplotlib.colors.to_rgba(
                FORA if v < 0 else REGIME[int(v) % len(REGIME)])
        return out

    fig = plt.figure(figsize=(15.2, 10.4))
    gsm = fig.add_gridspec(2, len(seasons), hspace=0.10, wspace=0.10,
                           left=0.062, right=0.985, top=0.900, bottom=0.470)
    gsb = fig.add_gridspec(1, len(ks), wspace=0.20,
                           left=0.062, right=0.985, top=0.335, bottom=0.085)

    for r, src in enumerate(["xavier", "chirps"]):
        for c, sn in enumerate(seasons):
            ax = fig.add_subplot(gsm[r, c])
            g, ext = to_grid_rgba(rgba(reg[r, c]), lat, lon)
            ax.set_facecolor(OCEAN)
            for rings in bd:
                for ring in rings:
                    ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                         edgecolor=LAND_EDGE, lw=.7, zorder=1))
            ax.imshow(g, origin="lower", extent=ext, interpolation="nearest",
                      aspect="equal", zorder=2)
            ax.set_xlim(ext[0] - 1.5, ext[1] + 1.5)
            ax.set_ylim(ext[2] - 1.0, ext[3] + 1.0)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#d8d7d2")
            if r == 0:
                ax.set_title(sn, fontsize=12, fontweight="semibold", color=INK, pad=7)
                n = int((reg[r, c] >= 0).sum())
                ax.text(0.03, 0.05, f"{n} pixels na máscara",
                        transform=ax.transAxes, fontsize=8.5, color=INK,
                        fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
            if c == 0:
                ax.text(-0.09, 0.5, "Xavier" if r == 0 else "CHIRPS",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=12, fontweight="semibold", color=INK)

    handles = [Patch(facecolor=REGIME[i], label=f"regime {i+1}") for i in range(n_reg)]
    handles.append(Patch(facecolor=FORA, label="fora da máscara"))
    axl = fig.add_axes([0.062, 0.395, 0.923, 0.045]); axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=9, handlelength=1.2, columnspacing=1.4, handletextpad=0.5,
               title=f"regimes do SOM em k={k_mapa} — CHIRPS alinhado ao Xavier "
                     "(Hungarian), porque numeração de SOM é arbitrária",
               title_fontsize=9.5)

    x = np.arange(len(seasons))
    for ki, k in enumerate(ks):
        ax = fig.add_subplot(gsb[0, ki])
        ax.fill_between(x, teto[ki].min(1), teto[ki].max(1), color=CINZA, alpha=.45,
                        zorder=1, label="TETO — mesmos dados, outra semente")
        ax.fill_between(x, nul[ki].min(1), nul[ki].max(1), color=RED, alpha=.15, zorder=1)
        ax.plot(x, np.median(nul[ki], 1), color=RED, lw=1.5, ls="--", zorder=3,
                label="NULO — fontes em surrogates")
        cores = [VERDE if o > n else AZUL for o, n in zip(obs[ki], nul[ki].max(1))]
        ax.plot(x, obs[ki], color=AZUL, lw=2.2, zorder=4)
        ax.scatter(x, obs[ki], s=80, c=cores, zorder=5, edgecolor="white", lw=1.3,
                   label="MEDIDO — Xavier vs CHIRPS")
        for i in range(len(seasons)):
            ax.annotate(f"{obs[ki, i]:.2f}", (x[i], obs[ki, i]),
                        textcoords="offset points", xytext=(0, -16), ha="center",
                        fontsize=8.5, color=INK, fontweight="semibold")
        ax.set_xticks(x); ax.set_xticklabels(seasons, fontsize=9)
        ax.set_title(f"k = {k} regimes", fontsize=10.5, fontweight="semibold",
                     color=INK, loc="left", pad=6)
        if ki == 0:
            ax.set_ylabel("Adjusted Rand Index", fontsize=9.5, color=INK_2)
            ax.legend(fontsize=8.2, frameon=False, loc="upper left",
                      bbox_to_anchor=(0.0, -0.16), ncol=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e8e7e2"); ax.set_axisbelow(True)

    lo = min(0.0, float(min(obs.min(), nul.min())) - 0.04)
    hi = float(max(teto.max(), obs.max(), nul.max())) * 1.20
    for a_ in fig.axes[2 * len(seasons) + 1:]:
        a_.set_ylim(lo, hi)
        a_.axhline(0, color="#d8d7d2", lw=1)

    passa = int((obs > nul.max(-1)).sum())
    fig.suptitle("SOM por trimestre climático e por fonte — bacia do Uruguai, "
                 "1996–2024\nα dos 29 MoVs no lag escolhido pela predominância, "
                 "só nos pixels em que Xavier e CHIRPS concordam (|Δα| < 0,1)",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.982)
    fig.text(0.5, 0.012,
             f"O ARI observado supera o melhor surrogate em {passa} de {obs.size} "
             "combinações (k × estação). O piso não é zero: as duas fontes medem a "
             "mesma chuva, então seus SOMs já coincidem sem nenhum sinal de MoV.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}\n")
    print(f"{'k':>3}{'estação':>9}{'TETO':>9}{'MEDIDO':>9}{'NULO med':>10}"
          f"{'NULO máx':>10}{'passa?':>9}")
    for ki, k in enumerate(ks):
        for si, sn in enumerate(seasons):
            n = nul[ki, si]
            print(f"{k:>3}{sn:>9}{teto[ki,si].mean():>9.3f}{obs[ki,si]:>9.3f}"
                  f"{np.median(n):>10.3f}{n.max():>10.3f}"
                  f"{('SIM' if obs[ki,si] > n.max() else 'não'):>9}")


if __name__ == "__main__":
    main()
