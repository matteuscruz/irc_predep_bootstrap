#!/usr/bin/env python3
"""
Exp 21 — o MoV vencedor pixel a pixel, e se o padrão significa algo.

Linha 1: o mapa observado. Linha 2: o mesmo procedimento em um surrogate, que
não tem relação nenhuma com a chuva. Se as duas linhas tiverem a mesma cara, a
organização espacial do mapa observado é herdada da autocorrelação da
precipitação — não é evidência de que aquele MoV governa aquele pedaço da bacia.
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
RED, CINZA = "#8a1f1e", "#c9c8c2"
SA = ROOT / "data/reference/south_america_110m.geojson"
PALETA = ["#2a78d6", "#e0562a", "#1f8a70", "#8e4ec6", "#c9a227", "#d4508f",
          "#3fb0c9", "#8c5a2b", "#6b8f3a", "#b23b3b", "#5b6bbf", "#c77b1e"]
OUTROS = "#94918a"


def borders():
    if not SA.exists():
        return []
    gj = json.loads(SA.read_text())
    return [[np.asarray(r) for p in (f["geometry"]["coordinates"]
             if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]])
             for r in p] for f in gj["features"]]


def to_grid(v, lat, lon):
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.zeros((len(ulat), len(ulon), 4))
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = v
    return g, [ulon.min() - .05, ulon.max() + .05, ulat.min() - .05, ulat.max() + .05]


def mapa(ax, rgba, lat, lon, bd, titulo, cor_tit, nota=None):
    g, ext = to_grid(rgba, lat, lon)
    ax.set_facecolor(OCEAN)
    for rings in bd:
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                 edgecolor=LAND_EDGE, lw=.7, zorder=1))
    ax.imshow(g, origin="lower", extent=ext, interpolation="nearest",
              aspect="equal", zorder=2)
    ax.set_xlim(ext[0] - 1.6, ext[1] + 1.6)
    ax.set_ylim(ext[2] - 1.1, ext[3] + 1.1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d8d7d2")
    if titulo:
        ax.set_title(titulo, fontsize=11.5, fontweight="semibold", color=cor_tit, pad=6)
    if nota:
        ax.text(0.03, 0.05, nota, transform=ax.transAxes, fontsize=9, color=INK,
                fontweight="semibold", zorder=5,
                bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp21_mov_espacial.nc")
    ap.add_argument("--surrogate", type=int, default=1, help="índice em shift (1..6)")
    ap.add_argument("--out", default="plots/exp21_mov_espacial.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    bm = ds["best_mov"].values                      # (shift, season, pixel)
    coer = ds["coerencia"].values                   # (shift, season)
    coer_emb = ds["coerencia_embaralhada"].values
    acordo = ds["acordo_fontes"].values
    ndom = ds["n_movs_dominantes"].values
    sh = ds["shift"].values
    bd = borders()

    # cores fixas para os MoVs mais frequentes no observado, resto cinza
    cnt = np.bincount(bm[0].ravel(), minlength=len(movs))
    top = list(np.argsort(cnt)[::-1][:len(PALETA)])
    cor_de = {m: PALETA[i] for i, m in enumerate(top)}

    def rgba(lbl):
        out = np.zeros((len(lbl), 4))
        for p, m in enumerate(lbl):
            c = matplotlib.colors.to_rgba(cor_de.get(int(m), OUTROS))
            out[p] = c
        return out

    fig = plt.figure(figsize=(15.4, 9.6))
    gsm = fig.add_gridspec(2, 4, hspace=0.10, wspace=0.28,
                           left=0.058, right=0.985, top=0.900, bottom=0.435)
    gsb = fig.add_gridspec(1, 2, wspace=0.26,
                           left=0.058, right=0.985, top=0.300, bottom=0.090)

    si_sur = args.surrogate
    for i, sn in enumerate(seasons):
        mapa(fig.add_subplot(gsm[0, i]), rgba(bm[0, i]), lat, lon, bd, sn, INK,
             f"{ndom[i]} MoVs \u22651% dos pixels")
        mapa(fig.add_subplot(gsm[1, i]), rgba(bm[si_sur, i]), lat, lon, bd, None, INK)

    fig.text(0.050, 0.783, "observado", rotation=90, va="center", ha="center",
             fontsize=11, fontweight="semibold", color=INK)
    fig.text(0.050, 0.552, f"surrogate\n(desloc. {sh[si_sur]} anos)", rotation=90,
             va="center", ha="center", fontsize=11, fontweight="semibold", color=RED)

    handles = [Patch(facecolor=cor_de[m], label=movs[m]) for m in top]
    handles.append(Patch(facecolor=OUTROS, label="demais MoVs"))
    axl = fig.add_axes([0.058, 0.345, 0.927, 0.035]); axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=13, frameon=False, fontsize=8.8,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.5)

    x = np.arange(len(seasons))
    w = 0.26

    # ── coerência ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gsb[0, 0])
    ax.bar(x - w, coer_emb, w, color=CINZA, label="rótulos embaralhados")
    ax.bar(x, coer[0], w, color="#2a78d6", label="observado")
    ax.bar(x + w, np.nanmax(coer[1:], axis=0), w, color=RED, alpha=.85,
           label="melhor surrogate")
    ax.set_xticks(x); ax.set_xticklabels(seasons, fontsize=9.5)
    ax.set_ylabel("vizinhos com o mesmo MoV", fontsize=9.5, color=INK_2)
    ax.set_ylim(0, max(0.48, float(np.nanmax(coer)) * 1.22))
    ax.legend(fontsize=8.5, ncol=3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.15))
    ax.set_title("Há blocos — mas o surrogate forma blocos iguais ou maiores",
                 fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=22)

    # ── acordo entre fontes ─────────────────────────────────────────────────
    ax = fig.add_subplot(gsb[0, 1])
    ax.bar(x, 100 * acordo, 0.5, color="#1f8a70")
    ax.axhline(100 / len(movs), color=RED, lw=1.4, ls="--", zorder=3)
    ax.text(len(seasons) - 0.55, 100 / len(movs) + 1.2,
            f"acaso = 1/{len(movs)} = {100/len(movs):.1f}%", fontsize=8.8,
            color=RED, ha="right", va="bottom")
    for i, a in enumerate(acordo):
        ax.text(i, 100 * a + 1.2, f"{100*a:.1f}%", ha="center", fontsize=9,
                color=INK, fontweight="semibold")
    ax.set_xticks(x); ax.set_xticklabels(seasons, fontsize=9.5)
    ax.set_ylabel("pixels em que Xavier e CHIRPS\nescolhem o mesmo MoV (%)",
                  fontsize=9.5, color=INK_2)
    ax.set_ylim(0, max(20.0, 100 * float(acordo.max()) * 1.35))
    ax.set_title("As duas fontes discordam da escolha em ~90% dos pixels",
                 fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=22)

    for a_ in (fig.axes[-2], fig.axes[-1]):
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=8.5)
        a_.grid(axis="y", color="#e8e7e2"); a_.set_axisbelow(True)

    fig.suptitle("PREDEP — bacia do Uruguai, 1996–2024 · qual MoV vence em cada pixel, "
                 "pelo critério de concordância entre Xavier e CHIRPS",
                 fontsize=13, fontweight="semibold", color=INK, y=0.965)
    fig.text(0.5, 0.014,
             "O surrogate passa pelo mesmo procedimento e não tem relação com a chuva. "
             "Os blocos que ele forma medem a autocorrelação espacial da precipitação, "
             "não o alcance de um modo de variabilidade.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}\n")
    for i, sn in enumerate(seasons):
        c = np.bincount(bm[0, i], minlength=len(movs))
        o = np.argsort(c)[::-1][:4]
        print(f"   {sn}: " + ", ".join(f"{movs[m]} {100*c[m]/len(lat):.0f}%" for m in o)
              + f"  ·  coer {coer[0,i]:.3f} vs surrog {np.nanmax(coer[1:,i]):.3f}")


if __name__ == "__main__":
    main()
