#!/usr/bin/env python3
"""
Mapa do desenho invertido: escolhe o MoV primeiro, lê o α depois.

Um único (lag, MoV) por estação, escolhido pela predominância com concordância
entre as fontes — média espacial de min(α_xavier, α_chirps). Como a escolha não
olha o α do pixel, o valor mapeado deixa de ser um máximo e passa a ser uma
estimativa comum, sem viés de seleção pixel a pixel.

O nulo aplica o MESMO procedimento (escolha + leitura) a cada surrogate de
deslocamento anual — inclusive deixando o nulo escolher o melhor entre os
mesmos 203 candidatos.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div_bluered", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f8c9c8", "#ef8b8a", "#d33534", "#8a1f1e"])
CMAP_WIN = ListedColormap(["#8a1f1e"])
for cm in (CMAP_SEQ, CMAP_DIV, CMAP_WIN):
    cm.set_bad((0, 0, 0, 0))
SA = ROOT / "data/reference/south_america_110m.geojson"


def borders():
    if not SA.exists():
        return []
    gj = json.loads(SA.read_text())
    return [[np.asarray(r) for p in (f["geometry"]["coordinates"]
             if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]])
             for r in p] for f in gj["features"]]


def to_grid(v, lat, lon):
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.full((len(ulat), len(ulon)), np.nan)
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = v
    return g, [ulon.min() - .05, ulon.max() + .05, ulat.min() - .05, ulat.max() + .05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp19_mov_predominante.nc")
    ap.add_argument("--out", default="plots/exp19_mov_predominante.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    a = ds["alpha"].values                       # (shift, season, pixel)
    sel_m, sel_l = ds["sel_mov"].values, ds["sel_lag"].values
    n_sur = a.shape[0] - 1
    chance = 1.0 / (n_sur + 1)
    bd = borders()

    obs = a[0]
    nul_med = np.nanmedian(a[1:], axis=0)
    exc = obs - nul_med
    vence = obs > np.nanmax(a[1:], axis=0)

    vmax = float(np.nanpercentile(np.concatenate([obs.ravel(), nul_med.ravel()]), 99))
    vmin = float(np.nanpercentile(np.concatenate([obs.ravel(), nul_med.ravel()]), 1))
    dlim = float(np.nanpercentile(np.abs(exc), 98))

    fig, axes = plt.subplots(len(seasons), 4, figsize=(15.5, 3.05 * len(seasons)))
    titles = ["α do MoV escolhido", "α do NULO (mediana)",
              "EXCESSO (obs − nulo)", "obs vence todos os surrogates"]
    im_seq = im_div = None
    for r, sn in enumerate(seasons):
        for c, (vals, kind) in enumerate([(obs[r], "seq"), (nul_med[r], "seq"),
                                          (exc[r], "div"),
                                          (np.where(vence[r], 1.0, np.nan), "win")]):
            ax = axes[r, c]
            g, ext = to_grid(vals, lat, lon)
            ax.set_facecolor(OCEAN)
            for rings in bd:
                for ring in rings:
                    ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                         edgecolor=LAND_EDGE, lw=.7, zorder=1))
            kw = {"seq": dict(cmap=CMAP_SEQ, vmin=vmin, vmax=vmax),
                  "div": dict(cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -dlim, dlim)),
                  "win": dict(cmap=CMAP_WIN, vmin=0, vmax=1)}[kind]
            im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                           interpolation="nearest", aspect="equal", zorder=2, **kw)
            if kind == "seq":
                im_seq = im
            elif kind == "div":
                im_div = im
            ax.set_xlim(ext[0] - 1.6, ext[1] + 1.6)
            ax.set_ylim(ext[2] - 1.1, ext[3] + 1.1)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#d8d7d2"); sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title(titles[c], fontsize=10.5, fontweight="semibold",
                             color=INK, pad=8)
            if c == 0:
                ax.text(-0.09, 0.5, sn, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12.5,
                        fontweight="semibold", color=INK)
                ax.text(0.03, 0.05, f"{movs[sel_m[0, r]]} · lag {sel_l[0, r]}",
                        transform=ax.transAxes, fontsize=9.5, color=INK,
                        fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
            if c == 2:
                ax.text(0.03, 0.05, f"{100*np.nanmean(exc[r] > 0):.0f}% > 0",
                        transform=ax.transAxes, fontsize=9, color=INK,
                        fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.8))
            if c == 3:
                f = np.nanmean(vence[r])
                cor = "#8a1f1e" if f > chance else INK_2
                ax.text(0.03, 0.05, f"{100*f:.1f}%  ·  acaso {100*chance:.1f}%",
                        transform=ax.transAxes, fontsize=9, color=cor,
                        fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1.8))

    fig.subplots_adjust(left=0.045, right=0.905, top=0.935, bottom=0.03,
                        wspace=0.05, hspace=0.10)
    cax1 = fig.add_axes([0.920, 0.56, 0.012, 0.30])
    cb1 = fig.colorbar(im_seq, cax=cax1); cb1.set_label("α", fontsize=9, color=INK_2)
    cax2 = fig.add_axes([0.920, 0.16, 0.012, 0.30])
    cb2 = fig.colorbar(im_div, cax=cax2)
    cb2.set_label("excesso sobre o nulo", fontsize=9, color=INK_2)
    for cb in (cb1, cb2):
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, length=2)

    fig.suptitle("Lógica invertida: escolher o MoV primeiro, ler o α depois — "
                 "bacia do Uruguai\n"
                 "um (lag, MoV) por estação, por predominância com concordância "
                 f"entre as fontes · nulo por {n_sur} deslocamentos anuais",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.978)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}")
    for r, sn in enumerate(seasons):
        print(f"   {sn}: {movs[sel_m[0, r]]} lag {sel_l[0, r]} · "
              f"α {np.nanmean(obs[r]):.3f} · nulo {np.nanmean(nul_med[r]):.3f} · "
              f"excesso {np.nanmean(exc[r]):+.3f} · vence {100*np.nanmean(vence[r]):.1f}% "
              f"(acaso {100*chance:.1f}%)")


if __name__ == "__main__":
    main()
