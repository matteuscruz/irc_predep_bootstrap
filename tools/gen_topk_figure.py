#!/usr/bin/env python3
"""
Exp 20 — a varredura de k: combinar os k MoVs mais predominantes ajuda?

Linha 1: o mapa em k=3 (os "3 melhores"), para comparar com o de k=1.
Linha 2: as duas curvas que respondem a pergunta — a taxa de vitória contra os
surrogates (nenhum k passa do acaso) e o α, que cai com k junto com o nulo,
lado a lado, mostrando que a queda é mecânica.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
RED = "#8a1f1e"
CMAP = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"])
CMAP.set_bad((0, 0, 0, 0))
SA = ROOT / "data/reference/south_america_110m.geojson"
CORES = {"DJF": "#2a78d6", "MAM": "#1f8a70", "JJA": "#c77b1e", "SON": "#8e4ec6"}


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
    ap.add_argument("--nc", default="results/exp20_topk.nc")
    ap.add_argument("--k-mapa", type=int, default=3)
    ap.add_argument("--out", default="plots/exp20_topk.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    ks = [int(k) for k in ds.k.values]
    a = ds["alpha"].values                          # (k, shift, season, pixel)
    n_sh = a.shape[1]
    acaso = 100.0 / n_sh

    vence = np.empty((len(ks), len(seasons)))
    a_obs = np.empty_like(vence)
    a_nul = np.empty_like(vence)
    for ki in range(len(ks)):
        for si in range(len(seasons)):
            o, nl = a[ki, 0, si], a[ki, 1:, si]
            vence[ki, si] = 100 * np.nanmean(o > np.nanmax(nl, axis=0))
            a_obs[ki, si] = np.nanmean(o)
            a_nul[ki, si] = np.nanmean(np.nanmedian(nl, axis=0))

    kmi = ks.index(args.k_mapa)
    obs_mapa = a[kmi, 0]
    vmin = float(np.nanpercentile(obs_mapa, 1))
    vmax = float(np.nanpercentile(obs_mapa, 99))
    bd = borders()

    fig = plt.figure(figsize=(15.4, 6.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.30, wspace=0.30,
                          left=0.055, right=0.905, top=0.875, bottom=0.115)

    for i, sn in enumerate(seasons):
        ax = fig.add_subplot(gs[0, i])
        g, ext = to_grid(obs_mapa[i], lat, lon)
        ax.set_facecolor(OCEAN)
        for rings in bd:
            for ring in rings:
                ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                     edgecolor=LAND_EDGE, lw=.7, zorder=1))
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                       cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest",
                       aspect="equal", zorder=2)
        ax.set_xlim(ext[0] - 1.6, ext[1] + 1.6)
        ax.set_ylim(ext[2] - 1.1, ext[3] + 1.1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d8d7d2")
        ax.set_title(sn, fontsize=12, fontweight="semibold", color=CORES[sn], pad=7)
        ax.text(0.03, 0.05, f"α médio {a_obs[kmi, i]:.3f}", transform=ax.transAxes,
                fontsize=9.5, color=INK, fontweight="semibold", zorder=5,
                bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=2))

    cax = fig.add_axes([0.917, 0.565, 0.010, 0.265])
    cb = fig.colorbar(im, cax=cax); cb.set_label("α (PREDEP)", fontsize=9.5, color=INK_2)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, length=2)

    x = np.arange(len(ks))

    # ── painel 1: a resposta ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    ax.axhspan(0, acaso, color="#f2f1ec", zorder=0)
    ax.axhline(acaso, color=RED, lw=1.4, ls="--", zorder=3)
    ax.text(len(ks) - 1.02, acaso + 0.45, f"acaso = 1/{n_sh} = {acaso:.1f}%",
            fontsize=8.8, color=RED, ha="right", va="bottom")
    for si, sn in enumerate(seasons):
        ax.plot(x, vence[:, si], "-o", color=CORES[sn], lw=1.5, ms=4.2,
                alpha=.75, label=sn, zorder=4)
    ax.plot(x, vence.mean(axis=1), "-o", color=INK, lw=2.4, ms=5.6,
            label="média", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("k  —  nº de candidatos (MoV × lag) combinados", fontsize=9.5, color=INK_2)
    ax.set_ylabel("pixels em que o observado\nsupera todos os surrogates (%)",
                  fontsize=9.5, color=INK_2)
    ax.set_ylim(0, max(30.0, vence.max() * 1.12))
    ax.legend(fontsize=8.5, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    ax.set_title("Nenhum k passa do acaso", fontsize=10.5, fontweight="semibold",
                 color=INK, loc="left", pad=22)

    # ── painel 2: por que α cai ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    for si, sn in enumerate(seasons):
        ax.plot(x, a_obs[:, si], "-o", color=CORES[sn], lw=1.6, ms=4.2, zorder=4)
        ax.plot(x, a_nul[:, si], "--", color=CORES[sn], lw=1.2, alpha=.55, zorder=3)
        ax.fill_between(x, a_nul[:, si], a_obs[:, si], color=CORES[sn], alpha=.10, zorder=2)
    ax.plot([], [], "-o", color=INK_2, lw=1.6, ms=4.2, label="observado")
    ax.plot([], [], "--", color=INK_2, lw=1.2, label="mediana dos surrogates")
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("k  —  nº de candidatos (MoV × lag) combinados", fontsize=9.5, color=INK_2)
    ax.set_ylabel("α médio na bacia", fontsize=9.5, color=INK_2)
    ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    ax.set_title("O α cai com k — mas o nulo cai junto", fontsize=10.5,
                 fontweight="semibold", color=INK, loc="left", pad=22)

    for a_ in fig.axes[len(seasons) + 1:]:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.tick_params(labelsize=8.5)
        a_.grid(axis="y", color="#e8e7e2"); a_.set_axisbelow(True)

    fig.suptitle("PREDEP — bacia do Uruguai, 1996–2024 · combinar os k MoVs mais "
                 "predominantes não recupera sinal",
                 fontsize=13, fontweight="semibold", color=INK, y=0.965)
    fig.text(0.5, 0.014,
             f"Mapas em k={args.k_mapa}. O nulo escolhe seus próprios k melhores entre os "
             "mesmos candidatos, então a comparação é justa. A queda do α com k é mecânica: "
             "cada candidato extra é menos predominante que o anterior.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}\n")
    print(f"{'k':>4}" + "".join(f"{s:>9}" for s in seasons) + f"{'média':>9}")
    for ki, k in enumerate(ks):
        print(f"{k:>4}" + "".join(f"{vence[ki,si]:>8.1f}%" for si in range(len(seasons)))
              + f"{vence[ki].mean():>8.1f}%")


if __name__ == "__main__":
    main()
