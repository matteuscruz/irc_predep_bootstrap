#!/usr/bin/env python3
"""
Mapa do α CRU (sem comparação com o nulo), na metodologia de MoV predominante.

É a convenção do artigo: reportar α diretamente, sem calibração nula. Torna os
valores comparáveis à Table 1 do artigo. A ressalva fica no rodapé da figura,
não no dado.
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
CMAP = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"])
CMAP.set_bad((0, 0, 0, 0))
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
    ap.add_argument("--out", default="plots/exp19_alpha_cru.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    obs = ds["alpha"].values[0]                  # deslocamento 0 = observado
    sel_m, sel_l = ds["sel_mov"].values, ds["sel_lag"].values
    bd = borders()

    vmin = float(np.nanpercentile(obs, 1))
    vmax = float(np.nanpercentile(obs, 99))

    # linha 2: ranking de predominância. Como há UM MoV por estação, um mapa
    # dele seria cor chapada — o informativo é por que ele foi escolhido e quem
    # ficou perto. A faixa cinza é o nível que os surrogates alcançam.
    concord = ds["concord"].values                 # (shift, season, lag, mov)
    n_top = 8

    fig, axes = plt.subplots(2, len(seasons), figsize=(4.0 * len(seasons), 7.8),
                             gridspec_kw={"height_ratios": [1.5, 1]})
    for i, sn in enumerate(seasons):
        ax = axes[0, i]
        g, ext = to_grid(obs[i], lat, lon)
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
        ax.set_title(f"{sn}\n{movs[sel_m[0, i]]} · lag {sel_l[0, i]}",
                     fontsize=11.5, fontweight="semibold", color=INK, pad=8)
        ax.text(0.03, 0.05, f"α médio {np.nanmean(obs[i]):.3f}",
                transform=ax.transAxes, fontsize=9.5, color=INK,
                fontweight="semibold", zorder=5,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))

        # ── ranking de predominância ────────────────────────────────────────
        ax = axes[1, i]
        flat = concord[0, i]                       # (lag, mov) observado
        ordem = np.argsort(flat.ravel())[::-1][:n_top]
        ll, mm = np.unravel_index(ordem, flat.shape)
        vals = flat[ll, mm]
        rot = [f"{movs[m]} · lag {ds.lag.values[l]}" for l, m in zip(ll, mm)]
        nulo_max = np.nanmax(concord[1:, i])       # melhor que os surrogates alcançam
        cores = ["#2a78d6" if v > nulo_max else "#c9c8c2" for v in vals]
        y = np.arange(n_top)[::-1]
        ax.barh(y, vals, color=cores, height=0.66)
        ax.axvline(nulo_max, color="#8a1f1e", lw=1.4, ls="--", zorder=3)
        ax.text(nulo_max, n_top - 0.2, " melhor surrogate", fontsize=8,
                color="#8a1f1e", va="top", ha="left")
        ax.set_yticks(y); ax.set_yticklabels(rot, fontsize=8.5)
        ax.set_xlim(0, max(float(np.nanmax(vals)), float(nulo_max)) * 1.18)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("predominância  (média de min(α Xavier, α CHIRPS))",
                      fontsize=8.5, color=INK_2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#e8e7e2"); ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.075, right=0.90, top=0.90, bottom=0.09,
                        wspace=0.42, hspace=0.20)
    cax = fig.add_axes([0.915, 0.50, 0.011, 0.36])
    cb = fig.colorbar(im, cax=cax); cb.set_label("α (PREDEP)", fontsize=9.5, color=INK_2)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, length=2)

    fig.suptitle("PREDEP — bacia do Uruguai, 1996–2024 · um MoV por estação, "
                 "escolhido por predominância com concordância entre Xavier e CHIRPS",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.965)
    fig.text(0.5, 0.012,
             "Valores brutos, sem calibração nula — a convenção do artigo. "
             "Nesta amostra (n efetivo ≈ 30) o α do acaso é da mesma ordem: "
             "ver plots/exp19_mov_predominante.png.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}")
    for i, sn in enumerate(seasons):
        o = obs[i]
        flat = concord[0, i]
        ordem = np.argsort(flat.ravel())[::-1][:3]
        ll, mm = np.unravel_index(ordem, flat.shape)
        top = ", ".join(f"{movs[m]}/lag{ds.lag.values[l]} {flat[l,m]:.3f}"
                        for l, m in zip(ll, mm))
        print(f"   {sn}: α médio {np.nanmean(o):.3f} · top3 predominância: {top} · "
              f"melhor surrogate {np.nanmax(concord[1:,i]):.3f}")


if __name__ == "__main__":
    main()
