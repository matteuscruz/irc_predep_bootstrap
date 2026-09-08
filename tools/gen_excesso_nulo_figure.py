#!/usr/bin/env python3
"""
Mapa do EXCESSO de α sobre o nulo — a única leitura interpretável do PREDEP
neste tamanho de amostra.

Com o estimador do artigo em n=87 o α do nulo é ~+0,34, e o α MÁXIMO (máximo
sobre 203 candidatos) chega a 0,92 sem sinal algum. O valor absoluto no mapa não
significa nada; o que significa é o quanto ele excede o α que o acaso produziria
NAQUELE MESMO pixel — estimado por surrogates de deslocamento anual do MoV, que
preservam a autocorrelação do índice e destroem o pareamento com a chuva.

Lê a saída de experiments/18_modal_mapa_excesso_nulo.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div_bluered", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f8c9c8", "#ef8b8a", "#d33534", "#8a1f1e"])
for cm in (CMAP_SEQ, CMAP_DIV):
    cm.set_bad((0, 0, 0, 0))
SA = ROOT / "data/reference/south_america_110m.geojson"


def borders():
    if not SA.exists():
        return []
    gj = json.loads(SA.read_text())
    out = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out.append([np.asarray(r) for p in polys for r in p])
    return out


def to_grid(v, lat, lon):
    ulat, ulon = np.unique(lat), np.unique(lon)
    g = np.full((len(ulat), len(ulon)), np.nan)
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = v
    return g, [ulon.min() - .05, ulon.max() + .05, ulat.min() - .05, ulat.max() + .05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp18_excesso_nulo.nc")
    ap.add_argument("--out", default="plots/exp18_excesso_nulo.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    bd = borders()

    amax = ds["amax"].values          # (fonte, deslocamento, estação, pixel)
    comb = ds["amax_comb"].values     # (fonte, deslocamento, pixel)
    # deslocamento 0 = observado; 1..N = surrogates
    # Referência do nulo = MEDIANA dos surrogates. NÃO usar o percentil 95: com
    # poucos surrogates ele vira o máximo, e comparar contra o máximo de 6
    # sorteios não é um teste de 5% — é de 1/7.
    obs = amax[:, 0]
    null_med = np.nanmedian(amax[:, 1:], axis=1)
    exc = obs - null_med
    # "vence": o observado supera TODOS os surrogates. Sob o nulo isso ocorre
    # com probabilidade 1/(n_sur+1).
    wins = obs > np.nanmax(amax[:, 1:], axis=1)
    obs_c = comb[:, 0]
    null_c = np.nanmedian(comb[:, 1:], axis=1)
    exc_c = obs_c - null_c
    wins_c = obs_c > np.nanmax(comb[:, 1:], axis=1)

    rows = [(seasons[i], obs[0, i], null_med[0, i], exc[0, i],
             wins[0, i] & wins[1, i]) for i in range(len(seasons))]
    rows.append(("COMBINADO", obs_c[0], null_c[0], exc_c[0], wins_c[0] & wins_c[1]))
    chance = 1.0 / (amax.shape[1])          # 1/(n_sur+1)

    vmax = float(np.nanpercentile(np.concatenate([obs.ravel(), null_med.ravel()]), 99))
    vmin = float(np.nanpercentile(np.concatenate([obs.ravel(), null_med.ravel()]), 1))
    dlim = float(np.nanpercentile(np.abs(np.concatenate([exc.ravel(), exc_c.ravel()])), 98))

    fig, axes = plt.subplots(len(rows), 4, figsize=(15.5, 3.0 * len(rows)))
    titles = ["α máximo observado", "α máximo do NULO (mediana)",
              "EXCESSO (obs − nulo)", "obs vence todos os surrogates,\nnas duas fontes"]
    im_seq = im_div = None
    for r, (name, o, nl, ex, venceu) in enumerate(rows):
        marca = np.where(venceu, 1.0, np.nan)
        for c, (vals, kind) in enumerate([(o, "seq"), (nl, "seq"),
                                          (ex, "div"), (marca, "win")]):
            ax = axes[r, c]
            g, ext = to_grid(vals, lat, lon)
            ax.set_facecolor(OCEAN)
            for rings in bd:
                for ring in rings:
                    ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                         edgecolor=LAND_EDGE, lw=.7, zorder=1))
            kw = {"seq": dict(cmap=CMAP_SEQ, vmin=vmin, vmax=vmax),
                  "div": dict(cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -dlim, dlim)),
                  "win": dict(cmap=CMAP_DIV, vmin=0, vmax=1)}[kind]
            im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                           interpolation="nearest", aspect="equal", zorder=2, **kw)
            if kind == "seq":
                im_seq = im
            elif kind == "div":
                im_div = im
            ax.set_xlim(ext[0] - 1.6, ext[1] + 1.6)
            ax.set_ylim(ext[2] - 1.1, ext[3] + 1.1)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#8b8a84" if name == "COMBINADO" else "#d8d7d2")
                sp.set_linewidth(1.4 if name == "COMBINADO" else 0.8)
            if r == 0:
                ax.set_title(titles[c], fontsize=10.5, fontweight="semibold",
                             color=INK, pad=8)
            if c == 0:
                ax.text(-0.09, 0.5, name, transform=ax.transAxes, rotation=90,
                        va="center", ha="center",
                        fontsize=11.5 if name == "COMBINADO" else 12.5,
                        fontweight="semibold", color=INK)
            if c == 2:
                txt = f"{100*np.nanmean(ex > 0):.0f}% > 0"
            elif c == 3:
                f = np.nanmean(venceu)
                txt = f"{100*f:.1f}%  ·  acaso {100*chance**2:.1f}%"
            else:
                txt = None
            if txt:
                ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=9,
                        color=INK, fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1.8))

    fig.subplots_adjust(left=0.045, right=0.905, top=0.945, bottom=0.03,
                        wspace=0.05, hspace=0.10)
    cax1 = fig.add_axes([0.920, 0.56, 0.012, 0.30])
    cb1 = fig.colorbar(im_seq, cax=cax1); cb1.set_label("α máximo", fontsize=9, color=INK_2)
    cax2 = fig.add_axes([0.920, 0.16, 0.012, 0.30])
    cb2 = fig.colorbar(im_div, cax=cax2)
    cb2.set_label("excesso sobre o nulo", fontsize=9, color=INK_2)
    for cb in (cb1, cb2):
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, length=2)

    n_sh = ds.sizes["shift"] - 1
    fig.suptitle("PREDEP: excesso sobre o nulo — bacia do Uruguai, 1996–2024\n"
                 f"{ds.sizes['pixel']:,} pixels · nulo por {n_sh} deslocamentos anuais "
                 "do MoV · estimador do artigo (Xavier na coluna do excesso)",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.985)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}")
    print(f"   (acaso: observado vencer {amax.shape[1]-1} surrogates = "
          f"{100*chance:.1f}%; nas duas fontes = {100*chance**2:.1f}%)")
    for name, o, nl, ex, venceu in rows:
        print(f"   {name:<10} obs {np.nanmean(o):.3f} · nulo {np.nanmean(nl):.3f} · "
              f"excesso {np.nanmean(ex):+.3f} · vence nas duas "
              f"{100*np.nanmean(venceu):.1f}%")


if __name__ == "__main__":
    main()
