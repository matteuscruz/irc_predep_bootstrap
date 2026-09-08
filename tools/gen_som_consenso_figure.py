#!/usr/bin/env python3
"""
Figura do exp11: o SOM salva a atribuição de MoV?

Linha 1 — três mapas de regime: Xavier (semente 42), Xavier (semente 777) e
CHIRPS. Os DOIS PRIMEIROS SÃO OS MESMOS DADOS; se o SOM fosse estável, seriam
idênticos.

Linha 2 — reprodutibilidade (ARI) contra coerência espacial, que é a leitura
central: os mapas parecem ótimos e não se reproduzem.

Os rótulos de regime são arbitrários entre execuções do SOM, então os mapas 2 e
3 têm seus rótulos REALINHADOS ao mapa 1 pelo algoritmo húngaro. Sem isso, boa
parte da diferença visual seria só permutação de rótulo, não estrutura.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Polygon
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.analysis.som_insights import assign_regimes, train_som  # noqa: E402

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"
REGIME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4",
                 "#4a3aa7", "#eda100", "#e34948", "#008300"]
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


def align(ref, other, k):
    """Realinha os rótulos de `other` aos de `ref` maximizando a coincidência.

    Rótulo de regime do SOM é arbitrário: o regime "3" de uma execução não tem
    relação com o "3" de outra. Sem este realinhamento a comparação visual
    mediria permutação de rótulo, não diferença de estrutura.
    """
    cm = np.zeros((k, k), int)
    for a, b in zip(ref, other):
        cm[a, b] += 1
    r, c = linear_sum_assignment(-cm)
    mapping = dict(zip(c, r))
    return np.array([mapping.get(x, x) for x in other])


def feats(ds, src):
    a = ds[f"alpha_{src}"].values
    x = np.nanmax(a, axis=(0, 1)).T
    mu, sd = x.mean(0), x.std(0)
    return (x - mu) / np.where(sd == 0, 1., sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp09b_uruguai_corrigido.nc")
    ap.add_argument("--out", default="plots/exp11_som_consenso.png")
    ap.add_argument("--n-regimes", type=int, default=7)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    k = args.n_regimes

    ds = xr.open_dataset(ROOT / args.nc)
    lat, lon = ds.latitude.values, ds.longitude.values
    Xx, Xc = feats(ds, "xavier"), feats(ds, "chirps")

    def reg(X, seed):
        som, _ = train_som(X, seed=seed, num_iteration=args.iters, use_epochs=True)
        return assign_regimes(som, X, n_regimes=k, seed=seed).pixel_regime

    print("treinando 3 SOMs...", flush=True)
    r_a, r_b, r_c = reg(Xx, 42), reg(Xx, 777), reg(Xc, 42)
    ari_seed, ari_src = adjusted_rand_score(r_a, r_b), adjusted_rand_score(r_a, r_c)
    r_b, r_c = align(r_a, r_b, k), align(r_a, r_c, k)

    # coerência espacial
    ulat, ulon = np.unique(lat), np.unique(lon)
    grid = np.full((len(ulat), len(ulon)), -1, int)
    grid[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = np.arange(len(lat))
    pairs = [(grid[a, b], grid[a + da, b + db])
             for a in range(len(ulat)) for b in range(len(ulon))
             for da, db in ((0, 1), (1, 0))
             if grid[a, b] >= 0 and a + da < len(ulat) and b + db < len(ulon)
             and grid[a + da, b + db] >= 0]
    pairs = np.array(pairs)

    def coher(lab):
        obs = (lab[pairs[:, 0]] == lab[pairs[:, 1]]).mean()
        rng = np.random.default_rng(0)
        null = np.mean([(lambda p: (p[pairs[:, 0]] == p[pairs[:, 1]]).mean())(rng.permutation(lab))
                        for _ in range(200)])
        return obs, null

    co_x, ch_x = coher(r_a)
    co_c, ch_c = coher(r_c)

    bd = borders()
    cmap = ListedColormap(REGIME_COLORS[:k]); cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-.5, k + .5), k)
    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.28, wspace=0.12)

    maps = [(r_a, "Xavier  ·  semente 42"), (r_b, "Xavier  ·  semente 777"),
            (r_c, "CHIRPS  ·  semente 42")]
    for i, (lab, title) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        g, ext = to_grid(lab.astype(float), lat, lon)
        ax.set_facecolor(OCEAN)
        for rings in bd:
            for ring in rings:
                ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                     edgecolor=LAND_EDGE, lw=.7, zorder=1))
        ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext, cmap=cmap,
                  norm=norm, interpolation="nearest", aspect="equal", zorder=2)
        ax.set_xlim(ext[0] - 1.6, ext[1] + 1.6); ax.set_ylim(ext[2] - 1.1, ext[3] + 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d8d7d2")
        ax.set_title(title, fontsize=11, fontweight="semibold",
                     color="#8a1f1e" if i < 2 else INK, pad=7)
    # colchete ligando os dois primeiros mapas, que são a MESMA base
    fig.text(0.375, 0.955, "MESMOS DADOS", ha="center", fontsize=10.5,
             fontweight="semibold", color="#8a1f1e")
    fig.add_artist(plt.Line2D([0.135, 0.615], [0.945, 0.945], color="#8a1f1e",
                              lw=1.2, transform=fig.transFigure))

    # ARI
    ax = fig.add_subplot(gs[1, 0])
    vals = [ari_seed, ari_src, 0.0]
    labs = ["mesma base,\noutra semente", "Xavier vs\nCHIRPS", "aleatório"]
    ax.bar(range(3), vals, color=["#6da7ec", "#d33534", "#c9c8c2"], width=.62)
    for i, v in enumerate(vals):
        ax.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=10, fontweight="semibold",
                color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("ARI (1 = idêntico)", fontsize=9.5, color=INK_2)
    ax.axhline(1.0, color=INK_3, lw=.8, ls=":")
    ax.text(2.45, 0.97, "reprodutível", fontsize=8.5, color=INK_3, ha="right", va="top")
    ax.set_title("Reprodutibilidade dos regimes", fontsize=10.5, loc="left",
                 fontweight="semibold", color=INK)
    ax.grid(axis="y", color="#e8e7e2"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # coerência espacial
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(2); w = .34
    ax.bar(x - w/2, [co_x*100, co_c*100], w, color="#256abf", label="observado")
    ax.bar(x + w/2, [ch_x*100, ch_c*100], w, color="#c9c8c2", label="acaso")
    for i, (o, c) in enumerate([(co_x, ch_x), (co_c, ch_c)]):
        ax.text(i - w/2, o*100 + 1.2, f"{o*100:.0f}%", ha="center", fontsize=9.5,
                fontweight="semibold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["Xavier", "CHIRPS"], fontsize=9.5)
    ax.set_ylabel("vizinhos no mesmo regime (%)", fontsize=9.5, color=INK_2)
    ax.set_title("Coerência espacial", fontsize=10.5, loc="left",
                 fontweight="semibold", color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", color="#e8e7e2"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # leitura
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 1.0, "A leitura", fontsize=10.5, fontweight="semibold", color=INK,
            va="top")
    ax.text(0, 0.90,
            "Os mapas são espacialmente\n"
            "coerentes (2,7× o acaso) mas\n"
            "NÃO se reproduzem — nem com\n"
            "os mesmos dados.\n\n"
            "A coerência é herdada: pixels\n"
            "vizinhos da chuva têm r = 0,994,\n"
            "logo caem no mesmo regime por\n"
            "construção.\n\n"
            "Não é subtreinamento: com\n"
            "600 épocas (12×) o ARI sobe\n"
            "só de 0,10 para 0,17.\n\n"
            "Aparência máxima,\n"
            "substância mínima.",
            fontsize=9.5, color=INK_2, va="top", linespacing=1.45)

    fig.legend(handles=[Patch(facecolor=c, label=f"regime {i}")
                        for i, c in enumerate(REGIME_COLORS[:k])],
               loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=k,
               frameon=False, fontsize=9)
    meth = ds.attrs.get("method", "bayesian_blocks")
    fig.suptitle("O SOM salva a atribuição de MoV?  —  bacia do Uruguai, "
                 f"{len(lat):,} pixels × 29 MoVs, {k} regimes  ·  binning {meth}",
                 fontsize=13, fontweight="semibold", color=INK, y=0.995)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}")
    print(f"   ARI mesma base = {ari_seed:.3f} · Xavier vs CHIRPS = {ari_src:.3f}")
    print(f"   coerência Xavier {co_x*100:.0f}% (acaso {ch_x*100:.0f}%) · "
          f"CHIRPS {co_c*100:.0f}% (acaso {ch_c*100:.0f}%)")


if __name__ == "__main__":
    main()
