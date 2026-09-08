#!/usr/bin/env python3
"""
Uma única imagem com o PREDEP máximo de cada fonte e a diferença entre elas.

Grade 4x3: uma linha por estação; colunas = alpha max Xavier, alpha max CHIRPS,
diferença (CHIRPS - Xavier). "Alpha máximo" é o max por pixel sobre todos os
MoVs x lags, que é a estatística robusta usada nos comparativos do projeto.

Lê a saída de experiments/09_common_window_uruguai.py (dims season/lag/mov/pixel).

Uso:
  python tools/gen_exp09_predep_max_figure.py \
      --nc results/exp09b_uruguai_corrigido.nc \
      --out plots/exp09b_predep_max.png
"""
import argparse
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                              ListedColormap, TwoSlopeNorm)
from matplotlib.patches import Patch, Polygon

# Magnitude: rampa de uma só matiz (claro->escuro). Diferença: divergente
# azul<->vermelho com cinza neutro no meio, sempre centrada em zero — assim
# "sem diferença" lê como ausência de cor, e nunca arco-íris.
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b"])
CMAP_DIV = LinearSegmentedColormap.from_list("div_bluered", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f8c9c8", "#ef8b8a", "#d33534", "#8a1f1e"])
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
MISSING = "#f0efec"
OCEAN, LAND, LAND_EDGE = "#dbe9f2", "#e9e4d8", "#a39c8c"

# Famílias de MoV. Agrupar é mais honesto que mostrar 29 índices: trocar de MoV
# DENTRO de uma família não é mudança de fenômeno, é o mesmo sinal por outro
# proxy. A Estratosfera (só QBO) vence 0.1% das vezes e entra em "Índico/outros".
FAMILIES = {
    "ENSO":               ["MEI", "NIN03", "NIN034", "NIN04", "NIN12", "ONI", "SOI"],
    "Atlântico":          ["AMO", "ATL3", "IASAS", "MGI", "SAODI", "SASDI", "TNA", "TSA"],
    "Modos anulares":     ["AAO", "AO", "NAO", "PSA1", "PSA2"],
    "Pacífico extratrop.": ["EPNP", "PDO", "PNA", "WP"],
    "Índico/outros":      ["IOD", "IOSD", "SIOD", "ITSMRG2", "QBO"],
}
# Matizes dos 5 primeiros slots categóricos, pulando o amarelo (que não se
# separa do laranja quando todos os pares aparecem juntos, como num mapa).
FAM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
# Concordância: rampa ordinal (mais escuro = mais concordante), não categórica.
AGREE_COLORS = ["#cde2fb", "#6da7ec", "#0d366b"]
AGREE_LABELS = ["MoV diferente", "mesma família", "mesmo MoV"]
SA_BORDERS = Path(__file__).resolve().parent.parent / "data/reference/south_america_110m.geojson"


def load_borders():
    """Contornos dos países (Natural Earth 110m) versionados no repo.

    Lidos com `json` — `geopandas.datasets` foi removido no GeoPandas 1.0.
    """
    if not SA_BORDERS.exists():
        print(f"⚠ {SA_BORDERS.name} ausente — mapas sem contorno de país")
        return []
    gj = json.loads(SA_BORDERS.read_text())
    out = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        out.append([np.asarray(r) for poly in polys for r in poly])
    return out


def add_basemap(ax, extent, borders):
    """Oceano, terra e fronteiras por trás dos dados."""
    ax.set_facecolor(OCEAN)
    for rings in borders:
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                 edgecolor=LAND_EDGE, linewidth=0.7, zorder=1))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
# NaN transparente: fora da bacia o mapa-base (terra/oceano) deve aparecer
for cm in (CMAP_SEQ, CMAP_DIV):
    cm.set_bad((0, 0, 0, 0))


def to_grid(values, lat, lon):
    """Devolve os valores por pixel ao formato (lat, lon) da grade 0.1°."""
    ulat = np.unique(lat)
    ulon = np.unique(lon)
    g = np.full((len(ulat), len(ulon)), np.nan, dtype=float)
    g[np.searchsorted(ulat, lat), np.searchsorted(ulon, lon)] = values
    extent = [ulon.min() - 0.05, ulon.max() + 0.05,
              ulat.min() - 0.05, ulat.max() + 0.05]
    return g, extent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp09b_uruguai_corrigido.nc")
    ap.add_argument("--out", default="plots/exp09b_predep_max.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--agree-threshold", type=float, default=0.10,
                    help="|Δ α| abaixo do qual as fontes são tidas como concordantes")
    args = ap.parse_args()

    ds = xr.open_dataset(args.nc)
    seasons = [str(s) for s in ds.season.values]
    lat, lon = ds.latitude.values, ds.longitude.values

    # alpha máximo por pixel, sobre MoVs x lags
    n_px = ds.sizes["pixel"]
    amax = {}
    for src in ("xavier", "chirps"):
        a = ds[f"alpha_{src}"].values.reshape(len(seasons), -1, n_px)
        amax[src] = np.nanmax(a, axis=1)
    diff = amax["chirps"] - amax["xavier"]

    # 4ª coluna: consenso — a média das duas fontes, mostrada apenas onde elas
    # concordam (|Δ| < limiar). É a média, não uma das fontes, porque onde as
    # duas concordam não há razão para privilegiar nenhuma.
    thr = args.agree_threshold
    agree = np.abs(diff) < thr
    consensus = np.where(agree, (amax["xavier"] + amax["chirps"]) / 2, np.nan)

    # MoV vencedor por pixel e sua família
    movs = [str(m) for m in ds.mov_name.values]
    fam_of = {m: i for i, (f, ms) in enumerate(FAMILIES.items()) for m in ms}
    missing = set(movs) - set(fam_of)
    if missing:
        raise ValueError(f"MoVs sem família: {sorted(missing)}")
    fam_idx = np.array([fam_of[m] for m in movs])

    best_mov, best_fam = {}, {}
    for src in ("xavier", "chirps"):
        a = ds[f"alpha_{src}"].values.reshape(len(seasons), -1, n_px)
        b = np.argmax(np.where(np.isfinite(a), a, -np.inf), axis=1)
        best_mov[src] = b % len(movs)          # eixo achatado é (lag, mov)
        best_fam[src] = fam_idx[best_mov[src]]

    same_mov = best_mov["xavier"] == best_mov["chirps"]
    same_fam = best_fam["xavier"] == best_fam["chirps"]
    agreement = np.where(same_mov, 2, np.where(same_fam, 1, 0))
    # mapa final: família só onde as duas fontes concordam no VALOR e na FAMÍLIA
    final_fam = np.where(agree & same_fam, best_fam["xavier"], np.nan)

    # ── bloco COMBINADO: colapsa as 4 estações ──────────────────────────────
    # α combinado = máximo sobre estação × lag × MoV (o "melhor caso" do pixel).
    # MoV combinado = quem produziu esse máximo.
    comb_amax, comb_mov, comb_fam = {}, {}, {}
    for src in ("xavier", "chirps"):
        flat = ds[f"alpha_{src}"].values.reshape(-1, n_px)   # (season*lag*mov, px)
        comb_amax[src] = np.nanmax(flat, axis=0)
        comb_mov[src] = np.argmax(np.where(np.isfinite(flat), flat, -np.inf),
                                  axis=0) % len(movs)
        comb_fam[src] = fam_idx[comb_mov[src]]
    comb_diff = comb_amax["chirps"] - comb_amax["xavier"]
    comb_agree = np.abs(comb_diff) < thr
    comb_cons = np.where(comb_agree,
                         (comb_amax["xavier"] + comb_amax["chirps"]) / 2, np.nan)
    comb_same_fam = comb_fam["xavier"] == comb_fam["chirps"]
    comb_final = np.where(comb_agree & comb_same_fam, comb_fam["xavier"], np.nan)
    # em quantas das 4 estações as fontes concordam (valor E família)
    n_seasons_agree = (agree & same_fam).sum(axis=0).astype(float)

    # escalas compartilhadas entre estações, para os painéis serem comparáveis
    vmax = float(np.nanpercentile(np.concatenate(
        [amax["xavier"].ravel(), amax["chirps"].ravel()]), 99))
    vmin = float(np.nanpercentile(np.concatenate(
        [amax["xavier"].ravel(), amax["chirps"].ravel()]), 1))
    dlim = float(np.nanpercentile(np.abs(diff), 98))

    borders = load_borders()
    _, data_extent = to_grid(amax["xavier"][0], lat, lon)
    mx, my = 1.6, 1.1                      # margem para o contexto aparecer
    map_extent = [data_extent[0] - mx, data_extent[1] + mx,
                  data_extent[2] - my, data_extent[3] + my]

    cmap_fam = ListedColormap(FAM_COLORS); cmap_fam.set_bad((0, 0, 0, 0))
    norm_fam = BoundaryNorm(np.arange(-0.5, len(FAMILIES) + 0.5), len(FAM_COLORS))
    cmap_agr = ListedColormap(AGREE_COLORS); cmap_agr.set_bad((0, 0, 0, 0))
    norm_agr = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

    cmap_cnt = ListedColormap(["#eaf2fd", "#cde2fb", "#9ec5f4", "#3987e5", "#0d366b"])
    cmap_cnt.set_bad((0, 0, 0, 0))
    norm_cnt = BoundaryNorm(np.arange(-0.5, 5.5), 5)

    nrow = 2 * len(seasons) + 2          # +1 bloco combinado
    fig, axes = plt.subplots(nrow, 4, figsize=(16.0, 2.9 * nrow))
    titles_a = ["Xavier", "CHIRPS", "diferença (CHIRPS − Xavier)",
                f"consenso  |Δ| < {thr:g}"]
    titles_m = ["MoV vencedor — Xavier", "MoV vencedor — CHIRPS",
                "concordância do MoV", "MoV final (valor + família)"]
    im_seq = im_div = None
    for i, s in enumerate(seasons):
        panels = [
            # linha do α
            [(amax["xavier"][i], "seq"), (amax["chirps"][i], "seq"),
             (diff[i], "div"), (consensus[i], "seq")],
            # linha do MoV
            [(best_fam["xavier"][i].astype(float), "fam"),
             (best_fam["chirps"][i].astype(float), "fam"),
             (agreement[i].astype(float), "agr"), (final_fam[i], "fam")],
        ]
        for sub, row in enumerate(panels):
            r = 2 * i + sub
            for c, (vals, kind) in enumerate(row):
                ax = axes[r, c]
                g, extent = to_grid(vals, lat, lon)
                add_basemap(ax, map_extent, borders)
                kw = {"seq": dict(cmap=CMAP_SEQ, vmin=vmin, vmax=vmax),
                      "div": dict(cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -dlim, dlim)),
                      "fam": dict(cmap=cmap_fam, norm=norm_fam),
                      "agr": dict(cmap=cmap_agr, norm=norm_agr)}[kind]
                im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=extent,
                               interpolation="nearest", aspect="equal", zorder=2, **kw)
                if kind == "div":
                    im_div = im
                elif kind == "seq":
                    im_seq = im
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7.5, length=2, color=INK_3, colors=INK_2)
                ax.set_xticks(np.arange(-58, -48, 2))
                ax.set_yticks(np.arange(-32, -25, 2))
                ax.set_xticklabels([f"{abs(v):.0f}°W" for v in np.arange(-58, -48, 2)]
                                   if r == nrow - 1 else [])
                ax.set_yticklabels([f"{abs(v):.0f}°S" for v in np.arange(-32, -25, 2)]
                                   if c == 0 else [])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#d8d7d2"); sp.set_linewidth(0.8)
                if r == 0:
                    ax.set_title(titles_a[c], color=INK, fontsize=10.5,
                                 fontweight="semibold", pad=8)
                if sub == 1 and i == 0:
                    ax.set_title(titles_m[c], color=INK_2, fontsize=9.5, pad=6)
                if c == 3:
                    frac = (np.mean(agree[i]) if sub == 0
                            else np.mean(agree[i] & same_fam[i]))
                    ax.text(0.03, 0.05, f"{100*frac:.0f}% dos pixels",
                            transform=ax.transAxes, fontsize=9, color=INK,
                            fontweight="semibold", zorder=5,
                            bbox=dict(facecolor="white", alpha=0.75,
                                      edgecolor="none", pad=1.8))
                if c == 0:
                    ax.text(-0.24, 0.5, s if sub == 0 else "", transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=13,
                            fontweight="semibold", color=INK)
                    ax.text(-0.10, 0.5, "α" if sub == 0 else "MoV",
                            transform=ax.transAxes, rotation=90, va="center",
                            ha="center", fontsize=9, color=INK_2)

    # bloco combinado, nas duas últimas linhas
    comb_panels = [
        [(comb_amax["xavier"], "seq"), (comb_amax["chirps"], "seq"),
         (comb_diff, "div"), (comb_cons, "seq")],
        [(comb_fam["xavier"].astype(float), "fam"),
         (comb_fam["chirps"].astype(float), "fam"),
         (n_seasons_agree, "cnt"), (comb_final, "fam")],
    ]
    comb_titles = [
        ["Xavier — todas as estações", "CHIRPS — todas as estações",
         "diferença", f"consenso  |Δ| < {thr:g}"],
        ["MoV vencedor — Xavier", "MoV vencedor — CHIRPS",
         "em quantas estações concordam", "MoV final combinado"],
    ]
    for sub, row in enumerate(comb_panels):
        r = 2 * len(seasons) + sub
        for c, (vals, kind) in enumerate(row):
            ax = axes[r, c]
            g, extent = to_grid(vals, lat, lon)
            add_basemap(ax, map_extent, borders)
            kw = {"seq": dict(cmap=CMAP_SEQ, vmin=vmin, vmax=vmax),
                  "div": dict(cmap=CMAP_DIV, norm=TwoSlopeNorm(0, -dlim, dlim)),
                  "fam": dict(cmap=cmap_fam, norm=norm_fam),
                  "cnt": dict(cmap=cmap_cnt, norm=norm_cnt)}[kind]
            im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=extent,
                           interpolation="nearest", aspect="equal", zorder=2, **kw)
            if kind == "cnt":
                im_cnt = im
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7.5, length=2, color=INK_3, colors=INK_2)
            ax.set_xticks(np.arange(-58, -48, 2))
            ax.set_yticks(np.arange(-32, -25, 2))
            ax.set_xticklabels([f"{abs(v):.0f}°W" for v in np.arange(-58, -48, 2)]
                               if r == nrow - 1 else [])
            ax.set_yticklabels([f"{abs(v):.0f}°S" for v in np.arange(-32, -25, 2)]
                               if c == 0 else [])
            for sp in ax.spines.values():
                sp.set_edgecolor("#8b8a84"); sp.set_linewidth(1.4)
            ax.set_title(comb_titles[sub][c], color=INK if sub == 0 else INK_2,
                         fontsize=10.5 if sub == 0 else 9.5,
                         fontweight="semibold" if sub == 0 else "normal", pad=7)
            if c == 3:
                frac = np.mean(comb_agree) if sub == 0 else np.mean(comb_agree & comb_same_fam)
                ax.text(0.03, 0.05, f"{100*frac:.0f}% dos pixels",
                        transform=ax.transAxes, fontsize=9, color=INK,
                        fontweight="semibold", zorder=5,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.8))
            if c == 0:
                ax.text(-0.24, 0.5, "COMBINADO" if sub == 0 else "",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=11.5, fontweight="semibold", color=INK)
                ax.text(-0.10, 0.5, "α" if sub == 0 else "MoV", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9, color=INK_2)

    fig.subplots_adjust(left=0.050, right=0.905, top=0.955, bottom=0.075,
                        wspace=0.06, hspace=0.14)
    cax1 = fig.add_axes([0.920, 0.60, 0.011, 0.28])
    cb1 = fig.colorbar(im_seq, cax=cax1)
    cb1.set_label("α máximo", fontsize=9, color=INK_2)
    cax2 = fig.add_axes([0.920, 0.22, 0.011, 0.28])
    cb2 = fig.colorbar(im_div, cax=cax2)
    cb2.set_label("Δ α máximo", fontsize=9, color=INK_2)
    for cb in (cb1, cb2):
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, length=2)

    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=f)
                        for c, f in zip(FAM_COLORS, FAMILIES)],
               loc="lower center", bbox_to_anchor=(0.42, 0.028), ncol=5,
               frameon=False, fontsize=9.5, title="família do MoV vencedor",
               title_fontsize=9.5)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l)
                        for c, l in zip(AGREE_COLORS, AGREE_LABELS)],
               loc="lower center", bbox_to_anchor=(0.42, -0.004), ncol=3,
               frameon=False, fontsize=9.5)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=str(k))
                        for k, c in enumerate(["#eaf2fd", "#cde2fb", "#9ec5f4",
                                               "#3987e5", "#0d366b"])],
               loc="lower center", bbox_to_anchor=(0.80, 0.004), ncol=5,
               frameon=False, fontsize=9.5,
               title="nº de estações em que as fontes concordam", title_fontsize=9.5)

    est = ds.attrs.get("exclude_self_pairs", "?")
    meth = ds.attrs.get("method", "bayesian_blocks")
    sel = ("ward (Algorithm 1 do artigo)" if meth == "ward"
           else "bayesian_blocks (fora do artigo)")
    fig.suptitle(f"PREDEP máximo por pixel — bacia do Uruguai, 1996–2024\n"
                 f"{ds.sizes['pixel']:,} pixels · {ds.sizes['mov_name']} MoVs × "
                 f"{ds.sizes['lag']} lags · binning {sel} · "
                 f"pares i==j {'excluídos' if est == 'True' else 'incluídos'}",
                 fontsize=13, fontweight="semibold", color=INK, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}")
    for r, s in enumerate(seasons):
        print(f"   {s}: Xavier {np.nanmean(amax['xavier'][r]):.3f} · "
              f"CHIRPS {np.nanmean(amax['chirps'][r]):.3f} · "
              f"Δ {np.nanmean(diff[r]):+.3f} · "
              f"corr {np.corrcoef(amax['xavier'][r], amax['chirps'][r])[0,1]:+.3f} · "
              f"consenso {100*np.mean(agree[r]):.0f}% · "
              f"mesmo MoV {100*np.mean(same_mov[r]):.0f}% · "
              f"mesma família {100*np.mean(same_fam[r]):.0f}% · "
              f"final {100*np.mean(agree[r] & same_fam[r]):.0f}%")
    print(f"   COMBINADO: Xavier {np.nanmean(comb_amax['xavier']):.3f} · "
          f"CHIRPS {np.nanmean(comb_amax['chirps']):.3f} · "
          f"consenso {100*np.mean(comb_agree):.0f}% · "
          f"mesma família {100*np.mean(comb_same_fam):.0f}% · "
          f"final {100*np.mean(comb_agree & comb_same_fam):.0f}%")
    print("   pixels por nº de estações concordantes: "
          + ", ".join(f"{k}: {100*np.mean(n_seasons_agree == k):.0f}%" for k in range(5)))


if __name__ == "__main__":
    main()
