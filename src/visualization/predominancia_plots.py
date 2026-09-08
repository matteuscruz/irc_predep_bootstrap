"""
Figuras da metodologia da predominância, prontas para a pipeline.

Duas funções públicas, uma por figura:

* `plot_predep_max` — α de cada fonte no candidato escolhido, a diferença, o
  consenso e o MoV predominante por pixel. Lê a saída do
  `experiments/22_modal_predep_max_novo.py`.
* `plot_som_regimes` — os regimes do SOM por trimestre e por fonte, em todos os
  pixels, mais a assinatura de MoV de cada regime. Lê a saída do
  `experiments/26_som_completo_assinaturas.py`.

Ambas aceitam um `Dataset` já aberto ou um caminho, devolvem o `Path` do PNG e
não dependem de estado global — é o que a pipeline precisa para chamá-las como
nó. As duas compartilham paleta e mapa-base via `predominancia_base`, para
poderem ser lidas lado a lado.

CLI:
  python -m src.visualization.predominancia_plots predep [--nc ...] [--out-dir plots]
  python -m src.visualization.predominancia_plots som    [--nc ...] [--k 4]
  python -m src.visualization.predominancia_plots todas
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from .predominancia_base import (AZUL, BORDA_EIXO, CINZA, CMAP_DIV, CMAP_SEQ,
                                 INK, INK_2, INK_3, MOV_OUTROS, MOV_PALETTE,
                                 RED, REGIME_COLORS, ROOT, VERDE, caixa,
                                 categorical_rgba, clean_axes, draw_basemap,
                                 fmt_lat, fmt_lon, load_borders, open_ds,
                                 mil, strip_axes, to_grid, to_grid_bool,
                                 to_grid_rgba)

CMAP_ASSINATURA = LinearSegmentedColormap.from_list("assinatura", [
    "#0d366b", "#256abf", "#9ec5f4", "#f0efec", "#f8c9c8", "#d33534", "#8a1f1e"])

DEFAULT_NC_PREDEP = "results/exp22_predep_max_novo.nc"
DEFAULT_NC_SOM = "results/exp26_som_completo.nc"


# ═══════════════════════════════════════════════════════════════════════════
# Figura 1 — PREDEP na metodologia da predominância
# ═══════════════════════════════════════════════════════════════════════════
def plot_predep_max(fonte=DEFAULT_NC_PREDEP, output_dir="plots", *,
                    tol: float = 0.10, filename: str = "exp22_predep_max_novo.png",
                    dpi: int = 150, bacia: str = "bacia do Uruguai",
                    periodo: str = "1996–2024") -> Path:
    """α por fonte, diferença, consenso e MoV predominante — cinco colunas.

    Diferente da versão antiga (`gen_exp09_predep_max_figure.py`), o (lag, MoV)
    é UM só por estação, escolhido na bacia inteira pela predominância com
    concordância, e as duas fontes são lidas NO MESMO candidato. A coluna de
    diferença passa a ser só diferença de valor, não diferença de escolha.

    A quinta coluna resolve o mesmo critério em cada pixel — não é o que decide
    nada, é o retrato de onde a escolha única vale.
    """
    ds = open_ds(fonte)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    lags = ds.lag.values
    bd = load_borders()

    a = ds["alpha"].values[:, 0]                 # (source, season, pixel)
    a_ano = ds["alpha_ano"].values[:, 0]         # (source, pixel)
    sl, sm = ds["sel_lag"].values[0], ds["sel_mov"].values[0]
    sl_a, sm_a = int(ds["sel_lag_ano"].values[0]), int(ds["sel_mov_ano"].values[0])
    cons, cons_a = ds["consenso"].values, ds["consenso_ano"].values
    bmp = np.concatenate([ds["best_mov_px"].values[0],
                          ds["best_mov_px_ano"].values[0][None]], axis=0)
    rep = np.concatenate([ds["representa"].values,
                          [float(np.mean(bmp[-1] == sm_a))]])

    xav = np.concatenate([a[0], a_ano[0][None]], axis=0)
    chi = np.concatenate([a[1], a_ano[1][None]], axis=0)
    dif = chi - xav
    consenso = np.where(np.abs(dif) < tol, (xav + chi) / 2, np.nan)
    linhas = seasons + ["ANO"]
    rot = [f"{movs[sm[i]]} · lag {lags[sl[i]]}" for i in range(len(seasons))]
    rot.append(f"{movs[sm_a]} · lag {lags[sl_a]}")

    vmin = float(np.nanpercentile(np.concatenate([xav, chi]), 1))
    vmax = float(np.nanpercentile(np.concatenate([xav, chi]), 99))
    dmax = float(np.nanpercentile(np.abs(dif), 99))
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    cnt = np.bincount(bmp.ravel(), minlength=len(movs))
    top_movs = list(np.argsort(cnt)[::-1][:len(MOV_PALETTE)])
    cor_de = {m: MOV_PALETTE[i] for i, m in enumerate(top_movs)}

    nr, nc = len(linhas), 5
    fig = plt.figure(figsize=(17.9, 3.05 * nr + 2.6))
    top, bot_maps = 0.945, 0.175
    gsm = fig.add_gridspec(nr, nc, hspace=0.10, wspace=0.10,
                           left=0.062, right=0.905, top=top, bottom=bot_maps)
    gsb = fig.add_gridspec(1, 1, left=0.062, right=0.520, top=0.122, bottom=0.034)

    titulos = ["Xavier", "CHIRPS", "diferença (CHIRPS − Xavier)",
               f"consenso  |Δ| < {tol:g}", "MoV predominante em cada pixel"]
    im_s = im_d = None
    for r in range(nr):
        for c in range(nc):
            ax = fig.add_subplot(gsm[r, c])
            if c == 4:
                g, ext = to_grid_rgba(
                    categorical_rgba(bmp[r], cor_de, MOV_OUTROS), lat, lon)
            else:
                dados = [xav, chi, dif, consenso][c][r]
                g, ext = to_grid(dados, lat, lon)
            draw_basemap(ax, bd, ext, pad_x=1.4, pad_y=1.0)
            if c == 4:
                ax.imshow(g, origin="lower", extent=ext, interpolation="nearest",
                          aspect="equal", zorder=2)
            elif c == 2:
                im_d = ax.imshow(np.ma.masked_invalid(g), origin="lower",
                                 extent=ext, cmap=CMAP_DIV, norm=dnorm,
                                 interpolation="nearest", aspect="equal", zorder=2)
            else:
                im_s = ax.imshow(np.ma.masked_invalid(g), origin="lower",
                                 extent=ext, cmap=CMAP_SEQ, vmin=vmin, vmax=vmax,
                                 interpolation="nearest", aspect="equal", zorder=2)
            strip_axes(ax, ticks=True)
            ax.tick_params(labelsize=7.5, length=2, colors=INK_2)
            ax.xaxis.set_major_formatter(fmt_lon)
            ax.yaxis.set_major_formatter(fmt_lat)
            if r < nr - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(titulos[c], fontsize=10.5, fontweight="semibold",
                             color=INK, pad=8)
            if c == 0:
                ax.text(-0.30, 0.5, linhas[r], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12.5,
                        fontweight="semibold", color=INK)
                ax.text(-0.19, 0.5, rot[r], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, color=INK_2)
            if c in (0, 1):
                caixa(ax, f"α {np.nanmean([xav, chi][c][r]):.3f}")
            elif c == 3:
                pc = 100 * (cons_a[0] if r == nr - 1 else cons[0, r])
                ps = 100 * (np.nanmax(cons_a[1:]) if r == nr - 1
                            else np.nanmax(cons[1:, r]))
                caixa(ax, f"{pc:.0f}% dos pixels\nsurrogate {ps:.0f}%",
                      cor=VERDE if pc > ps else RED, linespacing=1.35)
            elif c == 4:
                caixa(ax, f"{rot[r].split(' · ')[0]} vence em {100*rep[r]:.0f}%")

    h = (top - bot_maps) / nr
    cax = fig.add_axes([0.917, bot_maps + h * (nr - 2.9), 0.010, h * 2.6])
    cb = fig.colorbar(im_s, cax=cax)
    cb.set_label("α (PREDEP)", fontsize=9.5, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)
    cax2 = fig.add_axes([0.917, bot_maps + h * 0.2, 0.010, h * 2.0])
    cb2 = fig.colorbar(im_d, cax=cax2)
    cb2.set_label("Δ α  (CHIRPS − Xavier)", fontsize=9.5, color=INK_2)
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(labelsize=8, length=2)

    ax = fig.add_subplot(gsb[0, 0])
    x = np.arange(nr)
    obs = np.concatenate([cons[0], [cons_a[0]]]) * 100
    sur = np.concatenate([np.nanmax(cons[1:], axis=0), [np.nanmax(cons_a[1:])]]) * 100
    ax.bar(x - 0.16, obs, 0.32,
           color=[VERDE if o > s else AZUL for o, s in zip(obs, sur)],
           label="observado")
    ax.bar(x + 0.16, sur, 0.32, color=RED, alpha=0.85, label="melhor surrogate")
    ax.set_xticks(x)
    ax.set_xticklabels(linhas, fontsize=9)
    ax.set_ylabel(f"pixels com |Δ| < {tol:g} (%)", fontsize=9, color=INK_2)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.34))
    clean_axes(ax)
    venceu = [("no " + linhas[i]) if linhas[i] == "ANO" else linhas[i]
              for i in range(nr) if obs[i] > sur[i]]
    ax.set_title("O consenso entre as fontes só supera o surrogate em "
                 + (" e ".join(venceu) if venceu else "nenhuma linha"),
                 fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=14)

    handles = [Patch(facecolor=cor_de[m], label=movs[m]) for m in top_movs]
    handles.append(Patch(facecolor=MOV_OUTROS, label="demais MoVs"))
    axl = fig.add_axes([0.575, 0.030, 0.330, 0.098])
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=8.5,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.5,
               title=f"MoV predominante por pixel — a escolha da bacia vence em "
                     f"{100*rep.min():.0f}% a {100*rep.max():.0f}% deles",
               title_fontsize=9.5)

    fig.suptitle(f"PREDEP — {bacia}, {periodo} · um MoV por estação, escolhido "
                 "pela predominância com concordância\n"
                 f"{mil(len(lat))} pixels · {len(movs)} MoVs × {len(lags)} lags · "
                 "estimador do artigo, partição ward, n_boot = ⌈n log n⌉",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.987)
    fig.text(0.5, 0.006,
             "As duas fontes são lidas no MESMO candidato, então a coluna de "
             "diferença é só diferença de valor — na versão anterior cada pixel e "
             "cada fonte escolhiam o seu próprio máximo entre "
             f"{len(movs)*len(lags)} candidatos.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    return _salvar(fig, output_dir, filename, dpi)


# ═══════════════════════════════════════════════════════════════════════════
# Figura 2 — SOM por trimestre e fonte, com assinatura de MoV
# ═══════════════════════════════════════════════════════════════════════════
def plot_som_regimes(fonte=DEFAULT_NC_SOM, output_dir="plots", *, k: int = 4,
                     top_movs: int = 10, filename: str | None = None,
                     dpi: int = 150, bacia: str = "bacia do Uruguai") -> Path:
    """Regimes do SOM em todos os pixels + a assinatura de MoV de cada regime.

    O SOM treina só na máscara de concordância e projeta todo pixel no neurônio
    mais próximo — a máscara governa o ajuste, não a atribuição. Os pixels fora
    dela saem hachurados, porque ali a atribuição é extrapolação.

    A assinatura é o α médio nativo de cada MoV nos pixels do regime, mostrado
    como anomalia contra a média da estação: um índice com α alto em todos os
    regimes não distingue nada e aparece neutro. Um regime é um perfil sobre os
    29 índices, não um índice vencedor.
    """
    ds = open_ds(fonte)
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    ks = [int(v) for v in ds.k.values]
    if k not in ks:
        raise ValueError(f"k={k} não está no arquivo; disponíveis: {ks}")
    ki = ks.index(k)
    reg = ds["regime"].values[ki]                 # (source, season, pixel)
    fp = ds["assinatura"].values[ki]              # (source, season, regime, mov)
    msk = ds["mascara"].values
    ari_m, ari_o = ds["ari_mascara"].values[ki], ds["ari_fora"].values[ki]
    qerr = ds["erro_quantizacao"].values[ki]
    bd = load_borders()
    filename = filename or f"exp26_som_completo_k{k}.png"

    anom = np.full_like(fp, np.nan)
    for s in range(2):
        for si in range(len(seasons)):
            anom[s, si, :k] = fp[s, si, :k] - np.nanmean(fp[s, si, :k], axis=0)
    lim = float(np.nanpercentile(np.abs(anom[:, :, :k]), 98))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    spread = np.nanmean(np.nanstd(anom[:, :, :k], axis=2), axis=(0, 1))
    top = sorted(np.argsort(spread)[::-1][:top_movs], key=lambda m: -spread[m])

    fig = plt.figure(figsize=(15.6, 12.2))
    gsm = fig.add_gridspec(2, len(seasons), hspace=0.09, wspace=0.10,
                           left=0.062, right=0.985, top=0.925, bottom=0.560)
    gsf = fig.add_gridspec(1, len(seasons), wspace=0.14,
                           left=0.062, right=0.940, top=0.435, bottom=0.105)

    for r in range(2):
        for c, sn in enumerate(seasons):
            ax = fig.add_subplot(gsm[r, c])
            cor_de = {i: REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(k)}
            g, ext = to_grid_rgba(categorical_rgba(reg[r, c], cor_de, MOV_OUTROS),
                                  lat, lon)
            draw_basemap(ax, bd, ext)
            ax.imshow(g, origin="lower", extent=ext, interpolation="nearest",
                      aspect="equal", zorder=2)
            fora = to_grid_bool(~msk[c], lat, lon) & (g[..., 3] > 0)
            ax.contourf(np.linspace(ext[0], ext[1], fora.shape[1]),
                        np.linspace(ext[2], ext[3], fora.shape[0]),
                        fora.astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"], zorder=3)
            strip_axes(ax)
            if r == 0:
                ax.set_title(sn, fontsize=12, fontweight="semibold", color=INK, pad=7)
                caixa(ax, f"ARI {ari_m[c]:.2f} na máscara · {ari_o[c]:.2f} fora")
            else:
                qi = np.nanmean(qerr[r, c][msk[c]])
                qo = np.nanmean(qerr[r, c][~msk[c]])
                caixa(ax, f"erro BMU {qi:.1f} → {qo:.1f}", cor=INK_2, peso="normal")
            if c == 0:
                ax.text(-0.09, 0.5, "Xavier" if r == 0 else "CHIRPS",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=12, fontweight="semibold", color=INK)

    handles = [Patch(facecolor=REGIME_COLORS[i % len(REGIME_COLORS)],
                     label=f"regime {i+1}") for i in range(k)]
    handles.append(Patch(facecolor="white", edgecolor=INK_3, hatch="////",
                         label="fora da máscara — só projetado, não treinado"))
    axl = fig.add_axes([0.062, 0.487, 0.923, 0.042])
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=9, handlelength=1.4, columnspacing=1.4, handletextpad=0.5)

    im = None
    for c, sn in enumerate(seasons):
        ax = fig.add_subplot(gsf[0, c])
        M = anom[0, c][:k][:, top].T                      # (mov, regime) — Xavier
        im = ax.imshow(M, cmap=CMAP_ASSINATURA, norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"R{i+1}" for i in range(k)], fontsize=9)
        for i in range(k):
            ax.get_xticklabels()[i].set_color(REGIME_COLORS[i % len(REGIME_COLORS)])
            ax.get_xticklabels()[i].set_fontweight("semibold")
        if c == 0:
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([movs[m] for m in top], fontsize=8.5)
        else:
            ax.set_yticks([])
        ax.set_title(sn, fontsize=10.5, fontweight="semibold", color=INK, pad=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDA_EIXO)
        ax.tick_params(length=2)
        for i in range(k):                                # marca o dominante
            j = int(np.nanargmax(np.abs(M[:, i])))
            ax.add_patch(plt.Rectangle((i - .5, j - .5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.6, zorder=4))

    cax = fig.add_axes([0.950, 0.135, 0.010, 0.250])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("α do regime − α médio da estação", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(f"SOM por trimestre e por fonte, em todos os {mil(len(lat))} pixels "
                 f"— {bacia}, k = {k} regimes\n"
                 "treinado só onde Xavier e CHIRPS concordam, depois projetado na "
                 "bacia inteira",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.983)
    fig.text(0.5, 0.058,
             f"Assinatura (Xavier): os {top_movs} MoVs que mais separam os regimes, "
             "ordenados por poder de separação. O quadrado preto marca o MoV "
             "dominante de cada regime.",
             ha="center", fontsize=9, color=INK_2)
    fig.text(0.5, 0.020,
             "O erro BMU é a distância ao neurônio vencedor: ele sobe fora da "
             "máscara, que é onde a atribuição é extrapolação. Um regime é um perfil "
             f"sobre os {len(movs)} índices — o MoV destacado é o que mais o "
             "distingue, não o único que atua.",
             ha="center", fontsize=9, color=INK_3, style="italic")

    return _salvar(fig, output_dir, filename, dpi)


def gerar_todas(output_dir="plots", *, nc_predep=DEFAULT_NC_PREDEP,
                nc_som=DEFAULT_NC_SOM, k: int = 4, dpi: int = 150) -> list[Path]:
    """As duas figuras de uma vez — o ponto de entrada da pipeline.

    Uma figura que falta o `.nc` não derruba a outra: o erro é reportado e a
    lista devolvida traz só o que saiu.
    """
    saidas = []
    for nome, fn, kw in (("PREDEP", plot_predep_max, {"fonte": nc_predep}),
                         ("SOM", plot_som_regimes, {"fonte": nc_som, "k": k})):
        try:
            saidas.append(fn(output_dir=output_dir, dpi=dpi, **kw))
        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            print(f"  ⚠ {nome}: {type(e).__name__}: {e}")
    return saidas


def _salvar(fig, output_dir, filename, dpi) -> Path:
    out = Path(output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    caminho = out / filename
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {caminho}")
    return caminho


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figura", choices=["predep", "som", "todas"], nargs="?",
                    default="todas")
    ap.add_argument("--nc", help="sobrescreve o .nc da figura escolhida")
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--k", type=int, default=4, help="nº de regimes na figura do SOM")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.figura == "predep":
        plot_predep_max(args.nc or DEFAULT_NC_PREDEP, args.out_dir, dpi=args.dpi)
    elif args.figura == "som":
        plot_som_regimes(args.nc or DEFAULT_NC_SOM, args.out_dir, k=args.k,
                         dpi=args.dpi)
    else:
        gerar_todas(args.out_dir, k=args.k, dpi=args.dpi)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
