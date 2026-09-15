"""
Figuras de uma bacia, a partir do NetCDF que o `main.py` produz.

Duas figuras por bacia:

* `<bacia>_predep.png` — por estação: α de cada fonte no candidato escolhido, a
  diferença entre elas, o r², o consenso e o MoV predominante em cada pixel.
* `<bacia>_som.png` — os regimes do SOM por trimestre e por fonte, e a
  assinatura de MoV de cada regime.

**O consenso vem com o seu piso.** Ao lado de cada percentual aparece o que o
melhor surrogate alcançou, e a cor diz se o observado o supera. Sem isso o
número engana: nos surrogates do Uruguai o consenso chegou a 64%, então um
"65% dos pixels" sozinho não é evidência de nada.

Arquivos gerados antes dos surrogates ainda funcionam — a figura detecta a
ausência da dimensão `shift` e troca o rodapé por uma ressalva explícita, em vez
de mostrar um número sem referência como se tivesse uma.

Compartilha paleta e mapa-base com `predominancia_plots` via
`predominancia_base`, para as figuras poderem ser lidas juntas.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from .predominancia_base import (BORDA_EIXO, CMAP_DIV, CMAP_SEQ, INK, INK_2,
                                 INK_3, MOV_OUTROS, MOV_PALETTE, RED,
                                 REGIME_COLORS, VERDE, caixa, categorical_rgba,
                                 draw_basemap, fmt_lat, fmt_lon, load_borders,
                                 open_ds, strip_axes, to_grid,
                                 to_grid_bool, to_grid_rgba)
from .predominancia_plots import CMAP_ASSINATURA

RESSALVA_COM_NULO = (
    "Cada surrogate escolhe o SEU candidato pelo mesmo critério, então a "
    "comparação é justa. Verde = o observado supera o melhor surrogate; "
    "vermelho = não supera, e aí o número não é evidência.")
RESSALVA_SEM_NULO = (
    "Sem calibração nula: este arquivo não traz surrogates. Um "
    "consenso alto não é evidência sozinho — no Uruguai o melhor surrogate "
    "chegou a 64%. Recalcule para ter o piso.")


def _tem_nulo(ds):
    """O arquivo traz surrogates? Os primeiros .nc por bacia não traziam."""
    return "shift" in ds.dims and ds.sizes["shift"] > 1


def _escolha_ano(concord):
    """Um (lag, MoV) para o ano inteiro, pela predominância média das estações."""
    media = np.nanmean(concord, axis=0)                  # (lag, mov)
    plano = media.ravel()
    k = int(np.nanargmax(np.where(np.isfinite(plano), plano, -np.inf)))
    return divmod(k, media.shape[1])


def plot_bacia_predep(fonte, output_dir="plots/bacias", *, tol=0.10, dpi=150,
                      filename=None) -> Path:
    """α, r², diferença, consenso e MoV predominante — uma linha por estação."""
    ds = open_ds(fonte)
    bacia = ds.attrs.get("bacia", "bacia")
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    lags = ds.lag.values
    alpha = ds["alpha"].values                 # (fonte, est, lag, mov, pixel)
    r2 = ds["r2"].values
    com_nulo = _tem_nulo(ds)
    concord = ds["concordancia"].values
    sl, sm = ds["sel_lag"].values, ds["sel_mov"].values
    # A dimensão `shift` existe sempre que o arquivo vem da pipeline com nulo,
    # mesmo com um só deslocamento (--shifts 0). Testar só "tem mais de um"
    # deixava esse caso tratar (1, estação, ...) como (estação, ...).
    if "shift" in ds.dims:                      # (shift, ...) -> observado
        concord, sl, sm = concord[0], sl[0], sm[0]
    cons_sur = (np.nanmax(ds["consenso"].values[1:], axis=0)   # (estação,)
                if com_nulo else None)
    bd = load_borders()

    li_a, mi_a = _escolha_ano(concord)
    xav = np.concatenate([[alpha[0, si, sl[si], sm[si]] for si in range(len(seasons))],
                          [np.nanmean(alpha[0, :, li_a, mi_a], axis=0)]])
    chi = np.concatenate([[alpha[1, si, sl[si], sm[si]] for si in range(len(seasons))],
                          [np.nanmean(alpha[1, :, li_a, mi_a], axis=0)]])
    r2m = np.concatenate([[np.nanmean(r2[:, si, sl[si], sm[si]], axis=0)
                           for si in range(len(seasons))],
                          [np.nanmean(r2[:, :, li_a, mi_a], axis=(0, 1))]])
    dif = chi - xav
    consenso = np.where(np.abs(dif) < tol, (xav + chi) / 2, np.nan)

    # MoV predominante por pixel: o mesmo critério resolvido localmente
    conc_px = np.minimum(alpha[0], alpha[1])   # (est, lag, mov, pixel)
    n_mov = len(movs)
    bmp = []
    for si in range(len(seasons)):
        c = conc_px[si].reshape(-1, conc_px.shape[-1])
        bmp.append(np.argmax(np.where(np.isfinite(c), c, -np.inf), axis=0) % n_mov)
    c = np.nanmean(conc_px, axis=0).reshape(-1, conc_px.shape[-1])
    bmp.append(np.argmax(np.where(np.isfinite(c), c, -np.inf), axis=0) % n_mov)
    bmp = np.stack(bmp)

    linhas = seasons + ["ANO"]
    rot = [f"{movs[sm[si]]} · lag {lags[sl[si]]}" for si in range(len(seasons))]
    rot.append(f"{movs[mi_a]} · lag {lags[li_a]}")

    vmin = float(np.nanpercentile(np.concatenate([xav, chi]), 1))
    vmax = float(np.nanpercentile(np.concatenate([xav, chi]), 99))
    dmax = max(1e-6, float(np.nanpercentile(np.abs(dif), 99)))
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    r2max = float(np.nanpercentile(r2m, 99))

    cnt = np.bincount(bmp.ravel(), minlength=n_mov)
    top = list(np.argsort(cnt)[::-1][:len(MOV_PALETTE)])
    cor_de = {m: MOV_PALETTE[i] for i, m in enumerate(top)}

    nr, nc = len(linhas), 6
    fig = plt.figure(figsize=(20.6, 3.05 * nr + 2.4))
    # sem suptitle: os mapas começam mais acima
    top_f, bot_f = 0.975, 0.130
    gs = fig.add_gridspec(nr, nc, hspace=0.10, wspace=0.10,
                          left=0.055, right=0.905, top=top_f, bottom=bot_f)
    titulos = ["α Xavier", "α CHIRPS", "diferença (CHIRPS − Xavier)",
               "r² (média das fontes)", f"consenso  |Δα| < {tol:g}",
               "MoV predominante por pixel"]
    im_s = im_d = im_r = None
    for r in range(nr):
        for c_i in range(nc):
            ax = fig.add_subplot(gs[r, c_i])
            if c_i == 5:
                g, ext = to_grid_rgba(
                    categorical_rgba(bmp[r], cor_de, MOV_OUTROS), lat, lon)
            else:
                g, ext = to_grid([xav, chi, dif, r2m, consenso][c_i][r], lat, lon)
            draw_basemap(ax, bd, ext, pad_x=1.4, pad_y=1.0)
            if c_i == 5:
                ax.imshow(g, origin="lower", extent=ext, interpolation="nearest",
                          aspect="equal", zorder=2)
            elif c_i == 2:
                im_d = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                                 cmap=CMAP_DIV, norm=dnorm,
                                 interpolation="nearest", aspect="equal", zorder=2)
            elif c_i == 3:
                im_r = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                                 cmap="YlOrBr", vmin=0, vmax=r2max,
                                 interpolation="nearest", aspect="equal", zorder=2)
            else:
                im_s = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                                 cmap=CMAP_SEQ, vmin=vmin, vmax=vmax,
                                 interpolation="nearest", aspect="equal", zorder=2)
            strip_axes(ax, ticks=True)
            ax.tick_params(labelsize=7, length=2, colors=INK_2)
            ax.xaxis.set_major_formatter(fmt_lon)
            ax.yaxis.set_major_formatter(fmt_lat)
            if r < nr - 1:
                ax.set_xticklabels([])
            if c_i > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(titulos[c_i], fontsize=10.5, fontweight="semibold",
                             color=INK, pad=8)
            if c_i == 0:
                ax.text(-0.32, 0.5, linhas[r], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12.5,
                        fontweight="semibold", color=INK)
                ax.text(-0.21, 0.5, rot[r], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, color=INK_2)
            if c_i in (0, 1):
                caixa(ax, f"α {np.nanmean([xav, chi][c_i][r]):.3f}")
            elif c_i == 3:
                caixa(ax, f"r² {np.nanmean(r2m[r]):.3f}")
            elif c_i == 4:
                pc = 100 * np.mean(np.abs(dif[r]) < tol)
                if com_nulo and r < len(seasons):
                    ps = 100 * cons_sur[r]
                    caixa(ax, f"{pc:.0f}% dos pixels\nsurrogate {ps:.0f}%",
                          cor=VERDE if pc > ps else RED, linespacing=1.35)
                else:
                    caixa(ax, f"{pc:.0f}% dos pixels")
            elif c_i == 5:
                escolhido = sm[r] if r < len(seasons) else mi_a
                caixa(ax, f"{movs[escolhido]} vence em "
                          f"{100*np.mean(bmp[r] == escolhido):.0f}%")

    h = (top_f - bot_f) / nr
    for i, (im, rot_lab) in enumerate((
            (im_s, "α (PREDEP)"), (im_d, "Δα (CHIRPS − Xavier)"), (im_r, "r²"))):
        cax = fig.add_axes([0.917, bot_f + h * (nr - 1.6 - 1.75 * i), 0.009, h * 1.45])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(rot_lab, fontsize=9, color=INK_2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, length=2)

    handles = [Patch(facecolor=cor_de[m], label=movs[m]) for m in top]
    handles.append(Patch(facecolor=MOV_OUTROS, label="demais MoVs"))
    axl = fig.add_axes([0.055, 0.055, 0.850, 0.055])
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=13, frameon=False, fontsize=8.5,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.5)

    fig.text(0.5, 0.012, RESSALVA_COM_NULO if com_nulo else RESSALVA_SEM_NULO,
             ha="center", fontsize=9, color=INK_3, style="italic")
    return _salvar(fig, output_dir, filename or f"{bacia}_predep.png", dpi)


def plot_bacia_som(fonte, output_dir="plots/bacias", *, dpi=150, top_movs=6,
                   filename=None) -> Path:
    """Regimes do SOM por trimestre e fonte, e a assinatura de MoV de cada um."""
    ds = open_ds(fonte)
    bacia = ds.attrs.get("bacia", "bacia")
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    reg = ds["regime"].values
    fp = ds["assinatura"].values
    msk = ds["mascara"].values
    qerr = ds["erro_bmu"].values
    ari = ds["ari_fontes"].values
    k = int(ds.sizes["regime_id"])
    bd = load_borders()

    anom = np.full_like(fp, np.nan)
    for s in range(2):
        for si in range(len(seasons)):
            anom[s, si] = fp[s, si] - np.nanmean(fp[s, si], axis=0)
    lim = max(1e-6, float(np.nanpercentile(np.abs(anom), 98)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    spread = np.nanmean(np.nanstd(anom, axis=2), axis=(0, 1))
    ordem = sorted(np.argsort(spread)[::-1][:top_movs], key=lambda m: -spread[m])

    fig = plt.figure(figsize=(16.4, 12.4))
    gsm = fig.add_gridspec(2, len(seasons), hspace=0.09, wspace=0.10,
                           left=0.062, right=0.985, top=0.968, bottom=0.560)
    # Mesma left/right do gride de mapas: cada lista fica sob a sua estação.
    gsf = fig.add_gridspec(1, len(seasons), wspace=0.05,
                           left=0.062, right=0.985, top=0.430, bottom=0.115)
    cor_de = {i: REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(k)}

    for r in range(2):
        for c, sn in enumerate(seasons):
            ax = fig.add_subplot(gsm[r, c])
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
                caixa(ax, f"ARI entre fontes {ari[c]:.2f}")
            else:
                qi = np.nanmean(qerr[r, c][msk[c]])
                qo = np.nanmean(qerr[r, c][~msk[c]]) if (~msk[c]).any() else np.nan
                caixa(ax, f"erro BMU {qi:.1f} → {qo:.1f}", cor=INK_2, peso="normal")
            if c == 0:
                ax.text(-0.09, 0.5, "Xavier" if r == 0 else "CHIRPS",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=12, fontweight="semibold", color=INK)

    handles = [Patch(facecolor=cor_de[i], label=f"regime {i+1}") for i in range(k)]
    axl = fig.add_axes([0.062, 0.487, 0.923, 0.042])
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=9, handlelength=1.4, columnspacing=1.4, handletextpad=0.5)

    # Só a lista de nomes: nem barra nem número. A barra codificava magnitude,
    # que é justamente o que não se quer aqui — a pergunta é "quais índices
    # caracterizam este regime", não "quanto".
    #
    # DOIS filtros, e os dois são necessários:
    #   contraste > 0  — o índice é mais forte NESTE regime que na média da
    #                    estação. Sozinho, este critério deixava entrar índice
    #                    fraco: em SON o TSA tinha α=0,197, abaixo da média da
    #                    estação (0,256), e aparecia só por ser menos fraco ali.
    #   α > média      — o índice é forte em termos absolutos.
    # Sem o segundo, quem lê a lista sem os valores conclui que todos predizem
    # bem, o que não é verdade.
    for c, sn in enumerate(seasons):
        ax = fig.add_subplot(gsf[0, c])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.0, 1.04, sn, ha="left", va="bottom", fontsize=11.5,
                fontweight="semibold", color=INK, transform=ax.transAxes)
        limiar = float(np.nanmean(fp[0, c]))       # α médio da estação
        passo = 1.0 / k
        for r in range(k):
            v = anom[0, c, r]
            forte = fp[0, c, r]
            pos = [m for m in np.argsort(forte)[::-1]
                   if v[m] > 0 and forte[m] > limiar][:top_movs]
            y = 1.0 - (r + 0.5) * passo
            ax.add_patch(plt.Rectangle((0.0, y - 0.03), 0.035, 0.06,
                                       facecolor=cor_de[r], edgecolor="none",
                                       transform=ax.transAxes, clip_on=False))
            ax.text(0.055, y, f"R{r+1}", va="center", ha="left", fontsize=9.5,
                    fontweight="semibold", color=cor_de[r],
                    transform=ax.transAxes)
            if not pos:
                ax.text(0.14, y, "nenhum índice acima da média da estação",
                        va="center", ha="left", fontsize=8.4, color=INK_3,
                        style="italic", transform=ax.transAxes)
                continue
            # duas linhas: 6 nomes numa só não cabem na largura de uma estação
            metade = (len(pos) + 1) // 2
            ax.text(0.14, y + 0.045, "  ".join(movs[m] for m in pos[:metade]),
                    va="center", ha="left", fontsize=8.6, color=INK,
                    transform=ax.transAxes)
            ax.text(0.14, y - 0.045, "  ".join(movs[m] for m in pos[metade:]),
                    va="center", ha="left", fontsize=8.6, color=INK_2,
                    transform=ax.transAxes)

    fig.text(0.5, 0.052,
             f"Os MoVs que caracterizam cada regime (Xavier): até {top_movs} índices "
             "que são mais fortes naquele regime que na média da estação E têm α acima "
             "da média da estação — os dois critérios, para a lista não incluir índice "
             f"fraco. Ordenados por α. O SOM usa os {len(movs)} de uma vez e não "
             "seleciona nenhum. Valores exatos: main.py --assinaturas.",
             ha="center", fontsize=9, color=INK_2)
    fig.text(0.5, 0.020,
             RESSALVA_COM_NULO if _tem_nulo(ds) else RESSALVA_SEM_NULO,
             ha="center", fontsize=9, color=INK_3, style="italic")
    return _salvar(fig, output_dir, filename or f"{bacia}_som.png", dpi)


def gerar_figuras_bacia(fonte, output_dir="plots/bacias", **kw) -> list:
    """As duas figuras da bacia. Uma que falhe não impede a outra."""
    saidas = []
    for nome, fn in (("predep", plot_bacia_predep), ("som", plot_bacia_som)):
        try:
            saidas.append(fn(fonte, output_dir, **kw))
        except (KeyError, ValueError, OSError) as e:
            print(f"  ⚠ figura {nome}: {type(e).__name__}: {e}")
    return saidas


def _salvar(fig, output_dir, filename, dpi) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    caminho = out / filename
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {caminho}")
    return caminho
