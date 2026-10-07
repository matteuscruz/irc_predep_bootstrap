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
                      filename=None, metrica="alpha") -> Path:
    """Métrica no par escolhido, a outra métrica, diferença, consenso e MoV — por estação.

    `metrica="alpha"` é a figura do PREDEP; `metrica="r2"` a da regressão, no mesmo
    formato. Na do r², o par é escolhido pelo PRÓPRIO r² (mesma predominância com
    concordância) e a tolerância do consenso é a mesma regra relativa dos mapas do
    Brasil — 0,1 absoluto no r² deixaria passar quase todos os pixels. O r² não tem
    substitutos gravados, então essa figura não traz calibração nula.
    """
    from src.analysis.grid_predep import escolher_por_predominancia
    from src.pipeline.montagem import tolerancia_da_metrica

    if metrica not in ("alpha", "r2"):
        raise ValueError("metrica deve ser 'alpha' ou 'r2'")
    ds = open_ds(fonte)
    bacia = ds.attrs.get("bacia", "bacia")
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    lags = ds.lag.values
    a_cubo = ds["alpha"].values                # (fonte, est, lag, mov, pixel)
    r_cubo = ds["r2"].values
    if metrica == "alpha":
        alpha, r2 = a_cubo, r_cubo             # principal, secundária
        com_nulo = _tem_nulo(ds)
        concord = ds["concordancia"].values
        sl, sm = ds["sel_lag"].values, ds["sel_mov"].values
        # A dimensão `shift` existe sempre que o arquivo vem da pipeline com nulo,
        # mesmo com um só deslocamento (--shifts 0). Testar só "tem mais de um"
        # deixava esse caso tratar (1, estação, ...) como (estação, ...).
        if "shift" in ds.dims:                  # (shift, ...) -> observado
            concord, sl, sm = concord[0], sl[0], sm[0]
    else:
        alpha, r2 = r_cubo, a_cubo             # principal = r², secundária = α
        com_nulo = False
        sl, sm, concord = escolher_por_predominancia(r_cubo)
        tol = tolerancia_da_metrica(a_cubo, r_cubo, tol)
    cons_sur = (np.nanmax(ds["consenso"].values[1:], axis=0)   # (estação,)
                if com_nulo else None)
    nome_p, nome_s = ("α", "r²") if metrica == "alpha" else ("r²", "α")
    cmap_p, cmap_s = (CMAP_SEQ, "YlOrBr") if metrica == "alpha" else ("YlOrBr", CMAP_SEQ)
    casas = 3 if metrica == "alpha" else 4
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
    r2min = 0.0 if metrica == "alpha" else float(np.nanpercentile(r2m, 1))

    cnt = np.bincount(bmp.ravel(), minlength=n_mov)
    top = list(np.argsort(cnt)[::-1][:len(MOV_PALETTE)])
    cor_de = {m: MOV_PALETTE[i] for i, m in enumerate(top)}

    nr, nc = len(linhas), 6
    fig = plt.figure(figsize=(20.6, 3.05 * nr + 2.4))
    # sem suptitle: os mapas começam mais acima
    top_f, bot_f = 0.975, 0.130
    gs = fig.add_gridspec(nr, nc, hspace=0.10, wspace=0.10,
                          left=0.055, right=0.905, top=top_f, bottom=bot_f)
    titulos = [f"{nome_p} Xavier", f"{nome_p} CHIRPS", "diferença (CHIRPS − Xavier)",
               f"{nome_s} (média das fontes)", f"consenso  |Δ{nome_p}| < {tol:.3g}",
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
                                 cmap=cmap_s, vmin=r2min, vmax=r2max,
                                 interpolation="nearest", aspect="equal", zorder=2)
            else:
                im_s = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                                 cmap=cmap_p, vmin=vmin, vmax=vmax,
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
                caixa(ax, f"{nome_p} {np.nanmean([xav, chi][c_i][r]):.{casas}f}")
            elif c_i == 3:
                caixa(ax, f"{nome_s} {np.nanmean(r2m[r]):.{7 - casas}f}")
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
    rotulos_cb = ((im_s, "α (PREDEP)"), (im_d, "Δα (CHIRPS − Xavier)"), (im_r, "r²"))
    if metrica == "r2":
        rotulos_cb = ((im_s, "r² (regressão)"), (im_d, "Δr² (CHIRPS − Xavier)"),
                      (im_r, "α (PREDEP)"))
    for i, (im, rot_lab) in enumerate(rotulos_cb):
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

    ressalva = (RESSALVA_COM_NULO if com_nulo else RESSALVA_SEM_NULO)
    if metrica == "r2":
        ressalva = ("Par escolhido pelo próprio r² (predominância com concordância). "
                    f"Consenso com tolerância relativa ({tol:.3g}, a mesma regra do α). "
                    "Sem calibração nula: o r² dos substitutos não é gravado no .nc.")
    fig.text(0.5, 0.012, ressalva, ha="center", fontsize=9, color=INK_3, style="italic")
    padrao = f"{bacia}_predep.png" if metrica == "alpha" else f"{bacia}_regressao.png"
    return _salvar(fig, output_dir, filename or padrao, dpi)


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


def concordancia_regimes(ds):
    """Onde Xavier e CHIRPS caem no mesmo regime, e quanto disso o acaso explica.

    Os rótulos do CHIRPS já vêm alinhados aos do Xavier (Hungarian, em
    `som_regimes`), então "mesmo regime" é comparar o número. O acaso é o de
    duas partições independentes com os mesmos tamanhos de regime: Σ p_x(k)·p_c(k)
    por estação; a contagem de 0 a 4 estações segue uma Poisson-binomial disso.

    ATENÇÃO: é o acaso de rótulos independentes, não o nulo do MoV. As duas fontes
    medem a mesma chuva, então concordam acima dele mesmo sem sinal nenhum do
    índice — esse outro nulo só os surrogates dão.
    """
    reg = ds["regime"].values                            # (fonte, estação, pixel)
    k = int(ds.sizes["regime_id"])
    concorda = (reg[0] == reg[1]) & (reg[0] >= 0)        # (estação, pixel)
    acaso = []
    for si in range(reg.shape[1]):
        px = np.bincount(reg[0, si][reg[0, si] >= 0], minlength=k) / reg.shape[-1]
        pc = np.bincount(reg[1, si][reg[1, si] >= 0], minlength=k) / reg.shape[-1]
        acaso.append(float((px * pc).sum()))
    dist = np.array([1.0])
    for p in acaso:
        dist = np.convolve(dist, [1 - p, p])
    n = concorda.sum(axis=0)
    return {"concorda": concorda, "n_estacoes": n, "acaso": np.array(acaso),
            "dist_obs": np.bincount(n, minlength=reg.shape[1] + 1) / n.size,
            "dist_acaso": dist}


def _campos_concordados(cubo, seasons, movs, lags):
    """Par predominante de cada estação e do ano, e os campos das duas fontes nele.

    `cubo` é `(fonte, estação, lag, MoV, pixel)` — de α ou de r². O critério é o
    mesmo para as duas métricas: argmax da média espacial de min(fonte1, fonte2).
    """
    from src.analysis.grid_predep import escolher_por_predominancia

    sl, sm, conc = escolher_por_predominancia(cubo)
    li_a, mi_a = _escolha_ano(conc)
    campos = [(cubo[0, si, sl[si], sm[si]], cubo[1, si, sl[si], sm[si]])
              for si in range(len(seasons))]
    ano = np.nanmean(cubo[:, :, li_a, mi_a], axis=1)
    campos.append((ano[0], ano[1]))
    rotulos = [f"{seasons[si]} · {movs[sm[si]]} / lag {lags[sl[si]]}"
               for si in range(len(seasons))]
    rotulos.append(f"YEAR · {movs[mi_a]} / lag {lags[li_a]}")
    return campos, rotulos


def plot_predep_concordado(cubo_nc, output_dir="plots/som", *, metrica="alpha",
                           tol_alpha=0.10, dpi=150, filename=None) -> Path:
    """α ou r² no par predominante, só onde Xavier e CHIRPS concordam.

    É o "máximo" da metodologia da predominância: o valor no par (lag, MoV) que
    maximiza a média espacial de min(fonte1, fonte2) — uma escolha para a unidade
    inteira em cada estação —, e NÃO o máximo pixel a pixel sobre os 203 pares,
    que traria o viés de seleção de volta. O valor é a média das duas fontes onde
    elas concordam; cinza onde divergem. YEAR: o par de maior predominância média
    das estações, com o valor médio sazonal nele.

    A tolerância é RELATIVA, a mesma para as duas métricas: `tol_alpha` sobre o α
    mediano nos pares escolhidos. No α isso devolve `tol_alpha`; no r², que é uma
    ordem de grandeza menor, uma tolerância absoluta de 0,1 deixaria passar ~100%
    dos pixels e a máscara não filtraria nada.

    `cubo_nc` é o arquivo com os cubos de α e r² (o `brasil.nc` do `--juntar`).
    """
    from matplotlib.colors import ListedColormap

    if metrica not in ("alpha", "r2"):
        raise ValueError("metrica deve ser 'alpha' ou 'r2'")
    ds = open_ds(cubo_nc)
    nome = ds.attrs.get("bacia", "brasil")
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    lags = [int(v) for v in ds.lag.values]

    def mediana(campos):
        return float(np.nanmedian(np.concatenate([np.concatenate(c) for c in campos])))

    from src.pipeline.montagem import TOL_REL_R2

    if metrica == "alpha":
        campos, rotulos = _campos_concordados(ds["alpha"].values, seasons, movs, lags)
        tol = tol_alpha
    else:
        campos, rotulos = _campos_concordados(ds["r2"].values, seasons, movs, lags)
        tol = TOL_REL_R2 * mediana(campos[:-1])
    tol_rel = tol / mediana(campos[:-1])

    cons = [np.where(np.abs(c - x) < tol, (x + c) / 2, np.nan) for x, c in campos]
    todos = np.concatenate([v[np.isfinite(v)] for v in cons])
    vmin, vmax = (float(np.percentile(todos, q)) for q in (2, 98))
    bd = load_borders()
    cinza = ListedColormap([SEM_VALOR])
    rotulo_m = "α" if metrica == "alpha" else "r²"

    fig = plt.figure(figsize=(20.5, 6.0))
    gs = fig.add_gridspec(1, len(cons), wspace=0.05, left=0.01, right=0.99,
                          top=0.90, bottom=0.24)
    im = None
    for v, rot, ax in zip(cons, rotulos, [fig.add_subplot(gs[0, j])
                                          for j in range(len(cons))]):
        g, ext = to_grid(v, lat, lon)
        fundo = to_grid_bool(~np.isfinite(v), lat, lon)
        draw_basemap(ax, bd, ext)
        ax.imshow(np.where(fundo, 1.0, np.nan), origin="lower", extent=ext,
                  cmap=cinza, interpolation="nearest", zorder=2)
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext,
                       cmap=CMAP_SEQ if metrica == "alpha" else "YlOrBr",
                       vmin=vmin, vmax=vmax, interpolation="nearest", zorder=3)
        strip_axes(ax)
        ax.set_title(rot.split(" · ", 1)[0], fontsize=13, fontweight="semibold",
                     color=INK, pad=6)

    cax = fig.add_axes([0.30, 0.15, 0.40, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{rotulo_m} (média das duas fontes)", color=INK_2, fontsize=10)
    cb.ax.tick_params(labelsize=9, colors=INK_2)
    nulo = ("Sem calibração nula neste mapa: α nunca é lido contra zero, e o teste "
            "contra os substitutos é feito por bacia." if metrica == "alpha"
            else "Sem calibração nula para o r².")
    regra = ("tolerância fixa do α" if metrica == "alpha"
             else "1/3 do r² mediano")
    fig.text(0.5, 0.02,
             f"{rotulo_m} no par (lag, MoV) predominante do próprio {rotulo_m} — o de "
             f"maior média espacial de min(Xavier, CHIRPS) no Brasil —, onde "
             f"|CHIRPS − Xavier| < {tol:.3g} ({regra}, {tol_rel:.0%} do valor típico); "
             f"cinza = fontes divergem. YEAR: "
             f"o par de maior predominância média das estações. {nulo}",
             ha="center", fontsize=9, color=INK_2, style="italic")
    sufixo = "" if metrica == "alpha" else "_r2"
    return _salvar(fig, output_dir, filename or f"{nome}_concordancia{sufixo}.png", dpi)


def movs_do_regime(fp, top=4):
    """MoVs que caracterizam cada regime — o critério de `plot_bacia_som`.

    `fp` é a assinatura `(regime, MoV)` de uma estação. Um índice entra se for mais
    forte no regime que na média da estação E tiver α acima da média da estação;
    ordenados por α. Devolve uma lista de índices por regime.
    """
    base = np.nanmean(fp, axis=0)
    limiar = float(np.nanmean(fp))
    out = []
    for r in range(fp.shape[0]):
        if not np.isfinite(fp[r]).any():
            out.append([])
            continue
        out.append([int(m) for m in np.argsort(fp[r])[::-1]
                    if fp[r, m] - base[m] > 0 and fp[r, m] > limiar][:top])
    return out


def _soma_janela(a, r):
    """Soma de `a` numa janela (2r+1)×(2r+1) em torno de cada célula (bordas = 0)."""
    p = np.pad(a, r + 1)
    c = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n])[:a.shape[0], :a.shape[1]]


def suavizar_regimes(grade, terra, k, *, raio=3, limiar=0.30, com_forca=False):
    """Filtro de moda: junta os regimes em clusters sem inventar concordância.

    `grade` tem o regime onde as fontes concordam e -1 onde discordam; `terra` é a
    máscara de células com dado. Cada célula de terra recebe o regime mais votado
    entre as concordantes da janela, SE esses votos forem ao menos `limiar` das
    células de terra da janela; senão fica -1. Uma região onde as fontes discordam
    na maior parte continua discordante — o filtro só consolida o que já concorda.
    Com `com_forca`, devolve também a fração de votos do regime vencedor.
    """
    n_terra = _soma_janela(terra.astype(float), raio)
    votos = np.stack([_soma_janela((grade == r).astype(float), raio) for r in range(k)])
    melhor = votos.argmax(axis=0)
    frac = np.take_along_axis(votos, melhor[None], 0)[0] / np.maximum(n_terra, 1)
    out = np.where((frac >= limiar) & terra, melhor, -1)
    if com_forca:
        return out, np.where(out >= 0, frac, 0.0)
    return out


# Pixel sem valor (fontes divergem, ou fora de qualquer cluster). Cinza neutro e
# mais escuro que o bege dos países vizinhos: com o tom antigo (#e3e0d8) as áreas
# sem valor pareciam fora do mapa.
SEM_VALOR = "#b9b9b9"

# Paleta do painel combinado: SEM o azul, o laranja, o verde e o roxo dos regimes,
# para ninguém ler "o azul do combinado" como "o R1 das estações".
CORES_COMBINADO = ["#c9a227", "#d4508f", "#3fb0c9", "#8c5a2b", "#6b8f3a",
                   "#b23b3b", "#5b6bbf", "#e39b3a", "#2f4f4f", "#e6a8c9"]


def combinar_clusters(liso, forca, area_min=0.01, max_clusters=10):
    """Um mapa só com o cluster mais forte de cada célula entre as estações.

    `liso` e `forca` são `(estação, lat, lon)`: o regime suavizado e a fração de
    votos dele. Cada célula fica com o (estação, regime) de maior força; clusters
    com menos de `area_min` da terra, ou além dos `max_clusters` maiores, viram -1.
    Devolve `(mapa de ids, lista de (estação, regime) por id)`.
    """
    si_melhor = forca.argmax(axis=0)
    reg = np.take_along_axis(liso, si_melhor[None], 0)[0]
    ok = forca.max(axis=0) > 0
    chave = np.where(ok & (reg >= 0), si_melhor * 100 + reg, -1)
    valores, contagem = np.unique(chave[chave >= 0], return_counts=True)
    terra = max(int((liso >= -1).any(axis=0).sum()), 1)
    ordem = np.argsort(contagem)[::-1][:max_clusters]
    pares = [(int(valores[i]) // 100, int(valores[i]) % 100) for i in ordem
             if contagem[i] / terra >= area_min]
    ids = np.full(chave.shape, -1)
    for j, (si, r) in enumerate(pares):
        ids[chave == si * 100 + r] = j
    return ids, pares


def plot_concordancia_movs(som, output_dir="plots/som", *, dpi=150, filename=None,
                           top_movs=4, raio=3, limiar=0.30) -> Path:
    """Regimes em que as duas fontes concordam, como clusters, e os MoVs de cada um.

    O MoV entra pelo regime, não pixel a pixel: o máximo de 203 estimativas em cada
    pixel dá um mapa sal e pimenta. O mapa é suavizado por `suavizar_regimes`
    (janela de 2·raio+1 células); os percentuais vêm dos pixels ORIGINAIS.
    """
    ds = open_ds(som)
    nome = ds.attrs.get("bacia", "brasil")
    lat, lon = ds.latitude.values, ds.longitude.values
    seasons = [str(s) for s in ds.season.values]
    movs = [str(m) for m in ds.mov_name.values]
    reg = ds["regime"].values
    fp = np.nanmean(ds["assinatura"].values, axis=0)     # (estação, regime, MoV)
    k = int(ds.sizes["regime_id"])
    c = concordancia_regimes(ds)
    n_se = len(seasons)
    bd = load_borders()
    cinza = matplotlib.colors.to_rgba(SEM_VALOR)
    cores = [matplotlib.colors.to_rgba(REGIME_COLORS[i % len(REGIME_COLORS)])
             for i in range(k)]
    terra = to_grid_bool(np.ones(len(lat), bool), lat, lon)

    fig = plt.figure(figsize=(20.5, 7.6))
    gs = fig.add_gridspec(2, n_se + 1, height_ratios=[1, 0.55], wspace=0.05,
                          hspace=0.03, left=0.01, right=0.99, top=0.94, bottom=0.08)
    lisos, forcas = [], []
    for si, sn in enumerate(seasons):
        rot = np.where(c["concorda"][si], reg[0, si], -1).astype(float)
        g, ext = to_grid(rot, lat, lon)
        g = np.where(np.isfinite(g), g, -1).astype(int)
        liso, forca = suavizar_regimes(g, terra, k, raio=raio, limiar=limiar,
                                       com_forca=True)
        lisos.append(np.where(terra, liso, -2))
        forcas.append(forca)
        rgba = np.zeros(liso.shape + (4,))
        rgba[terra] = cinza
        for r in range(k):
            rgba[liso == r] = cores[r]

        ax = fig.add_subplot(gs[0, si])
        draw_basemap(ax, bd, ext)
        ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest", zorder=2)
        strip_axes(ax)
        ax.set_title(sn, fontsize=13, fontweight="semibold", color=INK, pad=6)

        axl = fig.add_subplot(gs[1, si])
        axl.axis("off")
        axl.set_xlim(0, 1)
        axl.set_ylim(0, 1)
        lista = movs_do_regime(fp[si], top_movs)
        for r in range(k):
            y = 0.90 - r * (0.80 / max(k - 1, 1))
            frac = np.mean(c["concorda"][si] & (reg[0, si] == r))
            axl.add_patch(plt.Rectangle((0.03, y - 0.055), 0.05, 0.11,
                                        facecolor=cores[r], edgecolor="none",
                                        transform=axl.transAxes, clip_on=False))
            axl.text(0.11, y, f"R{r + 1}", va="center", fontsize=10.5,
                     fontweight="semibold", color=cores[r], transform=axl.transAxes)
            txt = ("  ".join(movs[m] for m in lista[r]) if lista[r]
                   else "nenhum índice acima da média")
            axl.text(0.22, y, txt, va="center", fontsize=10,
                     color=INK if lista[r] else INK_3,
                     style="normal" if lista[r] else "italic",
                     transform=axl.transAxes)
            axl.text(0.98, y, f"{frac:.0%}", va="center", ha="right", fontsize=9,
                     color=INK_2, transform=axl.transAxes)

    # 5º painel: o cluster mais forte de cada célula entre as estações
    ids, pares = combinar_clusters(np.stack(lisos), np.stack(forcas))
    cores_c = [matplotlib.colors.to_rgba(CORES_COMBINADO[j % len(CORES_COMBINADO)])
               for j in range(len(pares))]
    rgba = np.zeros(ids.shape + (4,))
    rgba[terra] = cinza
    for j in range(len(pares)):
        rgba[ids == j] = cores_c[j]
    ax = fig.add_subplot(gs[0, n_se])
    draw_basemap(ax, bd, ext)
    ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest", zorder=2)
    strip_axes(ax)
    ax.set_title("YEAR", fontsize=13, fontweight="semibold", color=INK, pad=6)

    axl = fig.add_subplot(gs[1, n_se])
    axl.axis("off")
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    passo_y = 0.90 / max(len(pares), 1)
    for j, (si, r) in enumerate(pares):
        y = 0.95 - j * passo_y
        lista = movs_do_regime(fp[si], 3)[r]
        area = (ids == j).sum() / terra.sum()
        axl.add_patch(plt.Rectangle((0.03, y - 0.03), 0.05, 0.06,
                                    facecolor=cores_c[j], edgecolor="none",
                                    transform=axl.transAxes, clip_on=False))
        axl.text(0.11, y, f"{seasons[si]} R{r + 1}", va="center", fontsize=9.5,
                 fontweight="semibold", color=INK, transform=axl.transAxes)
        axl.text(0.36, y, "  ".join(movs[m] for m in lista) or "—", va="center",
                 fontsize=9, color=INK, transform=axl.transAxes)
        axl.text(0.98, y, f"{area:.0%}", va="center", ha="right", fontsize=8.5,
                 color=INK_2, transform=axl.transAxes)

    passo = abs(float(np.diff(np.unique(lat))[:1].mean())) if len(np.unique(lat)) > 1 else 0
    fig.text(0.5, 0.015,
             f"Mapas suavizados por moda (janela de {2 * raio + 1}×{2 * raio + 1} células, "
             f"≈{(2 * raio + 1) * passo:.1f}°): a célula recebe o regime mais votado entre "
             f"as concordantes da vizinhança se ele tiver ≥ {limiar:.0%} dela; cinza = "
             "fontes em regimes diferentes. Percentuais dos pixels originais. "
             "YEAR: em cada célula, o cluster (estação, regime) de maior fração de "
             "votos; clusters com ≥ 1% do Brasil, % = área no combinado.",
             ha="center", fontsize=9, color=INK_2, style="italic")
    metrica = ds.attrs.get("metrica", "alpha")
    sufixo = "" if metrica == "alpha" else f"_{metrica}"
    return _salvar(fig, output_dir,
                   filename or f"{nome}_concordancia_movs{sufixo}.png", dpi)


def gerar_figuras_bacia(fonte, output_dir="plots/bacias", **kw) -> list:
    """As três figuras da bacia — PREDEP, regressão e SOM. Uma que falhe não impede
    as outras."""
    from functools import partial

    saidas = []
    for nome, fn in (("predep", plot_bacia_predep),
                     ("regressao", partial(plot_bacia_predep, metrica="r2")),
                     ("som", plot_bacia_som)):
        try:
            saidas.append(fn(fonte, output_dir, **kw))
        except (KeyError, ValueError, OSError) as e:
            print(f"  ⚠ figura {nome}: {type(e).__name__}: {e}")
    return saidas


VARS_SOM = ("regime", "erro_bmu", "assinatura", "mascara", "ari_fontes", "ari_teto",
            "sel_lag", "sel_mov")


def extrair_som(fonte, destino) -> Path:
    """O mapeamento espacial do SOM da bacia, num .nc só dele.

    O `<bacia>.nc` completo carrega os cubos de α e r² (centenas de MB na
    Amazônia); este arquivo leva só o que descreve os regimes e o par escolhido.
    """
    ds = open_ds(fonte)
    manter = [v for v in VARS_SOM if v in ds.data_vars]
    out = ds[manter]
    for v in ("sel_lag", "sel_mov"):            # só o observado, se houver `shift`
        if v in out and "shift" in out[v].dims:
            out[v] = out[v].isel(shift=0)
    out = out.drop_dims([d for d in ("shift", "lag") if d in out.dims
                         and not any(d in out[v].dims for v in out.data_vars)])
    out.attrs.update({k: ds.attrs[k] for k in ds.attrs})
    out.attrs["conteudo"] = "mapeamento do SOM da bacia (extraído do <bacia>.nc)"
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(destino)
    return destino


def _salvar(fig, output_dir, filename, dpi) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    caminho = out / filename
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {caminho}")
    return caminho
