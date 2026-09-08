#!/usr/bin/env python3
"""
Exp 24 — o ARI entre Xavier e CHIRPS contra as duas referências que importam.

O gráfico é deliberadamente simples: para cada número de regimes, três coisas
no mesmo eixo. A faixa cinza é o TETO (o que dois SOMs sobre os MESMOS dados
alcançam entre si, variando só a semente) — nenhum ponto pode passar muito
dela. A faixa vermelha é o NULO: o mesmo ARI entre fontes medido em surrogates,
que preserva a chuva e destrói só o pareamento com o índice. O ponto azul é o
observado. Ele só significa alguma coisa acima da faixa vermelha.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8b8a84"
AZUL, RED, CINZA = "#2a78d6", "#8a1f1e", "#b8b6b0"
VERDE = "#1f8a70"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="results/exp24_som_consenso.nc")
    ap.add_argument("--out", default="plots/exp24_som_consenso.png")
    args = ap.parse_args()

    ds = xr.open_dataset(ROOT / args.nc)
    ks = [int(k) for k in ds.k.values]
    obs = ds["ari_observado"].values
    nul = ds["ari_nulo"].values          # (k, surrogate)
    teto = ds["ari_teto"].values         # (k, par)
    x = np.arange(len(ks))

    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    ax.fill_between(x, teto.min(1), teto.max(1), color=CINZA, alpha=.45, zorder=1,
                    label="TETO — mesmos dados, sementes diferentes")
    ax.plot(x, teto.mean(1), color="#6f6d68", lw=1.6, zorder=2)

    ax.fill_between(x, nul.min(1), nul.max(1), color=RED, alpha=.16, zorder=1)
    ax.plot(x, np.median(nul, 1), color=RED, lw=1.6, ls="--", zorder=3,
            label="NULO — Xavier vs CHIRPS em surrogates")
    for di in range(nul.shape[1]):
        ax.plot(x, nul[:, di], color=RED, lw=0.7, alpha=.35, zorder=2)

    cores = [VERDE if o > n else AZUL for o, n in zip(obs, nul.max(1))]
    ax.plot(x, obs, color=AZUL, lw=2.4, zorder=4)
    ax.scatter(x, obs, s=90, c=cores, zorder=5, edgecolor="white", linewidth=1.4,
               label="MEDIDO — Xavier vs CHIRPS observado")
    for i, o in enumerate(obs):
        ax.annotate(f"{o:.3f}", (x[i], o), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9,
                    color=INK, fontweight="semibold")

    ax.set_xticks(x); ax.set_xticklabels([f"{k} regimes" for k in ks], fontsize=10)
    ax.set_ylabel("Adjusted Rand Index", fontsize=10.5, color=INK_2)
    ax.set_ylim(min(0.0, float(np.min([obs.min(), nul.min()])) - 0.05),
                max(float(teto.max()), float(obs.max()), float(nul.max())) * 1.22)
    ax.axhline(0, color="#d8d7d2", lw=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#e8e7e2"); ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=1 if len(ks) < 3 else 2)

    venceu = int((obs > nul.max(1)).sum())
    fig.suptitle("O SOM concorda entre Xavier e CHIRPS — mas o surrogate concorda igual",
                 fontsize=13, fontweight="semibold", color=INK, y=0.99)
    fig.text(0.5, 0.005,
             f"O ARI observado fica dentro da faixa dos surrogates em "
             f"{len(ks)-venceu} dos {len(ks)} valores de k. Xavier e CHIRPS medem a "
             "mesma chuva, então seus SOMs coincidem mesmo sem sinal de MoV — é por "
             "isso que o piso do ARI não é zero.",
             ha="center", fontsize=9, color=INK_3, style="italic", wrap=True)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.845, bottom=0.135)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ {out}\n")
    print(f"{'k':>4}{'TETO':>9}{'MEDIDO':>9}{'NULO med':>10}{'NULO máx':>10}{'passa?':>9}")
    for i, k in enumerate(ks):
        print(f"{k:>4}{teto[i].mean():>9.3f}{obs[i]:>9.3f}{np.median(nul[i]):>10.3f}"
              f"{nul[i].max():>10.3f}{'SIM' if obs[i] > nul[i].max() else 'não':>9}")


if __name__ == "__main__":
    main()
