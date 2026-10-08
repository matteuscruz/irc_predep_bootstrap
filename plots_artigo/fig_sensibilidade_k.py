"""Sensibilidade dos regimes ao número de regimes K (PREDEP e r²).

    python plots_artigo/fig_sensibilidade_k.py

Entrada: <DADOS>/som/sensibilidade_k_{alpha,r2}.nc, gerados por
tools/som_sensibilidade_k.py. Saída: <SAIDA>/som/sensibilidade_k.png — uma linha por
métrica; colunas: estabilidade entre consensos (ARI), ρ médio entre as fontes,
variância explicada e silhouette dos neurônios. Linha tracejada em K = 4.
"""
import matplotlib.pyplot as plt
import xarray as xr

from comum import DADOS, SAIDA, exigir
from src.visualization.bacia_plots import _salvar

PAINEIS = [("estab_consenso", "Estabilidade entre sementes (ARI)"),
           ("rho_fontes", "Concordância Xavier × CHIRPS (ρ médio)"),
           ("var_explicada", "Variância explicada"),
           ("silhouette", "Silhouette (neurônios)")]
CORES = {"DJF": "#2c7fb8", "MAM": "#41ab5d", "JJA": "#e6550d", "SON": "#756bb1",
         "YEAR": "#252525"}

if __name__ == "__main__":
    arquivos = [exigir(DADOS / "som" / f"sensibilidade_k_{m}.nc") for m in ("alpha", "r2")]
    fig, axs = plt.subplots(len(arquivos), 4, figsize=(16, 3.4 * len(arquivos)),
                            squeeze=False)
    for li, f in enumerate(arquivos):
        d = xr.open_dataset(f)
        rot = "PREDEP α" if d.attrs["metrica"] == "alpha" else "r²"
        for ci, (v, t) in enumerate(PAINEIS):
            ax = axs[li, ci]
            for s, cor in CORES.items():
                ax.plot(d.K, d[v].sel(season=s), "o-", color=cor, label=s,
                        lw=2.4 if s == "YEAR" else 1.4, ms=4)
            ax.axvline(4, color="#999", ls="--", lw=1)
            ax.set_title(f"{rot} · {t}", fontsize=10)
            ax.set_xlabel("K (nº de regimes)")
            ax.set_xticks(d.K.values)
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _salvar(fig, SAIDA / "som", "sensibilidade_k.png", 150)
