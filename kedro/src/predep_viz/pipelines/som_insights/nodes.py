"""
Stage 4b — o bundle completo de insights do SOM.

Equivalente Kedro de `src/analysis/generate_som_insights.py`: a partir de um
SOM já treinado (nó `train_som_and_regimes` da pipeline `analysis`), gera os
7 plots de diagnóstico/interpretação e empacota tudo num PDF.

Nenhuma lógica de plot é reimplementada — todos os nós chamam
`src/visualization/som_plots.py` e `src/visualization/generate_plots_exp.py:_make_pdf`,
exatamente como o script de produção faz.
"""

import sys
from pathlib import Path

import pandas as pd

# kedro/src/predep_viz/pipelines/som_insights/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def render_som_insights_bundle(
    som_trained: dict, exp_label: int, exp_dir: str, diagnostics: bool
) -> pd.DataFrame:
    """
    Gera o conjunto completo de figuras do SOM para uma fonte e empacota em
    PDF — mesma saída de
    `python src/analysis/generate_som_insights.py --exp-n <N>`:

    - `som_regime_map`               mapa geográfico dos regimes (principal)
    - `som_regime_panels`            extensão espacial de cada regime
    - `som_geographic_component_planes`  α médio por MoV, no mapa
    - `som_atypicality_map`          erro de quantização por pixel
    - `som_boundary_map`             fronteiras entre regimes (U-matrix no mapa)
    - `som_regime_fingerprints`      assinatura MoV/lag/estação por regime
    - `som_regime_mov_table`         heatmap regime × MoV
    - (+ `som_umatrix`, `som_hitmap`, `som_component_planes` se
      `diagnostics: true` — os diagnósticos na grade do SOM, não no mapa)
    - `som_insights_exp{NN}.pdf`     tudo num PDF

    Saída em `plots/exp{exp_label:02d}/som/` — mesma convenção da produção
    (ontem: exp08 = Xavier, exp09 = CHIRPS).
    """
    from src.visualization import som_plots
    from src.visualization.cluster_plots import _load_basin_outlines
    from src.visualization.generate_plots_exp import _make_pdf

    bundle = som_trained["bundle"]
    som = som_trained["som"]
    regimes = som_trained["regimes"]
    fps = som_trained["fingerprints"]
    regime_grid = som_trained["regime_grid"]

    plots_dir = ROOT / "plots" / f"exp{exp_label:02d}" / "som"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mov0 = bundle.mov_names[0]
    outlines = _load_basin_outlines(
        ROOT / "results/predep_granular_brazil" / exp_dir / som_trained["source"] / mov0,
        mov0,
        fallback_dirs=sorted(
            (ROOT / "results/predep_granular_brazil").glob(f"exp*/{mov0}")
        ),
    )

    pngs = [
        som_plots.plot_regime_map(regime_grid, bundle, fps, plots_dir, exp_label,
                                  basin_outlines=outlines),
        som_plots.plot_regime_panels(regime_grid, bundle, fps, plots_dir, exp_label,
                                     basin_outlines=outlines),
        som_plots.plot_geographic_component_planes(bundle, plots_dir, exp_label,
                                                   basin_outlines=outlines,
                                                   value_label="α"),
        som_plots.plot_atypicality_map(som, regimes, bundle, plots_dir, exp_label,
                                       basin_outlines=outlines),
        som_plots.plot_boundary_map(som, regimes, bundle, plots_dir, exp_label,
                                    basin_outlines=outlines),
        som_plots.plot_regime_fingerprints(fps, bundle, plots_dir, exp_label,
                                           value_label="α"),
        som_plots.plot_regime_mov_heatmap(fps, bundle, plots_dir, exp_label,
                                          value_label="α"),
    ]
    if diagnostics:
        pngs += [
            som_plots.plot_umatrix(som, plots_dir, exp_label),
            som_plots.plot_hitmap(som, bundle.X, plots_dir, exp_label),
            som_plots.plot_component_planes(som, bundle, plots_dir, exp_label,
                                            value_label="α"),
        ]

    pdf_path = plots_dir / f"som_insights_exp{exp_label:02d}.pdf"
    _make_pdf(pngs, pdf_path)

    return pd.DataFrame(
        [
            {
                "fonte": som_trained["source"],
                "exp_label": f"exp{exp_label:02d}",
                "n_figuras": len(pngs),
                "pdf": pdf_path.name,
                "dir": str(plots_dir.relative_to(ROOT)),
            }
        ]
    )
