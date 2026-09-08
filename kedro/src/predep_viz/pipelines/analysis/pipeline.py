"""
Stages 4/5 — análise a jusante (SOM + figuras), sobre os dados reais que a
produção no Modal gerou.

Consome `predep_stack_{source}` produzido pela pipeline `predep_production`.
Fan-out por fonte na cadeia SOM (cada fonte treina o próprio SOM), depois
converge nos dois nós de comparação.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    build_som_features,
    compute_som_concordance,
    plot_predep_max_comparison,
    plot_som_comparison_figure,
    train_som_and_regimes,
)

SOURCES = ["xavier", "chirps"]


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=plot_predep_max_comparison,
            inputs=[
                "predep_stack_chirps",
                "predep_stack_xavier",
                "params:paths.plots_dir_predep_max",
                "params:analysis.margin",
            ],
            outputs="predep_max_stats",
            name="plot_predep_max_comparison",
            tags=["analise", "figuras"],
        )
    ]

    for source in SOURCES:
        nodes.extend(
            [
                node(
                    func=build_som_features,
                    inputs=[
                        f"predep_stack_{source}",
                        "params:exp_dir",
                        "params:som",
                    ],
                    outputs=f"som_features_{source}",
                    name=f"build_som_features_{source}",
                    tags=["analise", "som", source],
                ),
                node(
                    func=train_som_and_regimes,
                    inputs=[f"som_features_{source}", "params:som"],
                    outputs=f"som_trained_{source}",
                    name=f"train_som_and_regimes_{source}",
                    tags=["analise", "som", source],
                ),
            ]
        )

    nodes.extend(
        [
            node(
                func=compute_som_concordance,
                inputs=["som_trained_xavier", "som_trained_chirps", "params:som"],
                outputs="som_concordance",
                name="compute_som_concordance",
                tags=["analise", "som"],
            ),
            node(
                func=plot_som_comparison_figure,
                inputs=[
                    "som_trained_xavier",
                    "som_trained_chirps",
                    "som_concordance",
                    "params:paths.plots_dir_weekly",
                    "params:analysis.som_caveat",
                ],
                outputs="som_comparison_stats",
                name="plot_som_comparison_figure",
                tags=["analise", "som", "figuras"],
            ),
        ]
    )

    return Pipeline(nodes)
