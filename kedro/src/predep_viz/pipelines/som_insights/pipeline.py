"""
Stage 4b — bundle completo de insights do SOM (7 figuras + PDF por fonte).

Consome `som_trained_{source}` da pipeline `analysis` (não retreina o SOM),
então as duas precisam rodar na mesma sessão — ver as composições em
`pipeline_registry.py`.

`exp_label` define a pasta de saída (`plots/exp{NN}/som/`), mantendo a
convenção já usada na produção: exp08 = Xavier, exp09 = CHIRPS.
"""
from kedro.pipeline import Pipeline, node

from .nodes import render_som_insights_bundle

SOURCES = ["xavier", "chirps"]


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=render_som_insights_bundle,
                inputs=[
                    f"som_trained_{source}",
                    f"params:som_insights.exp_label_{source}",
                    "params:exp_dir",
                    "params:som_insights.diagnostics",
                ],
                outputs=f"som_insights_files_{source}",
                name=f"render_som_insights_bundle_{source}",
                tags=["som_insights", "figuras", source],
            )
            for source in SOURCES
        ]
    )
