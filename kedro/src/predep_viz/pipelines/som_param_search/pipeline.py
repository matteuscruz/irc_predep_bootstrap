"""
Stage 4c — busca de hiperparâmetros do SOM (fora do `__default__`: é cara).

Dois mecanismos independentes. Para rodar só um, use `--to-nodes` (que
inclui as dependências a montante) — `--tags` NÃO serve aqui, porque
filtra os nós sem trazer quem produz as entradas deles:

    kedro run --pipeline busca_parametros --to-nodes summarize_param_search  # Pareto
    kedro run --pipeline busca_parametros --to-nodes summarize_sweep         # varredura g/k

Consome `som_features_{source}` da pipeline `analysis` (reaproveita a matriz
de features já construída, não relê os NCs), então roda junto com ela — ver
a composição `busca_parametros` em `pipeline_registry.py`.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    search_topology_pareto,
    summarize_param_search,
    summarize_sweep,
    sweep_g_and_k,
)

SOURCES = ["xavier", "chirps"]


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []
    for source in SOURCES:
        nodes.extend(
            [
                node(
                    func=search_topology_pareto,
                    inputs=[f"som_features_{source}", "params:param_search.pareto"],
                    outputs=f"pareto_search_{source}",
                    name=f"search_topology_pareto_{source}",
                    tags=["param_search", "pareto", source],
                ),
                node(
                    func=sweep_g_and_k,
                    inputs=[
                        f"som_features_{source}",
                        "params:exp_dir",
                        "params:param_search.sweep",
                    ],
                    outputs=f"gk_sweep_{source}",
                    name=f"sweep_g_and_k_{source}",
                    tags=["param_search", "sweep", source],
                ),
            ]
        )

    nodes.extend(
        [
            node(
                func=summarize_param_search,
                inputs=["pareto_search_xavier", "pareto_search_chirps"],
                outputs="pareto_search_stats",
                name="summarize_param_search",
                tags=["param_search", "pareto"],
            ),
            node(
                func=summarize_sweep,
                inputs=["gk_sweep_xavier", "gk_sweep_chirps"],
                outputs="gk_sweep_stats",
                name="summarize_sweep",
                tags=["param_search", "sweep"],
            ),
        ]
    )
    return Pipeline(nodes)
