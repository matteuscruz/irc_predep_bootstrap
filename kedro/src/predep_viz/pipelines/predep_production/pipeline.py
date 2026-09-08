"""
Stage 2 — o cálculo PREDEP de produção (Modal), representado no grafo.

Fan-out por FONTE (xavier / chirps — 2 branches), com o MoV como dimensão
interna (lista de 23 em `params:movs`), espelhando o laço de lotes do
`main()` de `experiments/08_predep_parana_recompute.py`. Um nó por MoV
deixaria 46 caixas idênticas no grafo sem ganho de leitura.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    dispatch_modal_predep,
    load_predep_stack,
    prepare_modal_inputs,
    summarize_predep_stack,
)

SOURCES = ["xavier", "chirps"]


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []
    for source in SOURCES:
        nodes.extend(
            [
                node(
                    func=prepare_modal_inputs,
                    inputs=[
                        f"params:sources.{source}",
                        "params:movs",
                        "params:paths",
                    ],
                    outputs=f"modal_inputs_{source}",
                    name=f"prepare_modal_inputs_{source}",
                    tags=["producao", "modal", source],
                ),
                node(
                    func=dispatch_modal_predep,
                    inputs=[
                        f"modal_inputs_{source}",
                        "params:exp_dir",
                        "params:movs",
                        "params:run_params",
                    ],
                    outputs=f"dispatch_info_{source}",
                    name=f"dispatch_modal_predep_{source}",
                    tags=["producao", "modal", source],
                ),
                node(
                    func=load_predep_stack,
                    inputs=[
                        f"dispatch_info_{source}",
                        "params:exp_dir",
                        f"params:sources.{source}",
                    ],
                    outputs=f"predep_stack_{source}",
                    name=f"load_predep_stack_{source}",
                    tags=["producao", source],
                ),
                node(
                    func=summarize_predep_stack,
                    inputs=f"predep_stack_{source}",
                    outputs=f"predep_summary_{source}",
                    name=f"summarize_predep_stack_{source}",
                    tags=["producao", source],
                ),
            ]
        )
    return Pipeline(nodes)
