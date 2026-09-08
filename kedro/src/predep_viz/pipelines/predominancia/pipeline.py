"""
Metodologia da predominância: exp22/exp23 (Modal) → exp26 (SOM) → figuras.

Cadeia linear de propósito — cada passo depende do anterior, e o último
produz a tabela de métricas que serve para comparar execuções.
"""
from kedro.pipeline import Pipeline, node

from .nodes import gerar_figuras, preparar_grids, resumir_metricas, rodar_som


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=preparar_grids,
            inputs=["params:predominancia",
                    "params:predominancia.allow_modal_dispatch"],
            outputs="predominancia_grids",
            name="preparar_grids",
            tags=["predominancia", "modal"],
        ),
        node(
            func=rodar_som,
            inputs=["predominancia_grids", "params:predominancia",
                    "params:predominancia.forcar_som"],
            outputs="predominancia_som_nc",
            name="rodar_som",
            tags=["predominancia", "som"],
        ),
        node(
            func=gerar_figuras,
            inputs=["predominancia_grids", "predominancia_som_nc",
                    "params:predominancia.plots_dir", "params:predominancia.k_mapa"],
            outputs="predominancia_figuras",
            name="gerar_figuras",
            tags=["predominancia", "figuras"],
        ),
        node(
            func=resumir_metricas,
            inputs=["predominancia_grids", "predominancia_som_nc",
                    "params:predominancia.k_mapa"],
            outputs="predominancia_metricas",
            name="resumir_metricas",
            tags=["predominancia", "reprodutibilidade"],
        ),
    ])
