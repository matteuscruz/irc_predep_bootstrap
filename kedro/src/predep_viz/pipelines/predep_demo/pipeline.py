"""
Monta o grafo de nós da pipeline PREDEP.

Fan-out real no grafo: só por ESTAÇÃO (DJF/MAM/JJA/SON, 4 branches) — o
lag é tratado como dimensão interna de cada nó (dict {lag: valor}), não
como um nó por valor de lag, a pedido do usuário (1 nó por lag deixava o
grafo com ~130 nós sem ganho de clareza, já que os 6 lags passam pela
EXATA mesma lógica). `SEASONS` abaixo é construído em tempo de import do
Kedro (antes do carregamento de `parameters.yml`), então precisa bater com
os nomes de estação usados em `src/predep_calculation` /
`experiments/08_predep_parana_recompute.py` (DJF/MAM/JJA/SON) — não é
parametrizável via `parameters.yml` sem reestruturar o grafo.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    align_mov_all_lags,
    assemble_season_result,
    build_valid_xy_all_lags,
    combine_results,
    compute_anomaly,
    compute_monthly_climatology,
    compute_predep_all_lags,
    compute_regression_all_lags,
    extract_pixel_series,
    filter_season,
    load_mov_index,
    load_pixel_mask,
    load_precip_source,
)

SEASONS = ["DJF", "MAM", "JJA", "SON"]


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=load_precip_source,
            inputs=["params:source", "params:time_start", "params:time_end"],
            outputs="precip_source",
            name="load_precip_source",
            tags=["ingestao"],
        ),
        node(
            func=load_pixel_mask,
            inputs="params:mov_name",
            outputs="pixel",
            name="load_pixel_mask",
            tags=["ingestao"],
        ),
        node(
            func=extract_pixel_series,
            inputs=["precip_source", "pixel"],
            outputs="pixel_series",
            name="extract_pixel_series",
            tags=["ingestao"],
        ),
        node(
            func=compute_monthly_climatology,
            inputs="pixel_series",
            outputs="climatology",
            name="compute_monthly_climatology",
            tags=["anomalia"],
        ),
        node(
            func=compute_anomaly,
            inputs=["pixel_series", "climatology"],
            outputs="anomaly",
            name="compute_anomaly",
            tags=["anomalia"],
        ),
        node(
            func=load_mov_index,
            inputs="params:mov_name",
            outputs="mov_lookup",
            name="load_mov_index",
            tags=["ingestao"],
        ),
    ]

    combine_inputs = []
    for season in SEASONS:
        season_anom = f"anomaly_{season}"
        aligned = f"aligned_{season}"
        xy = f"xy_{season}"
        alpha = f"alpha_{season}"
        r2 = f"r2_{season}"
        result = f"result_{season}"

        nodes.extend(
            [
                node(
                    func=filter_season,
                    inputs=["anomaly", f"params:season_{season}"],
                    outputs=season_anom,
                    name=f"filter_season_{season}",
                    tags=["estacao", season],
                ),
                node(
                    func=align_mov_all_lags,
                    inputs=[season_anom, "mov_lookup", "params:lags"],
                    outputs=aligned,
                    name=f"align_mov_all_lags_{season}",
                    tags=["lag", season],
                ),
                node(
                    func=build_valid_xy_all_lags,
                    inputs=aligned,
                    outputs=xy,
                    name=f"build_valid_xy_all_lags_{season}",
                    tags=["lag", season],
                ),
                node(
                    func=compute_predep_all_lags,
                    inputs=[xy, "params:n_boot", "params:seed"],
                    outputs=alpha,
                    name=f"compute_predep_all_lags_{season}",
                    tags=["predep", season],
                ),
                node(
                    func=compute_regression_all_lags,
                    inputs=xy,
                    outputs=r2,
                    name=f"compute_regression_all_lags_{season}",
                    tags=["predep", season],
                ),
                node(
                    func=assemble_season_result,
                    inputs=[alpha, r2, xy, f"params:season_{season}"],
                    outputs=result,
                    name=f"assemble_season_result_{season}",
                    tags=["resultado", season],
                ),
            ]
        )
        combine_inputs.append(result)

    nodes.append(
        node(
            func=combine_results,
            inputs=combine_inputs,
            outputs="predep_results",
            name="combine_results",
            tags=["resultado"],
        )
    )

    return Pipeline(nodes)
