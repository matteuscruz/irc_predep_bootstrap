"""
Stage 0/1 — preparação dos dados (roda local).

Xavier mensal entra como dado externo (ver docstring de
`load_xavier_monthly`: não há produtor reproduzível no repo hoje); o CHIRPS
é convertido e regridado de verdade a partir do diário bruto, e a máscara
combinada do Paraná é construída a partir dos dois.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    build_parana_combined_mask,
    convert_chirps_daily_to_monthly,
    load_xavier_monthly,
    save_chirps_monthly,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_xavier_monthly,
                inputs="params:paths.xavier_monthly",
                outputs="xavier_monthly",
                name="load_xavier_monthly",
                tags=["data_prep", "entrada_externa"],
            ),
            node(
                func=convert_chirps_daily_to_monthly,
                inputs=["params:paths.chirps_raw", "xavier_monthly"],
                outputs="chirps_monthly",
                name="convert_chirps_daily_to_monthly",
                tags=["data_prep"],
            ),
            node(
                func=save_chirps_monthly,
                inputs=["chirps_monthly", "params:paths.chirps_monthly"],
                outputs="chirps_monthly_path",
                name="save_chirps_monthly",
                tags=["data_prep"],
            ),
            node(
                func=build_parana_combined_mask,
                inputs=[
                    "chirps_monthly_path",
                    "params:paths.xavier_monthly",
                    "params:paths.basin_dir",
                    "params:paths.parana_mask",
                ],
                outputs="parana_mask_info",
                name="build_parana_combined_mask",
                tags=["data_prep"],
            ),
        ]
    )
