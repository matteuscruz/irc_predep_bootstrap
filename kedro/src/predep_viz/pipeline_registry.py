"""Project pipelines.

Quatro pipelines reais:

- `data_prep`          Stage 0/1 — raw → mensal na grade comum → máscara do Paraná.
- `predep_production`  Stage 2 — o cálculo PREDEP de produção (Modal), em modo
                       resume por padrão (não gasta crédito).
- `analysis`           Stages 4/5 — SOM + figuras comparativas, sobre os dados
                       reais de produção.
- `predep_demo`        Visão didática nível-equação (1 MoV, 1 pixel), onde estão
                       os docstrings com as equações do paper.
- `predominancia`      Exp 22–26 — a metodologia da predominância com
                       concordância: mapas de α, SOM por trimestre e as duas
                       figuras finais. Independente das demais (parte das
                       fatias já calculadas no volume Modal), por isso fica
                       fora do `__default__`.

Mais duas composições, porque `analysis` consome datasets em MEMÓRIA
produzidos por `predep_production` — as duas precisam rodar na mesma sessão:

- `producao_analise`   predep_production + analysis (uso comum: retoma dos NCs
                       já calculados e refaz toda a análise/figuras).
- `__default__`        data_prep + predep_production + analysis — o fluxo real
                       ponta a ponta. `predep_demo` fica de fora do default por
                       ser um recorte independente, com seu próprio fluxo.

Registro EXPLÍCITO (em vez de `find_pipelines()`): a descoberta automática
filtra pelo nome pedido em `--pipeline`, o que quebra as composições — elas
não existem como pasta em `pipelines/`.
"""
from __future__ import annotations

from kedro.pipeline import Pipeline

from predep_viz.pipelines import (
    analysis,
    data_prep,
    predep_demo,
    predep_production,
    predominancia,
    som_insights,
    som_param_search,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_prep": data_prep.create_pipeline(),
        "predep_production": predep_production.create_pipeline(),
        "analysis": analysis.create_pipeline(),
        "predep_demo": predep_demo.create_pipeline(),
        "som_insights": som_insights.create_pipeline(),
        "som_param_search": som_param_search.create_pipeline(),
        "predominancia": predominancia.create_pipeline(),
    }
    pipelines["producao_analise"] = (
        pipelines["predep_production"] + pipelines["analysis"]
    )
    # `som_insights` e `som_param_search` consomem datasets em MEMÓRIA de
    # `analysis` (`som_trained_*` / `som_features_*`) — precisam da mesma sessão.
    pipelines["analise_completa"] = (
        pipelines["producao_analise"] + pipelines["som_insights"]
    )
    pipelines["busca_parametros"] = (
        pipelines["predep_production"]
        + pipelines["analysis"]
        + pipelines["som_param_search"]
    )
    pipelines["__default__"] = (
        pipelines["data_prep"]
        + pipelines["predep_production"]
        + pipelines["analysis"]
    )
    return pipelines
