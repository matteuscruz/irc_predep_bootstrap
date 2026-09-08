"""
Stage 4c — busca de hiperparâmetros do SOM.

O projeto tem DOIS mecanismos distintos de busca, ambos representados aqui:

1. **Pareto QE×TE + auto-K** (`--optimize --auto-k` do
   `generate_som_insights.py`) — é o que rodamos em 2026-08-31. Varre
   topologias candidatas ao redor da heurística, escolhe a que minimiza a
   distância à origem no plano (QE normalizado, TE normalizado), e depois
   escolhe K por silhouette média entre múltiplas sementes.
   Implementação real: `src/som_modeling/som_optimizer.py` e
   `src/som_modeling/som_macro_clustering.py`, acionados via
   `train_som(optimize=True)` / `assign_regimes(auto_k=True)`.

2. **Varredura g/k por estação** (`src/analysis/som_param_search.py:run_full_search`)
   — ferramenta separada, offline: sweep de g com detecção de cotovelo na
   curva de QE, e sweep de k com silhouette / Davies-Bouldin / score
   composto, rodando por estação e para o caso combinado.

Ambos são CAROS (treinam dezenas de SOMs), por isso esta pipeline fica fora
do `__default__` — roda sob demanda via `--pipeline som_param_search`.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# kedro/src/predep_viz/pipelines/som_param_search/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def search_topology_pareto(features: dict, search: dict) -> dict:
    """
    Mecanismo 1 — busca de topologia pela fronteira de Pareto (QE × TE) e
    escolha de K por silhouette (o caminho `--optimize --auto-k`).

    `search_sample_size` / `search_epochs` barateiam a VARREDURA (cada
    candidata treina numa subamostra, com poucas épocas); o modelo vencedor
    é sempre retreinado na base COMPLETA antes de ser retornado. Com 3000
    px / 2 épocas, a busca levou ~80s por fonte em 2026-08-31 (contra
    minutos se cada candidata treinasse na base inteira).

    A escolha de K usa um conjunto FIXO de 5 sementes (42,1,7,13,99) para a
    silhouette média — estabilidade da decisão, independente da `seed` do
    treino — e desempata pelo K mais granular dentro de 1σ.
    """
    from src.analysis.som_insights import assign_regimes, train_som

    bundle = features["bundle"]
    t0 = time.time()
    som, metrics = train_som(
        bundle.X,
        som_size=None,
        num_iteration=search["iterations"],
        seed=search["seed"],
        optimize=True,
        search_sample_size=search["search_sample_size"],
        search_epochs=search["search_epochs"],
    )
    regimes = assign_regimes(
        som, bundle.X, seed=search["seed"], auto_k=True, max_k=len(bundle.mov_names)
    )
    return {
        "source": features["source"],
        "som_size": metrics["som_size"],
        "qe": float(metrics["quantization_error"]),
        "te": float(metrics["topographic_error"]),
        "n_regimes": int(regimes.n_regimes),
        "k_note": (regimes.kmeans_results or {}).get("k_margin_note", ""),
        "elbow_k": (regimes.kmeans_results or {}).get("elbow_k"),
        "silhouette_k_raw": (regimes.kmeans_results or {}).get("silhouette_k_raw"),
        "pareto_results": metrics.get("pareto_results", []),
        "elapsed_s": round(time.time() - t0, 1),
    }


def sweep_g_and_k(features: dict, exp_dir: str, sweep: dict) -> dict:
    """
    Mecanismo 2 — varredura explícita de g e k
    (`src/analysis/som_param_search.py:run_full_search`).

    Para cada estação (e o caso combinado "all"): treina um SOM por g em
    `g_range` e acha o cotovelo da curva de QE pela segunda derivada; depois,
    com o g ótimo, varre `k_range` avaliando inertia, silhouette e
    Davies-Bouldin, e ranqueia por um score composto.

    `nc_suffix` é repassado porque os NCs da produção têm a fonte no nome
    (`{MOV}_predep_granular_seasonal_xavier.nc`); o parâmetro foi adicionado
    a `run_full_search`/`run_season_search` nesta sessão (default preserva o
    comportamento antigo).

    **Custo**: `len(g_range)` SOMs por estação. O default da ferramenta
    (g 5..20 × 5 estações = 80 treinos) leva ~20min nesta base; os ranges
    em `params:param_search.sweep` são reduzidos de propósito para a
    pipeline rodar em tempo razoável — aumente-os para a busca completa.
    """
    from src.analysis.som_param_search import run_full_search

    source = features["source"]
    exp_path = ROOT / "results/predep_granular_brazil" / exp_dir / source
    t0 = time.time()
    result = run_full_search(
        exp_path,
        g_range=sweep["g_range"],
        k_range=sweep["k_range"],
        lags=sweep["lags"],
        seasons=sweep["seasons"],
        include_all=sweep["include_all"],
        num_iteration=sweep["iterations"],
        seed=sweep["seed"],
        verbose=True,
        nc_suffix=f"_predep_granular_seasonal_{source}",
    )
    return {"source": source, "result": result, "elapsed_s": round(time.time() - t0, 1)}


def summarize_param_search(pareto_x: dict, pareto_c: dict) -> pd.DataFrame:
    """
    Resumo comparável dos dois lados para o mecanismo Pareto/auto-K: a
    topologia e o K escolhidos por fonte, com QE/TE do modelo final.

    Em 2026-08-31 este caminho deu Xavier 15×27/K=12 e CHIRPS 26×15/K=16 —
    topologias diferentes por fonte, mas a concordância local entre os SOMs
    ficou em 0.50/3, praticamente igual ao 0.49/3 dos parâmetros fixos
    (K=7, grade 21×21). Ou seja: a divergência entre as fontes NÃO é
    artefato da escolha de hiperparâmetro.
    """
    return pd.DataFrame(
        [
            {
                "fonte": p["source"],
                "som_size": p["som_size"],
                "qe": round(p["qe"], 3),
                "te": round(p["te"], 4),
                "k_escolhido": p["n_regimes"],
                "k_cotovelo": p["elbow_k"],
                "k_silhouette_bruto": p["silhouette_k_raw"],
                "n_candidatas_pareto": len(p["pareto_results"]),
                "tempo_s": p["elapsed_s"],
            }
            for p in (pareto_x, pareto_c)
        ]
    )


def summarize_sweep(sweep_x: dict, sweep_c: dict) -> pd.DataFrame:
    """
    Resumo da varredura g/k: para cada fonte e estação, o g do cotovelo da
    QE e o top-1 de k pelo score composto.
    """
    rows = []
    for s in (sweep_x, sweep_c):
        for season, season_result in s["result"].per_season.items():
            best = season_result.top3_configs[0] if season_result.top3_configs else None
            rows.append(
                {
                    "fonte": s["source"],
                    "estacao": season,
                    "g_cotovelo": season_result.g_sweep.optimal_g,
                    "k_composto": season_result.k_sweep.best_k_composite,
                    "k_silhouette": season_result.k_sweep.best_k_silhouette,
                    "k_davies_bouldin": season_result.k_sweep.best_k_davies_bouldin,
                    "qe_top1": round(best.quantization_error, 3) if best else None,
                    "silhouette_top1": round(best.silhouette, 3) if best else None,
                    "tempo_s": s["elapsed_s"],
                }
            )
    return pd.DataFrame(rows)
