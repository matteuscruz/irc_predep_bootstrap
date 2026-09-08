"""
Stage 2 — o cálculo PREDEP de PRODUÇÃO.

Esta é a pipeline real do projeto: `experiments/08_predep_parana_recompute.py`
rodando no **Modal**, um container por MoV (em lotes de `max_parallel`),
para cada fonte de precipitação. Escala: 7503 pixels × 23 MoVs × 13 lags ×
4 estações × 2 fontes — horas de compute distribuído, impossível de rodar
dentro de um `kedro run` local.

Por isso os nós aqui **representam** o job (mesmos passos, mesmos caminhos,
mesma semântica de skip-if-exists que o `main()` de produção já tem) e, por
padrão (`params:allow_modal_dispatch: false`), operam em **modo resume**:
verificam/carregam os NetCDFs que a produção já gerou, sem gastar crédito
Modal. Com `allow_modal_dispatch: true` o nó de dispatch chama o job de
verdade.
"""

import subprocess
import sys
from pathlib import Path

# kedro/src/predep_viz/pipelines/predep_production/nodes.py -> raiz do repo
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def _nc_path(exp_dir: str, source: str, mov: str) -> Path:
    return (
        ROOT
        / "results/predep_granular_brazil"
        / exp_dir
        / source
        / mov
        / f"{mov}_predep_granular_seasonal_{source}.nc"
    )


def prepare_modal_inputs(source: str, movs: list, paths: dict, mask_info=None) -> dict:
    """
    Espelha os uploads que a produção faz para o volume Modal `xavier-data`
    antes de lançar os containers (`experiments/08`: `upload_precip_data`,
    `upload_combined_mask`, `upload_mov_index`):

    - o NetCDF mensal da fonte (Xavier ou CHIRPS regridado);
    - a máscara combinada do Paraná (`parana_combined_mask.npz`);
    - um CSV normalizado por MoV (colunas `time`,`value`) — 23 MoVs.

    Aqui o nó apenas **verifica** que esses insumos existem localmente e
    descreve o que seria enviado; o upload real acontece dentro do
    `modal run` (nó seguinte), que é quem tem a sessão autenticada.

    `mask_info` existe só para criar a dependência explícita no grafo com a
    pipeline `data_prep` quando as duas rodam juntas.
    """
    precip = ROOT / (paths["xavier_monthly"] if source == "xavier" else paths["chirps_monthly"])
    mask = ROOT / paths["parana_mask"]
    missing = [str(p) for p in (precip, mask) if not p.exists()]

    mov_files = {}
    for mov in movs:
        matches = list((ROOT / paths["mov_indices_dir"]).glob(f"{mov}_indice*.csv"))
        if matches:
            mov_files[mov] = matches[0].name
        else:
            missing.append(f"índice do MoV {mov}")

    if missing:
        raise FileNotFoundError(
            f"Insumos ausentes para a fonte '{source}': {missing}"
        )

    return {
        "source": source,
        "precip_file": precip.name,
        "mask_file": mask.name,
        "n_movs": len(mov_files),
        "modal_volume": "xavier-data",
    }


def dispatch_modal_predep(
    modal_inputs: dict, exp_dir: str, movs: list, run_params: dict
) -> dict:
    """
    O passo de PRODUÇÃO: lança `experiments/08_predep_parana_recompute.py`
    no Modal — um container (8 CPU / 8 GB) por MoV, em lotes de
    `max_parallel`, cada um calculando `predep()` + `regression_calc()`
    pixel a pixel (7503 px × 13 lags × 4 estações) e gravando um NetCDF
    `alpha_core`/`r2` por MoV.

    Comportamento (igual ao `main()` de produção, que já pula MoVs cujo NC
    final existe):
    - `allow_modal_dispatch: false` (padrão) → **modo resume**: não chama o
      Modal; verifica quais NCs já existem em disco e falha com mensagem
      clara se faltar algum. É assim que esta pipeline roda localmente sem
      gastar crédito.
    - `allow_modal_dispatch: true` → executa
      `modal run experiments/08_predep_parana_recompute.py --movs "all"
      --sources "<source>" --seed ... --n-boot ... --max-parallel ...`.

    Nota operacional (2026-08-31): o workspace Modal está com o **spend
    limit excedido**, então o dispatch real falha até isso ser ajustado no
    dashboard — outro motivo do padrão ser o modo resume.
    """
    source = modal_inputs["source"]
    existing = [mov for mov in movs if _nc_path(exp_dir, source, mov).exists()]
    pending = [mov for mov in movs if mov not in existing]

    if not run_params.get("allow_modal_dispatch", False):
        if pending:
            raise FileNotFoundError(
                f"[{source}] modo resume: faltam {len(pending)} de {len(movs)} MoVs em "
                f"results/predep_granular_brazil/{exp_dir}/{source}/ ({pending[:5]}...). "
                "Rode a produção no Modal (allow_modal_dispatch: true) ou aponte "
                "params:exp_dir para um experimento já calculado."
            )
        return {
            "source": source,
            "mode": "resume",
            "dispatched": 0,
            "already_computed": len(existing),
        }

    cmd = [
        str(ROOT / ".venv/bin/python"), "-m", "modal", "run",
        str(ROOT / "experiments/08_predep_parana_recompute.py"),
        "--movs", "all",
        "--sources", source,
        "--seed", str(run_params["seed"]),
        "--n-boot", str(run_params["n_boot"]),
        "--max-parallel", str(run_params["max_parallel"]),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[{source}] modal run falhou (rc={result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    return {
        "source": source,
        "mode": "modal_dispatch",
        "dispatched": len(pending),
        "already_computed": len(existing),
    }


def load_predep_stack(dispatch_info: dict, exp_dir: str, source: str) -> dict:
    """
    Carrega e alinha os 23 NetCDFs por MoV que a produção gerou, usando a
    função real já usada pelos scripts de plot
    (`src/visualization/cluster_plots.py:_load_mov_stack`).

    Retorna os arrays empilhados `(n_movs, n_seasons, n_lags, n_lat, n_lon)`
    de `alpha_core` e `r2`, mais os eixos — a entrada da pipeline `analysis`.
    """
    from src.visualization.cluster_plots import _load_mov_stack

    exp_path = ROOT / "results/predep_granular_brazil" / exp_dir / source
    ac, r2, movs, lats, lons, lags = _load_mov_stack(
        exp_path, nc_suffix=f"_predep_granular_seasonal_{source}.nc"
    )
    if not movs:
        raise FileNotFoundError(f"Nenhum NC encontrado em {exp_path}")
    return {
        "alpha_core": ac,
        "r2": r2,
        "mov_names": movs,
        "lats": lats,
        "lons": lons,
        "lags": lags,
        "source": source,
    }


def summarize_predep_stack(stack: dict) -> "object":
    """
    Resumo tabular do que a produção entregou para esta fonte: nº de MoVs,
    dimensões da grade, lags/estações disponíveis e o footprint de pixels
    válidos — a checagem rápida de que os dois lados (Xavier e CHIRPS) são
    comparáveis pixel a pixel.
    """
    import numpy as np
    import pandas as pd

    ac = stack["alpha_core"]
    valid = int(np.isfinite(ac[0, 0, 0]).sum())
    return pd.DataFrame(
        [
            {
                "source": stack["source"],
                "n_movs": len(stack["mov_names"]),
                "n_seasons": ac.shape[1],
                "n_lags": ac.shape[2],
                "grid": f"{ac.shape[3]}x{ac.shape[4]}",
                "pixels_validos": valid,
                "lags": ",".join(str(x) for x in stack["lags"]),
            }
        ]
    )
