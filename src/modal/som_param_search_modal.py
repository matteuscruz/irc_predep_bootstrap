#!/usr/bin/env python3
"""
Wrapper Modal para busca automática de hiperparâmetros SOM do PREDEP.

Varre g (tamanho do grid SOM) e k (nº de regimes KMeans) para cada lag
individualmente (0, 1, 3, 6, 9, 12) e para todos os lags combinados, rodando
em paralelo via Modal. Para cada cenário, avalia por estação (DJF, MAM, JJA,
SON) e para todas as estações juntas.

Saída em volume:
  results/predep_som_param_search/{exp}/lag_0/param_search.json
  results/predep_som_param_search/{exp}/lag_1/param_search.json
  ...
  results/predep_som_param_search/{exp}/lag_all/param_search.json

Uso:
  modal run src/modal/som_param_search_modal.py --exp-name exp_brasil
  modal run src/modal/som_param_search_modal.py --exp-name exp_brasil \\
      --lags "0,1,3,6,9,12" --g-min 5 --g-max 15 --k-min 2 --k-max 12
  # dados já no volume (pular upload):
  modal run src/modal/som_param_search_modal.py --exp-name exp_brasil --no-upload
"""

import subprocess
import sys
from pathlib import Path

import modal

VOLUME_NAME = "irc-predep-data"
RESULTS_REL = "results/predep_granular_brazil"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "xarray[complete]", "netcdf4", "h5netcdf",
        "scipy", "scikit-learn", "matplotlib", "pillow", "minisom",
    )
)

if modal.is_local():
    _ROOT = Path(__file__).resolve().parents[2]
    image = (
        image
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "minisom"), remote_path="/root/minisom")
    )

app = modal.App("predep-som-param-search", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(volumes={"/data": volume}, cpu=8, memory=32768, timeout=7200)
def run_param_search(
    exp_n: int = 1,
    exp_name: str = "",
    movs: str = "",
    lags: str = "",
    lag_label: str = "all",
    g_min: int = 5,
    g_max: int = 15,
    k_min: int = 2,
    k_max: int = 12,
    num_iteration: int = 10000,
    seed: int = 42,
    standardize: bool = True,
    seasons: str = "",
    include_all: bool = True,
) -> dict:
    """
    Executa run_full_search no container Modal para um subconjunto de lags.

    lag_label é usado apenas para nomear o subdiretório de saída.
    Retorna o dict serializado do ParamSearchResult.
    """
    sys.path.insert(0, "/root")

    from src.analysis.som_param_search import run_full_search, save_results_json

    _exp_name = exp_name or f"exp{exp_n:02d}"
    exp_dir = Path("/data") / RESULTS_REL / _exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Experimento não encontrado no volume: {exp_dir}. "
            "Rode sem --no-upload para subir os dados."
        )

    mov_filter = [m.strip() for m in movs.split(",") if m.strip()] or None
    lags_list = [int(x.strip()) for x in lags.split(",") if x.strip()] or None
    seasons_list = [s.strip() for s in seasons.split(",") if s.strip()] or None
    g_range = list(range(g_min, g_max + 1))
    k_range = list(range(k_min, k_max + 1))

    print(f"[Modal] {_exp_name} | lag={lag_label} | MoVs={mov_filter}")
    print(f"        g_range={g_range} | k_range={k_range} | seasons={seasons_list}")

    result = run_full_search(
        exp_dir,
        g_range=g_range,
        k_range=k_range,
        mov_filter=mov_filter,
        lags=lags_list,
        standardize=standardize,
        num_iteration=num_iteration,
        seed=seed,
        seasons=seasons_list,
        include_all=include_all,
        verbose=True,
    )

    out_path = (
        Path("/data") / "results" / "predep_som_param_search"
        / _exp_name / f"lag_{lag_label}" / "param_search.json"
    )
    save_results_json(result, out_path)
    volume.commit()

    rec = result.overall_recommendation
    print(f"[Modal] lag={lag_label} → g={rec.g}, k={rec.k} "
          f"(score={rec.composite_score:.4f}, sil={rec.silhouette:.4f})")
    return result.to_dict()


@app.local_entrypoint()
def main(
    exp_n: int = 1,
    exp_name: str = "",
    movs: str = "",
    lags: str = "0,1,3,6,9,12",
    g_min: int = 5,
    g_max: int = 15,
    k_min: int = 2,
    k_max: int = 12,
    num_iteration: int = 10000,
    seed: int = 42,
    standardize: bool = True,
    seasons: str = "",
    include_all: bool = True,
    per_lag: bool = True,
    upload: bool = True,
    download: bool = True,
):
    ROOT = Path(__file__).resolve().parents[2]
    _exp_name = exp_name or f"exp{exp_n:02d}"
    remote_exp = f"/{RESULTS_REL}/{_exp_name}"

    # ── Fase 1: upload ───────────────────────────────────────────────────────────
    if upload:
        local_exp = ROOT / RESULTS_REL / _exp_name
        if not local_exp.exists():
            raise FileNotFoundError(
                f"Experimento local não encontrado: {local_exp}"
            )
        print(f"📤 Subindo {local_exp} → volume {VOLUME_NAME}:{remote_exp} ...")
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_exp), remote_exp)
        print("   ✅ upload concluído")

    # ── Fase 2: buscas em paralelo ───────────────────────────────────────────────
    lags_list = [int(x.strip()) for x in lags.split(",") if x.strip()]

    common = dict(
        exp_n=exp_n, exp_name=exp_name, movs=movs,
        g_min=g_min, g_max=g_max, k_min=k_min, k_max=k_max,
        num_iteration=num_iteration, seed=seed,
        standardize=standardize, seasons=seasons, include_all=include_all,
    )

    if per_lag:
        # Um spawn por lag individual + um para todos combinados
        spawn_jobs = [
            (f"lag{lag}", run_param_search.spawn(lags=str(lag), lag_label=f"lag{lag}", **common))
            for lag in lags_list
        ]
        spawn_jobs.append(
            ("all", run_param_search.spawn(lags=lags, lag_label="all", **common))
        )
        print(f"🚀 {len(spawn_jobs)} buscas lançadas em paralelo "
              f"({len(lags_list)} lags + all)...")

        print("\n📊 Aguardando resultados:")
        for lag_label, future in spawn_jobs:
            result = future.get()
            rec = result.get("overall_recommendation", {})
            print(f"  lag={lag_label:6s} → g={rec.get('g'):3d}, k={rec.get('k'):3d}  "
                  f"score={rec.get('composite_score', 0):.4f}")
    else:
        # Busca única com todos os lags combinados
        result = run_param_search.remote(
            lags=lags, lag_label="all", **common,
        )
        rec = result.get("overall_recommendation", {})
        print(f"\n★ g={rec.get('g')}, k={rec.get('k')}  "
              f"score={rec.get('composite_score', 0):.4f}")

    # ── Fase 3: download ─────────────────────────────────────────────────────────
    if download:
        remote_json = f"/results/predep_som_param_search/{_exp_name}"
        local_dest = ROOT / "results" / "predep_som_param_search"
        local_dest.mkdir(parents=True, exist_ok=True)
        print(f"\n📥 Baixando {VOLUME_NAME}:{remote_json} → {local_dest}/ ...")
        subprocess.run(
            ["modal", "volume", "get", "--force", VOLUME_NAME,
             remote_json, str(local_dest)],
            check=False,
        )
        print(f"   ✅ resultados em {local_dest / _exp_name}")
