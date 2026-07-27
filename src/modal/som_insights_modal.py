#!/usr/bin/env python3
"""
Wrapper Modal para os insights de SOM (MiniSom) do PREDEP.

Reusa AS MESMAS funções de src/analysis/som_insights.py e
src/visualization/som_plots.py — nada da lógica é reescrito. O pacote `src/`
é embarcado na imagem; o minisom vem do PyPI (no laptop é vendorizado).

Fluxo (dados locais → nuvem):
  1. (local_entrypoint) sobe results/predep_granular_brazil/exp{NN} para o Volume
  2. (remoto) lê /data/results/.../exp{NN}, treina o SOM, salva os PNG/PDF em
     /data/plots/exp{NN}/som e faz volume.commit()
  3. (local_entrypoint) baixa /plots/exp{NN}/som para plots/exp{NN}/som local

Uso (com o CLI Modal global):
  modal run src/modal/som_insights_modal.py --exp-n 1
  modal run src/modal/som_insights_modal.py --exp-n 1 --movs "ONI,NAO" \
      --som-size 12 --n-regimes 5 --no-standardize
  # pular upload/download (dados já no volume):
  modal run src/modal/som_insights_modal.py --exp-n 1 --no-upload --no-download

Observação: o SOM é leve (~40s para exp01). O Modal só vale a pena se os NCs
já moram num Volume ou para encadear na pipeline da nuvem.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import modal

VOLUME_NAME = "irc-predep-data"  # volume deste projeto (ajuste se necessário)
RESULTS_REL = "results/predep_granular_brazil"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "pandas", "xarray[complete]", "netcdf4", "h5netcdf",
        "scipy", "scikit-learn", "matplotlib", "pillow", "minisom",
    )
)

# O módulo é reimportado dentro do container (em /root/som_insights_modal.py,
# plano), onde parents[2] não existe. Por isso ROOT e o mount do pacote src/
# só são resolvidos localmente. No container, src/ já está em /root/src e o
# minisom vem do PyPI (o shim em som_insights cai no pacote instalado).
if modal.is_local():
    _ROOT = Path(__file__).resolve().parents[2]
    image = (
        image
        .add_local_dir(str(_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(_ROOT / "scripts"), remote_path="/root/scripts")
    )

app = modal.App("predep-som-insights", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(volumes={"/data": volume}, cpu=4, memory=16384, timeout=3600)
def run_som(
    exp_n: int = 1,
    movs: str = "",
    lags: str = "0,1,3,6,9,12",
    som_size: int = 0,
    n_regimes: int = 7,
    iterations: int = 10000,
    standardize: bool = True,
    diagnostics: bool = False,
):
    """Roda o pipeline SOM lendo/escrevendo no Volume montado em /data."""
    sys.path.insert(0, "/root")

    from src.analysis.som_insights import (
        build_feature_matrix, train_som, assign_regimes,
        regime_to_grid, regime_fingerprints,
    )
    from src.visualization import som_plots
    from src.visualization.cluster_plots import _load_basin_outlines

    results_base = Path("/data") / RESULTS_REL
    exp_dir = results_base / f"exp{exp_n:02d}"
    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Experimento não encontrado no volume: {exp_dir}. "
            f"Rode com upload (sem --no-upload)."
        )

    mov_filter = [m.strip() for m in movs.split(",") if m.strip()] or None
    lags_list = (None if lags.strip().lower() == "all"
                 else [int(x.strip()) for x in lags.split(",") if x.strip()])

    print(f"[Modal] Construindo features de exp{exp_n:02d}...")
    bundle = build_feature_matrix(exp_dir, mov_filter, lags_list,
                                  standardize=standardize)
    print(f"  Amostras: {bundle.X.shape[0]} | Features: {bundle.X.shape[1]} | "
          f"MoVs: {len(bundle.mov_names)}")

    print("[Modal] Treinando SOM...")
    som, metrics = train_som(bundle.X, som_size=som_size or None,
                             num_iteration=iterations)
    print(f"  {metrics}")

    regimes = assign_regimes(som, bundle.X, n_regimes=n_regimes)
    regime_grid = regime_to_grid(regimes, bundle)
    fps = regime_fingerprints(regimes, bundle)

    mov0 = bundle.mov_names[0]
    fallback = sorted(results_base.glob(f"exp*/{mov0}"))
    outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)

    out_dir = Path("/data") / "plots" / f"exp{exp_n:02d}" / "som"
    if out_dir.exists():             # limpa saídas antigas no volume (sem órfãos)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = [
        som_plots.plot_regime_map(regime_grid, bundle, fps, out_dir, exp_n,
                                  basin_outlines=outlines),
        som_plots.plot_regime_panels(regime_grid, bundle, fps, out_dir, exp_n,
                                     basin_outlines=outlines),
        som_plots.plot_geographic_component_planes(bundle, out_dir, exp_n,
                                                   basin_outlines=outlines),
        som_plots.plot_atypicality_map(som, regimes, bundle, out_dir, exp_n,
                                       basin_outlines=outlines),
        som_plots.plot_boundary_map(som, regimes, bundle, out_dir, exp_n,
                                    basin_outlines=outlines),
        som_plots.plot_regime_fingerprints(fps, bundle, out_dir, exp_n),
        som_plots.plot_regime_mov_heatmap(fps, bundle, out_dir, exp_n),
    ]
    if diagnostics:
        pngs += [
            som_plots.plot_umatrix(som, out_dir, exp_n),
            som_plots.plot_hitmap(som, bundle.X, out_dir, exp_n),
            som_plots.plot_component_planes(som, bundle, out_dir, exp_n),
        ]

    # PDF (Pillow) — mesmo empacotamento de scripts/generate_plots_exp.py.
    from PIL import Image
    pdf_path = out_dir / f"som_insights_exp{exp_n:02d}.pdf"
    imgs = [Image.open(str(p)).convert("RGB") for p in pngs]
    imgs[0].save(str(pdf_path), save_all=True, append_images=imgs[1:])
    for im in imgs:
        im.close()

    volume.commit()
    print(f"[Modal] Concluído. Saída em {out_dir} (volume {VOLUME_NAME}).")
    return {"metrics": metrics, "files": [p.name for p in pngs] + [pdf_path.name]}


@app.function(volumes={"/data": volume}, cpu=4, memory=16384, timeout=3600)
def run_som_parquets(
    exp_n: int = 1,
    lags: str = "0,1,3,6,9,12",
    n_regimes_start: int = 2,
    n_regimes_end: int = 10,
    feature_mode: str = "best_mov",
    seed: int = 42,
):
    """Gera som_pixels.parquet + som_meta.json no volume para cada n_regimes."""
    sys.path.insert(0, "/root")
    from scripts.som_to_parquet import export_experiment

    data_results = Path("/data") / "results" / "predep_granular_brazil"
    out_dir = Path("/data") / "results" / "predep_som"
    lags_list = [int(x.strip()) for x in lags.split(",") if x.strip()]

    for k in range(n_regimes_start, n_regimes_end + 1):
        print(f"\n[Modal] Exportando parquet n_regimes={k} ...")
        export_experiment(
            exp=f"exp{exp_n:02d}", out_dir=out_dir, n_regimes=k,
            seed=seed, lags=lags_list, feature_mode=feature_mode,
            results_dir=data_results,
        )

    volume.commit()
    print(f"[Modal] Parquets em {out_dir}/exp{exp_n:02d}/n*/")
    return {"exp": f"exp{exp_n:02d}", "n_range": [n_regimes_start, n_regimes_end]}


@app.local_entrypoint()
def main(
    exp_n: int = 1,
    movs: str = "",
    lags: str = "0,1,3,6,9,12",
    som_size: int = 0,
    n_regimes: int = 7,
    iterations: int = 10000,
    standardize: bool = True,
    diagnostics: bool = False,
    upload: bool = True,
    download: bool = True,
    parquets: bool = False,
    n_regimes_start: int = 2,
    n_regimes_end: int = 10,
    feature_mode: str = "best_mov",
):
    ROOT = Path(__file__).resolve().parents[2]  # roda localmente
    remote_exp = f"/{RESULTS_REL}/exp{exp_n:02d}"

    if upload:
        local_exp = ROOT / RESULTS_REL / f"exp{exp_n:02d}"
        if not local_exp.exists():
            raise FileNotFoundError(
                f"Experimento local não encontrado: {local_exp}")
        print(f"📤 Subindo {local_exp} → volume {VOLUME_NAME}:{remote_exp} ...")
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_exp), remote_exp)
        print("   ✅ upload concluído")

    if parquets:
        result = run_som_parquets.remote(
            exp_n=exp_n, lags=lags,
            n_regimes_start=n_regimes_start, n_regimes_end=n_regimes_end,
            feature_mode=feature_mode, seed=42,
        )
        print(f"\n📦 {result}")

        if download:
            # baixa results/predep_som/exp{NN}/ → predep_visualization/results/predep_som/exp{NN}/
            viz_root = ROOT.parent / "predep_visualization"
            local_dest = viz_root / "results" / "predep_som"
            local_dest.mkdir(parents=True, exist_ok=True)
            local_exp_som = local_dest / f"exp{exp_n:02d}"
            if local_exp_som.exists():
                shutil.rmtree(local_exp_som)
            remote_som = f"/results/predep_som/exp{exp_n:02d}"
            print(f"📥 Baixando {VOLUME_NAME}:{remote_som} → {local_dest}/ ...")
            subprocess.run(
                ["modal", "volume", "get", "--force", VOLUME_NAME,
                 remote_som, str(local_dest)],
                check=False,
            )
            print(f"   ✅ parquets em {local_exp_som}")
        return

    result = run_som.remote(
        exp_n=exp_n, movs=movs, lags=lags, som_size=som_size,
        n_regimes=n_regimes, iterations=iterations, standardize=standardize,
        diagnostics=diagnostics,
    )
    print(f"\n📊 {result}")

    if download:
        # modal volume get copia o dir remoto DENTRO do destino; aponta-se o
        # pai para resultar em plots/exp{NN}/som/ (sem aninhar som/som).
        local_parent = ROOT / "plots" / f"exp{exp_n:02d}"
        local_parent.mkdir(parents=True, exist_ok=True)
        local_som = local_parent / "som"
        if local_som.exists():
            shutil.rmtree(local_som)
        remote_som = f"/plots/exp{exp_n:02d}/som"
        print(f"📥 Baixando {VOLUME_NAME}:{remote_som} → {local_parent}/ ...")
        subprocess.run(
            ["modal", "volume", "get", "--force", VOLUME_NAME,
             remote_som, str(local_parent)],
            check=False,
        )
        print(f"   ✅ plots em {local_parent / 'som'}")
