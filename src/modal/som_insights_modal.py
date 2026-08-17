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
        # numpy<2 pinned para casar com o venv local (1.24.3): numpy>=2 usa
        # 'numpy._core' no pickle de np.float64/etc, que quebra a
        # desserialização do valor de retorno no processo local (numpy<2 só
        # tem 'numpy.core'). O trabalho remoto salva no volume antes do
        # return, mas sem o pin o `modal run` derruba com DeserializationError
        # depois de tudo já ter sido computado.
        "numpy<2", "pandas", "xarray[complete]", "netcdf4", "h5netcdf",
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
    )

app = modal.App("predep-som-insights", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _download_som_parquet(root: Path, out_key: str):
    """Baixa results/predep_som/{out_key}/ do volume para predep_visualization/."""
    viz_root = root.parent / "predep_visualization"
    local_dest = viz_root / "results" / "predep_som"
    local_dest.mkdir(parents=True, exist_ok=True)
    local_out = local_dest / out_key
    if local_out.exists():
        shutil.rmtree(local_out)
    remote_som = f"/results/predep_som/{out_key}"
    print(f"📥 Baixando {VOLUME_NAME}:{remote_som} → {local_dest}/ ...")
    subprocess.run(
        ["modal", "volume", "get", "--force", VOLUME_NAME,
         remote_som, str(local_dest)],
        check=False,
    )
    print(f"   ✅ em {local_out}")


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
    optimize: bool = False,
    auto_k: bool = False,
    variable: str = "alpha_core",
    mov_alpha_threshold: float = -1.0,
):
    """Roda o pipeline SOM lendo/escrevendo no Volume montado em /data.

    variable: "alpha_core" (PREDEP) ou "r2" (linha de base linear) — feature
    usada para treinar o SOM.
    mov_alpha_threshold: se >= 0, inclui apenas MoVs cujo score de α (PREDEP;
    média entre pixels do melhor lag×estação por pixel) exceda esse limiar.
    Sempre calculado sobre alpha_core, mesmo se variable="r2". Ignora `movs`
    se ambos forem passados. -1.0 (padrão) desativa o filtro.
    """
    sys.path.insert(0, "/root")

    from src.analysis.som_insights import (
        build_feature_matrix, train_som, assign_regimes,
        regime_to_grid, regime_fingerprints, movs_above_threshold,
    )
    from src.visualization import som_plots
    from src.visualization.cluster_plots import _load_basin_outlines
    from src.som_modeling.som_visualization import plot_pareto_curve, plot_elbow_curve, plot_dendrogram

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

    thr_tag = ""
    if mov_alpha_threshold is not None and mov_alpha_threshold >= 0:
        if mov_filter:
            print("[Modal] [AVISO] movs ignorado: mov_alpha_threshold define o filtro.")
        print(f"[Modal] Selecionando MoVs com α (PREDEP) > {mov_alpha_threshold}...")
        mov_filter, scores = movs_above_threshold(
            exp_dir, mov_alpha_threshold, lags=lags_list
        )
        for name in sorted(scores, key=scores.get, reverse=True):
            flag = "OK " if scores[name] > mov_alpha_threshold else "   "
            print(f"  [{flag}] {name:10s} score={scores[name]:.4f}")
        print(f"  → {len(mov_filter)}/{len(scores)} MoVs mantidos: {mov_filter}")
        if not mov_filter:
            raise ValueError("Nenhum MoV passou do limiar.")
        thr_tag = f"_gt{mov_alpha_threshold:g}".replace(".", "p")

    value_label = "α" if variable == "alpha_core" else "r²"

    print(f"[Modal] Construindo features de exp{exp_n:02d} (variável={variable})...")
    bundle = build_feature_matrix(exp_dir, mov_filter, lags_list,
                                  standardize=standardize, variable=variable)
    print(f"  Amostras: {bundle.X.shape[0]} | Features: {bundle.X.shape[1]} | "
          f"MoVs: {len(bundle.mov_names)}")

    print("[Modal] Treinando SOM...")
    som, metrics = train_som(bundle.X, som_size=som_size or None,
                             num_iteration=iterations, optimize=optimize)
    print(f"  {metrics}")

    regimes = assign_regimes(som, bundle.X, n_regimes=n_regimes, auto_k=auto_k,
                             max_k=len(bundle.mov_names))
    regime_grid = regime_to_grid(regimes, bundle)
    fps = regime_fingerprints(regimes, bundle)

    mov0 = bundle.mov_names[0]
    fallback = sorted(results_base.glob(f"exp*/{mov0}"))
    outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)

    som_subdir = ("som" if variable == "alpha_core" else f"som_{variable}") + thr_tag
    out_dir = Path("/data") / "plots" / f"exp{exp_n:02d}" / som_subdir
    if out_dir.exists():             # limpa saídas antigas no volume (sem órfãos)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = [
        som_plots.plot_regime_map(regime_grid, bundle, fps, out_dir, exp_n,
                                  basin_outlines=outlines),
        som_plots.plot_regime_panels(regime_grid, bundle, fps, out_dir, exp_n,
                                     basin_outlines=outlines),
        som_plots.plot_geographic_component_planes(bundle, out_dir, exp_n,
                                                   basin_outlines=outlines,
                                                   value_label=value_label),
        som_plots.plot_atypicality_map(som, regimes, bundle, out_dir, exp_n,
                                       basin_outlines=outlines),
        som_plots.plot_boundary_map(som, regimes, bundle, out_dir, exp_n,
                                    basin_outlines=outlines),
        som_plots.plot_regime_fingerprints(fps, bundle, out_dir, exp_n,
                                           value_label=value_label),
        som_plots.plot_regime_mov_heatmap(fps, bundle, out_dir, exp_n,
                                          value_label=value_label),
    ]
    
    if optimize and "pareto_results" in metrics:
        pareto_path = out_dir / f"som_pareto_curve_exp{exp_n:02d}.png"
        plot_pareto_curve(metrics["pareto_results"], save_path=str(pareto_path))
        pngs.append(pareto_path)
        
    if auto_k and regimes.kmeans_results:
        elbow_path = out_dir / f"som_elbow_curve_exp{exp_n:02d}.png"
        plot_elbow_curve(
            regimes.kmeans_results['k_values'], 
            regimes.kmeans_results['wss_values'], 
            elbow_k=regimes.n_regimes,
            save_path=str(elbow_path)
        )
        pngs.append(elbow_path)
        
    if auto_k and regimes.hierarchical_model:
        dendro_path = out_dir / f"som_dendrogram_exp{exp_n:02d}.png"
        plot_dendrogram(regimes.hierarchical_model, save_path=str(dendro_path))
        pngs.append(dendro_path)

    if diagnostics:
        pngs += [
            som_plots.plot_umatrix(som, out_dir, exp_n),
            som_plots.plot_hitmap(som, bundle.X, out_dir, exp_n),
            som_plots.plot_component_planes(som, bundle, out_dir, exp_n,
                                            value_label=value_label),
        ]

    # PDF (Pillow) — mesmo empacotamento de src/visualization/generate_plots_exp.py.
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
    exp_name: str = "",
    lags: str = "0,1,3,6,9,12",
    n_regimes_start: int = 2,
    n_regimes_end: int = 10,
    feature_mode: str = "best_mov",
    seed: int = 42,
    season: str = "",
    top_movs_k: int = 0,
    run_id: str = "",
    seasonal_transitions: bool = False,
    som_size: int = 0,
):
    """Gera som_pixels.parquet + som_meta.json no volume para cada n_regimes."""
    sys.path.insert(0, "/root")
    from src.analysis.som_to_parquet import export_experiment

    data_results = Path("/data") / "results" / "predep_granular_brazil"
    out_dir = Path("/data") / "results" / "predep_som"
    lags_list = [int(x.strip()) for x in lags.split(",") if x.strip()]
    exp_name = exp_name or f"exp{exp_n:02d}"

    for k in range(n_regimes_start, n_regimes_end + 1):
        print(f"\n[Modal] Exportando parquet n_regimes={k} ...")
        export_experiment(
            exp=exp_name, out_dir=out_dir, n_regimes=k,
            seed=seed, lags=lags_list, feature_mode=feature_mode,
            results_dir=data_results,
            season_filter=season or None,
            top_movs_k=top_movs_k,
            run_id=run_id or None,
            seasonal_transitions=seasonal_transitions,
            som_size=som_size or None,
        )

    volume.commit()
    out_key = run_id or exp_name
    print(f"[Modal] Parquets em {out_dir}/{out_key}/n*/")
    return {"exp": exp_name, "run_id": out_key,
            "n_range": [n_regimes_start, n_regimes_end]}


@app.function(volumes={"/data": volume}, cpu=8, memory=32768, timeout=21600,
             nonpreemptible=True)
def run_dual_parquet(
    exp_n: int = 1,
    exp_name: str = "",
    movs: str = "",
    lags: str = "0,1,3,6,9,12",
    mov_alpha_threshold: float = -1.0,
    seed: int = 42,
    iterations: int = 8,
    run_id: str = "",
    source: str = "nc",
    optimize: bool = True,
    auto_k: bool = True,
    n_regimes: int = 7,
    som_size: int = 0,
    use_epochs: bool = True,
    search_radius: int = 2,
    k_selection: str = "silhouette",
    standardize_method: str = "quantile",
    search_sample_size: int = 15000,
    search_epochs: int = 1,
    resume: bool = True,
    variables: str = "",
    k_margin_sigma: float = 1.0,
    basin_correct: bool = False,
):
    """
    Gera 1 parquet dual (PREDEP + r², mesmo filtro de MoVs) no volume, via
    export_dual_som — metodologia pesada (Pareto + silhouette) por padrão,
    ou rápida (grade heurística fixa + n_regimes fixo) se optimize/auto_k=False.

    movs: lista explícita de MoVs (vírgula) — tem prioridade sobre
    mov_alpha_threshold quando ambos são passados.

    use_epochs=True (padrão): `iterations` é o nº de ÉPOCAS completas do
    SOM, não um total fixo de atualizações — necessário pra treino
    adequado em datasets grandes (ver docstring de export_dual_som).
    timeout do container aumentado pra 6h pra acomodar o treino por época.

    search_sample_size/search_epochs: ver export_dual_som — a varredura
    de topologias do Pareto (~30-50 candidatas com search_radius=2) roda
    numa subamostra de `search_sample_size` pixels e `search_epochs`
    época(s), e só o modelo VENCEDOR é retreinado na base completa com
    `iterations` épocas. Sem isso, medido localmente em 70k pixels, 1
    época custa 113-320s conforme o tamanho da grade — a varredura
    completa na base inteira facilmente ultrapassa as 6h de timeout.

    resume=True (padrão): se o checkpoint de uma variável (PREDEP ou r²)
    já existe no volume (de uma tentativa anterior que estourou o tempo
    treinando a OUTRA variável), pula direto pro retreino/pula o retreino
    dela e só computa o que falta. O checkpoint de cada variável é
    commitado no volume assim que ela termina (não só no fim do job todo),
    então sobrevive mesmo que o container seja morto por timeout logo
    depois, no meio da segunda variável.

    source="parquet": lê de /data/results/predep_granular_brazil_src/{exp_name}
    (experimento Brasil-todo pré-computado, upload feito por `main()`).

    variables: "" (padrão) treina as DUAS (PREDEP e r², comportamento normal
    usado pela dashboard). Passe "r2" ou "alpha" para treinar só uma —
    bem mais rápido para testar uma mudança isolada; o parquet resultante
    só terá as colunas/meta dessa variável (não renderiza na visão
    lado-a-lado da dashboard, que espera as duas).

    basin_correct: só com source="parquet". Centra cada bacia na própria
    média antes da padronização global, corrigindo o viés sistemático de
    bacia que vem do α/r² ter sido calculado com ajuste independente por
    bacia (ver build_feature_matrix_from_parquet). Sem isso, bacias
    pequenas podem colapsar inteiras num único regime do SOM.
    """
    sys.path.insert(0, "/root")
    from src.analysis.som_to_parquet import export_dual_som

    data_results = Path("/data") / "results" / "predep_granular_brazil"
    out_dir = Path("/data") / "results" / "predep_som"
    lags_list = [int(x.strip()) for x in lags.split(",") if x.strip()]
    exp_name = exp_name or f"exp{exp_n:02d}"
    thr = None if mov_alpha_threshold is None or mov_alpha_threshold < 0 else mov_alpha_threshold
    mov_filter = [m.strip() for m in movs.split(",") if m.strip()] or None
    parquet_dir = Path("/data") / "results" / "predep_granular_brazil_src" / exp_name
    checkpoint_dir = Path("/data") / "checkpoints" / "predep_som" / (run_id or exp_name)
    variables_tuple = tuple(v.strip() for v in variables.split(",") if v.strip()) or None

    out_exp = export_dual_som(
        exp=exp_name, out_dir=out_dir, mov_alpha_threshold=thr,
        lags=lags_list, seed=seed, iterations=iterations,
        results_dir=data_results, run_id=run_id or None,
        source=source, parquet_dir=parquet_dir if source == "parquet" else None,
        optimize=optimize, auto_k=auto_k, n_regimes=n_regimes,
        som_size=som_size or None, mov_filter=mov_filter,
        use_epochs=use_epochs, search_radius=search_radius,
        k_selection=k_selection, standardize_method=standardize_method,
        search_sample_size=(search_sample_size or None),
        search_epochs=search_epochs, checkpoint_dir=checkpoint_dir,
        variables=variables_tuple, k_margin_sigma=k_margin_sigma,
        resume=resume, commit_cb=volume.commit,
        basin_correct=basin_correct,
    )

    volume.commit()
    out_key = run_id or exp_name
    print(f"[Modal] Parquet dual em {out_exp}")
    return {"exp": exp_name, "run_id": out_key, "mov_alpha_threshold": thr}


@app.local_entrypoint()
def main(
    exp_n: int = 1,
    exp_name: str = "",
    movs: str = "",
    lags: str = "0,1,3,6,9,12",
    som_size: int = 0,
    n_regimes: int = 7,
    iterations: int = 10000,
    dual_iterations: int = 8,
    standardize: bool = True,
    diagnostics: bool = False,
    optimize: bool = False,
    auto_k: bool = False,
    variable: str = "alpha_core",
    mov_alpha_threshold: float = -1.0,
    upload: bool = True,
    download: bool = True,
    parquets: bool = False,
    dual: bool = False,
    n_regimes_start: int = 2,
    n_regimes_end: int = 10,
    feature_mode: str = "best_mov",
    season: str = "",
    top_movs_k: int = 0,
    run_id: str = "",
    seasonal_transitions: bool = False,
    source: str = "nc",
    parquet_src: str = "",
    use_epochs: bool = True,
    search_radius: int = 2,
    k_selection: str = "silhouette",
    standardize_method: str = "quantile",
    search_sample_size: int = 15000,
    search_epochs: int = 1,
    resume: bool = True,
    variables: str = "",
    k_margin_sigma: float = 1.0,
    basin_correct: bool = False,
):
    ROOT = Path(__file__).resolve().parents[2]  # roda localmente
    _exp_name = exp_name or f"exp{exp_n:02d}"
    remote_exp = f"/{RESULTS_REL}/{_exp_name}"

    if upload and source != "parquet":
        local_exp = ROOT / RESULTS_REL / _exp_name
        if not local_exp.exists():
            raise FileNotFoundError(
                f"Experimento local não encontrado: {local_exp}")
        print(f"📤 Subindo {local_exp} → volume {VOLUME_NAME}:{remote_exp} ...")
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_exp), remote_exp)
        print("   ✅ upload concluído")

    if dual and source == "parquet" and upload:
        local_src = (Path(parquet_src) if parquet_src else
                    ROOT.parent / "predep_visualization" / RESULTS_REL / _exp_name)
        if not local_src.exists():
            raise FileNotFoundError(f"Parquet source não encontrado: {local_src}")
        remote_src = f"/results/predep_granular_brazil_src/{_exp_name}"
        print(f"📤 Subindo {local_src} → volume {VOLUME_NAME}:{remote_src} ...")
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local_src), remote_src)
        print("   ✅ upload (parquet) concluído")

    if dual:
        result = run_dual_parquet.remote(
            exp_n=exp_n, exp_name=exp_name, movs=movs, lags=lags,
            mov_alpha_threshold=mov_alpha_threshold, seed=42,
            iterations=dual_iterations, run_id=run_id, source=source,
            optimize=optimize, auto_k=auto_k, n_regimes=n_regimes,
            som_size=som_size, use_epochs=use_epochs,
            search_radius=search_radius, k_selection=k_selection,
            standardize_method=standardize_method,
            search_sample_size=search_sample_size,
            search_epochs=search_epochs, resume=resume, variables=variables,
            k_margin_sigma=k_margin_sigma, basin_correct=basin_correct,
        )
        print(f"\n📦 {result}")
        if download:
            _download_som_parquet(ROOT, run_id or _exp_name)
        return

    if parquets:
        result = run_som_parquets.remote(
            exp_n=exp_n, exp_name=exp_name, lags=lags,
            n_regimes_start=n_regimes_start, n_regimes_end=n_regimes_end,
            feature_mode=feature_mode, seed=42,
            season=season, top_movs_k=top_movs_k,
            run_id=run_id, seasonal_transitions=seasonal_transitions,
            som_size=som_size,
        )
        print(f"\n📦 {result}")
        if download:
            _download_som_parquet(ROOT, run_id or _exp_name)
        return

    result = run_som.remote(
        exp_n=exp_n, movs=movs, lags=lags, som_size=som_size,
        n_regimes=n_regimes, iterations=iterations, standardize=standardize,
        diagnostics=diagnostics, optimize=optimize, auto_k=auto_k,
        variable=variable, mov_alpha_threshold=mov_alpha_threshold,
    )
    print(f"\n📊 {result}")

    if download:
        # modal volume get copia o dir remoto DENTRO do destino; aponta-se o
        # pai para resultar em plots/exp{NN}/{som_subdir}/ (sem aninhar).
        thr_tag = ("" if mov_alpha_threshold is None or mov_alpha_threshold < 0
                   else f"_gt{mov_alpha_threshold:g}".replace(".", "p"))
        som_subdir = ("som" if variable == "alpha_core" else f"som_{variable}") + thr_tag
        local_parent = ROOT / "plots" / f"exp{exp_n:02d}"
        local_parent.mkdir(parents=True, exist_ok=True)
        local_som = local_parent / som_subdir
        if local_som.exists():
            shutil.rmtree(local_som)
        remote_som = f"/plots/exp{exp_n:02d}/{som_subdir}"
        print(f"📥 Baixando {VOLUME_NAME}:{remote_som} → {local_parent}/ ...")
        subprocess.run(
            ["modal", "volume", "get", "--force", VOLUME_NAME,
             remote_som, str(local_parent)],
            check=False,
        )
        print(f"   ✅ plots em {local_parent / 'som'}")
