#!/usr/bin/env python3
"""
Exporta os mapas espaciais do SOM (regimes, atipicidade, fronteiras e
component planes por MoV) como artefatos leves para a dashboard
predep_visualization, espelhando o src/data_processing/export/nc_to_parquet.py.

Pré-computa o SOM (mesmos hiperparâmetros do generate_som_insights.py, para
casar com os PNGs estáticos) e escreve:
    {out_dir}/{exp}/som_pixels.parquet   (1 linha por pixel)
    {out_dir}/{exp}/som_meta.json        (regimes, cores, escalas, métricas)

A dashboard apenas renderiza esses artefatos (não precisa de minisom/sklearn).

Uso:
    python src/analysis/som_to_parquet.py --exp exp01
    python src/analysis/som_to_parquet.py --exp exp01 --out-dir /caminho --n-regimes 7
"""

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import xarray as xr

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.som_insights import (  # noqa: E402
    build_feature_matrix, build_feature_matrix_from_parquet, train_som,
    assign_regimes, regime_fingerprints, auto_top_movs, build_precip_bundle,
    seasonal_regimes, movs_above_threshold, local_mov_importance,
)
from src.visualization.som_plots import (  # noqa: E402
    _make_regime_cmap, _regime_mov_matrix,
)
from src.visualization.cluster_plots import (  # noqa: E402
    _load_basin_outlines, BASIN_LABELS,
)

RESULTS_DIR = ROOT / "results" / "predep_granular_brazil"
DEFAULT_OUT = ROOT.parent / "predep_visualization" / "results" / "predep_som"
DEFAULT_LAGS = [0, 1, 3, 6, 9, 12]
PRECIP_NC = (ROOT / "data" / "processed" / "xavier_monthly"
             / "pr_monthly_1961_2019_BRAZIL_combined.nc")


def _basin_per_pixel(exp_dir, mov0, pix_lat, pix_lon):
    """Atribui cada pixel a uma bacia via máscaras dos NCs por bacia."""
    fallback = sorted((exp_dir.parent).glob(f"exp*/{mov0}"))
    outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)
    num_to_name = {v: k for k, v in BASIN_LABELS.items()}
    coord_to_basin = {}
    for number, blats, blons, mask in outlines:
        name = num_to_name.get(number, str(number))
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            coord_to_basin[(round(float(blats[r]), 2),
                            round(float(blons[c]), 2))] = name
    return [
        coord_to_basin.get((round(float(la), 2), round(float(lo), 2)),
                           "desconhecida")
        for la, lo in zip(pix_lat, pix_lon)
    ]


def _collapse_best_mov(bundle):
    """Colapsa as features para o melhor α de cada MoV (máx sobre lag×estação).

    Resultado: 1 feature por MoV ("quão preditivo este MoV é aqui, no seu
    melhor lag/estação"). O SOM passa a agrupar pixels por QUAIS MoVs predizem.
    """
    mov_ids = bundle.feature_index[:, 0]
    n_mov = len(bundle.mov_names)
    xr_best = np.column_stack([
        bundle.X_raw[:, mov_ids == m].max(axis=1) for m in range(n_mov)
    ])
    mean = xr_best.mean(axis=0)
    std = xr_best.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    x_z = (xr_best - mean) / std_safe
    # feature_index: 1 coluna por MoV (lag/season marcados como -1)
    feat_index = np.column_stack([
        np.arange(n_mov), -np.ones(n_mov), -np.ones(n_mov)
    ]).astype(int)
    return replace(
        bundle, X=x_z, X_raw=xr_best, feature_index=feat_index,
        feature_names=[f"{m}|best" for m in bundle.mov_names],
        scaler_mean=mean, scaler_std=std_safe,
    )


def export_experiment(exp, out_dir, n_regimes=7, seed=42, lags=None,
                      feature_mode="full", combo_k=3, results_dir=None,
                      season_filter=None, top_movs_k=0, run_id=None,
                      seasonal_transitions=False, som_size=None,
                      strict=False, mov_filter=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    exp_dir = results_dir / exp
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experimento não encontrado: {exp_dir}")
    lags = lags if lags is not None else DEFAULT_LAGS
    out_key = run_id or exp   # chave usada no caminho de saída

    print(f"\n{'='*60}\nSOM → parquet: {out_key}  (feature_mode={feature_mode})"
          f"\nSaída: {out_dir / out_key}\n{'='*60}")

    # ── pipeline SOM ───────────────────────────────────────────────────────
    if feature_mode == "precip_latlon":
        if not PRECIP_NC.exists():
            raise FileNotFoundError(f"NC de precipitação não encontrado: {PRECIP_NC}")
        bundle = build_precip_bundle(exp_dir, PRECIP_NC, standardize=True)
    else:
        if top_movs_k > 0:
            # Ranking sem filtro de MoV para selecionar top-k.
            _full = build_feature_matrix(exp_dir, mov_filter=None, lags=lags,
                                         standardize=False, strict=strict)
            mov_filter = auto_top_movs(_full, top_movs_k)
            print(f"  Top-{top_movs_k} MoVs: {mov_filter}")
        bundle = build_feature_matrix(exp_dir, mov_filter=mov_filter, lags=lags,
                                      standardize=True,
                                      season_filter=season_filter,
                                      strict=strict)
    if feature_mode == "best_mov":
        bundle = _collapse_best_mov(bundle)
    som, metrics = train_som(bundle.X, som_size=som_size, seed=seed)
    regimes = assign_regimes(som, bundle.X, n_regimes=n_regimes, seed=seed)
    fps = regime_fingerprints(regimes, bundle)
    print(f"  amostras={bundle.X.shape[0]} features={bundle.X.shape[1]} "
          f"MoVs={len(bundle.mov_names)} regimes={regimes.n_regimes}")

    # ── arrays por pixel (mesmas fórmulas dos plots estáticos) ─────────────
    W = som.get_weights()
    bmu = regimes.bmu_rc
    w_pix = W[bmu[:, 0], bmu[:, 1]]
    atypicality = np.linalg.norm(bundle.X - w_pix, axis=1)
    boundary = som.distance_map()[bmu[:, 0], bmu[:, 1]]

    pix_lat = bundle.lats[bundle.pixel_rc[:, 0]]
    pix_lon = bundle.lons[bundle.pixel_rc[:, 1]]
    # precip_latlon não tem MoV tradicional; usa lat/lon como fallback de bacia
    _mov0 = bundle.mov_names[0] if feature_mode != "precip_latlon" else None
    basin = (_basin_per_pixel(exp_dir, _mov0, pix_lat, pix_lon)
             if _mov0 else ["desconhecida"] * len(pix_lat))

    df = pd.DataFrame({
        "latitude": pix_lat.astype("float32"),
        "longitude": pix_lon.astype("float32"),
        "basin": pd.Categorical(basin),
        "regime": regimes.pixel_regime.astype("int8"),
        "atypicality": atypicality.astype("float32"),
        "boundary": boundary.astype("float32"),
    })

    # Transições sazonais (apenas feature_mode full, requer season_ids >= 0).
    if seasonal_transitions and feature_mode not in ("best_mov", "precip_latlon"):
        seas_regs = seasonal_regimes(som, regimes, bundle)
        for season, reg_arr in seas_regs.items():
            df[f"regime_{season}"] = reg_arr.astype("int8")

    mov_ids = bundle.feature_index[:, 0]
    comp_min, comp_max = np.inf, -np.inf
    for m, mov in enumerate(bundle.mov_names):
        cols = np.nonzero(mov_ids == m)[0]
        vals = bundle.X_raw[:, cols].mean(axis=1).astype("float32")
        df[f"{mov}_alpha"] = vals
        comp_min = min(comp_min, float(vals.min()))
        comp_max = max(comp_max, float(vals.max()))

    # ── R² por MoV (max sobre lag×estação) + best lag/season por pixel ───
    _NC_SFXR = "_predep_granular_seasonal.nc"
    rows_idx, cols_idx = bundle.pixel_rc[:, 0], bundle.pixel_rc[:, 1]
    r2_data: dict = {}
    mov_best_lag: dict = {}
    mov_best_season: dict = {}
    for mov in bundle.mov_names:
        nc_p = exp_dir / mov / f"{mov}{_NC_SFXR}"
        if not nc_p.exists():
            print(f"  [R²] NC não encontrado para {mov}, pulando.")
            continue
        with xr.open_dataset(nc_p) as _ds:
            r2_arr     = _ds["r2"].values           # (season, lag, lat, lon)
            al_arr     = _ds["alpha_core"].values   # (season, lag, lat, lon)
            _lats      = _ds["latitude"].values
            lags_nc    = [int(x) for x in _ds["lag"].values]
            seasons_nc = [str(x) for x in _ds["season"].values]
        if len(_lats) > 1 and _lats[0] > _lats[-1]:
            r2_arr = r2_arr[:, :, ::-1, :]
            al_arr = al_arr[:, :, ::-1, :]
        r2_pix = np.nanmax(r2_arr[:, :, rows_idx, cols_idx], axis=(0, 1))
        r2_data[f"{mov}_r2"] = r2_pix.astype("float32")
        # Best (season, lag) per pixel — restrito ao season_filter se definido
        if season_filter and season_filter in seasons_nc:
            _si = seasons_nc.index(season_filter)
            al_enrich   = al_arr[[_si], :, :, :]   # (1, n_lag, lat, lon)
            seasons_enrich = [season_filter]
        else:
            al_enrich   = al_arr
            seasons_enrich = seasons_nc
        al_pix = al_enrich[:, :, rows_idx, cols_idx]  # (n_season_e, n_lag, n_pixel)
        n_s, n_l, n_p = al_pix.shape
        al_flat = al_pix.reshape(n_s * n_l, n_p)
        valid = ~np.all(np.isnan(al_flat), axis=0)
        best_idx = np.zeros(n_p, dtype=int)
        if valid.any():
            best_idx[valid] = np.nanargmax(al_flat[:, valid], axis=0)
        mov_best_lag[mov]    = np.array([lags_nc[i % n_l]        for i in best_idx])
        mov_best_season[mov] = np.array([seasons_enrich[i // n_l] for i in best_idx])

    if r2_data:
        r2_stack = np.column_stack(list(r2_data.values()))
        r2_data["r2_max"] = np.nanmax(r2_stack, axis=1).astype("float32")
        for _cn, _arr in r2_data.items():
            df[_cn] = _arr
        print(f"  r2_max: min={df['r2_max'].min():.3f} max={df['r2_max'].max():.3f}")

    # ── metadados (cores/legenda fiéis ao estático) ────────────────────────
    _, _, colors = _make_regime_cmap(regimes.n_regimes)
    _, anomaly = _regime_mov_matrix(fps)
    regimes_meta = []
    for r in range(regimes.n_regimes):
        by_mov = fps[r]["by_mov"]
        order = np.argsort(by_mov)[::-1]
        top = int(np.argmax(anomaly[r])) if anomaly.shape[1] else 0
        # combinação = top-k MoVs que melhor preveem nesse regime (α bruto)
        combo = [bundle.mov_names[int(i)] for i in order[:combo_k]]
        regimes_meta.append({
            "id": r,
            "color_hex": mcolors.to_hex(colors[r]),
            "distinctive_mov": bundle.mov_names[top],
            "combo": combo,
            "combo_alpha": [round(float(by_mov[int(i)]), 4) for i in order[:combo_k]],
            "label": " + ".join(combo),
            "size": int(fps[r]["size"]),
        })

    # ── Enriquecer regimes com best-lag e dominant-season ─────────────────
    def _mode_arr(arr):
        if len(arr) == 0:
            return None
        vals, counts = np.unique(arr, return_counts=True)
        return vals[int(np.argmax(counts))]

    for rm in regimes_meta:
        r_id = rm["id"]
        regime_pix = np.where(regimes.pixel_regime == r_id)[0]
        combo_lags, combo_seasons = [], []
        for mov in rm["combo"]:
            if mov in mov_best_lag and len(regime_pix):
                combo_lags.append(int(_mode_arr(mov_best_lag[mov][regime_pix])))
                combo_seasons.append(str(_mode_arr(mov_best_season[mov][regime_pix])))
            else:
                combo_lags.append(None)
                combo_seasons.append(None)
        rm["combo_best_lag"] = combo_lags
        rm["combo_best_season"] = combo_seasons
        all_seasons_r = (
            np.concatenate([
                mov_best_season[m][regime_pix]
                for m in bundle.mov_names if m in mov_best_season
            ]) if mov_best_season and len(regime_pix) else np.array([])
        )
        rm["dominant_season"] = (
            str(_mode_arr(all_seasons_r)) if len(all_seasons_r) else None
        )

    meta = {
        "exp": exp,
        "run_id": out_key,
        "feature_mode": feature_mode,
        "season_filter": season_filter,
        "mov_names": bundle.mov_names,
        "lags": [int(x) for x in bundle.lags],
        "seasons": bundle.seasons,
        "n_regimes": int(regimes.n_regimes),
        "som_size": int(metrics["som_size"]),
        "quantization_error": float(metrics["quantization_error"]),
        "topographic_error": float(metrics["topographic_error"]),
        "component_alpha_vmin": comp_min,
        "component_alpha_vmax": comp_max,
        "atypicality_min": float(atypicality.min()),
        "atypicality_max": float(atypicality.max()),
        "boundary_min": float(boundary.min()),
        "boundary_max": float(boundary.max()),
        "r2_max_min": float(df["r2_max"].min()) if "r2_max" in df.columns else None,
        "r2_max_max": float(df["r2_max"].max()) if "r2_max" in df.columns else None,
        "regimes": regimes_meta,
        "seasonal_transitions": (
            seasonal_transitions
            and feature_mode not in ("best_mov", "precip_latlon")
        ),
    }

    out_exp = out_dir / out_key / f"n{n_regimes:02d}"
    out_exp.mkdir(parents=True, exist_ok=True)
    pq_path = out_exp / "som_pixels.parquet"
    df.to_parquet(pq_path, compression="zstd", index=False)
    with open(out_exp / "som_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mb = pq_path.stat().st_size / 1024 / 1024
    print(f"  som_pixels.parquet: {len(df):,} linhas | {size_mb:.2f} MB")
    print(f"  som_meta.json: {regimes.n_regimes} regimes | "
          f"QE={meta['quantization_error']:.3f} "
          f"TE={meta['topographic_error']:.3f}")
    print("  regimes: "
          + " | ".join(f"R{m['id']}=[{m['label']}] (n={m['size']})"
                       for m in regimes_meta))


def _checkpoint_path(checkpoint_dir, variable):
    return Path(checkpoint_dir) / f"_checkpoint_{variable}.pkl"


def _save_checkpoint(checkpoint_dir, variable, payload):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, variable)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # write-then-rename: nunca deixa um checkpoint truncado
    print(f"  [checkpoint] {variable} salvo em {path}")


def _load_checkpoint(checkpoint_dir, variable):
    path = _checkpoint_path(checkpoint_dir, variable)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        print(f"  [checkpoint] {variable} restaurado de {path} (pulando retreino)")
        return payload
    except Exception as e:
        print(f"  [checkpoint] falha ao ler {path} ({e}); recomputando {variable}.")
        return None


def export_dual_som(exp, out_dir, mov_alpha_threshold=None, lags=None,
                    seed=42, iterations=8, results_dir=None, run_id=None,
                    source="nc", parquet_dir=None, optimize=True, auto_k=True,
                    n_regimes=7, som_size=None, mov_filter=None,
                    use_epochs=True, search_radius=2, k_selection="silhouette",
                    standardize_method="quantile", search_sample_size=15000,
                    search_epochs=1, checkpoint_dir=None, resume=True,
                    commit_cb=None, variables=None, k_margin_sigma=1.0,
                    basin_correct=False):
    """
    Treina DOIS SOMs (PREDEP e r²) sobre o MESMO subconjunto de MoVs
    (definido por `mov_alpha_threshold`, sempre calculado sobre PREDEP) e
    exporta um único parquet com colunas prefixadas `predep_*` e `r2_*`,
    para renderização lado a lado na dashboard.

    Metodologia igual à dos plots estáticos (generate_som_insights.py /
    som_insights_modal.py): otimização de topologia via fronteira de Pareto
    (QE×TE) e nº de regimes via silhouette (mais estável que o cotovelo do
    WSS entre sementes) — não a metodologia "leve" (best_mov, grade fixa)
    usada por export_experiment().

    use_epochs: True (padrão) faz `iterations` ser o nº de ÉPOCAS completas
    de treino do SOM (len(X) atualizações cada), não um nº total fixo de
    atualizações — essencial para datasets grandes (ex.: 70k pixels do
    Brasil-todo), onde `iterations` interpretado como total sub-treina
    drasticamente o SOM (poucos % dos dados vistos). ATENÇÃO: por isso o
    padrão de `iterations` caiu de 10000 (herdado da semântica antiga, de
    "total de atualizações") para 8 — 10000 ÉPOCAS reais em ~70k pixels
    seria de meses de treino (medido: 1 época custa 113-320s dependendo
    da grade). 8 épocas já são ~560k atualizações de peso no total.
    search_radius: raio de busca de topologias candidatas no Pareto (menor
    = mais rápido, já que cada candidata agora treina por mais épocas).
    standardize_method: "zscore" ou "quantile" (RankGauss — recomendado
    para α/r², que têm distribuição fortemente assimétrica).

    search_sample_size/search_epochs: a varredura de topologias (Pareto)
    treina 1 SOM por candidata (~30-50 candidatas com search_radius=2) só
    para comparar QE/TE — não precisa do dataset completo nem de todas as
    épocas para isso, apenas de um ranking relativo razoável entre
    candidatas. Por padrão a varredura roda numa subamostra de até
    `search_sample_size` pixels (15k) com `search_epochs` época(s) (1),
    e o modelo VENCEDOR é sempre retreinado do zero na base COMPLETA com
    `iterations` épocas antes de ser usado. Sem isso, medido localmente,
    1 época em 70k pixels leva de ~113s (grade 20x20) a ~320s (grade
    40x30) — uma varredura completa de ~40-50 candidatas × 5 épocas na
    base inteira facilmente ultrapassa 6h (o timeout do Modal) só para 1
    das 2 variáveis (PREDEP/r²). Passe search_sample_size=None para
    desativar (comportamento antigo, caro).
    checkpoint_dir: diretório para salvar/retomar o progresso por variável
    (PREDEP e r² são independentes). Se None, usa out_dir/{run_id}/.checkpoints.
    Ao (re)rodar com resume=True (padrão), qualquer variável já concluída
    é lida do checkpoint em vez de retreinada — sobrevive a timeouts que
    interrompem o job no meio (ex.: PREDEP terminou, r² estourou o tempo).
    commit_cb: callback opcional (sem args) chamado logo após CADA variável
    ser salva no checkpoint — no Modal, passar volume.commit() aqui é o
    que de fato torna o checkpoint durável: escrever em /data sozinho não
    persiste no volume até um commit, e se o container for morto por
    timeout no meio da 2ª variável, sem commit explícito o checkpoint da
    1ª (mesmo já escrito em disco) se perderia também.

    source: "nc" (padrão; exp_dir = results_dir/exp, granular por bacia,
    coluna alpha_core) ou "parquet" (exp_dir = parquet_dir, 1 arquivo por
    MoV×estação, coluna alpha — usado pelo experimento Brasil-todo
    "exp_brasil" já pré-computado em predep_visualization/results/).

    mov_alpha_threshold=None → todos os MoVs (sem filtro).
    """
    if source == "parquet":
        if parquet_dir is None:
            raise ValueError("source='parquet' requer parquet_dir.")
        exp_dir = Path(parquet_dir)
        predep_var, builder = "alpha", build_feature_matrix_from_parquet
    else:
        if results_dir is None:
            results_dir = RESULTS_DIR
        exp_dir = Path(results_dir) / exp
        predep_var, builder = "alpha_core", build_feature_matrix
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experimento não encontrado: {exp_dir}")
    lags = lags if lags is not None else DEFAULT_LAGS
    _run_id = run_id or exp

    # variables: por padrão treina as DUAS (comportamento normal, exporta
    # o parquet dual completo usado pela dashboard). Passar variables=("r2",)
    # ou ("alpha",)/("alpha_core",) treina só uma — útil para testar rápido
    # uma mudança que afeta só uma das duas sem esperar a outra. Nesse caso
    # o parquet exportado só terá as colunas/meta dessa variável (não é
    # compatível com a visão lado-a-lado da dashboard, que espera as duas).
    all_variables = (predep_var, "r2")
    if variables is None:
        variables = all_variables
    else:
        variables = tuple(
            predep_var if v in ("alpha", "alpha_core") else v for v in variables
        )
        invalid = set(variables) - set(all_variables)
        if invalid:
            raise ValueError(f"variables inválido: {invalid}. Use {all_variables}.")

    print(f"\n{'='*60}\nSOM dual (PREDEP + R²) → parquet: {_run_id}"
          f"\nFonte: {source}  |  Limiar de α: {mov_alpha_threshold}  |  "
          f"Variáveis: {variables}\n{'='*60}")

    # mov_filter_by_variable: filtro de MoVs A APLICAR em cada variável.
    # O limiar de α (mov_alpha_threshold) é sempre calculado sobre PREDEP —
    # aplicar o MESMO valor numérico à escala de r² não faz sentido (r²
    # bruto raramente passa de 0.03-0.05 mesmo pros MoVs mais fortes,
    # enquanto α facilmente chega a 0.06-0.07; um limiar como 0.15
    # eliminaria quase todo MoV se comparado a r²). Por isso o limiar
    # decide apenas a LISTA DE NOMES usada no SOM de PREDEP; o SOM de r²
    # sempre usa TODOS os MoVs do experimento (sem filtro por limiar),
    # já que os valores de r² são pequenos o bastante para não precisarem
    # dessa poda — só um filtro EXPLÍCITO (--movs) restringe os dois.
    mov_filter_by_variable = {predep_var: mov_filter, "r2": mov_filter}
    if mov_filter:
        print(f"  MoVs (filtro explícito, aplicado a ambos): {len(mov_filter)} → {mov_filter}")
    elif mov_alpha_threshold is not None:
        predep_mov_filter, thr_scores = movs_above_threshold(
            exp_dir, mov_alpha_threshold, lags=lags, source=source
        )
        print(f"  MoVs > {mov_alpha_threshold} (só PREDEP): "
              f"{len(predep_mov_filter)}/{len(thr_scores)} → {predep_mov_filter}")
        if not predep_mov_filter:
            raise ValueError(f"Nenhum MoV passou do limiar {mov_alpha_threshold}.")
        print(f"  R²: sem filtro de limiar — usa todos os MoVs do experimento")
        mov_filter_by_variable = {predep_var: predep_mov_filter, "r2": None}

    if checkpoint_dir is None:
        checkpoint_dir = Path(out_dir) / _run_id / ".checkpoints"

    bundles = {}
    for variable in variables:
        cached = _load_checkpoint(checkpoint_dir, variable) if resume else None
        if cached is not None:
            bundles[variable] = cached
            continue

        t_var0 = time.time()
        print(f"\n  [{variable}] Construindo features "
              f"(padronização: {standardize_method})...")
        builder_kwargs = dict(mov_filter=mov_filter_by_variable[variable], lags=lags,
                              standardize=True, variable=variable)
        if source == "parquet":
            builder_kwargs["standardize_method"] = standardize_method
            builder_kwargs["basin_correct"] = basin_correct
        bundle = builder(exp_dir, **builder_kwargs)
        print(f"  [{variable}] amostras={bundle.X.shape[0]} "
              f"features={bundle.X.shape[1]} MoVs={len(bundle.mov_names)}")
        print(f"  [{variable}] Treinando SOM "
              f"({'otimização Pareto QE×TE' if optimize else 'grade heurística fixa'}, "
              f"{'por épocas' if use_epochs else 'por iterações totais'}, "
              f"busca em {'subamostra de ' + str(search_sample_size) if search_sample_size else 'base completa'} "
              f"× {search_epochs or iterations} época(s))...")

        n_done_holder = {"n": 0}

        def _progress_cb(done, total, rows, cols, elapsed, _t_var0=t_var0, _n=n_done_holder):
            _n["n"] = done
            print(f"    [Pareto {done}/{total}] {rows}x{cols} "
                  f"em {elapsed:.1f}s (decorrido total: {time.time() - _t_var0:.0f}s)")

        som, metrics = train_som(bundle.X, som_size=som_size,
                                 num_iteration=iterations, seed=seed,
                                 optimize=optimize, use_epochs=use_epochs,
                                 search_radius=search_radius,
                                 search_sample_size=search_sample_size,
                                 search_epochs=search_epochs,
                                 progress_cb=_progress_cb if optimize else None)
        print(f"  [{variable}] {metrics['som_size']}  "
              f"QE={metrics['quantization_error']:.4f} "
              f"TE={metrics['topographic_error']:.4f}  "
              f"(treino levou {time.time() - t_var0:.0f}s)")
        regimes = assign_regimes(som, bundle.X, n_regimes=n_regimes,
                                 seed=seed, auto_k=auto_k,
                                 k_selection=k_selection,
                                 k_margin_sigma=k_margin_sigma,
                                 max_k=len(bundle.mov_names))
        print(f"  [{variable}] Nº de regimes "
              f"({k_selection if auto_k else 'fixo'}): {regimes.n_regimes}")
        fps = regime_fingerprints(regimes, bundle)
        bundles[variable] = (bundle, som, metrics, regimes, fps)
        _save_checkpoint(checkpoint_dir, variable, bundles[variable])
        if commit_cb is not None:
            commit_cb()
            print(f"  [checkpoint] commit externo (volume) após {variable}")

    bundle0 = bundles[variables[0]][0]
    pix_lat = bundle0.lats[bundle0.pixel_rc[:, 0]]
    pix_lon = bundle0.lons[bundle0.pixel_rc[:, 1]]

    if source == "parquet" and bundle0.pixel_basin is not None:
        basin = list(bundle0.pixel_basin)
    else:
        mov0 = bundle0.mov_names[0]
        fallback = sorted((Path(results_dir or RESULTS_DIR)).glob(f"exp*/{mov0}"))
        outlines = _load_basin_outlines(exp_dir / mov0, mov0, fallback_dirs=fallback)
        num_to_name = {v: k for k, v in BASIN_LABELS.items()}
        coord_to_basin = {}
        for number, blats, blons, mask in outlines:
            name = num_to_name.get(number, str(number))
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                coord_to_basin[(round(float(blats[r]), 2),
                                round(float(blons[c]), 2))] = name
        basin = [
            coord_to_basin.get((round(float(la), 2), round(float(lo), 2)), "desconhecida")
            for la, lo in zip(pix_lat, pix_lon)
        ]

    df = pd.DataFrame({
        "latitude": pix_lat.astype("float32"),
        "longitude": pix_lon.astype("float32"),
        "basin": pd.Categorical(basin),
    })

    meta = {
        "exp": exp,
        "run_id": _run_id,
        "source": source,
        "mov_alpha_threshold": mov_alpha_threshold,
        "standardize_method": standardize_method,
        "use_epochs": use_epochs,
        "iterations": iterations,
        "k_selection": k_selection,
        "mov_names": bundle0.mov_names,
        "lags": [int(x) for x in bundle0.lags],
        "seasons": bundle0.seasons,
    }

    tags_by_variable = {predep_var: "predep", "r2": "r2"}
    for variable in variables:
        tag = tags_by_variable[variable]
        bundle, som, metrics, regimes, fps = bundles[variable]
        W = som.get_weights()
        bmu = regimes.bmu_rc
        w_pix = W[bmu[:, 0], bmu[:, 1]]
        atypicality = np.linalg.norm(bundle.X - w_pix, axis=1)
        boundary = som.distance_map()[bmu[:, 0], bmu[:, 1]]

        df[f"{tag}_regime"] = regimes.pixel_regime.astype("int8")
        df[f"{tag}_atypicality"] = atypicality.astype("float32")
        df[f"{tag}_boundary"] = boundary.astype("float32")

        # Explicabilidade local: por pixel, quais MoVs (e em que lag/estação)
        # melhor casam com o protótipo do neurônio vencedor — complementa a
        # assinatura GLOBAL do regime (distinctive_mov/combo) com um "porquê"
        # POR PIXEL, sem retreinar nada.
        importance = local_mov_importance(bundle, som, regimes, top_k=3)
        for k in range(importance["top_movs"].shape[1]):
            df[f"{tag}_explain_top{k + 1}"] = importance["top_movs"][:, k]
        df[f"{tag}_explain_lag"] = importance["top1_lag"].astype("int16")
        df[f"{tag}_explain_season"] = importance["top1_season"]

        mov_ids = bundle.feature_index[:, 0]
        lag_ids = bundle.feature_index[:, 1]
        season_ids = bundle.feature_index[:, 2]
        lags_arr = np.array(bundle.lags)
        seasons_arr = np.array(bundle.seasons)

        # Melhor (lag, estação) por pixel, por MoV — direto de X_raw (já
        # temos a matriz completa; não precisa reabrir NC/parquet por MoV).
        mov_best_lag, mov_best_season = {}, {}
        for m, mov in enumerate(bundle.mov_names):
            cols = np.nonzero(mov_ids == m)[0]
            sub = bundle.X_raw[:, cols]
            best_local = np.argmax(sub, axis=1)
            best_col = cols[best_local]
            mov_best_lag[mov] = lags_arr[lag_ids[best_col]]
            mov_best_season[mov] = seasons_arr[season_ids[best_col]]

        def _mode_arr(arr):
            if len(arr) == 0:
                return None
            vals, counts = np.unique(arr, return_counts=True)
            return vals[int(np.argmax(counts))]

        for m, mov in enumerate(bundle.mov_names):
            cols = mov_ids == m
            vals = bundle.X_raw[:, cols].mean(axis=1).astype("float32")
            df[f"{mov}_{tag}"] = vals

        _, _, colors = _make_regime_cmap(regimes.n_regimes)
        _, anomaly = _regime_mov_matrix(fps)
        regimes_meta = []
        for r in range(regimes.n_regimes):
            regime_pix = np.where(regimes.pixel_regime == r)[0]
            by_mov = fps[r]["by_mov"]
            # Ranking por MAGNITUDE BRUTA (independente da anomalia): qual
            # MoV tem o maior valor médio nesse regime, cheio ponto. Um MoV
            # como NINO pode ser o de MAIOR r²/α em TODOS os regimes (efeito
            # quase uniforme no Brasil) e ainda assim ter anomalia ~0 (não
            # DIFERE entre regimes) — nesse caso ele nunca aparece como
            # "distinctive_mov" abaixo, o que pode sugerir erradamente que
            # ele é irrelevante. dominant_mov cobre exatamente esse caso:
            # é sempre reportado, mesmo quando é igual entre regimes.
            raw_order = np.argsort(by_mov)[::-1]
            dominant_k = min(3, len(raw_order))
            dominant_movs = [bundle.mov_names[int(i)] for i in raw_order[:dominant_k]]
            dominant_values = [round(float(by_mov[int(i)]), 4) for i in raw_order[:dominant_k]]
            # Ranking por ANOMALIA (vs média entre regimes), não valor bruto:
            # MoVs fortes em todo lugar (ex.: PSA2/NIN03 em boa parte do
            # Brasil) dominariam o top-3 de quase todo regime se ordenado
            # por valor bruto, tornando os rótulos indistinguíveis entre
            # regimes. A anomalia mostra o que é DISTINTIVO de cada regime.
            has_anomaly = anomaly.shape[1] > 0
            order = np.argsort(anomaly[r])[::-1] if has_anomaly else np.argsort(by_mov)[::-1]
            # Tamanho do combo é autônomo: corta na maior queda (gap) entre
            # anomalias consecutivas, dentro da faixa positiva (MoVs acima
            # da média entre regimes) — um corte "anomalia > 0" simples
            # inclui ~metade dos MoVs por construção (a anomalia é centrada
            # em 0), gerando rótulos com 15-22 MoVs. O maior salto encontra
            # a quebra natural entre "distintivo" e "não distintivo" para
            # cada regime, sem número fixo.
            if has_anomaly:
                sorted_anom = anomaly[r, order]
                n_pos = max(1, int((sorted_anom > 0).sum()))
                gaps = -np.diff(sorted_anom[:n_pos + 1]) if n_pos < len(sorted_anom) \
                    else -np.diff(sorted_anom[:n_pos])
                cut = int(np.argmax(gaps)) + 1 if len(gaps) else 1
                combo_idx = [int(i) for i in order[:cut]]
            else:
                combo_idx = [int(order[0])]
            combo = [bundle.mov_names[i] for i in combo_idx]

            combo_lags, combo_seasons = [], []
            for mov in combo:
                if len(regime_pix):
                    combo_lags.append(int(_mode_arr(mov_best_lag[mov][regime_pix])))
                    combo_seasons.append(str(_mode_arr(mov_best_season[mov][regime_pix])))
                else:
                    combo_lags.append(None)
                    combo_seasons.append(None)
            all_seasons_r = (
                np.concatenate([mov_best_season[mv][regime_pix] for mv in combo])
                if combo and len(regime_pix) else np.array([])
            )
            dominant_season = (
                str(_mode_arr(all_seasons_r)) if len(all_seasons_r) else None
            )

            regimes_meta.append({
                "id": r,
                "color_hex": mcolors.to_hex(colors[r]),
                "distinctive_mov": bundle.mov_names[int(order[0])],
                "dominant_mov": dominant_movs[0],
                "dominant_movs": dominant_movs,
                "dominant_values": dominant_values,
                "combo": combo,
                "combo_alpha": [round(float(by_mov[i]), 4) for i in combo_idx],
                "combo_best_lag": combo_lags,
                "combo_best_season": combo_seasons,
                "dominant_season": dominant_season,
                "label": " + ".join(combo),
                "size": int(fps[r]["size"]),
            })

        kmr = regimes.kmeans_results or {}
        k_diagnostics = {
            "k_values": kmr.get("k_values"),
            "silhouette_means": kmr.get("silhouette_means"),
            "silhouette_stds": kmr.get("silhouette_stds"),
            "elbow_k": kmr.get("elbow_k"),
            "silhouette_k_raw": kmr.get("silhouette_k_raw"),
            "silhouette_k": kmr.get("silhouette_k"),
            "k_candidates_tied": kmr.get("k_candidates_tied"),
            "k_margin_note": kmr.get("k_margin_note"),
            "chosen_k": int(regimes.n_regimes),
        } if kmr else None

        meta[tag] = {
            "mov_names": bundle.mov_names,
            "n_regimes": int(regimes.n_regimes),
            "som_size": metrics["som_size"],
            "quantization_error": float(metrics["quantization_error"]),
            "topographic_error": float(metrics["topographic_error"]),
            "value_min": float(np.nanmin(bundle.X_raw)),
            "value_max": float(np.nanmax(bundle.X_raw)),
            "atypicality_min": float(atypicality.min()),
            "atypicality_max": float(atypicality.max()),
            "boundary_min": float(boundary.min()),
            "boundary_max": float(boundary.max()),
            "regimes": regimes_meta,
            "k_diagnostics": k_diagnostics,
        }

    out_exp = Path(out_dir) / _run_id
    out_exp.mkdir(parents=True, exist_ok=True)
    pq_path = out_exp / "som_pixels.parquet"
    df.to_parquet(pq_path, compression="zstd", index=False)
    with open(out_exp / "som_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mb = pq_path.stat().st_size / 1024 / 1024
    print(f"\n  som_pixels.parquet: {len(df):,} linhas | {size_mb:.2f} MB")
    if "predep" in meta:
        print(f"  PREDEP: K={meta['predep']['n_regimes']} grid={meta['predep']['som_size']}")
    if "r2" in meta:
        print(f"  R²:     K={meta['r2']['n_regimes']} grid={meta['r2']['som_size']}")

    # Job concluído com sucesso: checkpoints deixaram de ser necessários.
    for variable in variables:
        _checkpoint_path(checkpoint_dir, variable).unlink(missing_ok=True)

    return out_exp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp", required=True, help="Experimento (ex: exp01)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT),
                        help=f"Diretório de saída (padrão: {DEFAULT_OUT})")
    parser.add_argument("--n-regimes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lags", default="0,1,3,6,9,12",
                        help="Lags (vírgula) ou 'all' (padrão: 0,1,3,6,9,12)")
    parser.add_argument("--feature-mode",
                        choices=["full", "best_mov", "precip_latlon"],
                        default="full",
                        help="full: α por MoV×lag×estação. "
                             "best_mov: melhor α por MoV. "
                             "precip_latlon: lat+lon+precip sazonal.")
    parser.add_argument("--combo-k", type=int, default=3,
                        help="Nº de MoVs no rótulo de combinação (best_mov)")
    parser.add_argument("--season", default=None,
                        choices=["DJF", "MAM", "JJA", "SON"],
                        help="Treina o SOM apenas nesta estação meteorológica")
    parser.add_argument("--top-movs", type=int, default=0,
                        help="Usa os top-N MoVs por α médio (0 = todos)")
    parser.add_argument("--movs", default=None,
                        help="Lista de MoVs a usar (vírgula), ex: ONI,TNA,AAO")
    parser.add_argument("--run-id", default=None,
                        help="ID do run no caminho de saída (padrão: --exp)")
    parser.add_argument("--seasonal-transitions", action="store_true",
                        help="Adiciona regime_DJF/MAM/JJA/SON ao parquet "
                             "(apenas feature_mode=full)")
    parser.add_argument("--som-size", type=int, default=None,
                        help="Tamanho da grade SOM (g×g). Padrão: automático")
    parser.add_argument("--strict", action="store_true",
                        help="Usa apenas pixels com dados válidos para todos "
                             "os MoV×lag×estação (elimina NaN-fill artifacts)")
    parser.add_argument("--dual", action="store_true",
                        help="Modo dual: treina PREDEP e r² sobre o mesmo "
                             "filtro de MoVs (export_dual_som) em vez do "
                             "pipeline leve padrão (export_experiment).")
    parser.add_argument("--mov-alpha-threshold", type=float, default=None,
                        help="[--dual] Inclui apenas MoVs com α (PREDEP) "
                             "acima deste limiar (ex.: 0.15, 0.3). "
                             "Ignorado sem --dual.")
    parser.add_argument("--iterations", type=int, default=8,
                        help="[--dual] Nº de ÉPOCAS completas de treino do "
                             "SOM final (use_epochs=True; padrão: 8). NÃO "
                             "use valores grandes tipo 10000 aqui — isso é "
                             "o nº de passadas completas pela base inteira, "
                             "não um total de atualizações.")
    parser.add_argument("--source", choices=["nc", "parquet"], default="nc",
                        help="[--dual] 'parquet' usa o experimento Brasil-todo "
                             "já pré-computado (1 arquivo por MoV×estação, "
                             "colunas basin/latitude/longitude/lag/r2/alpha) "
                             "em vez dos NCs granulares por bacia.")
    parser.add_argument("--parquet-dir", default=None,
                        help="[--dual --source parquet] Caminho do experimento "
                             "parquet (ex.: predep_visualization/results/"
                             "predep_granular_brazil/exp_brasil)")
    parser.add_argument("--search-sample-size", type=int, default=15000,
                        help="[--dual] Nº de pixels usados na varredura de "
                             "topologias do Pareto (subamostra; o modelo "
                             "final é sempre retreinado na base completa). "
                             "0 desativa (usa a base completa, mais lento).")
    parser.add_argument("--search-epochs", type=int, default=1,
                        help="[--dual] Nº de épocas por candidata na "
                             "varredura de topologias (independente de "
                             "--iterations, usado só no retreino final).")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="[--dual] Diretório para salvar/retomar "
                             "progresso por variável (PREDEP/r²). Padrão: "
                             "{out_dir}/{run_id}/.checkpoints")
    parser.add_argument("--no-resume", action="store_true",
                        help="[--dual] Ignora checkpoints existentes e "
                             "recomputa tudo do zero.")
    parser.add_argument("--basin-correct", action="store_true",
                        help="[--dual --source parquet] Corrige viés "
                             "sistemático de bacia (centra cada bacia na "
                             "própria média antes da padronização global) "
                             "antes de treinar o SOM — evita que bacias "
                             "inteiras colapsem num único regime por causa "
                             "do α/r² ter sido calculado com ajuste "
                             "independente por bacia, não por sinal "
                             "climático real.")
    args = parser.parse_args()

    lags = (None if args.lags.strip().lower() == "all"
            else [int(x) for x in args.lags.split(",") if x.strip()])
    mov_filter = ([m.strip() for m in args.movs.split(",") if m.strip()]
                  if args.movs else None)

    if args.dual:
        export_dual_som(args.exp, Path(args.out_dir),
                        mov_alpha_threshold=args.mov_alpha_threshold,
                        lags=lags, seed=args.seed, iterations=args.iterations,
                        run_id=args.run_id, source=args.source,
                        parquet_dir=args.parquet_dir, mov_filter=mov_filter,
                        search_sample_size=(args.search_sample_size or None),
                        search_epochs=args.search_epochs,
                        checkpoint_dir=args.checkpoint_dir,
                        resume=not args.no_resume,
                        basin_correct=args.basin_correct)
        return
    export_experiment(args.exp, Path(args.out_dir),
                      n_regimes=args.n_regimes, seed=args.seed, lags=lags,
                      feature_mode=args.feature_mode, combo_k=args.combo_k,
                      season_filter=args.season, top_movs_k=args.top_movs,
                      run_id=args.run_id,
                      seasonal_transitions=args.seasonal_transitions,
                      som_size=args.som_size, strict=args.strict,
                      mov_filter=mov_filter)


if __name__ == "__main__":
    main()
