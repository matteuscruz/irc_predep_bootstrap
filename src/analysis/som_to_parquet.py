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
    build_feature_matrix, train_som, assign_regimes, regime_fingerprints,
    auto_top_movs, build_precip_bundle, seasonal_regimes,
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
    args = parser.parse_args()

    lags = (None if args.lags.strip().lower() == "all"
            else [int(x) for x in args.lags.split(",") if x.strip()])
    mov_filter = ([m.strip() for m in args.movs.split(",") if m.strip()]
                  if args.movs else None)
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
