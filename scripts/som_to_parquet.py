#!/usr/bin/env python3
"""
Exporta os mapas espaciais do SOM (regimes, atipicidade, fronteiras e
component planes por MoV) como artefatos leves para a dashboard
predep_visualization, espelhando o scripts/nc_to_parquet.py.

Pré-computa o SOM (mesmos hiperparâmetros do generate_som_insights.py, para
casar com os PNGs estáticos) e escreve:
    {out_dir}/{exp}/som_pixels.parquet   (1 linha por pixel)
    {out_dir}/{exp}/som_meta.json        (regimes, cores, escalas, métricas)

A dashboard apenas renderiza esses artefatos (não precisa de minisom/sklearn).

Uso:
    python scripts/som_to_parquet.py --exp exp01
    python scripts/som_to_parquet.py --exp exp01 --out-dir /caminho --n-regimes 7
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.som_insights import (  # noqa: E402
    build_feature_matrix, train_som, assign_regimes, regime_fingerprints,
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
                      feature_mode="full", combo_k=3, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    exp_dir = results_dir / exp
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experimento não encontrado: {exp_dir}")
    lags = lags if lags is not None else DEFAULT_LAGS

    print(f"\n{'='*60}\nSOM → parquet: {exp}  (feature_mode={feature_mode})"
          f"\nSaída: {out_dir / exp}\n{'='*60}")

    # ── pipeline SOM ───────────────────────────────────────────────────────
    bundle = build_feature_matrix(exp_dir, mov_filter=None, lags=lags,
                                  standardize=True)
    if feature_mode == "best_mov":
        bundle = _collapse_best_mov(bundle)
    som, metrics = train_som(bundle.X, seed=seed)
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
    basin = _basin_per_pixel(exp_dir, bundle.mov_names[0], pix_lat, pix_lon)

    df = pd.DataFrame({
        "latitude": pix_lat.astype("float32"),
        "longitude": pix_lon.astype("float32"),
        "basin": pd.Categorical(basin),
        "regime": regimes.pixel_regime.astype("int8"),
        "atypicality": atypicality.astype("float32"),
        "boundary": boundary.astype("float32"),
    })

    mov_ids = bundle.feature_index[:, 0]
    comp_min, comp_max = np.inf, -np.inf
    for m, mov in enumerate(bundle.mov_names):
        cols = np.nonzero(mov_ids == m)[0]
        vals = bundle.X_raw[:, cols].mean(axis=1).astype("float32")
        df[f"{mov}_alpha"] = vals
        comp_min = min(comp_min, float(vals.min()))
        comp_max = max(comp_max, float(vals.max()))

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

    meta = {
        "exp": exp,
        "feature_mode": feature_mode,
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
        "regimes": regimes_meta,
    }

    out_exp = out_dir / exp / f"n{n_regimes:02d}"
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
    parser.add_argument("--feature-mode", choices=["full", "best_mov"],
                        default="full",
                        help="full: α por MoV×lag×estação (552 dims). "
                             "best_mov: melhor α por MoV (23 dims) → clusters "
                             "por combinação de MoVs preditores.")
    parser.add_argument("--combo-k", type=int, default=3,
                        help="Nº de MoVs no rótulo de combinação (best_mov)")
    args = parser.parse_args()

    lags = (None if args.lags.strip().lower() == "all"
            else [int(x) for x in args.lags.split(",") if x.strip()])
    export_experiment(args.exp, Path(args.out_dir),
                      n_regimes=args.n_regimes, seed=args.seed, lags=lags,
                      feature_mode=args.feature_mode, combo_k=args.combo_k)


if __name__ == "__main__":
    main()
