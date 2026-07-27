"""
Busca automática de hiperparâmetros SOM (g) e KMeans (k) para o PREDEP.

Funções composáveis e independentes:
  sweep_g       — varre tamanhos de grade SOM, detecta cotovelo da QE.
  sweep_k       — varre nº de regimes KMeans, ranqueia por pontuação composta.
  run_season_search — sweep_g + sweep_k para uma estação (ou todas combinadas).
  run_full_search   — itera sobre todas as estações + "all", retorna resultado completo.

Uso típico:
    from src.analysis.som_param_search import run_full_search, print_results_table
    result = run_full_search("results/predep_granular_brazil/exp01",
                             mov_filter=["ONI","NAO","TNA"], lags=[0,3,6])
    print_results_table(result)
    save_results_json(result, "param_search.json")
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# MiniSom vendorizado — shim idêntico ao de som_insights.py.
_MINISOM_DIR = Path(__file__).resolve().parents[2] / "minisom"
if str(_MINISOM_DIR) not in sys.path:
    sys.path.insert(0, str(_MINISOM_DIR))
from minisom import MiniSom  # noqa: E402

from src.analysis.som_insights import (  # noqa: E402
    SEASONS,
    FeatureBundle,
    build_feature_matrix,
    train_som,
)

# ── Dataclasses de resultado ──────────────────────────────────────────────────


@dataclass
class GSweepEntry:
    g: int
    quantization_error: float
    topographic_error: float


@dataclass
class GSweepResult:
    entries: list
    optimal_g: int
    elbow_index: int
    second_derivative: list  # zero-padded; mesma len que entries


@dataclass
class KSweepEntry:
    k: int
    inertia: float
    silhouette: float       # NaN para k < 2
    davies_bouldin: float   # NaN para k < 2


@dataclass
class KSweepResult:
    entries: list
    best_k_silhouette: int
    best_k_davies_bouldin: int
    best_k_composite: int
    top3_k: list
    scores_normalized: list  # [{"k", "sil_n", "db_n", "iner_n", "composite"}]


@dataclass
class SeasonConfig:
    g: int
    k: int
    quantization_error: float
    topographic_error: float
    silhouette: float
    davies_bouldin: float
    inertia: float
    composite_score: float


@dataclass
class SeasonSearchResult:
    season: str            # "DJF" | "MAM" | "JJA" | "SON" | "all"
    g_sweep: GSweepResult
    k_sweep: KSweepResult  # rodado com optimal_g
    top3_configs: list     # list[SeasonConfig]


@dataclass
class ParamSearchResult:
    exp_dir: str
    mov_filter: object     # list[str] | None
    lags: list
    g_range: list
    k_range: list
    seasons_searched: list
    per_season: dict       # {season_str: SeasonSearchResult}
    overall_recommendation: SeasonConfig  # top-1 do caso "all" (ou melhor geral)

    def to_dict(self) -> dict:
        """Serializa para dict com valores Python nativos (JSON-safe)."""
        def _coerce(obj):
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_coerce(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, float) and (obj != obj):  # NaN
                return None
            return obj
        return _coerce(asdict(self))


# ── Detecção de cotovelo (sem dependência externa) ────────────────────────────


def _detect_elbow(values: list) -> int:
    """
    Retorna o índice do cotovelo em `values` via segunda derivada.

    Normaliza para [0,1], calcula d1 e d2; o cotovelo é argmax(d2) + 1.
    Com menos de 3 pontos retorna o índice de menor valor.
    """
    n = len(values)
    if n < 3:
        return int(np.argmin(values))

    v_min, v_max = min(values), max(values)
    span = v_max - v_min or 1e-12
    vn = [(v - v_min) / span for v in values]

    d1 = [vn[i + 1] - vn[i] for i in range(n - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(n - 2)]
    return int(np.argmax(d2)) + 1


def _second_derivative_padded(values: list) -> list:
    """Segunda derivada da curva normalizada, zero-padded ao tamanho de values."""
    n = len(values)
    if n < 3:
        return [0.0] * n
    v_min, v_max = min(values), max(values)
    span = v_max - v_min or 1e-12
    vn = [(v - v_min) / span for v in values]
    d1 = [vn[i + 1] - vn[i] for i in range(n - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(n - 2)]
    return [0.0, 0.0] + [float(v) for v in d2]


# ── Pontuação composta ────────────────────────────────────────────────────────


def _normalize(values: list) -> list:
    """Min-max normaliza lista; retorna [0]*n se todos iguais."""
    v_min, v_max = min(values), max(values)
    span = v_max - v_min or 1.0
    return [(v - v_min) / span for v in values]


def _composite_scores(entries: list) -> list:
    """
    Calcula pontuação composta para cada KSweepEntry válido (k >= 2).

    Fórmula (maior = melhor):
        composite = 0.5 * sil_n + 0.3 * (1 - db_n) + 0.2 * (1 - iner_n)

    Silhouette recebe peso 0.5 — métrica mais robusta para clusters de tamanhos
    desiguais (regimes geográficos do PREDEP). Davies-Bouldin invertido (0.3)
    e inércia invertida (0.2) completam o critério.

    Retorna lista de dicts {"k", "sil_n", "db_n", "iner_n", "composite"} na
    mesma ordem de `entries` (incluindo k=1 com composite=None se presente).
    """
    valid = [e for e in entries if not (e.silhouette != e.silhouette)]  # exclui NaN
    if not valid:
        return [{"k": e.k, "sil_n": None, "db_n": None,
                 "iner_n": None, "composite": None} for e in entries]

    sil_n = _normalize([e.silhouette for e in valid])
    db_n = _normalize([e.davies_bouldin for e in valid])
    iner_n = _normalize([e.inertia for e in valid])
    composites = {e.k: 0.5 * s + 0.3 * (1 - d) + 0.2 * (1 - i)
                  for e, s, d, i in zip(valid, sil_n, db_n, iner_n)}
    sil_map = {e.k: s for e, s in zip(valid, sil_n)}
    db_map = {e.k: d for e, d in zip(valid, db_n)}
    iner_map = {e.k: i for e, i in zip(valid, iner_n)}

    result = []
    for e in entries:
        if e.k in composites:
            result.append({"k": e.k,
                           "sil_n": round(sil_map[e.k], 6),
                           "db_n": round(db_map[e.k], 6),
                           "iner_n": round(iner_map[e.k], 6),
                           "composite": round(composites[e.k], 6)})
        else:
            result.append({"k": e.k, "sil_n": None, "db_n": None,
                           "iner_n": None, "composite": None})
    return result


# ── Funções públicas ──────────────────────────────────────────────────────────


def sweep_g(
    X: np.ndarray,
    g_range: list,
    sigma: Optional[float] = None,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> GSweepResult:
    """
    Treina um SOM para cada g em g_range e detecta o cotovelo da curva QE.

    Parameters
    ----------
    X : (n_pixels, n_features) — features pré-construídas (z-score recomendado).
    g_range : tamanhos de grade a testar, ex: list(range(5, 21)).
    sigma : raio de vizinhança (None → g/2, como em train_som).
    learning_rate, num_iteration, seed : hiperparâmetros do SOM.
    verbose : se True, imprime progresso.

    Returns
    -------
    GSweepResult com optimal_g detectado por segunda derivada da QE normalizada.
    """
    entries = []
    for g in g_range:
        if verbose:
            print(f"  sweep_g: g={g} ...", end=" ", flush=True)
        _sigma = sigma if sigma is not None else max(1.0, g / 2.0)
        _, metrics = train_som(
            X, som_size=g, sigma=_sigma,
            learning_rate=learning_rate,
            num_iteration=num_iteration, seed=seed,
        )
        entry = GSweepEntry(
            g=g,
            quantization_error=metrics["quantization_error"],
            topographic_error=metrics["topographic_error"],
        )
        entries.append(entry)
        if verbose:
            print(f"QE={entry.quantization_error:.4f}  TE={entry.topographic_error:.4f}")

    qe_vals = [e.quantization_error for e in entries]
    elbow_idx = _detect_elbow(qe_vals)
    d2 = _second_derivative_padded(qe_vals)

    return GSweepResult(
        entries=entries,
        optimal_g=g_range[elbow_idx],
        elbow_index=elbow_idx,
        second_derivative=d2,
    )


def sweep_k(
    som: MiniSom,
    k_range: list,
    seed: int = 42,
    verbose: bool = True,
) -> KSweepResult:
    """
    Roda KMeans nos neurônios do SOM para cada k em k_range.

    Registra inércia, silhouette e Davies-Bouldin e retorna top-3 k por
    pontuação composta:
        composite(k) = 0.5*sil_n + 0.3*(1-db_n) + 0.2*(1-iner_n)

    Parameters
    ----------
    som : SOM já treinado (MiniSom).
    k_range : valores de k a testar; k < 2 recebe silhouette/DB = NaN.
    seed : semente para KMeans.
    verbose : se True, imprime progresso.

    Returns
    -------
    KSweepResult com top3_k, best_k_* e scores_normalized por k.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    W = som.get_weights()
    W_flat = W.reshape(-1, W.shape[2])

    entries = []
    for k in k_range:
        if verbose:
            print(f"  sweep_k: k={k} ...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(W_flat)
        if k >= 2:
            sil = float(silhouette_score(W_flat, km.labels_))
            db = float(davies_bouldin_score(W_flat, km.labels_))
        else:
            sil = float("nan")
            db = float("nan")
        entry = KSweepEntry(k=k, inertia=float(km.inertia_),
                            silhouette=sil, davies_bouldin=db)
        entries.append(entry)
        if verbose:
            print(f"inertia={km.inertia_:.1f}  sil={sil:.4f}  DB={db:.4f}")

    scores = _composite_scores(entries)
    valid_scores = [(s["composite"], s["k"]) for s in scores
                    if s["composite"] is not None]
    valid_scores.sort(reverse=True)
    top3_k = [k for _, k in valid_scores[:3]]

    valid_entries = [e for e in entries if not (e.silhouette != e.silhouette)]
    best_sil = max(valid_entries, key=lambda e: e.silhouette).k if valid_entries else k_range[0]
    best_db = min(valid_entries, key=lambda e: e.davies_bouldin).k if valid_entries else k_range[0]
    best_comp = top3_k[0] if top3_k else k_range[0]

    return KSweepResult(
        entries=entries,
        best_k_silhouette=best_sil,
        best_k_davies_bouldin=best_db,
        best_k_composite=best_comp,
        top3_k=top3_k,
        scores_normalized=scores,
    )


def run_season_search(
    exp_dir,
    season: Optional[str],
    g_range: list,
    k_range: list,
    mov_filter: Optional[list] = None,
    lags: Optional[list] = None,
    standardize: bool = True,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> SeasonSearchResult:
    """
    Executa sweep_g → treina SOM ótimo → sweep_k para uma estação (ou todas).

    Parameters
    ----------
    exp_dir : caminho do experimento PREDEP.
    season : "DJF" | "MAM" | "JJA" | "SON" | None  (None = todas concatenadas).
    g_range, k_range : intervalos de busca.
    mov_filter : lista de MoVs (None = todos).
    lags : lags em meses (None = padrão de build_feature_matrix).
    standardize, learning_rate, num_iteration, seed : hiperparâmetros.
    verbose : imprime progresso.

    Returns
    -------
    SeasonSearchResult com top3_configs ordenado por composite_score decrescente.
    """
    season_label = season if season is not None else "all"
    if verbose:
        print(f"\n[season={season_label}] Construindo features...")
    bundle = build_feature_matrix(
        exp_dir, mov_filter=mov_filter, lags=lags,
        standardize=standardize, season_filter=season,
    )
    if verbose:
        print(f"  {bundle.X.shape[0]} pixels × {bundle.X.shape[1]} features")

    if verbose:
        print(f"[season={season_label}] sweep_g sobre {g_range}...")
    g_result = sweep_g(
        bundle.X, g_range,
        learning_rate=learning_rate,
        num_iteration=num_iteration,
        seed=seed, verbose=verbose,
    )
    optimal_g = g_result.optimal_g
    if verbose:
        print(f"  → optimal_g = {optimal_g}")

    if verbose:
        print(f"[season={season_label}] Treinando SOM final (g={optimal_g})...")
    som, _ = train_som(
        bundle.X, som_size=optimal_g,
        learning_rate=learning_rate,
        num_iteration=num_iteration, seed=seed,
    )

    if verbose:
        print(f"[season={season_label}] sweep_k sobre {k_range}...")
    k_result = sweep_k(som, k_range, seed=seed, verbose=verbose)

    # Busca a entrada de g-sweep correspondente ao optimal_g.
    g_entry = next(e for e in g_result.entries if e.g == optimal_g)

    # Monta top3_configs com os top3_k.
    score_map = {s["k"]: s["composite"] for s in k_result.scores_normalized
                 if s["composite"] is not None}
    k_entry_map = {e.k: e for e in k_result.entries}
    top3 = []
    for k in k_result.top3_k:
        ke = k_entry_map[k]
        top3.append(SeasonConfig(
            g=optimal_g, k=k,
            quantization_error=g_entry.quantization_error,
            topographic_error=g_entry.topographic_error,
            silhouette=ke.silhouette,
            davies_bouldin=ke.davies_bouldin,
            inertia=ke.inertia,
            composite_score=score_map.get(k, 0.0),
        ))
    top3.sort(key=lambda c: c.composite_score, reverse=True)

    return SeasonSearchResult(
        season=season_label,
        g_sweep=g_result,
        k_sweep=k_result,
        top3_configs=top3,
    )


def run_full_search(
    exp_dir,
    g_range: Optional[list] = None,
    k_range: Optional[list] = None,
    mov_filter: Optional[list] = None,
    lags: Optional[list] = None,
    standardize: bool = True,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    seed: int = 42,
    seasons: Optional[list] = None,
    include_all: bool = True,
    verbose: bool = True,
) -> ParamSearchResult:
    """
    Ponto de entrada principal. Roda run_season_search para cada estação e
    opcionalmente para o caso combinado ("all").

    Parameters
    ----------
    exp_dir : path do experimento.
    g_range : intervalo de g (padrão: list(range(5, 21))).
    k_range : intervalo de k (padrão: list(range(2, 16))).
    mov_filter : filtro de MoVs (None = todos).
    lags : lags em meses (None = padrão).
    standardize : z-score nas features.
    learning_rate, num_iteration, seed : hiperparâmetros SOM.
    seasons : subconjunto de estações (padrão: ["DJF","MAM","JJA","SON"]).
    include_all : se True, inclui caso "all" (season_filter=None).
    verbose : imprime progresso.

    Returns
    -------
    ParamSearchResult com per_season indexado por estação/"all" e
    overall_recommendation = top-1 config do caso "all" (ou melhor entre estações).
    """
    g_range = g_range or list(range(5, 21))
    k_range = k_range or list(range(2, 16))
    seasons = seasons or list(SEASONS)

    seasons_to_run = list(seasons)
    if include_all:
        seasons_to_run.append(None)  # None → "all"

    per_season: dict = {}
    for season in seasons_to_run:
        sr = run_season_search(
            exp_dir, season=season,
            g_range=g_range, k_range=k_range,
            mov_filter=mov_filter, lags=lags,
            standardize=standardize,
            learning_rate=learning_rate,
            num_iteration=num_iteration,
            seed=seed, verbose=verbose,
        )
        per_season[sr.season] = sr

    # overall_recommendation: top-1 do caso "all"; se ausente, melhor entre estações.
    if "all" in per_season and per_season["all"].top3_configs:
        overall = per_season["all"].top3_configs[0]
    else:
        candidates = [
            (sr.top3_configs[0].composite_score, sr.top3_configs[0])
            for sr in per_season.values() if sr.top3_configs
        ]
        overall = max(candidates, key=lambda x: x[0])[1] if candidates else SeasonConfig(
            g=g_range[0], k=k_range[0],
            quantization_error=0.0, topographic_error=0.0,
            silhouette=0.0, davies_bouldin=0.0,
            inertia=0.0, composite_score=0.0,
        )

    seasons_searched = [s if s is not None else "all" for s in seasons_to_run]
    return ParamSearchResult(
        exp_dir=str(exp_dir),
        mov_filter=mov_filter,
        lags=lags or [],
        g_range=g_range,
        k_range=k_range,
        seasons_searched=seasons_searched,
        per_season=per_season,
        overall_recommendation=overall,
    )


# ── Utilitários de saída ──────────────────────────────────────────────────────


def print_results_table(result: ParamSearchResult) -> None:
    """Imprime tabela resumo de ParamSearchResult em stdout."""
    header = (f"{'Estação':>8}  {'Rank':>4}  {'g':>4}  {'k':>4}  "
              f"{'QE':>7}  {'TE':>6}  {'Silhouette':>10}  {'DB':>6}  "
              f"{'Inertia':>9}  {'Score':>7}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for season in result.seasons_searched:
        sr = result.per_season.get(season)
        if not sr:
            continue
        for rank, cfg in enumerate(sr.top3_configs, start=1):
            print(f"{season:>8}  {rank:>4}  {cfg.g:>4}  {cfg.k:>4}  "
                  f"{cfg.quantization_error:>7.4f}  {cfg.topographic_error:>6.4f}  "
                  f"{cfg.silhouette:>10.4f}  {cfg.davies_bouldin:>6.4f}  "
                  f"{cfg.inertia:>9.1f}  {cfg.composite_score:>7.4f}")
    print(sep)
    rec = result.overall_recommendation
    print(f"\n★ Recomendação geral: g={rec.g}, k={rec.k}  "
          f"(score={rec.composite_score:.4f}, sil={rec.silhouette:.4f})")


def save_results_json(result: ParamSearchResult, path) -> None:
    """Salva ParamSearchResult como JSON (padrão do projeto: indent=2)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {path}")
