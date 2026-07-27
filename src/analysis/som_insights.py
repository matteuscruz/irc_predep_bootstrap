"""
Insights espaciais de PREDEP via Self-Organizing Maps (MiniSom).

Constrói uma matriz de features por pixel (α por MoV × lag, concatenado nas
estações), treina um único SOM sobre todos os pixels válidos do Brasil e agrupa
os neurônios em "regimes" de previsibilidade. Os regimes e suas assinaturas
alimentam os plots aditivos em src/visualization/som_plots.py.

Uso programático:
    from src.analysis.som_insights import (
        build_feature_matrix, train_som, assign_regimes,
        regime_to_grid, regime_fingerprints,
    )

A biblioteca MiniSom é vendorizada em <repo>/minisom e adicionada ao sys.path
abaixo (não está instalada via pip).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Importa as otimizações criadas
from src.som_modeling.som_optimizer import optimize_som_topology
from src.som_modeling.som_macro_clustering import perform_macro_clustering, get_som_weights

# --- Import shim do MiniSom vendorizado (repo_root/minisom) ---------------
_MINISOM_DIR = Path(__file__).resolve().parents[2] / "minisom"
if str(_MINISOM_DIR) not in sys.path:
    sys.path.insert(0, str(_MINISOM_DIR))
from minisom import MiniSom  # noqa: E402

SEASONS = ["DJF", "MAM", "JJA", "SON"]
_NC_SUFFIX = "_predep_granular_seasonal"
DEFAULT_FILL = 0.0  # α >= 0; pixel/feature sem sinal -> "sem previsibilidade"


def _to_str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


@dataclass
class FeatureBundle:
    X: np.ndarray            # (n_pixels, n_features) usado p/ treinar (z-score se standardize)
    X_raw: np.ndarray        # (n_pixels, n_features) α na escala nativa [0, 1]
    feature_names: list      # len n_features, ex: "ONI|lag3|DJF"
    feature_index: np.ndarray  # (n_features, 3) -> (mov_idx, lag_idx, season_idx)
    mov_names: list
    lags: list               # lags efetivamente usados (meses)
    seasons: list            # estações efetivamente usadas
    lats: np.ndarray         # ascendente (n_lat,)
    lons: np.ndarray         # (n_lon,)
    valid_mask: np.ndarray   # (n_lat, n_lon) bool — pixels usados como amostras
    pixel_rc: np.ndarray     # (n_pixels, 2) (row, col) no grid (n_lat, n_lon)
    scaler_mean: np.ndarray  # (n_features,) ou None
    scaler_std: np.ndarray   # (n_features,) ou None


@dataclass
class RegimeResult:
    bmu_rc: np.ndarray        # (n_pixels, 2) coords do BMU por pixel
    neuron_regime: np.ndarray  # (som_x, som_y) regime de cada neurônio
    pixel_regime: np.ndarray  # (n_pixels,) regime de cada pixel
    weights_flat: np.ndarray  # (som_x*som_y, n_features)
    kmeans_centers: np.ndarray  # (n_regimes, n_features)
    n_regimes: int
    kmeans_results: dict = None
    hierarchical_model: object = None


def build_feature_matrix(
    exp_dir,
    mov_filter=None,
    lags=None,
    standardize: bool = True,
    fill_value: float = DEFAULT_FILL,
    season_filter: str = None,
    strict: bool = False,
) -> FeatureBundle:
    """
    Lê os NCs Brazil-wide de cada MoV em exp_dir e monta a matriz de features.

    Cada amostra é um pixel do Brasil; cada feature é α (alpha_core) de um
    (MoV, lag, estação). A ordem das features é (MoV, lag, estação).
    """
    import xarray as xr

    exp_dir = Path(exp_dir)
    datasets, mov_names = [], []
    for mov_dir in sorted(exp_dir.iterdir()):
        if not mov_dir.is_dir() or mov_dir.name == "plots":
            continue
        if mov_filter and mov_dir.name not in mov_filter:
            continue
        nc = mov_dir / f"{mov_dir.name}{_NC_SUFFIX}.nc"
        if nc.exists():
            datasets.append(xr.open_dataset(nc))
            mov_names.append(mov_dir.name)

    if not datasets:
        raise FileNotFoundError(
            f"Nenhum NC Brazil-wide ('{{MoV}}{_NC_SUFFIX}.nc') encontrado em {exp_dir}"
        )

    # Grids iguais entre MoVs; outer-join garante alinhamento.
    aligned = xr.align(*datasets, join="outer", fill_value=np.nan)
    ac = np.stack([ds["alpha_core"].values for ds in aligned])  # (mov, season, lag, lat, lon)

    ds0 = aligned[0]
    nc_seasons = [_to_str(s) for s in ds0.season.values]
    nc_lags = [int(x) for x in ds0.lag.values]
    lats = np.array(ds0.latitude.values)
    lons = np.array(ds0.longitude.values)
    for ds in aligned:
        ds.close()
    for ds in datasets:
        ds.close()

    # Reordena estações para SEASONS (mantém só as presentes).
    seasons = [s for s in SEASONS if s in nc_seasons]
    s_idx = [nc_seasons.index(s) for s in seasons]
    ac = ac[:, s_idx, :, :, :]

    # Subconjunto de lags.
    if lags is not None:
        l_idx = [nc_lags.index(l) for l in lags if l in nc_lags]
    else:
        l_idx = list(range(len(nc_lags)))
    used_lags = [nc_lags[i] for i in l_idx]
    ac = ac[:, :, l_idx, :, :]

    # Latitude ascendente (idioma de cluster_plots / generate_best_mov_plots).
    if len(lats) > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        ac = ac[:, :, :, ::-1, :]

    n_mov, n_season, n_lag, n_lat, n_lon = ac.shape

    # Pixel válido = footprint dos dados (pelo menos uma combinação finita).
    # Mesma convenção de _load_basin_outlines. NaNs residuais viram fill_value,
    # então slabs (MoV, lag, estação) totalmente NaN apenas não contribuem.
    finite = ~np.isnan(ac)
    footprint = np.any(finite, axis=(0, 1, 2))   # (n_lat, n_lon)
    complete = np.all(finite, axis=(0, 1, 2))
    print(
        f"  Cobertura: footprint={int(footprint.sum())} px | "
        f"completos em todos MoV×lag×estação={int(complete.sum())} px"
    )

    valid_mask = complete if strict else footprint
    pixel_rc = np.argwhere(valid_mask)  # (n_pixels, 2)

    # Monta X_raw na ordem (mov, lag, season).
    cols, feature_names, feature_index = [], [], []
    for mi in range(n_mov):
        for li in range(n_lag):
            for sj in range(n_season):
                slab = ac[mi, sj, li]              # (n_lat, n_lon)
                cols.append(slab[valid_mask])      # (n_pixels,)
                feature_names.append(f"{mov_names[mi]}|lag{used_lags[li]}|{seasons[sj]}")
                feature_index.append((mi, li, sj))

    X_raw = np.column_stack(cols).astype(np.float64)
    X_raw = np.nan_to_num(X_raw, nan=fill_value)
    feature_index = np.array(feature_index, dtype=int)

    # Filtro de estação: mantém apenas features da estação solicitada.
    if season_filter is not None:
        if season_filter not in seasons:
            raise ValueError(
                f"season_filter='{season_filter}' não encontrado; "
                f"disponíveis: {seasons}"
            )
        sj = seasons.index(season_filter)
        keep = feature_index[:, 2] == sj
        X_raw = X_raw[:, keep]
        feature_names = [fn for fn, k in zip(feature_names, keep.tolist()) if k]
        feature_index = feature_index[keep]
        seasons = [season_filter]

    if standardize:
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)
        X = (X_raw - mean) / std_safe
        scaler_mean, scaler_std = mean, std_safe
    else:
        X = X_raw.copy()
        scaler_mean = scaler_std = None

    return FeatureBundle(
        X=X, X_raw=X_raw, feature_names=feature_names, feature_index=feature_index,
        mov_names=mov_names, lags=used_lags, seasons=seasons,
        lats=lats, lons=lons, valid_mask=valid_mask, pixel_rc=pixel_rc,
        scaler_mean=scaler_mean, scaler_std=scaler_std,
    )


def train_som(
    X: np.ndarray,
    som_size=None,
    sigma=None,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    seed: int = 42,
    optimize: bool = False
):
    """Treina um SOM sobre X. Retorna (som, metrics)."""
    n, input_len = X.shape
    
    if optimize:
        print("    [+] Otimizando topologia via Fronteira de Pareto (QE vs TE)...")
        som, metrics_dict, pareto_results = optimize_som_topology(
            X, learning_rate=learning_rate, num_iteration=num_iteration, random_seed=seed
        )
        metrics = {
            "som_size": f"{metrics_dict['rows']}x{metrics_dict['cols']}",
            "n_samples": n,
            "n_features": input_len,
            "quantization_error": metrics_dict['QE'],
            "topographic_error": metrics_dict['TE'],
            "pareto_results": pareto_results
        }
        return som, metrics

    # Abordagem fixa original
    if not som_size:
        g = int(np.ceil(np.sqrt(5.0 * np.sqrt(n))))
        g = int(np.clip(g, 8, 30))
    else:
        g = int(som_size)
    if sigma is None:
        sigma = max(1.0, g / 2.0)

    som = MiniSom(
        g, g, input_len, sigma=sigma, learning_rate=learning_rate,
        neighborhood_function="gaussian", topology="rectangular",
        activation_distance="euclidean", random_seed=seed,
    )
    som.pca_weights_init(X)
    som.train_random(X, num_iteration, verbose=False)

    metrics = {
        "som_size": g,
        "n_samples": n,
        "n_features": input_len,
        "quantization_error": float(som.quantization_error(X)),
        "topographic_error": float(som.topographic_error(X)),
    }
    return som, metrics


def assign_regimes(som: MiniSom, X: np.ndarray, n_regimes: int = 7, seed: int = 42, auto_k: bool = False):
    """Agrupa os neurônios em regimes e rotula cada pixel pelo seu BMU."""
    W = som.get_weights()                 # (gx, gy, input_len)
    gx, gy, input_len = W.shape
    W_flat = W.reshape(-1, input_len)
    
    kmeans_results = None
    hierarchical_model = None

    if auto_k:
        print("    [+] Buscando o 'Cotovelo' ideal para o número de macro-clusters (K-Means)...")
        kmeans_results = perform_macro_clustering(som, method='kmeans', k=None, max_k=15, random_seed=seed)
        km = kmeans_results['model']
        n_regimes = kmeans_results['elbow_k']
        print(f"    [+] Cotovelo encontrado: K = {n_regimes}")
        
        # Gera modelo hierarquico apenas para ter o Dendrograma
        hierarchical_model = perform_macro_clustering(som, method='hierarchical', k=n_regimes)['model']
    else:
        from sklearn.cluster import KMeans
        n_regimes = int(min(n_regimes, W_flat.shape[0]))
        km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10).fit(W_flat)
        
    neuron_regime = km.labels_.reshape(gx, gy)

    win = som.win_map(X, return_indices=True)   # {(i,j): [idx,...]}
    n_pixels = X.shape[0]
    bmu_rc = np.zeros((n_pixels, 2), dtype=int)
    pixel_regime = np.full(n_pixels, -1, dtype=int)
    for (i, j), idxs in win.items():
        for k in idxs:
            bmu_rc[k] = (i, j)
            pixel_regime[k] = neuron_regime[i, j]

    return RegimeResult(
        bmu_rc=bmu_rc, neuron_regime=neuron_regime, pixel_regime=pixel_regime,
        weights_flat=W_flat, kmeans_centers=km.cluster_centers_, n_regimes=n_regimes,
        kmeans_results=kmeans_results, hierarchical_model=hierarchical_model
    )


def regime_to_grid(regime_result: RegimeResult, bundle: FeatureBundle) -> np.ndarray:
    """Espalha pixel_regime de volta ao grid (n_lat, n_lon); NaN fora do footprint."""
    grid = np.full((len(bundle.lats), len(bundle.lons)), np.nan)
    rc = bundle.pixel_rc
    grid[rc[:, 0], rc[:, 1]] = regime_result.pixel_regime
    return grid


def auto_top_movs(bundle: FeatureBundle, k: int) -> list:
    """Top-k MoVs por α médio bruto (sobre todas features e pixels)."""
    n_mov = len(bundle.mov_names)
    mov_ids = bundle.feature_index[:, 0]
    scores = np.array([
        bundle.X_raw[:, mov_ids == m].mean()
        if np.any(mov_ids == m) else 0.0
        for m in range(n_mov)
    ])
    return [bundle.mov_names[i] for i in np.argsort(scores)[::-1][:min(k, n_mov)]]


_SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],  "SON": [9, 10, 11],
}


def build_precip_bundle(
    exp_dir,
    precip_nc,
    seasons=None,
    standardize: bool = True,
) -> FeatureBundle:
    """
    FeatureBundle cujas features são lat, lon e médias sazonais de precipitação
    (mm/mês) do dataset Xavier. Pixels restritos ao footprint do PREDEP (exp_dir).

    Features: ["lat", "lon", "precip_DJF", "precip_MAM", "precip_JJA", "precip_SON"]
    (ou subconjunto se seasons for fornecido).
    """
    import xarray as xr
    import pandas as pd

    exp_dir = Path(exp_dir)
    precip_nc = Path(precip_nc)
    seasons = seasons or SEASONS

    # Footprint do PREDEP para obter grade e máscara válida.
    _ref = build_feature_matrix(exp_dir, standardize=False)
    lats, lons = _ref.lats, _ref.lons
    valid_mask, pixel_rc = _ref.valid_mask, _ref.pixel_rc
    pix_lats = lats[pixel_rc[:, 0]]
    pix_lons = lons[pixel_rc[:, 1]]

    # Carrega precipitação Xavier.
    ds = xr.open_dataset(precip_nc)
    pr_lats = ds.latitude.values.astype(float)
    pr_lons = ds.longitude.values.astype(float)

    # Índices nearest-neighbor para cada pixel PREDEP na grade Xavier.
    lat_idx = np.argmin(np.abs(pr_lats[:, None] - pix_lats[None, :]), axis=0)
    lon_idx = np.argmin(np.abs(pr_lons[:, None] - pix_lons[None, :]), axis=0)
    pr_vals = ds["pr"].values.astype(np.float64)  # (time, lat, lon)
    pr_pixels = pr_vals[:, lat_idx, lon_idx]       # (time, n_pixels)
    ds.close()

    time_months = pd.DatetimeIndex(
        xr.open_dataset(precip_nc).time.values
    ).month.to_numpy()

    # Monta features.
    cols, feature_names, feature_index = [], [], []
    # lat e lon normalizados (z-score feito na etapa de standardize abaixo).
    cols.append(pix_lats)
    feature_names.append("lat")
    feature_index.append((0, -1, -1))
    cols.append(pix_lons)
    feature_names.append("lon")
    feature_index.append((1, -1, -1))

    for si, season in enumerate(seasons):
        s_months = _SEASON_MONTHS[season]
        mask_t = np.isin(time_months, s_months)
        precip_s = pr_pixels[mask_t, :].mean(axis=0)
        cols.append(precip_s)
        feature_names.append(f"precip_{season}")
        feature_index.append((2 + si, -1, si))

    X_raw = np.column_stack(cols).astype(np.float64)
    feature_index = np.array(feature_index, dtype=int)
    # mov_names = feature names (lat, lon, precip_DJF, ...)
    mov_names = feature_names[:]

    if standardize:
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)
        X = (X_raw - mean) / std_safe
        scaler_mean, scaler_std = mean, std_safe
    else:
        X = X_raw.copy()
        scaler_mean = scaler_std = None

    return FeatureBundle(
        X=X, X_raw=X_raw, feature_names=feature_names, feature_index=feature_index,
        mov_names=mov_names, lags=[], seasons=list(seasons),
        lats=lats, lons=lons, valid_mask=valid_mask, pixel_rc=pixel_rc,
        scaler_mean=scaler_mean, scaler_std=scaler_std,
    )


def seasonal_regimes(
    som: MiniSom, regimes: RegimeResult, bundle: FeatureBundle,
) -> dict:
    """
    Calcula o regime de cada pixel POR estação, usando apenas as features
    daquela estação (distância parcial ao BMU no SOM treinado).

    Requer bundle em feature_mode 'full' (season_idx >= 0). Retorna
    {season_str: np.ndarray de shape (n_pixels,)}.
    """
    W = som.get_weights()   # (g, g, n_features)
    gy = W.shape[1]
    season_ids = bundle.feature_index[:, 2]
    result = {}
    for sj, season in enumerate(bundle.seasons):
        keep = season_ids == sj
        if not np.any(keep):
            continue
        X_s = bundle.X[:, keep]         # (n_pixels, n_s_feats)
        W_s = W[:, :, keep]             # (g, g, n_s_feats)
        # distância euclidiana parcial: (n_pixels, g*g)
        diff = X_s[:, None, None, :] - W_s[None, :, :, :]
        dist = np.linalg.norm(diff, axis=-1).reshape(X_s.shape[0], -1)
        bmu_flat = dist.argmin(axis=1)
        regime_s = regimes.neuron_regime.ravel()[bmu_flat]
        result[season] = regime_s
    return result


def regime_fingerprints(regime_result: RegimeResult, bundle: FeatureBundle) -> dict:
    """
    Para cada regime, médias de α (escala nativa) por MoV, por lag e por estação,
    mais o tamanho (nº de pixels). Base dos plots de assinatura/heatmap.
    """
    X_raw = bundle.X_raw
    fidx = bundle.feature_index
    mov_ids, lag_ids, season_ids = fidx[:, 0], fidx[:, 1], fidx[:, 2]
    n_mov = len(bundle.mov_names)
    n_lag = len(bundle.lags)
    n_season = len(bundle.seasons)

    out = {}
    for r in range(regime_result.n_regimes):
        sel = regime_result.pixel_regime == r
        size = int(sel.sum())
        if size == 0:
            mean_feat = np.zeros(X_raw.shape[1])
        else:
            mean_feat = X_raw[sel].mean(axis=0)
        by_mov = np.array([
            mean_feat[mov_ids == m].mean() if np.any(mov_ids == m) else 0.0
            for m in range(n_mov)
        ])
        by_lag = np.array([
            mean_feat[lag_ids == l].mean() if np.any(lag_ids == l) else 0.0
            for l in range(n_lag)
        ])
        by_season = np.array([
            mean_feat[season_ids == s].mean() if np.any(season_ids == s) else 0.0
            for s in range(n_season)
        ])
        out[r] = {"size": size, "by_mov": by_mov, "by_lag": by_lag, "by_season": by_season}
    return out
