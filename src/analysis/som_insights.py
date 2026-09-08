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
from typing import Dict

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
    variable: str = "alpha_core"  # variável do NC usada como feature (alpha_core|r2)
    pixel_basin: np.ndarray = None  # (n_pixels,) nome da bacia, se disponível (fonte parquet)


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
    variable: str = "alpha_core",
    nc_suffix: str = _NC_SUFFIX,
) -> FeatureBundle:
    """
    Lê os NCs Brazil-wide de cada MoV em exp_dir e monta a matriz de features.

    Cada amostra é um pixel do Brasil; cada feature é o valor de `variable`
    ("alpha_core" ou "r2") de um (MoV, lag, estação). A ordem das features é
    (MoV, lag, estação). Usar variable="r2" produz um bundle diretamente
    comparável ao de alpha_core (mesmo footprint/pixels, mesma ordem de
    features), permitindo treinar um SOM equivalente sobre r² para
    comparação lado a lado com o SOM de PREDEP.

    `nc_suffix` permite apontar para NCs com sufixo diferente (ex.:
    "_predep_granular_seasonal_chirps" para o experimento CHIRPS).
    """
    import xarray as xr

    exp_dir = Path(exp_dir)
    datasets, mov_names = [], []
    for mov_dir in sorted(exp_dir.iterdir()):
        if not mov_dir.is_dir() or mov_dir.name == "plots":
            continue
        if mov_filter and mov_dir.name not in mov_filter:
            continue
        nc = mov_dir / f"{mov_dir.name}{nc_suffix}.nc"
        if nc.exists():
            datasets.append(xr.open_dataset(nc))
            mov_names.append(mov_dir.name)

    if not datasets:
        raise FileNotFoundError(
            f"Nenhum NC Brazil-wide ('{{MoV}}{nc_suffix}.nc') encontrado em {exp_dir}"
        )

    # Grids iguais entre MoVs; outer-join garante alinhamento.
    aligned = xr.align(*datasets, join="outer", fill_value=np.nan)
    ac = np.stack([ds[variable].values for ds in aligned])  # (mov, season, lag, lat, lon)

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
        scaler_mean=scaler_mean, scaler_std=scaler_std, variable=variable,
    )


def build_feature_matrix_from_parquet(
    exp_dir,
    mov_filter=None,
    lags=None,
    standardize: bool = True,
    fill_value: float = DEFAULT_FILL,
    season_filter: str = None,
    variable: str = "alpha",
    standardize_method: str = "quantile",
    basin_correct: bool = False,
) -> FeatureBundle:
    """
    Variante de build_feature_matrix para a fonte em parquet do experimento
    Brasil-todo (results/predep_granular_brazil/exp_brasil no repo
    predep_visualization): 1 arquivo por MoV × estação, colunas
    [basin, latitude, longitude, lag, r2, alpha]. Cobre a extensão completa
    do país (o cálculo pixel-a-pixel de PREDEP/R² já está pronto; aqui só
    remontamos a matriz de features).

    variable: "alpha" (PREDEP) ou "r2". Note que o nome da coluna aqui é
    "alpha", diferente de "alpha_core" usado por build_feature_matrix (NC).

    standardize_method: "zscore" (padrão) ou "quantile"/"rank". α e r² têm
    distribuição fortemente assimétrica (mediana baixa, cauda longa até 1);
    z-score comprime a maioria dos pixels perto da média e deixa a variância
    dominada pelos poucos valores extremos, o que tende a produzir um
    regime "fundo" dominando o mapa no SOM. "quantile" aplica uma
    transformação rank-based por feature (QuantileTransformer,
    output_distribution="normal" — "RankGauss"), preservando a ordem mas
    removendo a assimetria/cauda, o que tende a distribuir os regimes de
    forma mais equilibrada espacialmente.

    basin_correct: se True, aplica uma correção de "efeito de lote" por
    bacia ANTES da padronização global — cada bacia é centrada na própria
    média (X - média_da_bacia + média_global), feature a feature, sobre
    X_raw. Isso remove um viés sistemático de nível de bacia que vem do
    cálculo de α/r² ter sido feito com um ajuste (regressão + bootstrap)
    independente por bacia, sem apagar a variação real dentro de cada
    bacia. Diagnóstico: sem isso, bacias pequenas e climaticamente
    "isoladas" no cálculo (ex.: Uruguai) podem colapsar inteiras num único
    regime do SOM mesmo sem nenhuma diversidade real — um sintoma de
    divisão administrativa vazando pro resultado, não de sinal climático.
    """
    import pandas as pd

    exp_dir = Path(exp_dir)
    mov_dirs = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir() and (mov_filter is None or d.name in mov_filter)
    )
    if not mov_dirs:
        raise FileNotFoundError(f"Nenhum MoV encontrado em {exp_dir}")
    mov_names = [d.name for d in mov_dirs]
    seasons = [s for s in SEASONS if season_filter is None or s == season_filter]

    frames = {}
    lats = lons = lag_set = None
    for mov_dir, mov in zip(mov_dirs, mov_names):
        for season in seasons:
            f = mov_dir / f"{season}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(
                f, columns=["basin", "latitude", "longitude", "lag", variable]
            )
            frames[(mov, season)] = df
            if lats is None:
                lats = np.sort(df["latitude"].unique())
                lons = np.sort(df["longitude"].unique())
                lag_set = sorted(df["lag"].unique())

    if not frames:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {exp_dir}")

    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    used_lags = [l for l in lag_set if lags is None or l in lags]
    n_lat, n_lon = len(lats), len(lons)
    n_mov, n_season, n_lag = len(mov_names), len(seasons), len(used_lags)

    ac = np.full((n_mov, n_season, n_lag, n_lat, n_lon), np.nan)
    basin_grid = np.full((n_lat, n_lon), "desconhecida", dtype=object)

    for mi, mov in enumerate(mov_names):
        for sj, season in enumerate(seasons):
            df = frames.get((mov, season))
            if df is None:
                continue
            ri_all = df["latitude"].map(lat_idx).to_numpy()
            ci_all = df["longitude"].map(lon_idx).to_numpy()
            lag_all = df["lag"].to_numpy()
            val_all = df[variable].to_numpy()
            for li, lag in enumerate(used_lags):
                sel = lag_all == lag
                ac[mi, sj, li, ri_all[sel], ci_all[sel]] = val_all[sel]
            if mi == 0 and sj == 0:
                basin_grid[ri_all, ci_all] = df["basin"].to_numpy()

    finite = ~np.isnan(ac)
    footprint = np.any(finite, axis=(0, 1, 2))
    complete = np.all(finite, axis=(0, 1, 2))
    print(f"  Cobertura (parquet): footprint={int(footprint.sum())} px | "
          f"completos={int(complete.sum())} px")

    valid_mask = footprint
    pixel_rc = np.argwhere(valid_mask)

    cols, feature_names, feature_index = [], [], []
    for mi in range(n_mov):
        for li in range(n_lag):
            for sj in range(n_season):
                slab = ac[mi, sj, li]
                cols.append(slab[valid_mask])
                feature_names.append(
                    f"{mov_names[mi]}|lag{used_lags[li]}|{seasons[sj]}"
                )
                feature_index.append((mi, li, sj))

    X_raw = np.column_stack(cols).astype(np.float64)
    X_raw = np.nan_to_num(X_raw, nan=fill_value)
    feature_index = np.array(feature_index, dtype=int)
    pixel_basin = basin_grid[valid_mask]

    if basin_correct:
        global_mean = X_raw.mean(axis=0)
        for b in np.unique(pixel_basin):
            mask = pixel_basin == b
            X_raw[mask] = X_raw[mask] - X_raw[mask].mean(axis=0) + global_mean

    if standardize and standardize_method in ("quantile", "rank"):
        from sklearn.preprocessing import QuantileTransformer
        n_quant = max(10, min(1000, X_raw.shape[0]))
        qt = QuantileTransformer(
            output_distribution="normal", n_quantiles=n_quant,
            subsample=min(200_000, X_raw.shape[0]), random_state=42,
        )
        X = qt.fit_transform(X_raw)
        scaler_mean, scaler_std = None, None
    elif standardize:
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
        scaler_mean=scaler_mean, scaler_std=scaler_std, variable=variable,
        pixel_basin=pixel_basin,
    )


def train_som(
    X: np.ndarray,
    som_size=None,
    sigma=None,
    learning_rate: float = 0.5,
    num_iteration: int = 10000,
    seed: int = 42,
    optimize: bool = False,
    use_epochs: bool = False,
    search_radius: int = 3,
    search_sample_size: int = None,
    search_epochs: int = None,
    progress_cb=None,
):
    """Treina um SOM sobre X. Retorna (som, metrics).

    use_epochs=False preserva o comportamento histórico (num_iteration =
    nº total de atualizações de peso, não por amostra — MiniSom's
    train()/train_random() sem use_epochs). Para datasets grandes isso
    sub-treina (ex.: num_iteration=2000 em 70k amostras ≈ 2,8% dos dados
    vistos). use_epochs=True faz num_iteration ser de fato o nº de épocas
    completas (len(X) atualizações cada).

    search_sample_size/search_epochs: ver optimize_som_topology — quando
    optimize=True, permitem que a varredura de topologias (cara: 1 SOM
    por candidata) rode numa subamostra/com poucas épocas, enquanto o
    modelo final (retornado) é sempre retreinado do zero na base COMPLETA
    com num_iteration épocas. Isso é o principal alavancador de custo
    para datasets grandes (~70k pixels): sem isso, a varredura de ~30-50
    candidatas × épocas completas pode facilmente ultrapassar horas.
    """
    n, input_len = X.shape

    if optimize:
        print("    [+] Otimizando topologia via Fronteira de Pareto (QE vs TE)...")
        som, metrics_dict, pareto_results = optimize_som_topology(
            X, learning_rate=learning_rate, num_iteration=num_iteration,
            random_seed=seed, use_epochs=use_epochs, search_radius=search_radius,
            search_sample_size=search_sample_size, search_epochs=search_epochs,
            progress_cb=progress_cb,
        )
        metrics = {
            "som_size": f"{metrics_dict['rows']}x{metrics_dict['cols']}",
            "n_samples": n,
            "n_features": input_len,
            "quantization_error": float(metrics_dict['QE']),
            "topographic_error": float(metrics_dict['TE']),
            "search_quantization_error": float(metrics_dict.get('search_QE', metrics_dict['QE'])),
            "search_topographic_error": float(metrics_dict.get('search_TE', metrics_dict['TE'])),
            "pareto_results": [
                {**r, "QE": float(r["QE"]), "TE": float(r["TE"])}
                for r in pareto_results
            ],
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
    if use_epochs:
        som.train(X, num_iteration, random_order=True, use_epochs=True)
    else:
        som.train_random(X, num_iteration, verbose=False)

    metrics = {
        "som_size": g,
        "n_samples": n,
        "n_features": input_len,
        "quantization_error": float(som.quantization_error(X)),
        "topographic_error": float(som.topographic_error(X)),
    }
    return som, metrics


def assign_regimes(som: MiniSom, X: np.ndarray, n_regimes: int = 7, seed: int = 42,
                   auto_k: bool = False, k_selection: str = "silhouette",
                   k_margin_sigma: float = 1.0, max_k: int = 15):
    """Agrupa os neurônios em regimes e rotula cada pixel pelo seu BMU.

    k_selection: 'silhouette' (padrão, mais estável entre sementes) ou
    'elbow' (WSS/inertia, mais sensível à semente para poucos pontos como
    os neurônios do SOM). Só tem efeito quando auto_k=True.
    k_margin_sigma: exige que o K vencedor da silhouette bata o runner-up
    por pelo menos essa margem (em desvios combinados entre sementes);
    caso contrário, entre os K empatados escolhe o maior (mais granular)
    em vez do argmax cru. Ver select_k_with_margin em som_macro_clustering.py.
    max_k: teto da busca de K (só com auto_k=True). Recomendado = nº de
    MoVs usados no SOM (len(bundle.mov_names)) — mais que isso não faz
    muito sentido semântico (cada MoV pode no máximo "dominar" seu próprio
    regime) e evita que a regra de desempate (escolhe o maior K empatado)
    trave no teto artificial em vez do K real. Se o K escolhido bater
    exatamente em max_k, é sinal de que o teto pode estar truncando a
    busca — vale checar com um teto maior.
    """
    W = som.get_weights()                 # (gx, gy, input_len)
    gx, gy, input_len = W.shape
    W_flat = W.reshape(-1, input_len)

    kmeans_results = None
    hierarchical_model = None

    if auto_k:
        print(f"    [+] Buscando K ideal para os regimes (critério: {k_selection}, max_k={max_k})...")
        kmeans_results = perform_macro_clustering(
            som, method='kmeans', k=None, max_k=max_k, random_seed=seed,
            k_selection=k_selection, k_margin_sigma=k_margin_sigma,
        )
        km = kmeans_results['model']
        n_regimes = kmeans_results['k']
        print(f"    [+] K escolhido: {n_regimes}  "
              f"(cotovelo={kmeans_results['elbow_k']}, "
              f"silhouette={kmeans_results['silhouette_k']}, "
              f"raw={kmeans_results['silhouette_k_raw']})")
        print(f"    [+] {kmeans_results['k_margin_note']}")

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


def local_mov_importance(
    bundle: FeatureBundle, som: MiniSom, regimes: RegimeResult, top_k: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Explicabilidade LOCAL por pixel: decompõe a distância euclidiana
    pixel→BMU (a mesma soma que já vira "atipicidade") por MoV, em vez de
    agregada. Não retreina nada — só reaproveita X, os pesos do SOM e o
    BMU de cada pixel, já calculados.

    Para cada pixel, soma (X - peso_do_BMU)² restrita às colunas de cada
    MoV. Quanto MENOR essa soma para um MoV, mais "explicado" o pixel está
    por aquele MoV nessa alocação específica — o padrão lag×estação dele
    ali é quase idêntico ao protótipo do neurônio vencedor. É o
    complemento LOCAL (por pixel) do que `regime_fingerprints` já dá no
    nível GLOBAL (assinatura média do regime inteiro).

    Retorna dict com:
      mov_diff: (n_pixels, n_mov) soma de diffs² por MoV, ordem de mov_names
      top_movs: (n_pixels, top_k) nomes dos MoVs mais explicativos (menor
                diff primeiro)
      top1_lag, top1_season: (n_pixels,) lag/estação de melhor casamento
                dentro do MoV top1, por pixel
    """
    W = som.get_weights()
    w_pix = W[regimes.bmu_rc[:, 0], regimes.bmu_rc[:, 1]]   # (n_pixels, n_features)
    sq_diff = (bundle.X - w_pix) ** 2                        # (n_pixels, n_features)

    mov_ids = bundle.feature_index[:, 0]
    lag_ids = bundle.feature_index[:, 1]
    season_ids = bundle.feature_index[:, 2]
    lags_arr = np.array(bundle.lags)
    seasons_arr = np.array(bundle.seasons)

    n_pixels = bundle.X.shape[0]
    n_mov = len(bundle.mov_names)
    mov_diff = np.zeros((n_pixels, n_mov))
    for m in range(n_mov):
        cols = mov_ids == m
        mov_diff[:, m] = sq_diff[:, cols].sum(axis=1)

    order = np.argsort(mov_diff, axis=1)                     # menor diff primeiro
    top_idx = order[:, :min(top_k, n_mov)]
    top_movs = np.array(bundle.mov_names)[top_idx]           # (n_pixels, top_k)

    top1 = top_idx[:, 0]
    best_lag = np.empty(n_pixels, dtype=int)
    best_season = np.empty(n_pixels, dtype=object)
    for m in range(n_mov):
        pix_mask = top1 == m
        if not pix_mask.any():
            continue
        cols = np.nonzero(mov_ids == m)[0]
        local_best = np.argmin(sq_diff[np.ix_(pix_mask, cols)], axis=1)
        best_lag[pix_mask] = lags_arr[lag_ids[cols[local_best]]]
        best_season[pix_mask] = seasons_arr[season_ids[cols[local_best]]]

    return {
        "mov_diff": mov_diff,
        "top_movs": top_movs,
        "top1_lag": best_lag,
        "top1_season": best_season,
    }


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


def mov_alpha_scores(bundle: FeatureBundle) -> dict:
    """
    Score por MoV = média (entre pixels) do α máximo por pixel (melhor
    combinação lag×estação daquele MoV naquele pixel). Mesma agregação de
    _collapse_best_mov (som_to_parquet.py), reaproveitada aqui para decidir
    quais MoVs entram no SOM por um limiar de α (ver movs_above_threshold).
    """
    mov_ids = bundle.feature_index[:, 0]
    scores = {}
    for m, name in enumerate(bundle.mov_names):
        cols = mov_ids == m
        if not np.any(cols):
            scores[name] = float("-inf")
            continue
        best_per_pixel = bundle.X_raw[:, cols].max(axis=1)
        scores[name] = float(best_per_pixel.mean())
    return scores


def movs_above_threshold(exp_dir, threshold: float, lags=None, strict: bool = False,
                         source: str = "nc"):
    """
    MoVs cujo score de α (mov_alpha_scores, sempre sobre PREDEP,
    independentemente da variável do SOM a treinar) excede `threshold`.

    source: "nc" (build_feature_matrix, coluna alpha_core) ou "parquet"
    (build_feature_matrix_from_parquet, coluna alpha — experimento Brasil-todo).

    Returns
    -------
    (mov_filter, scores) — mov_filter: list[str] de MoVs acima do limiar
    (ordenados por score decrescente); scores: dict {mov: score} com TODOS
    os MoVs avaliados, para inspeção/relatório.
    """
    if source == "parquet":
        ref = build_feature_matrix_from_parquet(exp_dir, mov_filter=None, lags=lags,
                                                 standardize=False, variable="alpha")
    else:
        ref = build_feature_matrix(exp_dir, mov_filter=None, lags=lags,
                                   standardize=False, strict=strict,
                                   variable="alpha_core")
    scores = mov_alpha_scores(ref)
    kept = sorted((m for m, s in scores.items() if s > threshold),
                  key=lambda m: scores[m], reverse=True)
    return kept, scores


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
