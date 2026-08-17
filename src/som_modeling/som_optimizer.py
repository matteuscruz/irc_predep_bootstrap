import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import math
import itertools
from typing import Tuple, List, Dict, Any, Optional


def get_heuristic_grid_size(n_samples: int) -> int:
    """
    Calcula o número total de neurônios (M) usando a heurística M ≈ 5 * sqrt(n).
    """
    return int(round(5 * math.sqrt(n_samples)))


def get_pca_aspect_ratio(X: np.ndarray) -> float:
    """
    Calcula a proporção (Aspect Ratio) baseada nos dois maiores autovalores do PCA.
    A proporção é dada por sqrt(lambda_1) / sqrt(lambda_2) ou similar, onde 
    lambda_1 e lambda_2 são as variâncias explicadas.
    Retorna o aspect ratio esperado = max(lambda) / min(lambda) para determinar a forma.
    
    Returns:
        float: Razão (λ1 / λ2)
    """
    pca = PCA(n_components=2)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    # Aspect ratio: sqrt(var1) / sqrt(var2) -> desvio padrao
    # Assumindo eigenvalues[0] >= eigenvalues[1]
    if eigenvalues[1] == 0:
        return 1.0
    return math.sqrt(eigenvalues[0]) / math.sqrt(eigenvalues[1])


def generate_candidate_topologies(target_M: int, aspect_ratio: float, search_radius: int = 2) -> List[Tuple[int, int]]:
    """
    Gera configurações de grade (linhas x colunas) próximas ao target_M que respeitam
    aproximadamente o aspect_ratio.
    """
    # x * y = target_M e x / y = aspect_ratio
    # y = x / aspect_ratio
    # x * (x / aspect_ratio) = target_M -> x^2 = target_M * aspect_ratio
    # x = sqrt(target_M * aspect_ratio)
    
    base_rows = max(1, int(round(math.sqrt(target_M * aspect_ratio))))
    base_cols = max(1, int(round(target_M / base_rows)))
    
    candidates = set()
    
    # Busca configurações em um raio próximo
    for r in range(max(1, base_rows - search_radius), base_rows + search_radius + 1):
        for c in range(max(1, base_cols - search_radius), base_cols + search_radius + 1):
            candidates.add((r, c))
            # Tambem a versao transposta para testar
            candidates.add((c, r))
            
    return list(candidates)


def optimize_som_topology(
    X: np.ndarray,
    sigma_ratio: float = 1.0,
    learning_rate: float = 0.5,
    num_iteration: int = 1000,
    random_seed: int = 42,
    use_epochs: bool = False,
    search_radius: int = 3,
    search_sample_size: Optional[int] = None,
    search_epochs: Optional[int] = None,
    progress_cb: Optional[Any] = None,
) -> Tuple[MiniSom, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Nível 1 do Two-Level Clustering: Encontra as dimensões ótimas da grade SOM.
    Avalia métricas conflitantes: Erro de Quantização (QE) e Erro Topológico (TE)
    e seleciona um modelo que equilibra as duas (fronteira de Pareto).

    Args:
        X: Matriz numpy de entrada (Z-score normalizado).
        sigma_ratio: Proporção inicial do sigma em relação ao raio do SOM (ex: 1.0 -> sigma cobre quase todo o SOM inicialmente).
        learning_rate: Taxa de aprendizado inicial.
        num_iteration: Número de épocas/iterações de treino usadas no TREINO
            FINAL (modelo vencedor, na base completa). Se use_epochs=False
            (padrão, retrocompatível), MiniSom interpreta como Nº TOTAL de
            atualizações de peso, não por amostra — para datasets grandes
            isso sub-treina drasticamente (ex.: 2000 atualizações em 70k
            amostras cobre ~2,8% dos dados). Use use_epochs=True para que
            num_iteration seja de fato o nº de épocas completas.
        random_seed: Semente aleatória para reprodutibilidade.
        use_epochs: Se True, treina por num_iteration ÉPOCAS completas
            (len(X) atualizações cada), como o docstring original já dizia
            mas a implementação não fazia.
        search_radius: Raio de busca de topologias candidatas ao redor da
            forma-base (heurística M≈5√n + aspect ratio via PCA).
        search_sample_size: Se definido (< n_samples), a etapa cara de
            varredura de topologias (treina 1 SOM por candidata só para
            comparar QE/TE) usa uma subamostra aleatória de X em vez da
            base completa. O custo por candidata é ~linear em n_samples,
            então isso é o maior alavancador de custo. O modelo vencedor é
            SEMPRE retreinado do zero na base COMPLETA (X inteiro) com
            num_iteration épocas antes de ser retornado — a subamostragem
            só afeta a comparação relativa entre candidatas, não a
            qualidade do modelo final.
        search_epochs: Nº de épocas usadas em CADA candidata durante a
            varredura (independente de num_iteration, que é reservado ao
            retreino final). Poucas épocas bastam para ranquear
            candidatas relativamente. Se None, usa num_iteration.
        progress_cb: Callback opcional `f(done, total, rows, cols, elapsed_s)`
            chamado após cada candidata treinada, para logging/checkpointing
            externo do progresso da varredura.

    Returns:
        best_som: Instância do MiniSom treinado ótimo (retreinado na base completa).
        best_metrics: Dicionário com QE e TE do modelo ótimo (na base completa) e sua topologia.
        results: Lista com os resultados de todas as topologias testadas na varredura (na subamostra, se aplicável) para traçar a curva de Pareto.
    """
    import time as _time

    n_samples, n_features = X.shape

    target_M = get_heuristic_grid_size(n_samples)
    aspect_ratio = get_pca_aspect_ratio(X)

    candidates = generate_candidate_topologies(target_M, aspect_ratio, search_radius=search_radius)

    if search_sample_size and search_sample_size < n_samples:
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(n_samples, size=search_sample_size, replace=False)
        X_search = X[sample_idx]
    else:
        X_search = X

    _search_epochs = search_epochs if search_epochs is not None else num_iteration

    results = []
    total = len(candidates)

    for done, (rows, cols) in enumerate(candidates, start=1):
        t0 = _time.time()
        # Sigma é tipicamente ajustado baseado no tamanho do mapa.
        sigma = max(1.0, min(rows, cols) / 2.0 * sigma_ratio)

        som = MiniSom(
            x=rows,
            y=cols,
            input_len=n_features,
            sigma=sigma,
            learning_rate=learning_rate,
            topology='hexagonal', # 'hexagonal' ou 'rectangular'
            neighborhood_function='gaussian',
            random_seed=random_seed
        )

        som.pca_weights_init(X_search)
        som.train(X_search, _search_epochs, random_order=True, use_epochs=use_epochs)

        qe = som.quantization_error(X_search)
        te = som.topographic_error(X_search)

        key = f"{rows}x{cols}"
        results.append({
            'rows': rows,
            'cols': cols,
            'grid_size': rows * cols,
            'QE': qe,
            'TE': te,
            'model_key': key
        })

        if progress_cb is not None:
            progress_cb(done, total, rows, cols, _time.time() - t0)

    # Heurística para escolha do Pareto:
    # Normalizar QE e TE para [0,1] e pegar a topologia que minimiza a distância à origem (0,0)
    qes = np.array([r['QE'] for r in results])
    tes = np.array([r['TE'] for r in results])

    qe_min, qe_max = qes.min(), qes.max()
    te_min, te_max = tes.min(), tes.max()

    qe_norm = (qes - qe_min) / (qe_max - qe_min + 1e-9)
    te_norm = (tes - te_min) / (te_max - te_min + 1e-9)

    distances = np.sqrt(qe_norm**2 + te_norm**2)
    best_idx = np.argmin(distances)

    best_result = dict(results[best_idx])
    rows, cols = best_result['rows'], best_result['cols']

    # Retreino final do vencedor na base COMPLETA com o nº de épocas de verdade.
    sigma = max(1.0, min(rows, cols) / 2.0 * sigma_ratio)
    best_som = MiniSom(
        x=rows,
        y=cols,
        input_len=n_features,
        sigma=sigma,
        learning_rate=learning_rate,
        topology='hexagonal',
        neighborhood_function='gaussian',
        random_seed=random_seed
    )
    best_som.pca_weights_init(X)
    best_som.train(X, num_iteration, random_order=True, use_epochs=use_epochs)

    best_result['QE'] = float(best_som.quantization_error(X))
    best_result['TE'] = float(best_som.topographic_error(X))
    best_result['search_QE'] = float(results[best_idx]['QE'])
    best_result['search_TE'] = float(results[best_idx]['TE'])

    return best_som, best_result, results
