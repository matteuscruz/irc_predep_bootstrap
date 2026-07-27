import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import math
import itertools
from typing import Tuple, List, Dict, Any


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
    random_seed: int = 42
) -> Tuple[MiniSom, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Nível 1 do Two-Level Clustering: Encontra as dimensões ótimas da grade SOM.
    Avalia métricas conflitantes: Erro de Quantização (QE) e Erro Topológico (TE)
    e seleciona um modelo que equilibra as duas (fronteira de Pareto).
    
    Args:
        X: Matriz numpy de entrada (Z-score normalizado).
        sigma_ratio: Proporção inicial do sigma em relação ao raio do SOM (ex: 1.0 -> sigma cobre quase todo o SOM inicialmente).
        learning_rate: Taxa de aprendizado inicial.
        num_iteration: Número de épocas/iterações de treino.
        random_seed: Semente aleatória para reprodutibilidade.
        
    Returns:
        best_som: Instância do MiniSom treinado ótimo.
        best_metrics: Dicionário com QE e TE do modelo ótimo e sua topologia.
        results: Lista com os resultados de todas as topologias testadas para traçar a curva de Pareto.
    """
    n_samples, n_features = X.shape
    
    target_M = get_heuristic_grid_size(n_samples)
    aspect_ratio = get_pca_aspect_ratio(X)
    
    candidates = generate_candidate_topologies(target_M, aspect_ratio, search_radius=3)
    
    results = []
    trained_models = {}
    
    for (rows, cols) in candidates:
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
        
        som.pca_weights_init(X)
        som.train(X, num_iteration, random_order=True)
        
        qe = som.quantization_error(X)
        te = som.topographic_error(X)
        
        key = f"{rows}x{cols}"
        trained_models[key] = som
        results.append({
            'rows': rows,
            'cols': cols,
            'grid_size': rows * cols,
            'QE': qe,
            'TE': te,
            'model_key': key
        })
        
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
    
    best_result = results[best_idx]
    best_som = trained_models[best_result['model_key']]
    
    return best_som, best_result, results
