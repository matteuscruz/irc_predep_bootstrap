import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from minisom import MiniSom
from typing import Tuple, List, Dict, Optional, Any


def get_som_weights(som: MiniSom) -> np.ndarray:
    """Extrai e achata os pesos da grade SOM para macro-clustering."""
    weights = som.get_weights()
    x, y, dim = weights.shape
    return weights.reshape(x * y, dim)


def calculate_wss_kmeans(weights: np.ndarray, max_k: int = 15, random_seed: int = 42) -> Tuple[List[int], List[float]]:
    """
    Calcula o WSS (Within-Cluster Sum of Squares) / Inertia para diferentes valores de K.
    """
    k_values = list(range(2, max_k + 1))
    wss_values = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        kmeans.fit(weights)
        wss_values.append(kmeans.inertia_)
        
    return k_values, wss_values


def find_elbow_point(k_values: List[int], wss_values: List[float]) -> int:
    """
    Encontra o 'cotovelo' da curva WSS usando o ponto de maior distância
    para a reta que conecta o primeiro e o último ponto da curva.
    """
    p1 = np.array([k_values[0], wss_values[0]])
    p2 = np.array([k_values[-1], wss_values[-1]])
    
    distances = []
    for i in range(len(k_values)):
        p3 = np.array([k_values[i], wss_values[i]])
        # Distância de p3 à reta que passa por p1 e p2
        dist = np.abs(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
        
    best_idx = np.argmax(distances)
    return k_values[best_idx]


def perform_macro_clustering(
    som: MiniSom, 
    method: str = 'kmeans', 
    k: Optional[int] = None, 
    max_k: int = 15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Nível 2 do Two-Level Clustering: Agrupa os neurônios do SOM.
    
    Args:
        som: Modelo MiniSom treinado.
        method: 'kmeans' ou 'hierarchical'.
        k: Número de clusters desejado. Se None (e method='kmeans'), busca o cotovelo automaticamente.
        max_k: Número máximo de K para buscar o cotovelo.
        
    Returns:
        Dicionário com os rótulos dos neurônios, o modelo de clustering ajustado,
        e no caso do k-means com busca, a curva de WSS.
    """
    weights = get_som_weights(som)
    results = {'method': method}
    
    if method == 'kmeans':
        if k is None:
            k_values, wss_values = calculate_wss_kmeans(weights, max_k=max_k, random_seed=random_seed)
            k = find_elbow_point(k_values, wss_values)
            results['k_values'] = k_values
            results['wss_values'] = wss_values
            results['elbow_k'] = k
            
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        model.fit(weights)
        labels = model.labels_
        
    elif method == 'hierarchical':
        if k is None:
            # Hierárquico geralmente requer um K predefinido se quisermos os labels
            # Caso contrário, definir default.
            k = 7
        
        model = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=True)
        labels = model.fit_predict(weights)
        
        # Para dendrograma, o modelo treinado será útil
    else:
        raise ValueError(f"Método de clustering não suportado: {method}")
        
    results['k'] = k
    results['labels'] = labels
    results['model'] = model
    
    return results


def assign_bmu_labels(X: np.ndarray, som: MiniSom, cluster_labels: np.ndarray) -> np.ndarray:
    """
    Mapeia os labels de macro-clusters dos neurônios de volta para as amostras originais.
    
    Args:
        X: Dados originais (Z-score normalizado).
        som: Modelo MiniSom treinado.
        cluster_labels: Array 1D com o label de cluster para cada neurônio do SOM (achatado).
        
    Returns:
        Array 1D com o label de macro-cluster associado a cada amostra em X.
    """
    x_dim, y_dim = som.get_weights().shape[:2]
    
    # Reformatar cluster_labels de volta para grade 2D se não estiver
    grid_labels = cluster_labels.reshape(x_dim, y_dim)
    
    sample_labels = []
    for x_i in X:
        win_x, win_y = som.winner(x_i)
        sample_labels.append(grid_labels[win_x, win_y])
        
    return np.array(sample_labels)
