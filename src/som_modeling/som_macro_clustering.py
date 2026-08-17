import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from minisom import MiniSom
from typing import Tuple, List, Dict, Optional, Any

DEFAULT_KMEANS_SEEDS = (42, 1, 7, 13, 99)


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


def calculate_silhouette_kmeans(
    weights: np.ndarray, max_k: int = 15, seeds=DEFAULT_KMEANS_SEEDS,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Silhouette score do K-Means nos neurônios, para K=2..max_k, com MÚLTIPLAS
    sementes por K (a atribuição de clusters do K-Means — e portanto tanto o
    WSS quanto o K escolhido pelo cotovelo — pode oscilar de semente para
    semente; a silhouette é mais estável para esse tipo de dado e é reportada
    como média ± desvio-padrão entre sementes).
    """
    k_values = list(range(2, max_k + 1))
    sil_means, sil_stds = [], []
    for k in k_values:
        vals = []
        for sd in seeds:
            km = KMeans(n_clusters=k, random_state=sd, n_init=10).fit(weights)
            vals.append(silhouette_score(weights, km.labels_))
        sil_means.append(float(np.mean(vals)))
        sil_stds.append(float(np.std(vals)))
    return k_values, sil_means, sil_stds


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


def select_k_with_margin(
    k_values: List[int], sil_means: List[float], sil_stds: List[float],
    margin_sigma: float = 1.0,
) -> Dict[str, Any]:
    """
    Escolhe o K vencedor da silhouette exigindo uma margem mínima sobre o
    runner-up, em vez de um argmax cru.

    Motivação: a silhouette média entre sementes tem incerteza (sil_stds);
    um argmax puro pode escolher K=2 mesmo que K=4 esteja a 0,1% de
    distância — um "empate técnico" indistinguível de ruído entre
    sementes, mas apresentado como se fosse uma escolha inequívoca.

    Regra: um K é considerado empatado com o melhor se
    sil_means[k] >= sil_means[best] - margin_sigma * (sil_stds[best] + sil_stds[k])
    (sobreposição de ~margin_sigma desvios combinados). Entre os K
    empatados, escolhe o MAIOR — favorece estrutura mais granular/
    informativa em vez do default trivial (menor K, geralmente K=2).

    Returns
    -------
    dict com 'k' (escolhido pós-margem), 'k_raw' (argmax cru, sem margem),
    'tied' (lista de K's empatados com o melhor) e 'note' (explicação curta).
    """
    means = np.asarray(sil_means, dtype=float)
    stds = np.asarray(sil_stds, dtype=float)
    best_idx = int(np.argmax(means))
    k_raw = k_values[best_idx]

    threshold = means[best_idx] - margin_sigma * (stds[best_idx] + stds)
    tied_idx = [i for i in range(len(k_values)) if means[i] >= threshold[i]]
    tied = [k_values[i] for i in tied_idx]

    chosen = max(tied) if len(tied) > 1 else k_raw
    if len(tied) > 1:
        note = (f"K={k_raw} e K={tied} empatados dentro de {margin_sigma:g}σ "
                f"combinado; K={chosen} escolhido (mais granular)")
    else:
        note = f"K={k_raw} vence com margem clara sobre os demais"

    return {"k": chosen, "k_raw": k_raw, "tied": tied, "note": note}


def perform_macro_clustering(
    som: MiniSom,
    method: str = 'kmeans',
    k: Optional[int] = None,
    max_k: int = 15,
    random_seed: int = 42,
    k_selection: str = 'silhouette',
    kmeans_seeds=DEFAULT_KMEANS_SEEDS,
    k_margin_sigma: float = 1.0,
) -> Dict[str, Any]:
    """
    Nível 2 do Two-Level Clustering: Agrupa os neurônios do SOM.

    Args:
        som: Modelo MiniSom treinado.
        method: 'kmeans' ou 'hierarchical'.
        k: Número de clusters desejado. Se None (e method='kmeans'), busca automaticamente.
        max_k: Número máximo de K para buscar.
        k_selection: 'silhouette' (padrão) ou 'elbow'. O cotovelo do WSS com
            uma única semente oscila bastante para K-Means em poucos pontos
            (os neurônios do SOM); a silhouette média entre múltiplas
            sementes (kmeans_seeds) é mais estável e é o critério
            recomendado. O elbow ainda é computado e reportado para
            comparação/diagnóstico, mesmo quando não é o critério de escolha.
        kmeans_seeds: sementes usadas para a silhouette média por K.

    Returns:
        Dicionário com os rótulos dos neurônios, o modelo de clustering ajustado,
        e no caso do k-means com busca, as curvas de WSS e silhouette.
    """
    weights = get_som_weights(som)
    results = {'method': method}

    if method == 'kmeans':
        if k is None:
            k_values, wss_values = calculate_wss_kmeans(weights, max_k=max_k, random_seed=random_seed)
            elbow_k = find_elbow_point(k_values, wss_values)
            results['k_values'] = k_values
            results['wss_values'] = wss_values
            results['elbow_k'] = elbow_k

            _, sil_means, sil_stds = calculate_silhouette_kmeans(
                weights, max_k=max_k, seeds=kmeans_seeds
            )
            results['silhouette_means'] = sil_means
            results['silhouette_stds'] = sil_stds

            margin_result = select_k_with_margin(
                k_values, sil_means, sil_stds, margin_sigma=k_margin_sigma
            )
            silhouette_k = margin_result['k']
            results['silhouette_k'] = silhouette_k
            results['silhouette_k_raw'] = margin_result['k_raw']
            results['k_candidates_tied'] = margin_result['tied']
            results['k_margin_note'] = margin_result['note']

            k = silhouette_k if k_selection == 'silhouette' else elbow_k
            results['k_selection'] = k_selection

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
