import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from scipy.cluster.hierarchy import dendrogram
from typing import List, Dict, Any, Optional
import os


def plot_pareto_curve(results: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plota a Fronteira de Pareto (Erro de Quantização x Erro Topológico).
    """
    qes = [r['QE'] for r in results]
    tes = [r['TE'] for r in results]
    labels = [r['model_key'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(qes, tes, c='blue', alpha=0.6, s=100)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (qes[i], tes[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.title('Fronteira de Pareto do SOM: QE vs TE')
    plt.xlabel('Erro de Quantização (QE)')
    plt.ylabel('Erro Topológico (TE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_elbow_curve(k_values: List[int], wss_values: List[float], elbow_k: Optional[int] = None, save_path: Optional[str] = None):
    """
    Plota o gráfico do Método do Cotovelo (K x WSS).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wss_values, 'bo-', linewidth=2, markersize=8)
    
    if elbow_k:
        elbow_idx = k_values.index(elbow_k)
        plt.plot(elbow_k, wss_values[elbow_idx], 'r*', markersize=15, label=f'Cotovelo Estimado (K={elbow_k})')
        plt.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.5)
        plt.legend()
        
    plt.title('Método do Cotovelo - K-Means nos Pesos do SOM')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Soma dos Quadrados Intra-Cluster (WSS)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_dendrogram(model: Any, save_path: Optional[str] = None, **kwargs):
    """
    Plota um dendrograma para Clustering Hierárquico.
    O modelo precisa ser AgglomerativeClustering (sem n_clusters definido idealmente,
    ou com compute_distances=True se k estiver definido).
    """
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    plt.figure(figsize=(10, 6))
    plt.title('Dendrograma Hierárquico - Pesos do SOM')
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Neurônios do SOM (Índice)")
    plt.ylabel("Distância de Ligação (Ward)")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_umatrix(som: MiniSom, save_path: Optional[str] = None):
    """
    Plota a U-Matrix (Matriz de Distâncias Unificadas) da grade do SOM.
    Zonas escuras = clusters (vales). Zonas claras = fronteiras (cristas).
    """
    u_matrix = som.distance_map()
    plt.figure(figsize=(8, 8))
    # Usando colormap 'bone_r' conforme documentação original
    plt.pcolor(u_matrix.T, cmap='bone_r')
    plt.colorbar(label='Distância Euclidiana Média p/ Vizinhos')
    
    plt.title('U-Matrix (Mapa de Fronteiras do SOM)')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_geographic_clusters(
    df_clean: Any, 
    labels: np.ndarray, 
    lat_col: str = 'lat', 
    lon_col: str = 'lon', 
    save_path: Optional[str] = None
):
    """
    Integração geoespacial: Mapeia as amostras originais (bacias/pixels) com seus
    respectivos labels finais. Utiliza matplotlib basemap simples (scatter) para 
    evitar complexidade de dependências geopandas, mas pode ser expandido.
    """
    df_plot = df_clean.copy()
    df_plot['macro_cluster'] = labels
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        df_plot[lon_col], 
        df_plot[lat_col], 
        c=df_plot['macro_cluster'], 
        cmap='tab10', 
        s=10, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Regime/Macro-Cluster (K-Means)')
    plt.title('Zonas de Influência dos MoVs (Regimes Climáticos)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
