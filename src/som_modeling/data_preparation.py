import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


def prepare_som_data(
    df: pd.DataFrame, 
    feature_columns: List[str], 
    id_columns: Optional[List[str]] = None,
    handle_nans: str = 'drop'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepara os dados espaciais (bacias ou pixels) para treinamento do SOM.
    
    A abordagem implementada usa a 'Opção A' refinada: 
    - Cada linha é uma localização espacial (bacia/pixel).
    - As colunas (features) são as métricas de sensibilidade (ex: coeficientes do PREDEP, 
      correlação) para cada Modo de Variabilidade (MoV) em diferentes Lags/Estações.
      
    Args:
        df: DataFrame original contendo os dados.
        feature_columns: Lista com o nome das colunas que serão usadas como features no SOM.
        id_columns: Colunas identificadoras (ex: lat, lon, basin_id) para guardar a referência espacial.
        handle_nans: Como tratar valores ausentes ('drop' ou 'impute').
        
    Returns:
        X_scaled: Matriz numpy 2D normalizada (Z-score) pronta para o SOM.
        df_clean: DataFrame limpo (e possivelmente com linhas removidas) que mantém a correspondência 
                  direta de índice com X_scaled para reconstrução espacial posterior.
    """
    # 1. Isolar features e identificadores
    if id_columns is None:
        id_columns = []
        
    cols_to_keep = id_columns + feature_columns
    df_clean = df[cols_to_keep].copy()
    
    # 2. Tratamento de NaNs robusto
    # Em dados climatológicos/espaciais, a falta de dados pode ocorrer no oceano ou fora do domínio.
    if handle_nans == 'drop':
        df_clean = df_clean.dropna(subset=feature_columns)
    elif handle_nans == 'impute':
        # Imputa com a média espacial daquela feature
        df_clean[feature_columns] = df_clean[feature_columns].fillna(df_clean[feature_columns].mean())
    else:
        raise ValueError("handle_nans deve ser 'drop' ou 'impute'")
        
    df_clean = df_clean.reset_index(drop=True)
    
    # 3. Extrair matriz de features pura
    X_raw = df_clean[feature_columns].values
    
    # 4. Normalização (Z-score)
    # Fundamental pois o SOM usa distância euclidiana. Sem Z-score, 
    # MoVs com magnitudes absolutas maiores dominariam o mapa topológico.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return X_scaled, df_clean
