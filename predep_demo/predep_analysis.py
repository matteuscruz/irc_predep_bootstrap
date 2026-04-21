"""
Módulo principal para cálculo do PREDEP (Predictive Dependence) e Correlações
para as estações meteorológicas.
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as ss
from astropy.stats import bayesian_blocks
from pathlib import Path
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ecdf(data):
    """Implementação simples de ECDF."""
    sorted_data = np.sort(data)
    n = len(data)
    
    def _ecdf(x):
        return np.searchsorted(sorted_data, x, side='right') / n
    
    return _ecdf


def predep(X, Y, n_boot=10000):
    """
    Calcula PREDEP (Predictive Dependence) entre duas variáveis contínuas.
    
    PREDEP mede quanto o conhecimento de Y reduz a incerteza sobre X,
    usando estimação de densidade de kernel e particionamento adaptativo.
    
    Parameters
    ----------
    X : array-like
        Primeira variável (geralmente o preditor - MoV index)
    Y : array-like
        Segunda variável (geralmente a resposta - precipitação)
    n_boot : int, optional
        Número de amostras de bootstrap (padrão: 10000)
    
    Returns
    -------
    float
        Valor PREDEP α ∈ [0, 1]
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        BX1 = np.random.choice(X, n_boot)
        BX2 = np.random.choice(X, n_boot)
        DX = BX1 - BX2
        s_x = float(np.squeeze(ss.gaussian_kde(DX).pdf(0)))

        edges_y = bayesian_blocks(Y)
        ecdf_y = ecdf(Y)

        s_x_mid_y = 0.0
        if edges_y.shape[0] > 1:
            for i in range(1, edges_y.shape[0]):
                bg = edges_y[i - 1]
                ed = edges_y[i]
                X_mid_Y = X[(Y >= bg) & (Y < ed)]
                if X_mid_Y.shape[0] == 0:
                    continue

                BX_mid_Y1 = np.random.choice(X_mid_Y, n_boot)
                BX_mid_Y2 = np.random.choice(X_mid_Y, n_boot)
                DX_mid_Y = BX_mid_Y1 - BX_mid_Y2

                p_range = ecdf_y(ed) - ecdf_y(bg)

                p_x_mid_y = float(np.squeeze(ss.gaussian_kde(DX_mid_Y).pdf(0)))
                s_x_mid_y += p_range * p_x_mid_y
            alpha_est = (s_x_mid_y - s_x) / s_x_mid_y
            return float(alpha_est)
        else:
            return np.nan
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
        logger.debug(f"Erro em predep: {e}")
        return np.nan


def main():
    # Caminhos
    NC_PATH = Path("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/data/ERA5_INMET_Paired_ROI_4_Stations.nc")
    OUTPUT_DIR = Path("/home/paulo/Área de Trabalho/IRB/Base%20Vendaval/05_analysis/plots/predep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Carregando dados de {NC_PATH}...")
    ds = xr.open_dataset(NC_PATH)
    df = ds.to_dataframe().reset_index()
    
    target = 'rajada_max_final_ms'
    
    # Covariáveis identificadas
    data_cols = [
        'ws_mean', 'ws_max', 'ws_std', 'ws_p90', 'sin_dir_mean', 'cos_dir_mean', 
        'msl_mean', 'msl_min', 'msl_range', 't2m_mean', 't2m_max', 't2m_min', 
        't2m_range', 'rh_mean', 'rh_p10', 'td_dep_mean', 'tp_sum', 'tp_max', 
        'd_msl', 'tp_lag1', 'tp_lag2', 'tp_roll3'
    ]
    
    # Filtrar apenas as features que existem no dataframe
    features = [c for c in data_cols if c in df.columns]
    
    results = []
    stations = df['station'].unique()
    
    # Utilizamos n_boot=5000 por padrão para acelerar, mas o valor pode ser alterado para 10000
    N_BOOT = 5000 
    
    for station in stations:
        logger.info(f"=== Processando estação: {station} ===")
        df_station = df[df['station'] == station]
        
        for feature in features:
            logger.info(f"  Calculando métricas para feature: {feature}")
            pair_df = df_station[[feature, target]].dropna()
            
            if len(pair_df) < 50:
                logger.warning(f"    [{feature}] Poucos dados válidos ({len(pair_df)}). Pulando.")
                continue
            
            # Usando todos os dados conforme solicitado pelo usuário
            pair_df_sample = pair_df
                
            X = pair_df_sample[feature].values
            Y = pair_df_sample[target].values
            
            t0 = time.time()
            # 1. PREDEP
            val_predep = predep(X, Y, n_boot=N_BOOT)
            
            # 2. Correlação (Pearson)
            corr_pearson, p_pearson = ss.pearsonr(X, Y)
            
            # 3. Correlação (Spearman)
            corr_spearman, p_spearman = ss.spearmanr(X, Y)
            
            t1 = time.time()
            
            results.append({
                'Station': station,
                'Feature': feature,
                'PREDEP': round(val_predep, 5) if pd.notna(val_predep) else np.nan,
                'Pearson': round(corr_pearson, 4),
                'Spearman': round(corr_spearman, 4),
                'N_samples': len(X),
                'Calc_Time_s': round(t1 - t0, 2)
            })
            
    df_results = pd.DataFrame(results)
    
    # Salvar a matriz geral de resultados
    csv_path = OUTPUT_DIR / "predep_correlations_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nResultados totais salvos em {csv_path}\n")
    
    # Apresentando o Ranking
    print("\n" + "="*50)
    print("RANKING DE VARIÁVEIS POR ESTAÇÃO (Baseado no PREDEP)")
    print("="*50)
    
    for station in stations:
        df_st = df_results[df_results['Station'] == station].copy()
        # Ordenar pelas features com maior Poder Preditivo primeiro
        df_ranked = df_st.sort_values(by='PREDEP', ascending=False)
        
        print(f"\n>> Estação {station} <<")
        print(df_ranked[['Feature', 'PREDEP', 'Pearson', 'Spearman', 'N_samples']].head(7).to_string(index=False))
        print("-" * 50)

    # 4. Gerar Gráficos de Ranking
    plot_rankings(df_results, OUTPUT_DIR)


def plot_rankings(df_results, output_dir):
    """Gera gráficos de barras para o ranking de PREDEP por estação."""
    logger.info("Gerando gráficos de ranking...")
    stations = df_results['Station'].unique()
    
    # Criar um subplot para cada estação
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 6 * len(stations)))
    if len(stations) == 1:
        axes = [axes]
        
    for i, station in enumerate(stations):
        ax = axes[i]
        df_st = df_results[df_results['Station'] == station].copy()
        df_st = df_st.sort_values(by='PREDEP', ascending=False).dropna(subset=['PREDEP'])
        
        if df_st.empty:
            ax.text(0.5, 0.5, f"Estação {station}: Sem dados de PREDEP válidos", 
                    ha='center', va='center')
            continue
            
        sns.barplot(data=df_st, x='PREDEP', y='Feature', ax=ax, palette='viridis')
        ax.set_title(f"Ranking de Poder Preditivo (PREDEP) - Estação {station}")
        ax.set_xlabel("PREDEP (Alpha)")
        ax.set_ylabel("Covariável")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = output_dir / "predep_ranking_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de ranking salvo em {plot_path}")


if __name__ == "__main__":
    main()
