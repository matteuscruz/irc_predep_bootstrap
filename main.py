"""
Script Principal - PREDEP Bootstrap
Interface de entrada para diferentes tipos de análise.
"""

import pandas as pd
import xarray as xr
import numpy as np
from src.predep_calculation.core import predep


def calculate_predep_simple(X, Y, n_boot=10000):
    """Calcula PREDEP entre duas variáveis."""
    return predep(X, Y, n_boot=n_boot)


def load_data():
    """Carrega dados de entrada."""
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    precip_data = xr.open_dataset("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    return mov_data, precip_data


def main():
    """Função principal."""
    print("PREDEP - Análise de Dependência Preditiva")
    print("=" * 40)
    
    mov_data, precip_data = load_data()
    
    print(f"Dados carregados:")
    print(f"- MoVs: {list(mov_data.columns)}")
    print(f"- Precipitação: {precip_data.dims}")
    
    print("\nExperimentos disponíveis:")
    print("- python experiments/preliminary_results.py  # Resultados preliminares ATL3")
    print("- python experiments/initial_test.py         # Teste ATL3 vs MAM")
    print("- python experiments/full_analysis.py        # Análise completa")


if __name__ == "__main__":
    main()