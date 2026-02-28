#!/usr/bin/env python3
"""
Validação de Dados - Verificar antes de executar
"""

import pandas as pd
import xarray as xr
from pathlib import Path
import sys

def validate_data():
    """Valida se os dados estão prontos para análise."""
    print("=" * 80)
    print("VALIDAÇÃO DE DADOS")
    print("=" * 80)
    print()
    
    errors = []
    warnings = []
    
    # 1. Verificar arquivos
    print("📁 Verificando arquivos...")
    
    mov_file = Path("data/input/merged_indices.csv")
    precip_file = Path("data/input/seasonal_precipitation_anomalies_1982_2020.nc")
    
    if not mov_file.exists():
        errors.append(f"Arquivo não encontrado: {mov_file}")
    else:
        print(f"   ✅ {mov_file}")
    
    if not precip_file.exists():
        errors.append(f"Arquivo não encontrado: {precip_file}")
    else:
        print(f"   ✅ {precip_file}")
    
    if errors:
        print()
        for error in errors:
            print(f"   ❌ {error}")
        print()
        print("❌ VALIDAÇÃO FALHOU - Arquivos faltando")
        return False
    
    print()
    
    # 2. Carregar e validar MoV
    print("📊 Validando índices MoV...")
    try:
        mov_data = pd.read_csv(mov_file, low_memory=False)
        print(f"   Shape: {mov_data.shape}")
        print(f"   Colunas: {len(mov_data.columns)}")
        
        # Verificar MoVs esperados
        expected_movs = ['ATL3', 'NAO_Indice', 'NIÑO_3.4_Indice', 'PDO_Indice', 
                        'AMO_Indice', 'TNA_Indice', 'TSA_Indice', 'IOD_Indice', 
                        'SAM_Indice', 'ONI_Indice', 'Nino3_Indice', 'Nino4_Indice',
                        'SOI_STANDARDIZED_Indice']
        
        found_movs = []
        missing_movs = []
        
        for mov in expected_movs:
            if mov in mov_data.columns:
                n_valid = mov_data[mov].notna().sum()
                print(f"   ✅ {mov:10s}: {n_valid} valores válidos")
                found_movs.append(mov)
            else:
                print(f"   ⚠️  {mov:10s}: NÃO ENCONTRADO")
                missing_movs.append(mov)
        
        if missing_movs:
            warnings.append(f"MoVs não encontrados: {missing_movs}")
        
        print(f"   Total: {len(found_movs)}/{len(expected_movs)} MoVs disponíveis")
        
    except Exception as e:
        errors.append(f"Erro ao ler MoV: {str(e)}")
        print(f"   ❌ Erro: {str(e)}")
    
    print()
    
    # 3. Carregar e validar precipitação
    print("🌧️  Validando precipitação...")
    try:
        precip_data = xr.open_dataset(precip_file)
        print(f"   Dimensões: {dict(precip_data.dims)}")
        print(f"   Variáveis: {list(precip_data.data_vars)}")
        
        if 'pr' not in precip_data:
            errors.append("Variável 'pr' não encontrada no NetCDF")
            print(f"   ❌ Variável 'pr' não encontrada")
        else:
            print(f"   ✅ Variável 'pr' encontrada")
            print(f"   Shape: {precip_data['pr'].shape}")
        
        if 'time' not in precip_data.dims:
            errors.append("Dimensão 'time' não encontrada")
            print(f"   ❌ Dimensão 'time' não encontrada")
        else:
            n_times = len(precip_data.time)
            print(f"   ✅ Timesteps: {n_times}")
            
            # Verificar se tem 4 estações por ano
            if n_times % 4 != 0:
                warnings.append(f"Número de timesteps ({n_times}) não é múltiplo de 4")
        
        precip_data.close()
        
    except Exception as e:
        errors.append(f"Erro ao ler precipitação: {str(e)}")
        print(f"   ❌ Erro: {str(e)}")
    
    print()
    
    # 4. Verificar diretórios de saída
    print("📂 Verificando diretórios de saída...")
    
    output_dirs = [
        "results/high_performance",
        "results/full_analysis",
        "logs"
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    print()
    
    # 5. Resumo
    print("=" * 80)
    
    if errors:
        print("❌ VALIDAÇÃO FALHOU")
        print()
        print("Erros:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False
    
    if warnings:
        print("⚠️  VALIDAÇÃO COM AVISOS")
        print()
        print("Avisos:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
        print("Você pode prosseguir, mas verifique os avisos acima.")
    else:
        print("✅ VALIDAÇÃO COMPLETA - TUDO OK!")
        print()
        print("Você pode executar:")
        print("  1. Teste inicial: python experiments/high_performance_test.py")
        print("  2. Análise completa: python experiments/full_analysis_optimized.py")
    
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = validate_data()
    sys.exit(0 if success else 1)
