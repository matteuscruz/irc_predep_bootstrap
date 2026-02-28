#!/usr/bin/env python3
"""
Setup para executar análise PREDEP no Modal com dados reais
Este arquivo mostra como adaptar o script para seus dados específicos
"""

import modal

# ====== CONFIGURAÇÃO DE DADOS REAIS ======

def setup_real_data_functions():
    """
    Configure estas funções para carregar seus dados reais no Modal.
    Você tem algumas opções:
    """
    
    # OPÇÃO 1: Upload dos arquivos para o volume Modal
    def upload_data_to_modal():
        """
        Faça upload dos seus arquivos XAVIER e AAO para o volume Modal
        """
        volume = modal.Volume.from_name("xavier-data")
        
        # Upload dos arquivos XAVIER
        volume.put_file(
            "data/raw/xavier_daily/pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc",
            "/xavier_data/pr_1981_2000.nc"
        )
        volume.put_file(
            "data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc", 
            "/xavier_data/pr_2001_2019.nc"
        )
        
        # Upload do arquivo AAO
        volume.put_file(
            "data/raw/mov_indices/AAO_indice data.csv",
            "/aao_data/AAO_indice_data.csv"
        )
    
    # OPÇÃO 2: Carregar de URLs públicas
    def load_xavier_from_urls():
        """
        Se seus dados estão em URLs públicas, baixe diretamente no Modal
        """
        import urllib.request
        import xarray as xr
        
        # Exemplo de download
        url1 = "https://exemplo.com/xavier_1981_2000.nc"
        url2 = "https://exemplo.com/xavier_2001_2019.nc" 
        
        # Baixar para /tmp no Modal
        urllib.request.urlretrieve(url1, "/tmp/xavier_1981_2000.nc")
        urllib.request.urlretrieve(url2, "/tmp/xavier_2001_2019.nc")
        
        # Processar como no script original
        # ...
    
    # OPÇÃO 3: Integrar com cloud storage (S3, GCS, etc.)
    def load_from_cloud_storage():
        """
        Conectar com S3, Google Cloud Storage, etc.
        """
        # Exemplo para AWS S3
        import boto3
        s3 = boto3.client('s3')
        
        # Download dos arquivos
        s3.download_file('meu-bucket', 'xavier/pr_1981_2000.nc', '/tmp/pr_1981_2000.nc')
        s3.download_file('meu-bucket', 'xavier/pr_2001_2019.nc', '/tmp/pr_2001_2019.nc')
        
        # Processar...


# ====== SUBSTITUIÇÕES NO SCRIPT PRINCIPAL ======

def real_load_xavier_seasonal_data():
    """
    Substitua a função load_xavier_data_from_url() no script principal por esta:
    """
    import xarray as xr
    from pathlib import Path
    
    print("📥 Carregando dados XAVIER reais...")
    
    # Arquivos no volume Modal (após upload)
    xavier_files = [
        "/data/xavier_data/pr_1981_2000.nc",
        "/data/xavier_data/pr_2001_2019.nc"
    ]
    
    # Limites do RS (região expandida)
    RS_LAT_MIN, RS_LAT_MAX = -36.0, -25.0
    RS_LON_MIN, RS_LON_MAX = -60.0, -47.0
    
    season_groups = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5], 
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }
    
    seasonal_data = {}
    
    for file_path in xavier_files:
        if Path(file_path).exists():
            print(f"   Processando: {Path(file_path).name}...")
            
            # Carregar com chunking para eficiência
            ds = xr.open_dataset(file_path, chunks={'time': 500})
            
            # Filtrar região do RS
            rs_data = ds.sel(
                latitude=slice(RS_LAT_MIN, RS_LAT_MAX),
                longitude=slice(RS_LON_MIN, RS_LON_MAX)
            )
            
            print(f"      Região RS: {len(rs_data.latitude)} lat × {len(rs_data.longitude)} lon")
            
            # Processar cada estação
            for season, months in season_groups.items():
                season_mask = rs_data.time.dt.month.isin(months)
                season_subset = rs_data.where(season_mask, drop=True)
                
                if len(season_subset.time) > 0:
                    seasonal_means = season_subset.pr.groupby(season_subset.time.dt.year).mean('time')
                    
                    if season in seasonal_data:
                        seasonal_data[season] = xr.concat([seasonal_data[season], seasonal_means], dim='year')
                    else:
                        seasonal_data[season] = seasonal_means
            
            ds.close()
            del ds, rs_data
    
    # Calcular anomalias
    print("   📊 Calculando anomalias...")
    for season in seasonal_data.keys():
        data = seasonal_data[season]
        climatology = data.mean('year')
        anomalies = data - climatology
        seasonal_data[season] = anomalies
        print(f"      {season}: {len(anomalies.year)} anos de anomalias")
    
    return seasonal_data


def real_load_aao_data():
    """
    Substitua a função load_aao_data() no script principal por esta:
    """
    import pandas as pd
    
    print("📥 Carregando dados AAO reais...")
    
    # Arquivo no volume Modal (após upload)
    aao_file = "/data/aao_data/AAO_indice_data.csv"
    
    mov_data = pd.read_csv(aao_file)
    mov_data['time'] = pd.to_datetime(mov_data['time'])
    mov_data['Year'] = mov_data['time'].dt.year
    mov_data['Month'] = mov_data['time'].dt.month
    
    mov_monthly = mov_data.groupby(['Year', 'Month'])['value'].mean().reset_index()
    mov_monthly = mov_monthly.sort_values(['Year', 'Month'])
    mov_values = mov_monthly['value'].dropna().values
    
    print(f"   AAO: {len(mov_values)} meses de dados ({mov_data['Year'].min()}-{mov_data['Year'].max()})")
    
    return mov_values


# ====== INSTRUÇÕES DE EXECUÇÃO ======

"""
PASSO A PASSO PARA EXECUTAR NO MODAL:

1. INSTALAR MODAL:
   pip install modal

2. FAZER LOGIN:
   modal token new

3. PREPARAR DADOS:
   - Opção A: Fazer upload para volume Modal
   - Opção B: Colocar em cloud storage público
   - Opção C: Usar URLs públicas

4. MODIFICAR O SCRIPT:
   - Substitua as funções de carregamento de dados simulados
   - Por suas funções reais (exemplos acima)

5. EXECUTAR:
   modal run experiments/modal_xavier_rs_analysis.py

6. MONITORAR:
   - Logs aparecem em tempo real
   - Interface web Modal mostra progresso

7. BAIXAR RESULTADOS:
   modal volume get xavier-data results/

CONFIGURAÇÕES RECOMENDADAS PARA MODAL:
- GPU: "any" ou "A10G" para processamento rápido
- Memory: 16GB+ para datasets grandes
- Timeout: 6+ horas para análise completa
- Volume: Para persistir dados entre execuções

CUSTOS ESTIMADOS:
- GPU A10G: ~$0.60/hora
- CPU memory: ~$0.05/GB/hora  
- Análise completa: ~$5-15 dependendo da duração

VANTAGENS DO MODAL:
✅ Escalabilidade automática
✅ GPUs potentes disponíveis
✅ Sem configuração de infraestrutura
✅ Logs e monitoramento integrados
✅ Fácil compartilhamento de resultados
"""