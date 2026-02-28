#!/usr/bin/env python3
"""
Converte dados diários XAVIER em acumulados mensais.
Processa todos os arquivos XAVIER e gera NetCDF mensal para o Rio Grande do Sul.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Limites do Rio Grande do Sul
RS_LAT_MIN, RS_LAT_MAX = -36.0, -25.0
RS_LON_MIN, RS_LON_MAX = -60.0, -47.0


def process_xavier_to_monthly(input_file, output_file, region=None):
    """
    Converte dados diários em acumulados mensais.
    
    Parameters
    ----------
    input_file : str
        Arquivo NetCDF diário
    output_file : str
        Arquivo NetCDF mensal de saída
    region : dict, optional
        {'lat_min': -36, 'lat_max': -25, 'lon_min': -60, 'lon_max': -47}
    """
    print(f"\n📂 Processando: {Path(input_file).name}")
    
    # Carregar dados
    ds = xr.open_dataset(input_file, chunks={'time': 365})
    
    # Filtrar região se especificado
    if region:
        ds = ds.sel(
            latitude=slice(region['lat_min'], region['lat_max']),
            longitude=slice(region['lon_min'], region['lon_max'])
        )
        print(f"   Região filtrada: {len(ds.latitude)} lat × {len(ds.longitude)} lon")
    
    print(f"   Período: {ds.time.min().dt.year.item()}-{ds.time.max().dt.year.item()}")
    print(f"   Dias: {len(ds.time)}")
    
    # Converter para acumulado mensal (soma dos dias)
    print("   🔄 Calculando acumulados mensais...")
    monthly = ds.resample(time='1M').sum(dim='time')
    
    # Adicionar metadados
    monthly['pr'].attrs['units'] = 'mm/month'
    monthly['pr'].attrs['long_name'] = 'Monthly accumulated precipitation'
    monthly['pr'].attrs['description'] = 'Sum of daily precipitation values'
    monthly.attrs['source'] = f'Processed from {Path(input_file).name}'
    monthly.attrs['processing'] = 'Daily to monthly accumulation'
    
    print(f"   Meses gerados: {len(monthly.time)}")
    
    # Salvar
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    monthly.to_netcdf(output_path)
    print(f"   ✅ Salvo: {output_path}")
    
    ds.close()
    return monthly


def main():
    """Processa todos os arquivos XAVIER."""
    print("="*70)
    print("CONVERSÃO XAVIER: DIÁRIO → MENSAL ACUMULADO")
    print("="*70)
    
    # Arquivos de entrada
    xavier_files = [
        {
            'input': 'data/raw/xavier_daily/pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.2-002.nc',
            'output': 'data/processed/xavier_monthly/pr_monthly_1961_1980_RS.nc',
            'period': '1961-1980'
        },
        {
            'input': 'data/raw/xavier_daily/pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc',
            'output': 'data/processed/xavier_monthly/pr_monthly_1981_2000_RS.nc',
            'period': '1981-2000'
        },
        {
            'input': 'data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc',
            'output': 'data/processed/xavier_monthly/pr_monthly_2001_2019_RS_001.nc',
            'period': '2001-2019 (arquivo 1)'
        },
        {
            'input': 'data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-004.nc',
            'output': 'data/processed/xavier_monthly/pr_monthly_2001_2019_RS_004.nc',
            'period': '2001-2019 (arquivo 2)'
        }
    ]
    
    # Região do RS
    rs_region = {
        'lat_min': RS_LAT_MIN,
        'lat_max': RS_LAT_MAX,
        'lon_min': RS_LON_MIN,
        'lon_max': RS_LON_MAX
    }
    
    print(f"\n🎯 Região: Rio Grande do Sul")
    print(f"   Lat: {RS_LAT_MIN}° a {RS_LAT_MAX}°")
    print(f"   Lon: {RS_LON_MIN}° a {RS_LON_MAX}°")
    
    # Processar cada arquivo
    monthly_datasets = []
    
    for file_info in tqdm(xavier_files, desc="Arquivos", unit="file"):
        if not Path(file_info['input']).exists():
            print(f"\n⚠️  Arquivo não encontrado: {file_info['input']}")
            continue
        
        try:
            monthly = process_xavier_to_monthly(
                file_info['input'],
                file_info['output'],
                region=rs_region
            )
            monthly_datasets.append(monthly)
        except Exception as e:
            print(f"\n❌ Erro ao processar {file_info['period']}: {e}")
            continue
    
    # Combinar todos os períodos
    if len(monthly_datasets) > 0:
        print("\n" + "="*70)
        print("📊 COMBINANDO TODOS OS PERÍODOS")
        print("="*70)
        
        combined = xr.concat(monthly_datasets, dim='time')
        combined = combined.sortby('time')
        
        # Remover duplicatas se houver
        _, unique_indices = np.unique(combined.time.values, return_index=True)
        combined = combined.isel(time=np.sort(unique_indices))
        
        output_combined = Path('data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc')
        combined.to_netcdf(output_combined)
        
        print(f"\n✅ ARQUIVO COMBINADO CRIADO!")
        print(f"   📁 {output_combined}")
        print(f"   📅 Período: {combined.time.min().dt.year.item()}-{combined.time.max().dt.year.item()}")
        print(f"   📊 Total de meses: {len(combined.time)}")
        print(f"   🗺️  Grade: {len(combined.latitude)} lat × {len(combined.longitude)} lon")
    
    print("\n" + "="*70)
    print("✅ CONVERSÃO CONCLUÍDA!")
    print("="*70)
    print(f"📁 Arquivos mensais em: data/processed/xavier_monthly/")
    print(f"📊 Formato: NetCDF com acumulados mensais (mm/mês)")


if __name__ == "__main__":
    main()
