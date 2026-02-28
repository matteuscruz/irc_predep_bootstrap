"""
Módulo para processamento e acumulação mensal dos dados de precipitação Xavier Daily.

Este módulo processa os dados diários de precipitação em formato NetCDF,
convertendo-os para acumulados mensais de forma eficiente para grandes datasets.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir avisos específicos
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class XavierPrecipitationProcessor:
    """
    Classe para processar dados de precipitação Xavier Daily em batches.
    
    Converte dados diários para acumulados mensais de forma eficiente
    em memoria para datasets grandes.
    """
    
    def __init__(self, input_path: str, output_path: str, chunk_size: str = "100MB"):
        """
        Inicializar o processador.
        
        Parameters:
        -----------
        input_path : str
            Caminho para a pasta com os arquivos NetCDF de entrada
        output_path : str  
            Caminho para salvar os arquivos processados
        chunk_size : str
            Tamanho dos chunks para processamento (default: "100MB")
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        
        # Criar diretório de saída se não existir
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Definir arquivos de entrada
        self.input_files = {
            "1961-1980": "pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.2-002.nc",
            "1981-2000": "pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc", 
            "2001-2019": "pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc",
            "2020-2024": "final_2020_ate_2024-004.nc"
        }
        
        logger.info(f"Processador inicializado:")
        logger.info(f"  - Entrada: {self.input_path}")
        logger.info(f"  - Saída: {self.output_path}")
        logger.info(f"  - Chunk size: {self.chunk_size}")
    
    def check_input_files(self) -> Dict[str, bool]:
        """Verificar quais arquivos de entrada existem."""
        existing_files = {}
        total_size = 0
        
        for period, filename in self.input_files.items():
            file_path = self.input_path / filename
            exists = file_path.exists()
            existing_files[period] = exists
            
            if exists:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                logger.info(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"  ✗ {filename} não encontrado")
        
        logger.info(f"Tamanho total dos dados: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        return existing_files
    
    def open_dataset_chunked(self, file_path: Path) -> Optional[xr.Dataset]:
        """
        Abrir um dataset NetCDF com chunking para processamento eficiente.
        
        Parameters:
        -----------
        file_path : Path
            Caminho para o arquivo NetCDF
            
        Returns:
        --------
        xr.Dataset ou None
            Dataset carregado com chunks ou None se erro
        """
        try:
            # Abrir com chunks para processamento eficiente
            ds = xr.open_dataset(
                file_path, 
                chunks={'time': 100, 'lat': 50, 'lon': 50}  # Chunks otimizados
            )
            
            logger.info(f"Dataset aberto: {file_path.name}")
            logger.info(f"  - Dimensões: {dict(ds.dims)}")
            logger.info(f"  - Variáveis: {list(ds.data_vars)}")
            logger.info(f"  - Período: {ds.time.min().values} a {ds.time.max().values}")
            
            return ds
            
        except Exception as e:
            logger.error(f"Erro ao abrir {file_path}: {e}")
            return None
    
    def calculate_monthly_accumulation(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Calcular acumulação mensal de precipitação.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset com dados diários
            
        Returns:
        --------
        xr.Dataset
            Dataset com acumulados mensais
        """
        logger.info("Calculando acumulação mensal...")
        
        # Verificar se variável de precipitação existe
        if 'pr' not in ds.data_vars:
            raise ValueError("Variável 'pr' não encontrada no dataset")
        
        # Calcular acumulado mensal usando resample
        # Converter mm/dia para mm/mês (somar todos os dias do mês)
        monthly_ds = ds.resample(time='MS').sum(dim='time', skipna=True)
        
        # Adicionar atributos descritivos
        if 'pr' in monthly_ds.data_vars:
            monthly_ds['pr'].attrs.update({
                'long_name': 'Monthly accumulated precipitation',
                'units': 'mm/month',
                'description': 'Monthly accumulated precipitation from daily Xavier dataset',
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_units': 'mm/day',
                'processing_method': 'sum_of_daily_values'
            })
        
        # Atualizar atributos globais
        monthly_ds.attrs.update({
            'title': 'Monthly Accumulated Precipitation - Xavier et al.',
            'processing_info': 'Converted from daily to monthly accumulated values',
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_temporal_resolution': 'daily',
            'new_temporal_resolution': 'monthly'
        })
        
        logger.info(f"Acumulação mensal concluída:")
        logger.info(f"  - Timesteps originais: {ds.dims['time']}")
        logger.info(f"  - Timesteps mensais: {monthly_ds.dims['time']}")
        logger.info(f"  - Período: {monthly_ds.time.min().values} a {monthly_ds.time.max().values}")
        
        return monthly_ds
    
    def save_monthly_dataset(self, ds: xr.Dataset, period: str) -> Path:
        """
        Salvar dataset mensal em arquivo NetCDF.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Dataset com dados mensais
        period : str
            Período dos dados (ex: "1961-1980")
            
        Returns:
        --------
        Path
            Caminho do arquivo salvo
        """
        output_filename = f"xavier_pr_monthly_{period.replace('-', '_')}.nc"
        output_file = self.output_path / output_filename
        
        logger.info(f"Salvando dados mensais: {output_filename}")
        
        # Configurações de encoding para otimizar tamanho do arquivo
        encoding = {
            'pr': {
                'zlib': True,
                'complevel': 6,
                'shuffle': True,
                'dtype': 'float32'
            },
            'time': {'dtype': 'float64'},
            'lat': {'dtype': 'float32'},
            'lon': {'dtype': 'float32'}
        }
        
        # Salvar dataset
        ds.to_netcdf(
            output_file,
            encoding=encoding,
            format='NETCDF4'
        )
        
        # Verificar tamanho do arquivo salvo
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Arquivo salvo: {output_file} ({file_size_mb:.1f} MB)")
        
        return output_file
    
    def process_single_file(self, period: str) -> Optional[Path]:
        """
        Processar um único arquivo do período especificado.
        
        Parameters:
        -----------
        period : str
            Período para processar (ex: "1961-1980")
            
        Returns:
        --------
        Path ou None
            Caminho do arquivo processado ou None se erro
        """
        if period not in self.input_files:
            logger.error(f"Período '{period}' não reconhecido")
            return None
        
        filename = self.input_files[period]
        input_file = self.input_path / filename
        
        if not input_file.exists():
            logger.error(f"Arquivo não encontrado: {input_file}")
            return None
        
        logger.info(f"Processando período: {period}")
        logger.info(f"Arquivo: {filename}")
        
        # Verificar se arquivo de saída já existe
        output_filename = f"xavier_pr_monthly_{period.replace('-', '_')}.nc"
        output_file = self.output_path / output_filename
        
        if output_file.exists():
            logger.warning(f"Arquivo de saída já existe: {output_filename}")
            overwrite = input("Sobrescrever? (y/n): ").lower().strip() == 'y'
            if not overwrite:
                logger.info("Processamento cancelado pelo usuário")
                return None
        
        try:
            # Abrir dataset com chunks
            ds = self.open_dataset_chunked(input_file)
            if ds is None:
                return None
            
            # Calcular acumulação mensal
            monthly_ds = self.calculate_monthly_accumulation(ds)
            
            # Forçar computação e carregamento na memória (para dados pequenos)
            # Para datasets muito grandes, pode ser necessário processamento chunk por chunk
            monthly_ds = monthly_ds.compute()
            
            # Salvar dados processados
            output_path = self.save_monthly_dataset(monthly_ds, period)
            
            # Limpeza de memória
            ds.close()
            del ds, monthly_ds
            gc.collect()
            
            logger.info(f"Processamento de {period} concluído com sucesso!")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro no processamento de {period}: {e}")
            return None
    
    def process_all_files(self) -> Dict[str, Optional[Path]]:
        """
        Processar todos os arquivos disponíveis.
        
        Returns:
        --------
        dict
            Dicionário com resultado do processamento para cada período
        """
        logger.info("Iniciando processamento de todos os arquivos...")
        
        # Verificar arquivos disponíveis
        existing_files = self.check_input_files()
        
        results = {}
        processed_count = 0
        
        for period, exists in existing_files.items():
            if exists:
                logger.info(f"\n--- Processando {period} ---")
                result = self.process_single_file(period)
                results[period] = result
                
                if result is not None:
                    processed_count += 1
                    logger.info(f"✓ {period} processado com sucesso")
                else:
                    logger.error(f"✗ Erro no processamento de {period}")
            else:
                results[period] = None
                logger.warning(f"⚠ Arquivo de {period} não encontrado")
        
        logger.info(f"\n=== RESUMO DO PROCESSAMENTO ===")
        logger.info(f"Total de arquivos processados: {processed_count}/{len(existing_files)}")
        logger.info(f"Arquivos de saída salvos em: {self.output_path}")
        
        # Listar arquivos gerados
        output_files = list(self.output_path.glob("xavier_pr_monthly_*.nc"))
        if output_files:
            logger.info("Arquivos gerados:")
            total_size = 0
            for file in output_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
            logger.info(f"Tamanho total: {total_size:.1f} MB")
        
        return results
    
    def create_validation_report(self, processed_files: Dict[str, Optional[Path]]) -> str:
        """
        Criar relatório de validação dos dados processados.
        
        Parameters:
        -----------
        processed_files : dict
            Dicionário com caminhos dos arquivos processados
            
        Returns:
        --------
        str
            Caminho do relatório de validação
        """
        report_file = self.output_path / "validation_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Validação - Dados Xavier Monthly\n\n")
            f.write(f"**Data de processamento:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"**Processador:** XavierPrecipitationProcessor\n\n")
            
            f.write("## Resumo do Processamento\n\n")
            
            success_count = sum(1 for path in processed_files.values() if path is not None)
            f.write(f"- **Arquivos processados:** {success_count}/{len(processed_files)}\n")
            f.write(f"- **Diretório de saída:** `{self.output_path}`\n\n")
            
            f.write("## Detalhes por Período\n\n")
            
            for period, output_path in processed_files.items():
                f.write(f"### Período {period}\n")
                
                if output_path is not None and output_path.exists():
                    f.write(f"- **Status:** ✅ Processado com sucesso\n")
                    f.write(f"- **Arquivo de saída:** `{output_path.name}`\n")
                    
                    try:
                        # Carregar dataset para validação
                        ds = xr.open_dataset(output_path)
                        
                        f.write(f"- **Dimensões:** {dict(ds.dims)}\n")
                        f.write(f"- **Período temporal:** {ds.time.min().values} a {ds.time.max().values}\n")
                        
                        if 'pr' in ds.data_vars:
                            pr_data = ds['pr']
                            f.write(f"- **Estatísticas da precipitação mensal:**\n")
                            f.write(f"  - Média: {float(pr_data.mean().values):.2f} mm/mês\n")
                            f.write(f"  - Mínimo: {float(pr_data.min().values):.2f} mm/mês\n")
                            f.write(f"  - Máximo: {float(pr_data.max().values):.2f} mm/mês\n")
                            f.write(f"  - Desvio padrão: {float(pr_data.std().values):.2f} mm/mês\n")
                        
                        ds.close()
                        
                    except Exception as e:
                        f.write(f"- **Erro na validação:** {e}\n")
                    
                else:
                    f.write(f"- **Status:** ❌ Falha no processamento\n")
                
                f.write("\n")
            
            f.write("## Uso dos Dados Processados\n\n")
            f.write("Os dados mensais podem ser carregados usando:\n\n")
            f.write("```python\n")
            f.write("import xarray as xr\n")
            f.write("ds = xr.open_dataset('caminho/para/xavier_pr_monthly_PERIODO.nc')\n")
            f.write("```\n\n")
        
        logger.info(f"Relatório de validação salvo: {report_file}")
        return str(report_file)


def main():
    """Função principal para executar o processamento via linha de comando."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Processar dados Xavier Daily para acumulados mensais')
    parser.add_argument('--input', '-i', required=True, help='Caminho para pasta com dados de entrada')
    parser.add_argument('--output', '-o', required=True, help='Caminho para pasta de saída')
    parser.add_argument('--period', '-p', help='Período específico para processar (ex: 1961-1980)')
    parser.add_argument('--chunk-size', '-c', default='100MB', help='Tamanho dos chunks (default: 100MB)')
    
    args = parser.parse_args()
    
    # Criar processador
    processor = XavierPrecipitationProcessor(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size
    )
    
    if args.period:
        # Processar período específico
        result = processor.process_single_file(args.period)
        if result:
            print(f"Processamento concluído: {result}")
        else:
            print("Erro no processamento")
    else:
        # Processar todos os arquivos
        results = processor.process_all_files()
        
        # Criar relatório de validação
        processor.create_validation_report(results)


if __name__ == "__main__":
    main()