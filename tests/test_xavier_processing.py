"""
Testes unitários para o módulo xavier_pr_accumulated.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import sys

# Adicionar src ao path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data_processing.transform.xavier_pr_accumulated import XavierPrecipitationProcessor


class TestXavierPrecipitationProcessor(unittest.TestCase):
    """Testes para a classe XavierPrecipitationProcessor."""
    
    def setUp(self):
        """Configurar testes com dados temporários."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = Path(self.temp_dir) / "input"
        self.output_path = Path(self.temp_dir) / "output"
        
        # Criar diretórios
        self.input_path.mkdir(parents=True)
        self.output_path.mkdir(parents=True)
        
        # Criar dados de teste
        self.create_test_data()
        
        # Criar processador
        self.processor = XavierPrecipitationProcessor(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            chunk_size="10MB"
        )
    
    def tearDown(self):
        """Limpar dados temporários."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Criar dados NetCDF de teste."""
        # Criar coordenadas
        time = xr.cftime_range('2020-01-01', '2020-12-31', freq='D', calendar='standard')
        lat = np.arange(-10, -5, 0.5)  # 10 pontos
        lon = np.arange(-50, -45, 0.5)  # 10 pontos
        
        # Criar dados de precipitação sintéticos
        np.random.seed(42)
        pr_data = np.random.exponential(2.0, size=(len(time), len(lat), len(lon)))
        
        # Criar dataset
        ds = xr.Dataset(
            {
                'pr': (['time', 'lat', 'lon'], pr_data, {
                    'units': 'mm/day',
                    'long_name': 'Daily precipitation'
                })
            },
            coords={
                'time': time,
                'lat': lat,
                'lon': lon
            }
        )
        
        # Salvar arquivo de teste
        test_file = self.input_path / "final_2020_ate_2024-004.nc"
        ds.to_netcdf(test_file)
    
    def test_processor_initialization(self):
        """Testar inicialização do processador."""
        self.assertEqual(str(self.processor.input_path), str(self.input_path))
        self.assertEqual(str(self.processor.output_path), str(self.output_path))
        self.assertEqual(self.processor.chunk_size, "10MB")
    
    def test_check_input_files(self):
        """Testar verificação de arquivos de entrada."""
        existing_files = self.processor.check_input_files()
        
        # Deve encontrar o arquivo de teste 2020-2024
        self.assertTrue(existing_files['2020-2024'])
        
        # Outros arquivos não existem
        self.assertFalse(existing_files['1961-1980'])
        self.assertFalse(existing_files['1981-2000'])
        self.assertFalse(existing_files['2001-2019'])
    
    def test_open_dataset_chunked(self):
        """Testar abertura de dataset com chunks."""
        test_file = self.input_path / "final_2020_ate_2024-004.nc"
        
        ds = self.processor.open_dataset_chunked(test_file)
        
        self.assertIsNotNone(ds)
        self.assertIn('pr', ds.data_vars)
        self.assertIn('time', ds.dims)
        self.assertIn('lat', ds.dims)
        self.assertIn('lon', ds.dims)
        
        ds.close()
    
    def test_calculate_monthly_accumulation(self):
        """Testar cálculo de acumulação mensal."""
        test_file = self.input_path / "final_2020_ate_2024-004.nc"
        ds = self.processor.open_dataset_chunked(test_file)
        
        monthly_ds = self.processor.calculate_monthly_accumulation(ds)
        
        # Verificar dimensões
        self.assertIn('pr', monthly_ds.data_vars)
        self.assertEqual(len(monthly_ds.time), 12)  # 12 meses em 2020
        
        # Verificar atributos
        self.assertEqual(monthly_ds['pr'].attrs['units'], 'mm/month')
        self.assertIn('Monthly accumulated', monthly_ds['pr'].attrs['long_name'])
        
        # Verificar que valores são maiores (acumulados vs diários)
        daily_mean = ds['pr'].mean().values
        monthly_mean = monthly_ds['pr'].mean().values
        self.assertGreater(monthly_mean, daily_mean)
        
        ds.close()
    
    def test_process_single_file(self):
        """Testar processamento de arquivo único."""
        # Processar arquivo de teste
        result = self.processor.process_single_file("2020-2024")
        
        # Verificar que arquivo foi criado
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())
        
        # Verificar conteúdo do arquivo
        processed_ds = xr.open_dataset(result)
        
        self.assertIn('pr', processed_ds.data_vars)
        self.assertEqual(processed_ds['pr'].attrs['units'], 'mm/month')
        self.assertEqual(len(processed_ds.time), 12)  # 12 meses
        
        processed_ds.close()
    
    def test_process_nonexistent_file(self):
        """Testar processamento de arquivo inexistente."""
        result = self.processor.process_single_file("1961-1980")
        self.assertIsNone(result)
    
    def test_validation_report(self):
        """Testar criação de relatório de validação."""
        # Processar arquivo primeiro
        results = {"2020-2024": self.processor.process_single_file("2020-2024")}
        
        # Criar relatório
        report_path = self.processor.create_validation_report(results)
        
        # Verificar que relatório foi criado
        report_file = Path(report_path)
        self.assertTrue(report_file.exists())
        
        # Verificar conteúdo básico do relatório
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('Relatório de Validação', content)
        self.assertIn('2020-2024', content)
        self.assertIn('Processado com sucesso', content)


if __name__ == '__main__':
    unittest.main()