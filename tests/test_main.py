"""
Testes unitários para funções do main.py.
"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Adicionar src ao path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import calculate_predep_lag


class TestMainFunctions:
    """Testes para funções do main.py."""
    
    def setup_method(self):
        """Setup para cada teste."""
        # Criar dados mock
        self.mov_data = pd.DataFrame({
            'ATL3': np.random.normal(0, 1, 50),
            'NAO': np.random.normal(0, 1, 50)
        })
        
        # Criar dataset mock de precipitação
        time_data = pd.date_range('1982-01-01', periods=40, freq='3M')
        self.precip_data = xr.Dataset({
            'pr': (['time', 'latitude', 'longitude'], 
                   np.random.normal(0, 1, (40, 10, 10)))
        }, coords={
            'time': time_data,
            'latitude': np.linspace(-10, 10, 10),
            'longitude': np.linspace(-50, -40, 10)
        })
    
    def test_calculate_predep_lag_valid(self):
        """Teste com dados válidos."""
        result = calculate_predep_lag(
            self.mov_data, self.precip_data, 'ATL3', 'MAM', 0, 1000
        )
        
        assert result is not None
        assert 'mov_index' in result
        assert 'season' in result
        assert 'lag' in result
        assert 'predep' in result
        assert result['mov_index'] == 'ATL3'
        assert result['season'] == 'MAM'
        assert result['lag'] == 0
    
    def test_calculate_predep_lag_invalid_mov(self):
        """Teste com MoV inválido."""
        result = calculate_predep_lag(
            self.mov_data, self.precip_data, 'INVALID', 'MAM', 0, 1000
        )
        
        assert result is None
    
    def test_calculate_predep_lag_high_lag(self):
        """Teste com lag muito alto."""
        result = calculate_predep_lag(
            self.mov_data, self.precip_data, 'ATL3', 'MAM', 100, 1000
        )
        
        assert result is None
    
    def test_calculate_predep_lag_constant_data(self):
        """Teste com dados constantes."""
        # Criar dados constantes
        constant_data = pd.DataFrame({
            'ATL3': np.ones(50)
        })
        
        result = calculate_predep_lag(
            constant_data, self.precip_data, 'ATL3', 'MAM', 0, 1000
        )
        
        # Deve retornar None devido à baixa variabilidade
        assert result is None
    
    def test_calculate_predep_lag_different_seasons(self):
        """Teste com diferentes estações."""
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        
        for season in seasons:
            result = calculate_predep_lag(
                self.mov_data, self.precip_data, 'ATL3', season, 0, 1000
            )
            
            if result is not None:
                assert result['season'] == season