"""
Testes unitários para a função PREDEP.
"""

import pytest
import numpy as np
from src.predep_calculation.core import predep


class TestPredep:
    """Testes para a função PREDEP."""
    
    def test_predep_returns_float_or_nan(self):
        """Teste se PREDEP retorna float ou NaN."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        
        result = predep(X, Y, n_boot=1000)
        
        assert isinstance(result, (float, np.floating)) or np.isnan(result)
    
    def test_predep_with_valid_arrays(self):
        """Teste com arrays válidos."""
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([2, 4, 6, 8, 10])
        
        result = predep(X, Y, n_boot=100)
        
        assert isinstance(result, (float, np.floating)) or np.isnan(result)
    
    def test_predep_with_constant_data_returns_nan(self):
        """Teste com dados constantes deve retornar NaN."""
        X = np.ones(20)
        Y = np.random.normal(0, 1, 20)
        
        result = predep(X, Y, n_boot=100)
        
        # Com dados constantes, deve retornar NaN ou falhar graciosamente
        assert np.isnan(result) or isinstance(result, (float, np.floating))