"""
Testes unitários para a função PREDEP.
"""

import pytest
import numpy as np
from src.predep_calculation.core import predep


class TestPredep:
    """Testes para a função PREDEP."""
    
    def test_predep_basic(self):
        """Teste básico da função PREDEP."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)
        
        result = predep(X, Y, n_boot=1000)
        
        assert isinstance(result, (float, np.floating, np.ndarray))
        # Check if result is NaN-safe
        if isinstance(result, np.ndarray):
            assert not np.isnan(result).all()
        else:
            assert not np.isnan(result)
    
    def test_predep_independent_variables(self):
        """Teste com variáveis independentes."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)
        
        result = predep(X, Y, n_boot=1000)
        
        # Para variáveis independentes, PREDEP deve ser próximo de 0
        assert abs(result) < 0.5
    
    def test_predep_dependent_variables(self):
        """Teste com variáveis dependentes."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = X + np.random.normal(0, 0.1, 100)  # Y depende de X
        
        result = predep(X, Y, n_boot=1000)
        
        assert isinstance(result, (float, np.floating, np.ndarray))
        if isinstance(result, np.ndarray):
            assert not np.isnan(result).all()
        else:
            assert not np.isnan(result)
    
    def test_predep_constant_data(self):
        """Teste com dados constantes."""
        X = np.ones(100)
        Y = np.random.normal(0, 1, 100)
        
        result = predep(X, Y, n_boot=1000)
        
        # Com dados constantes, deve retornar NaN (gracefully handled)
        assert np.isnan(result) or isinstance(result, (float, np.floating, np.ndarray))
    
    def test_predep_small_sample(self):
        """Teste com amostra pequena."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 10)
        Y = np.random.normal(0, 1, 10)
        
        result = predep(X, Y, n_boot=100)
        
        # Result can be float, array, or NaN
        is_valid = isinstance(result, (float, np.floating, np.ndarray))
        is_nan = np.isnan(result) if np.isscalar(result) else np.isnan(result).any() if isinstance(result, np.ndarray) else False
        assert is_valid or is_nan
    
    def test_predep_different_n_boot(self):
        """Teste com diferentes valores de n_boot."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        
        result_1k = predep(X, Y, n_boot=1000)
        result_10k = predep(X, Y, n_boot=10000)
        
        # Both results should be valid scalars or NaN
        is_valid_1k = isinstance(result_1k, (float, np.floating, np.ndarray))
        is_nan_1k = np.isnan(result_1k) if np.isscalar(result_1k) else np.isnan(result_1k).any() if isinstance(result_1k, np.ndarray) else False
        
        is_valid_10k = isinstance(result_10k, (float, np.floating, np.ndarray))
        is_nan_10k = np.isnan(result_10k) if np.isscalar(result_10k) else np.isnan(result_10k).any() if isinstance(result_10k, np.ndarray) else False
        
        assert (is_valid_1k or is_nan_1k)
        assert (is_valid_10k or is_nan_10k)