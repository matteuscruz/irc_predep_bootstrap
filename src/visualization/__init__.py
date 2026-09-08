"""
Módulo de visualização PREDEP
Funções seguem especificações de plots.md
"""

from .predep_plots import (
    plot_4x4_optimal_lags,
    plot_12x4_all_lags,
)
from .predominancia_plots import (
    gerar_todas,
    plot_predep_max,
    plot_som_regimes,
)

__all__ = [
    'plot_4x4_optimal_lags',
    'plot_12x4_all_lags',
    'plot_predep_max',
    'plot_som_regimes',
    'gerar_todas',
]
