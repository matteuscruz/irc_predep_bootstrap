"""
Configurações para scripts de visualização
"""

from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Listas
MOV_LIST = ['ATL3', 'NAO', 'PDO', 'AMO', 'ENSO', 'IOD', 'SAM', 'TNA', 'TSA']
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
LAG_RANGE = list(range(13))

# Configurações cartográficas
BRAZIL_EXTENT = [-80, -34, -35, 6]
COASTLINE_COLOR = 'black'
COASTLINE_LINEWIDTH = 0.8

# Nomes dos MoVs
MOV_NAMES = {
    'ATL3': 'Atlantic Niño (ATL3)',
    'NAO': 'North Atlantic Oscillation (NAO)',
    'PDO': 'Pacific Decadal Oscillation (PDO)',
    'AMO': 'Atlantic Multidecadal Oscillation (AMO)',
    'ENSO': 'El Niño-Southern Oscillation (ENSO)',
    'IOD': 'Indian Ocean Dipole (IOD)',
    'SAM': 'Southern Annular Mode (SAM)',
    'TNA': 'Tropical North Atlantic (TNA)',
    'TSA': 'Tropical South Atlantic (TSA)',
}

def get_mov_name(mov_code):
    """Retorna nome completo do MoV."""
    return MOV_NAMES.get(mov_code, mov_code)

def setup_logging():
    """Configura logging básico."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)
