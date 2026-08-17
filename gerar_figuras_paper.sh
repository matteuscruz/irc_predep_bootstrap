#!/bin/bash

# Ativa o ambiente virtual
source .venv/bin/activate

echo "Gerando Figura 1 (Overview)..."
python3 src/visualization/generate_paper_figures.py

echo "Gerando Figura 2 (Lag Ótimo)..."
python3 src/visualization/fig2_optimal_lag.py

echo "Gerando Figura 3 (Heatmap)..."
python3 src/visualization/fig3_heatmap.py

echo "Gerando Figura 4 (Multi-Lag)..."
# Você pode alterar os MoVs e o idioma passando as flags, ex: --lang en --movs PDO,AMO
python3 src/visualization/fig4_multilag.py

echo "Gerando Figura 5 (PREDEP vs R2)..."
python3 src/visualization/fig5_predep_vs_r2.py

echo "Gerando Figura 6 (Perfis Regionais)..."
python3 src/visualization/fig6_regional.py

echo "Todas as figuras foram geradas em plots/paper_figures/!"
