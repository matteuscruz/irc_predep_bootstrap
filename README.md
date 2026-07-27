# PREDEP / MoV × Precipitação

Análise de dependência preditiva (PREDEP) e regressão entre índices de modos de
variabilidade (MoV) e precipitação. Os experimentos rodam no Modal (cloud) e são
disparados localmente via `modal run`.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt modal
modal token new   # autenticar no Modal
```

## Dados (não versionados)

Os dados não são incluídos no repositório. Estrutura esperada em `data/`:

| Caminho | Descrição |
|---|---|
| `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc` | Precipitação mensal Xavier (1961–2019) |
| `data/raw/mov_indices/*_indice*.csv` | Séries mensais de cada MoV |
| `data/clusters/<bacia>/*.shp` | Shapefiles das 12 bacias hidrográficas |

Bacias disponíveis: `amazonica`, `atlantico_leste`, `atlantico_nordeste`,
`atlantico_nordeste_oriental`, `atlantico_sudeste`, `atlantico_sul`, `paraguai`,
`parana`, `parnaiba`, `sao_francisco`, `tocantins`, `uruguai`.

## Experimentos

### Experimento 06 — PREDEP granular por bacia

```bash
modal run experiments/06_predep_comparisson_brazil_granular.py [flags]
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--movs` | `"ONI"` | MoVs separados por vírgula, ou `"all"` |
| `--basins` | `"all"` | Bacias separadas por vírgula, ou `"all"` |
| `--exp-n` | `0` | Reusar experimento existente (`0` = cria novo) |
| `--lags` | `"1-12"` | Lags a calcular. Ex: `"0,1,3,6,9,12"` |
| `--lag-zero` | `false` | Adiciona lag 0 a resultados existentes (pula bacias que já têm) |
| `--n-boot` | `5000` | Amostras bootstrap por pixel |
| `--no-use-anomaly` | — | Usa precipitação bruta em vez de anomalias |
| `--test` | `false` | Smoke test: apenas o primeiro MoV e a primeira bacia |

```bash
# Todos os MoVs, todas as bacias — novo experimento
modal run experiments/06_predep_comparisson_brazil_granular.py --movs "all" --basins "all"

# Continuar experimento existente (pula combinações já calculadas)
modal run experiments/06_predep_comparisson_brazil_granular.py --movs "all" --basins "all" --exp-n 21

# Lags selecionados para um MoV específico
modal run experiments/06_predep_comparisson_brazil_granular.py --movs "SAM" --basins "all" --lags "0,1,3,6,9,12"

# Adicionar lag 0 a experimento já existente
modal run experiments/06_predep_comparisson_brazil_granular.py --movs "all" --basins "all" --exp-n 21 --lag-zero
```

Os resultados são salvos em `results/predep_granular_brazil/expN/`.

## Geração de plots (local)

### PNGs por bacia e Brasil completo

```bash
# Experimento mais recente — todos os MoVs
python scripts/generate_plots_exp.py

# Experimento específico, subconjunto de MoVs e lags
python scripts/generate_plots_exp.py --exp-n 21 --movs "ONI,SAM" --lags "1,3,6,9,12"
```

Flags disponíveis: `--exp-n`, `--movs`, `--lags`.
Saída em `results/predep_granular_brazil/expN/plots/{MOV}/`.

### PDFs por bacia

```bash
python scripts/generate_plots_exp.py --exp-n 21 --pdf-basin parana
```

Gera três PDFs em `results/predep_granular_brazil/expN/plots/PDF/`:
- `{bacia}_PREDEP.pdf` — α ótimo + α por lag (um MoV por página)
- `{bacia}_REGRESSAO.pdf` — r² ótimo + r² por lag
- `{bacia}_COMPARACAO.pdf` — diferença PREDEP−r² + mapa vencedor

### Comparação multi-MoV

O notebook `notebooks/best_mov_comparison.ipynb` mostra, por pixel, qual MoV
teve o maior PREDEP (α) e r² para cada estação e lag.

Configure a célula de configuração e execute todas as células:

```python
EXP_N = 21
LAGS  = [0, 1, 3, 6, 9, 12]   # None = todos
SAVE  = True
```

Saída em `results/predep_granular_brazil/expN/plots/best_mov/`:
`best_mov_Geral.png`, `best_mov_Lag_N.png`, `best_mov_expN.pdf`.

Para exploração interativa por MoV individual, use `notebooks/plot_predep_granular_results.ipynb`.

## Estrutura do repositório

| Diretório | Conteúdo |
|---|---|
| `experiments/` | Entrypoints Modal |
| `src/` | PREDEP, regressão, processamento e visualização |
| `scripts/` | Geração de plots e automação |
| `notebooks/` | Exploração local e visualização interativa |
| `data/` | Dados de entrada (não versionados) |
| `results/` | Saídas dos experimentos (não versionados) |
| `tests/` | Testes unitários e Modal |
| `docs/` | Metodologia |

## Troubleshooting

```bash
modal token current                                  # verificar autenticação
python -c "import modal; print(modal.__version__)"   # verificar instalação
modal volume ls                                      # listar volumes
modal volume get xavier-data results/ ./results/     # baixar resultados
```
