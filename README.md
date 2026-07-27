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

> **Perfil Modal:** os dados do projeto ficam no volume `irc-predep-data` do perfil
> `matteuscruz`. Se o perfil ativo for outro, use
> `MODAL_PROFILE=matteuscruz modal run ...` em todos os comandos Modal.

## Dados (não versionados)

Os dados não são incluídos no repositório. Estrutura esperada em `data/`:

| Caminho | Descrição |
|---|---|
| `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc` | Precipitação mensal Xavier (1961–2019, 708 meses) |
| `data/raw/mov_indices/*_indice*.csv` | Séries mensais de cada MoV |
| `data/clusters/<bacia>/*.shp` | Shapefiles das 12 bacias hidrográficas |

Bacias disponíveis: `amazonica`, `atlantico_leste`, `atlantico_nordeste`,
`atlantico_nordeste_oriental`, `atlantico_sudeste`, `atlantico_sul`, `paraguai`,
`parana`, `parnaiba`, `sao_francisco`, `tocantins`, `uruguai`.

## Experimentos PREDEP

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
| `--lag-zero` | `false` | Adiciona lag 0 a resultados existentes |
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
```

Os resultados são salvos em `results/predep_granular_brazil/expN/`.

## Geração de plots (local)

### PNGs por bacia e Brasil completo

```bash
python src/visualization/generate_plots_exp.py                              # experimento mais recente
python src/visualization/generate_plots_exp.py --exp-n 21 --movs "ONI,SAM" --lags "1,3,6"
```

Saída em `results/predep_granular_brazil/expN/plots/{MOV}/`.

### PDFs por bacia

```bash
python src/visualization/generate_plots_exp.py --exp-n 21 --pdf-basin parana
```

Gera em `results/predep_granular_brazil/expN/plots/PDF/`:
- `{bacia}_PREDEP.pdf`, `{bacia}_REGRESSAO.pdf`, `{bacia}_COMPARACAO.pdf`

### Comparação multi-MoV

O notebook `notebooks/best_mov_comparison.ipynb` mostra, por pixel, qual MoV teve o
maior PREDEP (α) e r² para cada estação e lag.

---

## Análise SOM — Self-Organizing Maps

O pipeline SOM agrupa pixels do Brasil por **assinatura de previsibilidade climática**
usando o MiniSom (vendorizado em `minisom/`). Cada pixel é uma amostra descrita por
α (PREDEP) de cada MoV × lag × estação. Os neurônios do SOM são clusterizados via
KMeans nos "regimes" exibidos nos mapas.

### 23 MoVs disponíveis (exp01 — bacia do Paraná)

`AAO`, `AMO`, `AO`, `EPNP`, `IASAS`, `ITSMRG2`, `MEI`, `NAO`,
`NIN03`, `NIN034`, `NIN04`, `NIN12`, `ONI`, `PDO`, `PNA`, `QBO`,
`SAODI`, `SASDI`, `SIOD`, `SOI`, `TNA`, `TSA`, `WP`

### Plots estáticos (local, ~40 s)

```bash
python src/analysis/generate_som_insights.py --exp-n 1
python src/analysis/generate_som_insights.py --exp-n 1 --movs "ONI,TNA,AAO" --n-regimes 5
python src/analysis/generate_som_insights.py --exp-n 1 --som-diagnostics   # inclui U-matrix, hitmap
```

Saída em `plots/exp{NN}/som/` + PDF. Os 7 plots padrão são todos mapas geográficos
(regime, painéis por regime, component planes, atipicidade, fronteiras, fingerprints, heatmap).

### Exportação para a dashboard

```bash
python src/analysis/som_to_parquet.py --exp exp01 [flags]
```

| Flag | Padrão | Descrição |
|---|---|---|
| `--exp` | obrigatório | Experimento de origem (ex: `exp01`) |
| `--feature-mode` | `full` | Modo de features (ver tabela abaixo) |
| `--n-regimes` | `7` | Número de regimes (clusters KMeans) |
| `--lags` | `0,1,3,6,9,12` | Lags a usar (vírgula ou `all`) |
| `--season` | — | Treinar apenas nesta estação: `DJF`, `MAM`, `JJA`, `SON` |
| `--top-movs` | `0` | Usar top-N MoVs por α médio (`0` = todos) |
| `--run-id` | igual a `--exp` | ID do diretório de saída (ex: `exp01_DJF`) |
| `--seasonal-transitions` | — | Adiciona `regime_DJF/MAM/JJA/SON` ao parquet |
| `--seed` | `42` | Semente do SOM e KMeans |
| `--combo-k` | `3` | MoVs no rótulo de combinação (`best_mov`) |
| `--out-dir` | `../predep_visualization/results/predep_som` | Destino |

**Modos de features (`--feature-mode`):**

| Modo | Features | Agrupa por |
|---|---|---|
| `full` | α por MoV × lag × estação (~552 dims) | Perfil completo de previsibilidade |
| `best_mov` | melhor α de cada MoV (23 dims) | Quais MoVs preveem naquela região |
| `precip_latlon` | lat, lon, precip sazonal Xavier (6 dims) | Clima pluviométrico e localização |

```bash
# Modo padrão — agrupa por perfil completo
python src/analysis/som_to_parquet.py --exp exp01

# Modo best_mov — agrupa por combinação de MoVs preditores
python src/analysis/som_to_parquet.py --exp exp01 --feature-mode best_mov

# Por trimestre: top-4 MoVs, lags 1 e 3 (gera exp01_DJF, exp01_MAM, etc.)
for SEASON in DJF MAM JJA SON; do
  python src/analysis/som_to_parquet.py --exp exp01 --season $SEASON \
    --top-movs 4 --lags 1,3 --run-id exp01_$SEASON
done

# Lat/lon + precipitação (clusters geográficos coesos)
python src/analysis/som_to_parquet.py --exp exp01 \
  --feature-mode precip_latlon --run-id exp01_precip

# Transições sazonais (adiciona regime por estação ao parquet)
python src/analysis/som_to_parquet.py --exp exp01 \
  --feature-mode full --seasonal-transitions --run-id exp01_transitions
```

Saída: `{out-dir}/{run-id}/n{k:02d}/som_pixels.parquet` + `som_meta.json`.
Para gerar n_regimes 2–10 de uma vez, use o Modal (ver abaixo).

### Rodar SOM no Modal

O wrapper `src/modal/som_insights_modal.py` usa o volume `irc-predep-data`
(dados já carregados) e escreve os artefatos de volta no volume, depois baixa
localmente.

**Plots estáticos (PNG + PDF):**

```bash
MODAL_PROFILE=matteuscruz modal run src/modal/som_insights_modal.py \
  --exp-n 1 --no-upload
```

**Parquets para a dashboard (n_regimes 2–10):**

```bash
MODAL_PROFILE=matteuscruz modal run src/modal/som_insights_modal.py \
  --exp-n 1 --no-upload --parquets \
  --feature-mode best_mov \
  --n-regimes-start 2 --n-regimes-end 10
```

**Por trimestre meteorológico (top-4 MoVs, lags 1 e 3):**

```bash
for SEASON in DJF MAM JJA SON; do
  MODAL_PROFILE=matteuscruz modal run src/modal/som_insights_modal.py \
    --exp-n 1 --no-upload --parquets \
    --season $SEASON --top-movs 4 --lags 1,3 \
    --n-regimes-start 2 --n-regimes-end 10 \
    --run-id exp01_$SEASON
done
```

Parâmetros completos do entrypoint:

| Flag | Padrão | Descrição |
|---|---|---|
| `--exp-n` | `1` | Número do experimento |
| `--parquets` | — | Gera parquets em vez de plots estáticos |
| `--no-upload` | — | Pula upload dos NCs (dados já no volume) |
| `--no-download` | — | Pula download local |
| `--feature-mode` | `best_mov` | Modo de features (parquets) |
| `--n-regimes-start/end` | `2` / `10` | Range de k para parquets |
| `--season` | — | Filtrar por estação meteorológica |
| `--top-movs` | `0` | Top-N MoVs por α médio |
| `--run-id` | `exp{NN}` | ID do diretório de saída |
| `--seasonal-transitions` | — | Adicionar regime por estação |
| `--n-regimes` | `7` | Regimes para plots estáticos |
| `--diagnostics` | — | Incluir U-matrix, hitmap nos plots |

---

## Dashboard interativa

O repositório irmão `predep_visualization/` contém uma dashboard Dash/Plotly
servida em [predep-visualization.onrender.com](https://predep-visualization.onrender.com).

### Exportar resultados PREDEP para a dashboard

```bash
# Converte NCs de um experimento para parquet (por MoV × bacia × estação)
python src/data_processing/export/nc_to_parquet.py --exp exp501

# Experimentos 501–506 (um MoV cada, Brasil completo)
for N in 501 502 503 504 505 506; do
  python src/data_processing/export/nc_to_parquet.py --exp exp$N
done
```

Saída: `../predep_visualization/results/predep_granular_brazil/exp{NN}/{MoV}/{SEASON}.parquet`.

### Exportar artefatos SOM para a dashboard

```bash
python src/analysis/som_to_parquet.py --exp exp01 --feature-mode best_mov
```

Saída: `../predep_visualization/results/predep_som/exp01/n{k:02d}/som_pixels.parquet` + `som_meta.json`.

A dashboard detecta automaticamente todos os diretórios `predep_som/exp*/n*/`
e os exibe na aba **"SOM (regimes)"** com seletor de n_regimes, modo de visão
(regimes, atipicidade, fronteiras, component planes) e dropdown de estação
(quando o parquet contém colunas `regime_DJF/MAM/JJA/SON`).

---

## Estrutura do repositório

| Diretório / Arquivo | Conteúdo |
|---|---|
| `experiments/` | Entrypoints Modal (PREDEP, regressão) |
| `src/analysis/` | PREDEP, regressão, SOM (`som_insights.py`, `generate_som_insights.py`, `som_to_parquet.py`) |
| `src/visualization/` | Plots PREDEP, SOM, cluster (`cluster_plots.py`, `som_plots.py`, `generate_plots_exp.py`) |
| `src/data_processing/export/` | Exporta resultados NC para parquet (`nc_to_parquet.py`, `brasil_nc_to_parquet.py`, `merge_basins_local.py`) |
| `src/modal/` | Wrappers Modal (`run_experiment.py`, `som_insights_modal.py`) |
| `notebooks/` | Exploração local e visualização |
| `minisom/` | MiniSom vendorizado |
| `data/` | Dados de entrada (não versionados) |
| `results/` | Saídas dos experimentos (não versionados) |
| `plots/` | Plots gerados localmente (não versionados) |
| `tests/` | Testes unitários e Modal |
| `docs/` | Metodologia |

## Troubleshooting

```bash
modal token current                                    # verificar autenticação
MODAL_PROFILE=matteuscruz modal volume ls              # listar volumes do perfil correto
MODAL_PROFILE=matteuscruz modal volume get irc-predep-data /results/predep_granular_brazil ./results/
python -c "import modal; print(modal.__version__)"    # verificar instalação
```
