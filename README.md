# PREDEP / MoV x Precipitacao

Repositorio para analises PREDEP e regressao entre indices de modos de variabilidade (MoV) e precipitacao. Os experimentos pesados rodam no Modal (cloud) e sao disparados localmente via `modal run`.

## Requisitos

- Python >= 3.9
- pip
- Dependencias do projeto (em `requirements.txt`)
- Conta no Modal (obrigatoria para executar experimentos)

### Instalacao local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install modal
```

## Modal: autenticacao e verificacao

1. Crie uma conta no Modal.
2. Gere token local:

```bash
modal token new
```

3. Verifique a autenticacao:

```bash
modal token current
```

4. Smoke test:

```bash
modal run tests/test_modal.py
```

## Dados (nao versionados)

Os dados nao sao versionados no Git por tamanho/privacidade. Coloque tudo localmente antes de executar.

### Pipeline base (entrada generica)

- `data/input/merged_indices.csv`
- `data/input/seasonal_precipitation_anomalies_1982_2020.nc`

### Modal: Brasil (experimentos 02 e 03)

Arquivos locais esperados:

- `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc`
- `data/raw/mov_indices/TNA_indice data.csv`
- `data/raw/mov_indices/*_indice*.csv` (outros MoVs)

Se o arquivo mensal nao existir, os scripts tentam gerar a partir dos diarios:

- `data/raw/xavier_daily/pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.2-002.nc`
- `data/raw/xavier_daily/pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.2-003.nc`
- `data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-001.nc`
- `data/raw/xavier_daily/pr_20010101_20191231_BR-DWGD_UFES_UTEXAS_v_3.2.2-004.nc`

### Modal: Uruguai por clusters (experimento 04)

- `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc`
- `data/raw/mov_indices/*_indice*.csv`
- `data/clusters/uruguai/URUGUAI_recortado.shp`
- `data/processed/uruguai_cluster_masks.npz` (gerado localmente)

### Modal: Brasil granular por bacia (experimento 06)

- `data/processed/xavier_monthly/pr_monthly_1961_2019_BRAZIL_combined.nc`
- `data/raw/mov_indices/*_indice*.csv`
- `data/clusters/<bacia>/*.shp` (amazonica, atlantico_leste, atlantico_nordeste, atlantico_sudeste, atlantico_sul, paraguai, paranaiba, sao_francisco, tocantins, uruguai)

### Geracao de mensal RS (opcional)

Para produzir o arquivo mensal do RS:

```bash
python tools/convert_xavier_daily_to_monthly.py
```

Gera: `data/processed/xavier_monthly/pr_monthly_1961_2019_RS_combined.nc`

## Execucao dos experimentos no Modal

Todos os comandos abaixo exigem token valido.

```bash
modal run experiments/02_predep_xavier_brazil_analysis.py
modal run experiments/03_regression_xavier_brazil_analysis.py
modal run experiments/04_predep_uruguai_clusters.py --test
modal run experiments/04_predep_uruguai_clusters.py
modal run experiments/06_predep_comparisson_brazil_granular.py --movs "ONI" --basins "uruguai"
```

## Volumes e paths no Modal

Os experimentos usam o volume `xavier-data`. Principais paths remotos:

- `/data/input/pr_monthly_1961_2019_BRAZIL_combined.nc`
- `/data/input/<MOV>_indice_data.csv`
- `/data/input/uruguai_cluster_masks.npz`
- `/data/input/basin_masks/<bacia>.npz`

Comandos uteis:

```bash
modal volume ls
modal volume create xavier-data
modal volume put xavier-data <arquivo_local> /input/<nome_remoto>
modal volume get xavier-data results/ ./local_results/
```

## Saidas (locais apos download)

- `results/xavier_brazil/`
- `results/regression/`
- `results/predep_uruguai/expN/`
- `results/predep_granular_brazil/expN/`

## Estrutura do repositorio

- `experiments/`: entrypoints Modal
- `src/`: implementacao PREDEP, regressao, processamento e visualizacao
- `scripts/`: automacao e monitoramento
- `tools/`: utilitarios pontuais
- `data/`: insumos e dados processados (fora do Git)
- `results/`: saidas (fora do Git)
- `notebooks/`: exploracao local
- `docs/`: metodologia
- `tests/`: testes unitarios e Modal

## Troubleshooting rapido

- Token: `modal token current`
- Modal instalado: `python -c "import modal; print(modal.__version__)"`
- Dados faltando: verifique `data/` e o README em `data/`
- Volume vazio: refaca upload com `modal volume put` e rode novamente
