# PREDEP Bootstrap - Análise de Dependência Preditiva

[![Tests](https://github.com/username/irc_predep_bootstrap/workflows/tests/badge.svg)](https://github.com/username/irc_predep_bootstrap/actions)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=irc_predep_bootstrap&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=irc_predep_bootstrap)

Sistema para análise de dependência preditiva (PREDEP) entre índices de Modos de Variabilidade (MoV) oceânica e precipitação sazonal, com foco em processamento paralelo e análise climática.

## 📋 Visão Geral

O PREDEP (Predictive Dependence) é uma métrica estatística que quantifica o quanto o conhecimento de uma variável Y reduz a incerteza sobre uma variável X. Este projeto implementa:

- **Cálculo PREDEP** entre índices MoV e precipitação sazonal
- **Processamento paralelo** para análises em larga escala
- **Análise de lags temporais** (0-12 meses)
- **Validação estatística** com bootstrap
- **Visualização** e relatórios automatizados

## 🚀 Instalação

### Pré-requisitos
- Python 3.8+
- Git

### Instalação via pip
```bash
git clone https://github.com/username/irc_predep_bootstrap.git
cd irc_predep_bootstrap
pip install -r requirements.txt
```

### Instalação via uv (recomendado)
```bash
git clone https://github.com/username/irc_predep_bootstrap.git
cd irc_predep_bootstrap
uv sync
```

## 🧪 Experimentos

### Experimento Preliminar (Recomendado)

**Análise Espacial PREDEP - ATL3 vs MAM**

```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Executar experimento
python experiments/preliminary_exp.py
```

**O que faz:**
- Calcula PREDEP espacial entre ATL3 e precipitação MAM
- Lags: 0-12 meses
- Bootstrap: 5.000 amostras
- Grade: 393 lat × 391 lon (~153.000 pontos)
- Gera mapas automaticamente

**Tempo estimado:** ~10-15 minutos

**Saídas:**
- `results/spatial/ATL3_MAM_spatial.nc` - Dados completos em NetCDF
- `results/spatial/ATL3_MAM_spatial_maps.png` - 13 mapas (um por lag)

**Visualizar resultados:**
```bash
# Abrir imagem
xdg-open results/spatial/ATL3_MAM_spatial_maps.png

# Ou copiar para área de trabalho
cp results/spatial/ATL3_MAM_spatial_maps.png ~/Desktop/
```

**Documentação completa:** `experiments/PRELIMINARY_EXP_DOC.md`

---

### Executar Análise Preliminar (Legado)

```bash
# Resultados preliminares - ATL3 para todas as estações
python experiments/preliminary_results.py
```

O experimento executa:
- Cálculo PREDEP para ATL3 vs todas as estações (DJF, MAM, JJA, SON)
- Lags de 0-12 meses
- Bootstrap com 10.000 amostras
- Geração automática de gráficos
- Salva resultados em CSV e NetCDF

**Saídas:**
- `results/preliminary/ATL3_results.csv` - Resultados tabulares
- `results/preliminary/ATL3_results.nc` - Resultados em NetCDF
- `results/preliminary/ATL3_grid.png` - Grade 2x2 por estação
- `results/preliminary/ATL3_comparison.png` - Comparação entre estações

### Estrutura de Experimentos

```
experiments/
├── preliminary_results.py   # Resultados preliminares ATL3
├── initial_test.py          # Teste ATL3 vs MAM
├── full_analysis.py         # Análise completa
└── results/                 # Resultados dos experimentos
```

### Executar Experimentos Adicionais

```bash
# Teste inicial (ATL3 vs MAM)
python experiments/initial_test.py

# Análise completa (todos MoVs)
python experiments/full_analysis.py
```

### Criar Novo Experimento

```python
# experiments/meu_experimento.py
from src.predep_calculation.core import predep
import pandas as pd

def meu_experimento():
    # Carregar dados
    mov_data = pd.read_csv("data/input/merged_indices.csv")
    
    # Preparar variáveis
    X = mov_data['ATL3'].dropna().values
    Y = mov_data['NAO'].dropna().values
    
    # Calcular PREDEP
    alpha = predep(X, Y, n_boot=10000)
    print(f"PREDEP α = {alpha:.4f}")
    
    return alpha

if __name__ == "__main__":
    meu_experimento()
```

Executar: `.venv/bin/python experiments/meu_experimento.py`

## 📁 Estrutura do Projeto

```
irc_predep_bootstrap/
├── src/                          # Código fonte
│   ├── predep_calculation/       # Algoritmo PREDEP
│   ├── data_processing/          # Processamento de dados
│   ├── utils/                    # Utilitários
│   └── visualization/            # Visualizações
├── data/                         # Dados
│   ├── input/                    # Dados de entrada
│   ├── processed/                # Dados processados
│   └── raw/                      # Dados brutos
├── config/                       # Configurações
├── scripts/                      # Scripts utilitários
├── tests/                        # Testes
├── notebooks/                    # Jupyter notebooks
├── results/                      # Resultados
└── output/                       # Saídas finais
```

## 🔧 Configuração

### Arquivos de Configuração

- `config/analysis_config.yaml`: Parâmetros da análise
- `config/high_performance.yaml`: Configurações de performance
- `config/paths_config.yaml`: Caminhos dos dados

### Dados de Entrada

Coloque seus dados em:
- `data/input/merged_indices.csv`: Índices MoV
- `data/input/seasonal_precipitation_anomalies_1982_2020.nc`: Dados de precipitação

## 📈 Funcionalidades

### Algoritmo PREDEP
- Implementação original do algoritmo PREDEP
- Estimação de densidade via KDE
- Particionamento adaptativo com Bayesian Blocks
- Validação por bootstrap

### Processamento Paralelo
- Execução paralela com `joblib`
- Otimização para múltiplos cores
- Monitoramento de progresso

### Análise Temporal
- Lags de 0-12 meses
- Análise sazonal (DJF, MAM, JJA, SON)
- Múltiplos índices MoV

### Validação e Testes
- Testes unitários com `pytest`
- Cobertura de código
- Integração contínua

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Testes com cobertura
pytest --cov=src --cov-report=html

# Testes específicos
pytest tests/test_predep.py -v
```

## 📊 Resultados

Os resultados são salvos em múltiplos formatos:

- **NetCDF**: `output/predep_seasonal_results.nc`
- **CSV**: `predep_full_results.csv`
- **Relatórios**: `results/final_results/`

### Interpretação dos Resultados

- **α = 0**: Variáveis independentes
- **α = 1**: Dependência preditiva perfeita
- **0 < α < 1**: Grau de dependência preditiva

## 🔍 Monitoramento

### Logs
Os logs são salvos em `logs/` com diferentes níveis:
- INFO: Progresso geral
- DEBUG: Detalhes técnicos
- ERROR: Erros e exceções

### Performance
Use `scripts/monitored_run.py` para monitoramento em tempo real:

```bash
python scripts/monitored_run.py
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### Padrões de Código
- Use `black` para formatação
- Siga PEP 8
- Adicione testes para novas funcionalidades
- Documente funções públicas

## 📚 Documentação Adicional

- [Metodologia de Bootstrap](documentation/bootstrap_methodology.md) - Detalhes técnicos do algoritmo
- [Notebooks Exploratórios](notebooks/exploratory/)
- [Configurações Avançadas](config/)
- [Scripts Utilitários](scripts/)

## 🐛 Problemas Conhecidos

- Dados com baixa variabilidade podem retornar NaN
- Análise completa requer ~8GB RAM
- Processamento paralelo pode sobrecarregar sistemas com poucos cores

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

---

**Nota**: Este é um projeto de pesquisa científica. Para uso em produção, considere validação adicional dos resultados.