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

## 📊 Uso Rápido

### Teste Inicial
Execute um teste rápido com o índice ATL3 e precipitação MAM:

```bash
python main.py
```

### Análise Completa
Para executar a análise completa de todos os MoVs e estações:

```python
from main import run_full_analysis
results = run_full_analysis()
```

### Uso Programático
```python
from src.predep_calculation.core import predep
import numpy as np

# Dados de exemplo
x = np.random.normal(0, 1, 100)
y = 0.5 * x + np.random.normal(0, 0.5, 100)

# Calcular PREDEP
alpha = predep(x, y, n_boot=10000)
print(f"PREDEP α = {alpha:.4f}")
```

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

- [Notebooks Exploratórios](notebooks/exploratory/)
- [Configurações Avançadas](config/)
- [Scripts Utilitários](scripts/)

## 🐛 Problemas Conhecidos

- Dados com baixa variabilidade podem retornar NaN
- Análise completa requer ~8GB RAM
- Processamento paralelo pode sobrecarregar sistemas com poucos cores

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 👥 Autores

- **Seu Nome** - *Desenvolvimento inicial* - [GitHub](https://github.com/username)

## 🙏 Agradecimentos

- Equipe do IRC (International Research Centre)
- Comunidade científica de análise climática
- Desenvolvedores das bibliotecas utilizadas

## 📞 Contato

- Email: seu.email@exemplo.com
- LinkedIn: [Seu Perfil](https://linkedin.com/in/seuperfil)
- Issues: [GitHub Issues](https://github.com/username/irc_predep_bootstrap/issues)

---

**Nota**: Este é um projeto de pesquisa científica. Para uso em produção, considere validação adicional dos resultados.