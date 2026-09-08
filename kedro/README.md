# PREDEP Pipeline Viz

Projeto Kedro **isolado** (não mexe no `src/` de produção do repo principal)
criado só para visualizar, com Kedro-Viz, os passos reais de cálculo do
PREDEP — construção dos dados (ingestão, anomalia, alinhamento de lag,
recorte por estação) e a equação em si — para validação visual contra o
paper de referência (`../references/PREDEP.pdf`).

Roda numa fatia pequena e representativa (1 MoV, 1 pixel real da bacia do
Paraná, 4 estações, 6 lags — configurável em `conf/base/parameters.yml`),
reaproveitando as funções REAIS de `src/predep_calculation/core.py` e
`src/regression_calculation/core.py` do repo principal (via `sys.path`, não
duplicadas aqui).

## Uso

```bash
cd kedro_predep_pipeline
../.venv/bin/python -m kedro run          # roda a pipeline
../.venv/bin/python -m kedro viz run      # abre o Kedro-Viz local
../.venv/bin/python -m kedro viz build    # gera site estático em build/
```

## Ressalvas de metodologia (ver docstring de `compute_predep` em `nodes.py`)

Duas diferenças encontradas entre nosso `predep()` de produção e o
Algoritmo 1 do paper (documentadas, **não corrigidas** — decisão de
2026-08-31):

1. Binning de X: usamos `bayesian_blocks` (astropy); o paper recomenda
   Hierarchical Clustering com k=√n.
2. `n_boot`: usamos um valor fixo; o paper especifica `⌈n·log(n)⌉`
   (dependente do tamanho da amostra).
