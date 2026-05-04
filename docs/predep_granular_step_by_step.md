# Passo a passo do PREDEP e da regressão no Exp 06

Este documento resume o fluxo completo do experimento granular do Brasil por bacia em [experiments/06_predep_comparisson_brazil_granular.py](../experiments/06_predep_comparisson_brazil_granular.py), desde a preparação dos dados até o cálculo de `alpha_core` e `r²`.

## 1. Objetivo do experimento

O script calcula, ponto a ponto, duas métricas entre um índice oceânico/atmosférico de variabilidade climática, chamado aqui de MoV, e a precipitação mensal Xavier:

- `alpha_core`: métrica PREDEP, que busca quantificar quanto o conhecimento de `X` reduz a incerteza sobre `Y`.
- `r²`: baseline de regressão linear simples para comparação.

O processamento é feito por estação do ano e por lag de 1 a 12 meses, e pode ser executado para uma bacia isolada ou para todas as bacias do Brasil.

## 2. Entrada dos dados

O experimento usa três fontes principais:

- Precipitação mensal Xavier em NetCDF, carregada em [process_mov()](../experiments/06_predep_comparisson_brazil_granular.py#L187).
- Índices MoV em CSV, descobertos por [discover_mov_indices()](../experiments/06_predep_comparisson_brazil_granular.py#L418).
- Máscaras de bacia derivadas de shapefiles locais, criadas por [compute_basin_mask()](../experiments/06_predep_comparisson_brazil_granular.py#L507) e enviadas ao volume Modal por [upload_basin_mask()](../experiments/06_predep_comparisson_brazil_granular.py#L544).

## 3. Recorte espacial

O fluxo usa uma bbox padrão para o Brasil:

- latitude de `-34` a `6`
- longitude de `-75` a `-34`

Quando a execução é por bacia, o shapefile correspondente define um recorte ainda mais restrito. A máscara booleana da bacia é usada depois no worker para ignorar pixels fora da área de interesse.

## 4. Preparação da precipitação

Em [process_mov()](../experiments/06_predep_comparisson_brazil_granular.py#L187), o NetCDF mensal é aberto e recortado para a área desejada. A partir do tempo do dataset, o script extrai:

- anos
- meses
- latitude
- longitude
- valores de precipitação `pr_values`

Em seguida, ele calcula a climatologia mensal de referência com a média de cada mês do calendário ao longo de todos os anos em [experiments/06_predep_comparisson_brazil_granular.py#L251].

## 5. Separação por estação e cálculo de anomalia

O experimento trabalha com quatro estações:

- DJF
- MAM
- JJA
- SON

Para cada estação, o script seleciona os meses correspondentes em [experiments/06_predep_comparisson_brazil_granular.py#L289]. Depois calcula a anomalia como:

```text
anomalia = precipitação da estação - climatologia do mês calendário
```

Esse cálculo aparece em [experiments/06_predep_comparisson_brazil_granular.py#L297]. O resultado é uma série sazonal de anomalias por pixel.

## 6. Preparação do MoV e alinhamento temporal

O CSV do MoV é lido e agregado por ano e mês em [experiments/06_predep_comparisson_brazil_granular.py#L260]. O código constrói um dicionário `mov_lookup` com chaves do tipo `(ano, mês)`.

Depois, para cada lag, a função [ _precompute_mov_alignment() ](../experiments/06_predep_comparisson_brazil_granular.py#L76) faz o alinhamento temporal:

1. Para cada observação da estação, subtrai o lag do mês corrente.
2. Se o mês ficar menor ou igual a zero, ele volta para o ano anterior e soma 12 ao mês.
3. Busca o valor do MoV em `mov_lookup` usando `(ano_alinhado, mês_alinhado)`.
4. Se o valor não existir, insere `NaN`.

Isso produz um vetor do MoV já alinhado ao tempo da precipitação para aquele lag específico.

## 7. Cálculo pixel a pixel

O cálculo efetivo acontece no worker [ _compute_lag_worker() ](../experiments/06_predep_comparisson_brazil_granular.py#L94). Ele recebe:

- o lag atual
- o MoV alinhado
- o número de bootstraps
- o modo debug
- a máscara da bacia

O worker percorre cada pixel da grade e executa a mesma lógica:

1. Extrai a série temporal de precipitação daquele pixel.
2. Remove pares com valores ausentes no MoV ou na precipitação.
3. Exige pelo menos 10 observações válidas.
4. Exige pelo menos 3 valores únicos em cada variável.
5. Se os critérios forem satisfeitos, calcula PREDEP e `r²`.

Os resultados daquele lag são guardados em mapas 2D `ac_map` e `r2_map`.

## 8. Como o PREDEP é calculado

O PREDEP está implementado em [predep()](../src/predep_calculation/core.py#L6).

### 8.1 Ideia estatística

O objetivo é estimar:

```text
alpha = (S_{Y|X} - S_Y) / S_{Y|X}
```

onde:

- `S_Y` é uma medida de densidade marginal de `Y`.
- `S_{Y|X}` é a mesma ideia, mas agora condicionada ao valor de `X`.

Em termos práticos, valores mais próximos de 0 sugerem pouca ou nenhuma dependência, e valores mais altos indicam que `X` reduz mais a incerteza sobre `Y`.

### 8.2 Passos internos

1. Converte `X` e `Y` para arrays numéricos.
2. Faz bootstrap de `Y` por amostragem com reposição e calcula diferenças entre duas amostras sorteadas de `Y`.
3. Estima `S_Y` com KDE sobre essas diferenças e avalia a densidade em zero.
4. Particiona `X` com Bayesian Blocks, o que cria bins adaptativos em vez de bins fixos.
5. Para cada bin de `X`, seleciona os valores de `Y` correspondentes.
6. Faz bootstrap local de `Y` dentro de cada bin e estima a densidade com KDE em zero.
7. Pondera cada contribuição pela fração de probabilidade daquele bin em `X`.
8. Combina tudo para obter `S_{Y|X}`.
9. Retorna `alpha = (S_{Y|X} - S_Y) / S_{Y|X}`.

### 8.3 Interpretação prática

- `alpha` perto de `0`: o MoV pouco explica a variabilidade da precipitação.
- `alpha` maior: o MoV carrega mais informação sobre a precipitação naquele pixel, estação e lag.

## 9. Como o r² é calculado

O baseline linear está em [regression_calc()](../src/regression_calculation/core.py#L5).

### 9.1 Passos

1. Converte `X` e `Y` para arrays numéricos.
2. Verifica se os dois vetores têm mesmo tamanho.
3. Rejeita casos com menos de 3 pontos ou com menos de 3 valores únicos em qualquer variável.
4. Executa uma regressão linear simples com `linregress`.
5. Retorna o quadrado do coeficiente de correlação, `rvalue²`.

### 9.2 Papel no experimento

O `r²` não substitui o PREDEP. Ele serve como comparação linear para mostrar quando a relação entre MoV e precipitação pode ser explicada por uma dependência simples e quando a métrica não linear captura algo adicional.

## 10. Paralelização

Depois de preparar a anomalia da estação, o código coloca o array em uma variável global compartilhada entre processos e monta uma lista de argumentos por lag em [experiments/06_predep_comparisson_brazil_granular.py#L306]. Em seguida, usa `multiprocessing.Pool` com 8 workers para processar os 12 lags em paralelo.

Cada worker retorna um mapa completo de `alpha_core` e `r²` para um lag específico.

## 11. Consolidação da saída

Os mapas retornados por estação e lag são armazenados em arrays 4D:

- `ac_arr[season, lag, latitude, longitude]`
- `r2_arr[season, lag, latitude, longitude]`

Isso é montado dentro de [process_mov()](../experiments/06_predep_comparisson_brazil_granular.py#L187) e salvo como NetCDF por bacia em:

```text
{mov_name}_{basin_name}_predep_granular_seasonal.nc
```

Depois, [merge_basin_results()](../experiments/06_predep_comparisson_brazil_granular.py#L566) junta os arquivos das bacias com `xr.merge(..., join="outer")` e salva um NetCDF final para o Brasil todo.

## 12. Resumo do fluxo completo

1. O script identifica MoVs e bacias.
2. Sobe os dados para o volume do Modal.
3. Gera a máscara espacial da bacia.
4. Abre a precipitação mensal Xavier e calcula a climatologia mensal.
5. Separa os dados por estação e calcula anomalias.
6. Lê o MoV e alinha o índice por lag.
7. Processa pixel a pixel, filtrando pares válidos.
8. Calcula PREDEP e `r²` para cada pixel, estação e lag.
9. Salva os resultados em NetCDF por bacia.
10. Mescla as bacias em um arquivo final do Brasil todo.

## 13. Arquivos principais consultados

- [experiments/06_predep_comparisson_brazil_granular.py](../experiments/06_predep_comparisson_brazil_granular.py)
- [src/predep_calculation/core.py](../src/predep_calculation/core.py)
- [src/regression_calculation/core.py](../src/regression_calculation/core.py)
- [docs/predep_function.md](predep_function.md)