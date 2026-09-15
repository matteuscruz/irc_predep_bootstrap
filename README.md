# PREDEP / MoV × Precipitação — como executar a pipeline

A pipeline calcula, para cada bacia hidrográfica, o PREDEP (α) e o r² entre os 29
índices de modos de variabilidade (MoVs) e a precipitação de duas fontes (Xavier e
CHIRPS), escolhe o MoV predominante, compara com surrogates, treina o SOM e gera as
figuras. O ponto de entrada é o `main.py`, e o cálculo roda no **Modal**.

## 1. Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt modal pyshp
modal token new                  # autentica no Modal
modal profile activate matteuscruz
```

Use sempre o Python do `.venv`. O despacho para o Modal funciona mesmo se a pilha
científica local estiver quebrada, mas `--listar`, `--assinaturas` e
`--executor local` precisam de `xarray` funcionando.

## 2. Dados de entrada (uma vez só)

O Modal lê dois itens do volume `predep-common-window`:

| no volume | origem local |
|---|---|
| `/predep_janela_1982_2024.nc` | `data/processed/common_window/predep_janela_1982_2024.nc` |
| `/clusters/<bacia>/` | `data/clusters/<bacia>/` (shapefiles) |

Se o volume já tem os dois, pule para a seção 3. Para gerar e enviar do zero:

```bash
# mensal do CHIRPS a partir do zarr diário (data/raw/chirps_daily/chirps_v2_brazil.zarr)
python tools/build_monthly_from_raw.py --only chirps

# janela comum Xavier ∩ CHIRPS ∩ 29 MoVs
python -m src.data_processing.janela_comum --inicio 1982-01 --fim 2024-12

# envio ao Modal
modal volume create predep-common-window
modal volume put predep-common-window data/processed/common_window/predep_janela_1982_2024.nc /predep_janela_1982_2024.nc
modal volume put predep-common-window data/clusters /clusters
```

## 3. Rodar

```bash
# uma bacia
python main.py --bacia uruguai --movs all --anomalias

# várias bacias
python main.py --bacia parana,uruguai,paraguai --movs all --anomalias

# todas
python main.py --bacia all --movs all --anomalias
```

Antes de gastar crédito, confira o que será despachado:

```bash
python main.py --bacia uruguai --movs all --anomalias --dry-run
python main.py --listar          # pixels e custo estimado por bacia
```

Bacias: `amazonica`, `atlantico_leste`, `atlantico_nordeste`,
`atlantico_nordeste_oriental`, `atlantico_sudeste`, `atlantico_sul`, `paraguai`,
`parana`, `paranaiba`, `sao_francisco`, `tocantins`, `uruguai`.

Para cada bacia são 196 fatias (4 estações × 7 lags × 7 deslocamentos: o observado
e 6 surrogates), cada uma no seu container. Fatias já calculadas são puladas, então
uma corrida interrompida pode ser retomada com o mesmo comando.

## 4. Saídas

Ao terminar, o `main.py` baixa tudo para o repositório:

| arquivo | conteúdo |
|---|---|
| `results/bacias/<janela>/<bacia>.nc` | α, r², escolha por predominância, surrogates e SOM |
| `plots/bacias/<janela>/<bacia>_predep.png` | α Xavier × CHIRPS, diferença, r², consenso |
| `plots/bacias/<janela>/<bacia>_som.png` | regimes do SOM e MoVs de cada regime |

`<janela>` é o nome da janela, com o sufixo `_anom` quando se usa `--anomalias`
(ex.: `predep_janela_1982_2024_anom`). Corridas com e sem anomalias ficam em pastas
separadas.

## 5. Juntar as bacias

Depois de calcular as bacias, junte tudo num único arquivo para o Brasil:

```bash
python main.py --juntar --anomalias
```

Gera `results/bacias/<janela>/brasil.nc`. Use as mesmas opções (`--anomalias`,
`--janela`) da corrida que gerou as bacias. Bacias que ainda não foram calculadas
aparecem listadas como ausentes.

## 6. Outros comandos

```bash
python main.py --bacia uruguai --anomalias --so-figuras    # só redesenha as figuras
python main.py --bacia uruguai --anomalias --assinaturas   # MoVs de cada regime, em CSV
```

## Opções do `main.py`

| opção | padrão | descrição |
|---|---|---|
| `--bacia` | `all` | bacia, lista separada por vírgula, ou `all` |
| `--movs` | `all` | MoV, lista separada por vírgula, ou `all` |
| `--anomalias` | — | remove a climatologia mensal da chuva antes do PREDEP |
| `--janela` | `predep_janela_1982_2024.nc` | arquivo da janela no volume Modal |
| `--lag-min` / `--lag-max` | `0` / `6` | intervalo de lags |
| `--shifts` | `0,3,5,7,9,11,13` | deslocamentos dos surrogates, em anos (0 = observado) |
| `--n-regimes` | — | regimes do SOM |
| `--seed` | `20260908` | semente |
| `--juntar` | — | junta as bacias prontas num `brasil.nc` |
| `--so-figuras` | — | não recalcula, só redesenha as figuras |
| `--assinaturas` | — | imprime e salva os MoVs de cada regime |
| `--sem-baixar` | — | deixa os resultados só no volume Modal |
| `--dry-run` | — | mostra o que seria feito, sem calcular |
| `--listar` | — | pixels e custo estimado por bacia |
| `--executor` | `modal` | `local` só para depuração de uma bacia pequena |

## Problemas comuns

**`Volume 'predep-common-window' not found`** — o perfil Modal ativo não é o que tem
o volume. Confira com `modal profile current`; se estiver na conta certa e o volume
não existir, recrie-o conforme a seção 2.

**Baixar manualmente um resultado:**

```bash
modal volume get predep-common-window /results/predep_janela_1982_2024_anom/bacias/uruguai.nc results/bacias/predep_janela_1982_2024_anom/uruguai.nc
```
