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

Ao terminar, o `main.py` baixa tudo para `artifacts/<config>/` — sempre, mesmo se
a corrida falhar no meio (vem o que ficou pronto). Um download que falha não apaga a
cópia local anterior.

| arquivo | conteúdo |
|---|---|
| `raw/basins/<bacia>.nc` | α, r², escolha por predominância, surrogates e SOM |
| `plots/basins/predep/<bacia>.png` | α Xavier × CHIRPS, diferença, r², consenso, MoV por pixel |
| `plots/basins/regression/<bacia>.png` | o mesmo para o r², no par escolhido pelo próprio r² |
| `som/<bacia>.nc` | mapeamento espacial do SOM da bacia (regimes, máscara, assinaturas) |
| `som/plots/basins/<bacia>.png` | regimes do SOM da bacia e MoVs de cada regime |

`<config>` é o nome da janela com `_anom` quando se usa `--anomalias` e `_nb<valor>`
quando se usa `--n-boot` (ex.: `predep_janela_1982_2024_anom_nb10000`). Corridas
com configurações diferentes ficam em pastas separadas e nunca se sobrescrevem.

## 5. Juntar as bacias

Depois de calcular as bacias, junte tudo num único arquivo para o Brasil:

```bash
python main.py --juntar --anomalias
```

Gera `artifacts/<config>/raw/basins/brasil.nc`. Use as mesmas opções (`--anomalias`,
`--janela`, `--n-boot`) da corrida que gerou as bacias. Bacias que ainda não foram calculadas
aparecem listadas como ausentes.

## 6. Só o SOM (`--som`)

Calcula apenas o SOM, lendo os `<bacia>.nc` que já existem — sem fatias, sem PREDEP
e sem surrogates. As bacias precisam ter sido calculadas antes (seção 3).

```bash
# SOM único de algumas bacias
python main.py --som --regiao sul_sudeste --bacia atlantico_sul,atlantico_sudeste,parana,uruguai --anomalias

# SOM do Brasil inteiro (exige as 12 bacias calculadas)
python main.py --som --bacia all --anomalias
```

O nome da saída é o de `--regiao`; sem ele, `brasil` quando são as 12 bacias, ou os
nomes das bacias unidos por `_`. Saídas: `artifacts/<config>/som/<nome>.nc` e
`artifacts/<config>/som/plots/<nome>.png`.

## 7. Região completa (`--regiao`)

Trata um conjunto de bacias como uma unidade só e refaz, a partir das fatias, a
escolha do MoV (com os surrogates) e o SOM. Não recalcula PREDEP.

```bash
python main.py --regiao sul_sudeste --bacia atlantico_sul,atlantico_sudeste,parana,uruguai --anomalias
```

Saídas: `artifacts/<config>/raw/regions/<regiao>.nc`,
`plots/regions/{predep,regression}/<regiao>.png` e `som/plots/regions/<regiao>.png`.

Diferença para o `--juntar`: o `--juntar` só empilha os resultados de cada bacia
(cada uma com a sua escolha e o seu SOM); `--som` e `--regiao` fazem um SOM único.

## 8. Outros comandos

```bash
python main.py --bacia uruguai --anomalias --so-figuras    # só redesenha as figuras
python main.py --bacia uruguai --anomalias --assinaturas   # MoVs de cada regime, em CSV
```

## 9. Produtos finais do paper

Configuração usada no artigo: `predep_janela_1982_2024_anom_nb10000` (anomalias,
`--n-boot 10000`), lags 0–6, 8, 10 e 12. Os passos abaixo partem das 12 bacias já
calculadas e juntadas (seções 3 e 5).

```bash
# 1. lags 8, 10 e 12 — só o observado, um lag por vez, no Modal
for L in 8 10 12; do
  python main.py --bacia all --anomalias --n-boot 10000 \
      --lag-min $L --lag-max $L --shifts 0 --pular-som --pular-figuras --sim
done
# baixe as fatias s*_l{8,10,12}_d0.npz do volume para data/fatias_modal/<config>/<bacia>/

# 2. cubo nacional com todos os lags → raw/basins/brasil_lags.nc
python tools/juntar_lags.py

# 3. PREDEP e r² máximos por estação e lag (+ .nc para refazer só a figura)
python tools/fig_predep_max_lags.py --metrica alpha
python tools/fig_predep_max_lags.py --metrica r2
python tools/fig_predep_max_lags.py --metrica r2 --do-nc     # só redesenha

# 4. SOM de consenso (20 sementes, 4 estações + ano), treino no Modal
python tools/som_consenso_local.py --n-boot 10000 --entrada brasil_lags.nc \
    --com-ano --metrica alpha --modal
python tools/som_consenso_local.py --n-boot 10000 --entrada brasil_lags.nc \
    --com-ano --metrica r2 --modal

# 5. sensibilidade ao número de regimes (K = 2..8), no Modal
python tools/som_sensibilidade_k.py --metrica alpha
python tools/som_sensibilidade_k.py --metrica r2
```

Atenção: rodar o passo 1 com um lag só monta e baixa `<bacia>.nc` contendo apenas
aquele lag, sobrescrevendo os de 0–6. Faça backup de `raw/basins/` antes; o
`juntar_lags.py` lê os lags 0–6 do `brasil.nc` e os novos direto das fatias.

| saída | conteúdo |
|---|---|
| `raw/basins/brasil_lags.nc` | α e r² (fonte, estação, lag, MoV, pixel), lags 0–6, 8, 10, 12 |
| `plots/brasil/predep_max_lags/{predep,r2}_max_<estação>.png` | máximo entre os 29 MoVs por lag e o MoV que o produz; opacidade = concordância entre as fontes |
| `plots/brasil/predep_max_lags/{predep,r2}_max_lags.nc` | `valor_max`, `mov_max`, `dif_fontes` |
| `som/brasil_consenso_todos_lags[_r2].nc` | regimes, confiança, partições das 20 sementes, assinaturas por fonte, ρ |
| `som/brasil_consenso_todos_lags[_r2]/` | o mesmo, um `.nc` por estação e para o ano |
| `som/plots/brasil_consenso_todos_lags[_r2][_intersecao].png` | mapa de regimes; `_intersecao` lista só MoVs destacados nas duas fontes |
| `paper/dados/som/sensibilidade_k_{alpha,r2}.nc`, `paper/figuras/som/sensibilidade_k.png` | métricas por K |

Opções do SOM de consenso: `--modal` treina um container por estação (5 no total);
`--entrada-som minimo` usa min(α_Xav, α_CHI) por MoV em vez da média (teste de
sensibilidade, só com `--modal`); sem `--modal` roda local (horas).

A pasta `paper/` reúne cópias das figuras (`figuras/`) e dos `.nc` (`dados/`) usados
no artigo; os artigos estão em `predep-climate-variability/` (`main.tex` e
`oficial_paper/main.tex`), com as figuras em `figs/`.

## Opções do `main.py`

| opção | padrão | descrição |
|---|---|---|
| `--bacia` | `all` | bacia, lista separada por vírgula, ou `all` |
| `--movs` | `all` | MoV, lista separada por vírgula, ou `all` |
| `--anomalias` | — | remove a climatologia mensal da chuva antes do PREDEP |
| `--janela` | — | arquivo da janela no volume Modal |
| `--lag-min` / `--lag-max` | `0` / `6` | intervalo contínuo de lags (para 8, 10, 12 rode um por vez; seção 9) |
| `--shifts` | `0,3,5,7,9,11,13` | deslocamentos dos surrogates, em anos (0 = observado) |
| `--n-regimes` | — | regimes do SOM |
| `--seed` | `20260908` | semente |
| `--n-boot` | `0` | reamostragens por estimativa de α; `0` = ⌈n ln n⌉ (627). Outro valor vai para uma pasta própria |
| `--juntar` | — | junta as bacias prontas num `brasil.nc` |
| `--som` | — | só o SOM das bacias de `--bacia`, a partir dos `<bacia>.nc` prontos |
| `--regiao` | — | nome da região: escolha, surrogates e SOM únicos a partir das fatias |
| `--so-figuras` | — | não recalcula, só redesenha as figuras |
| `--assinaturas` | — | imprime e salva os MoVs de cada regime |
| `--sem-baixar` | — | deixa os resultados só no volume Modal |
| `--dry-run` | — | mostra o que seria feito, sem calcular |
| `--listar` | — | pixels e custo estimado por bacia, em núcleo-hora |
| `--executor` | `modal` | `local` só para depuração de uma bacia pequena |

## Problemas comuns

**`Volume 'predep-common-window' not found`** — o perfil Modal ativo não é o que tem
o volume. Confira com `modal profile current`; se estiver na conta certa e o volume
não existir, recrie-o conforme a seção 2.

**Baixar manualmente um resultado:**

```bash
modal volume get predep-common-window /results/predep_janela_1982_2024_anom/bacias/uruguai.nc artifacts/predep_janela_1982_2024_anom/raw/basins/uruguai.nc
```
