# Metodologia da predominância — o que mudou e o que os mapas novos mostram

Documento de trabalho, 2026-09-07. Cobre a troca do critério de seleção de MoV, os três
experimentos que a testaram (20, 21, 22) e as figuras que substituem
`plots/exp09b_predep_max.png`.

Escopo: bacia do Uruguai, 1.570 pixels, janela comum 1996–2024 (348 meses), 29 MoVs × 7
lags = 203 candidatos por estação. PREDEP com o estimador do artigo (pares `i == j`
permitidos), partição `ward` com k=√N, `n_boot = ⌈n·log n⌉`.

---

## 1. O problema com a seleção anterior

A figura `exp09b_predep_max.png` mostrava, em cada pixel, o **α máximo sobre os 203
candidatos**, calculado separadamente para Xavier e para CHIRPS. Isso embute dois vieses
que se somam:

**O máximo de 203 sorteios é alto por construção.** Mesmo sem nenhuma relação entre chuva
e índice, o maior de 203 valores ruidosos fica bem acima da média deles. O mapa resultante
é sempre azul-escuro, e isso não é evidência de nada.

**As duas fontes escolhiam candidatos diferentes.** O argmax coincidia em apenas 17–23%
dos pixels. Logo a coluna "diferença (CHIRPS − Xavier)" não comparava valores: comparava
`α(candidato A, CHIRPS)` contra `α(candidato B, Xavier)`. Diferença de valor e diferença
de escolha vinham somadas e não dava para separá-las.

**E as linhas de "MoV vencedor" eram ilegíveis** — um mosaico com 17 a 22 índices
diferentes reivindicando ≥1% da bacia.

## 2. O critério novo: predominância com concordância

Proposto pelo usuário. Inverte a ordem das duas operações:

> Antes: tomar o α máximo em cada pixel e depois perguntar de qual MoV ele veio.
> Agora: escolher primeiro o MoV em que as duas fontes mais concordam, e só então ler o α.

Formalmente, para cada estação escolhe-se **um único** par (lag, MoV) para a bacia inteira:

```python
concord = np.nanmean(np.minimum(alpha_xavier, alpha_chirps), axis=-1)  # média espacial
lag, mov = argmax sobre (lag, MoV) de concord
```

O `min` das duas fontes é o ponto central: um candidato só pontua alto se **ambas** as
fontes o acharem forte. Um candidato que o Xavier ama e o CHIRPS ignora pontua pelo
CHIRPS. A média espacial vem depois, então a escolha usa os 1.570 pixels de uma vez em vez
de decidir com o n efetivo de um pixel só.

Feita a escolha, **as duas fontes são lidas no mesmo candidato**. A coluna de diferença
passa a ser só diferença de valor.

### O que isso corrigiu

| | seleção por pixel (antes) | predominância (agora) |
|---|---|---|
| α médio | 0,78–0,92 | 0,40–0,46 |
| excesso sobre o nulo | −0,015 | +0,017 |
| mapa legível | não | sim |
| diferença entre fontes | valor + escolha | só valor |

O α caiu pela metade e isso é a correção funcionando: o valor anterior era o máximo de 203
sorteios. O excesso sobre o nulo trocou de sinal.

## 3. Calibração nula

Todo número aqui é lido contra **surrogates por deslocamento anual** (`SHIFTS = [0, 3, 5,
7, 9, 11, 13]`; 0 = observado, os outros 6 = nulo). O deslocamento preserva a
autocorrelação da série do índice e destrói o pareamento com a chuva.

Regra que vale para tudo neste documento: **o surrogate passa exatamente pelo mesmo
procedimento, inclusive escolhendo o seu próprio melhor candidato.** Comparar um observado
que foi selecionado contra um nulo que não foi inventa sinal.

Com 6 surrogates, a resolução por pixel é 1/7 = 14,3%. É a principal limitação em aberto.

---

## 4. Exp 20 — combinar os k melhores candidatos ajuda?

`experiments/20_modal_topk.py` → `results/exp20_topk.nc` → `plots/exp20_topk.png`
(`tools/gen_topk_figure.py`).

Pergunta: em vez de um candidato, usar a média dos k mais predominantes. Deveria reduzir a
variância por pixel. Varredura k ∈ {1, 2, 3, 5, 8, 12, 20}, com o nulo escolhendo os seus
próprios k.

Fração de pixels em que o observado supera **todos** os 6 surrogates (acaso = 14,3%):

| k | 1 | 2 | 3 | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|
| média das 4 estações | 13,5% | 13,4% | 12,5% | 12,3% | 12,0% | 13,0% | 14,7% |
| razão sobre o acaso | 0,95× | 0,94× | 0,87× | 0,86× | 0,91× | 0,91× | 1,03× |

**Nenhum k passa do acaso. k=3 é ligeiramente pior que k=1.**

Duas leituras a fixar:

- **A queda do α com k (0,455 → 0,430 → 0,359) é mecânica.** Cada candidato extra é menos
  predominante que o anterior. O nulo cai na mesma proporção (0,411 → 0,399 → 0,351) e a
  faixa entre as duas curvas mantém a espessura. Nunca ler essa queda como perda de sinal.
- **DJF e SON apontam para lados opostos** (1,41× → 0,46× e 0,66× → 1,94× de k=1 a k=20).
  Sinal concentrado num índice cresce com k pequeno e decai ao diluir — a forma do DJF.
  SON subindo até k=20, quando a "escolha" já cobre 10% dos candidatos, está medindo quase
  o α médio da estação. Duas estações se movendo em direções opostas sob o mesmo
  procedimento é assinatura de ruído.

**Decisão: k=1.**

## 5. Exp 21 — o MoV pixel a pixel é sal e pimenta?

`experiments/21_modal_mov_espacial.py` → `results/exp21_mov_espacial.nc` →
`plots/exp21_mov_espacial.png` (`tools/gen_mov_espacial_figure.py`).

O critério novo escolhe um MoV para a bacia toda, então o mapa dele seria cor chapada.
Para ver os MoVs espacialmente a escolha volta a ser por pixel — mas pelo critério de
concordância, não pelo α máximo de uma fonte só.

Coerência = fração de pares de pixels vizinhos (4-vizinhança, 2.977 pares) que escolhem o
mesmo MoV:

| | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| rótulos embaralhados no espaço (200 permutações) | 0,069 | 0,113 | 0,076 | 0,095 |
| **observado** | **0,260** | **0,401** | **0,305** | **0,370** |
| melhor surrogate | 0,311 | 0,418 | 0,395 | 0,415 |
| z (obs vs embaralhado) | 42,7 | 57,7 | 44,9 | 52,5 |

**Não é sal e pimenta** — z de 43 a 58 contra o embaralhamento, com manchas visíveis (TSA
com 27% dos pixels em MAM, MEI com 18% em SON). **E mesmo assim não vale nada:** o
surrogate forma blocos iguais ou maiores nas quatro estações. A organização vem da
autocorrelação espacial da precipitação — pixels vizinhos têm chuva com r = 0,994, recebem
α quase idêntico para qualquer índice e caem no mesmo argmax.

**Xavier e CHIRPS escolhem o mesmo MoV em 8,7% a 11,6% dos pixels** (acaso 1/29 = 3,4%).
Em ~90% dos pixels as duas fontes discordam de quem manda ali.

Regra que sai daqui: a referência para julgar um mapa de rótulos é o **surrogate**, nunca o
embaralhamento espacial. O embaralhamento só testa contra ruído independente, que nunca foi
a hipótese alternativa relevante.

## 6. Exp 22 — os mapas do exp09b refeitos

`experiments/22_modal_predep_max_novo.py` → `results/exp22_predep_max_novo.nc` →
`plots/exp22_predep_max_novo.png` (`tools/gen_predep_max_novo_figure.py`).

Mesma leitura da figura original — α de cada fonte, a diferença, o consenso — na
metodologia nova. Cinco linhas (as quatro estações mais um ANO) × cinco colunas. As duas
linhas de "MoV vencedor" do original desapareceram: como a escolha é única por estação,
ela virou o subtítulo da linha.

A linha **ANO** usa um único (lag, MoV) para o ano inteiro, escolhido pela predominância
média nas quatro estações. Não é o máximo sobre estações do exp09b, que somava mais um
argmax ao viés.

| linha | MoV · lag | α Xavier | α CHIRPS | Δ médio | consenso ¹ | melhor surrogate |
|---|---|---|---|---|---|---|
| DJF | PNA · lag 3 | 0,434 | 0,477 | +0,043 | 36,1% | 62,4% |
| MAM | TSA · lag 3 | 0,413 | 0,442 | +0,030 | 62,2% | 67,3% |
| JJA | NIN04 · lag 4 | 0,398 | 0,400 | +0,002 | 52,7% | 66,0% |
| SON | MEI · lag 3 | 0,462 | 0,428 | −0,034 | 66,5% | **64,0%** |
| **ANO** | **MEI · lag 4** | 0,313 | 0,331 | +0,018 | **94,1%** | **85,7%** |

¹ fração de pixels com |α_CHIRPS − α_Xavier| < 0,1.

**O consenso só supera o surrogate em SON e no ANO.** Em DJF, MAM e JJA um índice sem
relação nenhuma com a chuva produz mais acordo entre as fontes do que o índice observado —
o que reforça que o acordo entre Xavier e CHIRPS mede sobretudo o quanto as duas medem a
mesma chuva, não o quanto o índice prevê.

O ANO é o resultado mais forte que temos: |Δ| médio de 0,042 e 94,1% de consenso contra
85,7% do surrogate. A agregação sobre as quatro estações é o que reduz a variância — a
mesma razão pela qual a agregação espacial já tinha ajudado.

MEI aparece nas duas linhas de melhor desempenho (SON e ANO), e TSA lidera MAM tanto no
lag 3 quanto no lag 1. Liderança em lags vizinhos é assinatura de sinal real e não de
sorteio; DJF e JJA não têm essa assinatura — seus rankings são praticamente planos.

### A quinta coluna: a distribuição espacial da predominância

A escolha da metodologia é única por estação, então um mapa dela seria cor chapada. A
quinta coluna mostra outra coisa: o **mesmo critério de predominância resolvido em cada
pixel** (`argmax` local de `min(α_xavier, α_chirps)`). Não é o que a metodologia usa para
decidir — serve para ver o quanto a escolha única representa a região.

| | DJF | MAM | JJA | SON | ANO |
|---|---|---|---|---|---|
| escolha da bacia | PNA | TSA | NIN04 | MEI | MEI |
| vence localmente em | 12,1% | 26,8% | 8,3% | 18,1% | 14,6% |
| idem, com o mesmo lag | 8,3% | 12,0% | 7,8% | 10,8% | 8,8% |

**O candidato escolhido para a bacia vence localmente em 8% a 27% dos pixels.** MAM é o
caso mais coeso — TSA vence em mais de um quarto da bacia, e a mancha é contígua no
centro-oeste. JJA é o mais frágil: NIN04 vence em 8,3%, pouco acima do que qualquer um dos
29 índices teria por sorteio.

Isso **não invalida a escolha única** — ela é feita de propósito com os 1.570 pixels
juntos, justamente porque a decisão local tem n efetivo ≈ 30 e é dominada por ruído (é o
que o exp21 mediu). Mas fixa a leitura correta do mapa: a coluna mostra onde o índice
escolhido é de fato o mais forte, não uma regionalização climática. Pelo exp21, o
surrogate produz manchas do mesmo tamanho, então os blocos coloridos medem a
autocorrelação espacial da chuva.

---

## 6b. Exp 23–25 — o SOM concorda entre Xavier e CHIRPS?

`experiments/23_modal_export_som_features.py` (Modal, exporta as features) →
`experiments/25_som_por_trimestre.py` (local, treina) → `results/exp25_som_trimestre.nc`
→ `plots/exp25_som_trimestre.png` (`tools/gen_som_trimestre_figure.py`).

Oito SOMs: 4 trimestres × {Xavier, CHIRPS}. Duas correções sobre o exp11.

**A feature não seleciona nada.** O exp11 usava `nanmax` sobre estação e lag — o mesmo
viés de seleção que a metodologia da predominância removeu, levado para dentro do SOM.
Aqui o vetor do pixel é o α dos 29 MoVs no lag escolhido para a bacia naquela estação
(DJF lag 3, MAM lag 3, JJA lag 4, SON lag 3).

**A máscara de concordância.** Só entram os pixels com |α_CHIRPS − α_Xavier| < 0,1 no
candidato escolhido — a quarta coluna do exp22. São 566 pixels em DJF (36,1%), 977 em MAM
(62,2%), 828 em JJA (52,7%) e 1.044 em SON (66,5%). Cada surrogate gera **a sua própria**
máscara; fixar a do observado avaliaria o nulo num recorte escolhido pelo observado.

### O piso do ARI não é zero

É a correção que muda a leitura. O exp11 comparava ARI(Xavier, CHIRPS) contra 0. Mas as
duas fontes medem a MESMA chuva: dois SOMs treinados nelas coincidem bem acima de zero sem
nenhum sinal climático. O piso certo é empírico — o mesmo ARI medido em cada surrogate.

| k | estação | TETO ¹ | MEDIDO | NULO mediano | NULO máx | passa? |
|---|---|---|---|---|---|---|
| 4 | DJF | 0,480 | **0,491** | 0,235 | 0,260 | **SIM** |
| 4 | MAM | 0,331 | 0,242 | 0,307 | 0,451 | não |
| 4 | JJA | 0,344 | **0,430** | 0,258 | 0,366 | **SIM** |
| 4 | SON | 0,672 | **0,469** | 0,388 | 0,462 | **SIM** |
| 5 | DJF | 0,343 | 0,294 | 0,243 | 0,325 | não |
| 5 | MAM | 0,327 | 0,179 | 0,276 | 0,319 | não |
| 5 | JJA | 0,438 | 0,353 | 0,252 | 0,358 | não |
| 5 | SON | 0,622 | 0,433 | 0,375 | 0,499 | não |
| 6 | DJF | 0,370 | 0,250 | 0,196 | 0,276 | não |
| 6 | MAM | 0,332 | 0,200 | 0,275 | 0,343 | não |
| 6 | JJA | 0,426 | 0,262 | 0,283 | 0,333 | não |
| 6 | SON | 0,573 | **0,457** | 0,342 | 0,418 | **SIM** |

¹ TETO = ARI entre dois SOMs sobre os MESMOS dados, mudando só a semente.

**Passa em 4 de 12 combinações, e quais delas depende de k.** Isso é o achado, não um
detalhe: um resultado real não trocaria de estação ao mudar o número de regimes. Em k=4
passam DJF, JJA e SON; em k=5 nenhuma; em k=6 só SON.

**A única margem convincente é DJF em k=4**: 0,491 contra 0,260 do melhor surrogate. SON em
k=4 passa por 0,007 (0,469 contra 0,462), o que não sobrevive a nenhum teste sério. E DJF
falha em k=5 e k=6.

**O TETO limita tudo.** Em SON ele chega a 0,672, mas em MAM fica em 0,33 — o próprio SOM
reatribui um terço dos pixels só trocando a semente. Nenhum ARI observado pode passar muito
disso, então a instabilidade do SOM já consome boa parte do sinal que se procura.

**Conclusão: o SOM não valida a concordância entre as fontes.** Ele concorda — os mapas das
duas linhas da figura têm estrutura parecida — mas concorda porque as duas fontes medem a
mesma chuva, que é exatamente o que o surrogate reproduz. É o mesmo mecanismo de
[[predep-map-smoothness-is-inherited]] em mais uma forma.

---

## 6c. Exp 26 — SOM em todos os pixels, e a assinatura de MoV de cada regime

`experiments/26_som_completo_assinaturas.py` → `results/exp26_som_completo.nc` →
`plots/exp26_som_completo.png` (`tools/gen_som_completo_figure.py`).

### Preenchendo a bacia inteira

A máscara passa a governar o **treino**, não a atribuição: o SOM aprende só onde as duas
fontes concordam e depois todo pixel é projetado no neurônio mais próximo. É a separação
padrão entre ajustar e aplicar um modelo. Recusar a atribuição fora da máscara jogava fora
34% a 64% da bacia sem necessidade.

Dois cuidados obrigatórios:

- **A padronização usa média e desvio DA MÁSCARA.** Com as estatísticas globais o espaço de
  features muda entre treino e projeção e o BMU deixa de significar o que significava.
- **O erro de quantização vira camada de confiança.** Distância ao neurônio vencedor,
  guardada por pixel; na figura os pixels fora da máscara vêm hachurados.

### O que o preenchimento revelou

| k=4 | ARI na máscara | ARI fora | erro BMU dentro → fora |
|---|---|---|---|
| DJF | 0,491 | 0,421 | 4,46 → 4,93 |
| MAM | 0,242 | **0,415** | 4,64 → 4,90 |
| JJA | 0,430 | **0,483** | 4,51 → 5,08 |
| SON | 0,469 | 0,308 | 4,38 → 4,73 |

**O erro sobe só 8% a 13% fora da máscara** — os pixels excluídos não são estranhos ao
espaço aprendido. E em MAM e JJA as fontes concordam **mais** fora da máscara do que dentro.

A razão é que a máscara e o SOM medem coisas diferentes. A máscara filtra divergência no
**valor** de α do candidato escolhido; o SOM usa o **perfil** dos 29 índices, padronizado.
Duas fontes podem discordar de quanto vale α e ainda concordar sobre qual índice se destaca
em relação aos outros. **Consequência prática: a máscara de concordância não é o filtro
certo para o SOM.**

Preencher não resgata o resultado. No ARI de todos os pixels em k=4, contra o nulo: DJF
passa (0,446 vs 0,287), JJA passa por pouco (0,465 vs 0,418), MAM e SON falham. Mesmo
quadro do exp25.

### As assinaturas — quais MoVs definem cada regime

Um regime é um **perfil sobre os 29 índices**, não um índice vencedor. Reportar só o argmax
por regime repetiria o erro que a metodologia da predominância corrigiu. A assinatura é o α
médio nativo de cada MoV entre os pixels do regime, mostrado como anomalia contra a média
da estação: um índice com α alto em todos os regimes não distingue nada e aparece neutro.

Os três MoVs que mais definem cada regime (Xavier, k=4):

| estação | regime | assinatura |
|---|---|---|
| DJF | R1 (660 px) | ITSMRG2 −0,085 · NIN034 +0,084 · NIN12 +0,079 |
| DJF | R2 (371 px) | PNA +0,056 · AMO −0,055 · PSA2 +0,050 |
| DJF | R3 (254 px) | NIN034 −0,095 · PNA −0,088 · ONI −0,078 |
| DJF | R4 (285 px) | **AMO +0,163 · TNA +0,113 · MGI +0,109** |
| MAM | R1 (324 px) | NIN03 +0,086 · ONI +0,047 · NIN12 +0,042 |
| MAM | R2 (483 px) | NIN03 −0,076 · AO −0,069 · ONI −0,059 |
| MAM | R3 (360 px) | AO +0,102 · EPNP +0,084 · SOI +0,067 |
| MAM | R4 (403 px) | TNA +0,071 · TSA −0,064 · NIN03 −0,062 |
| JJA | R1 (269 px) | NIN04 −0,101 · ATL3 +0,078 · EPNP −0,067 |
| JJA | R2 (671 px) | **NIN04 +0,126** · EPNP +0,083 · AO +0,080 |
| JJA | R3 (453 px) | ITSMRG2 −0,103 · AO −0,099 · MEI +0,075 |
| JJA | R4 (177 px) | ITSMRG2 +0,100 · ATL3 −0,054 · MEI −0,053 |
| SON | R1 (479 px) | **MEI +0,134** · WP −0,067 · ATL3 +0,066 |
| SON | R2 (327 px) | NIN04 −0,091 · SASDI +0,079 · MEI −0,069 |
| SON | R3 (269 px) | **MEI −0,130** · PDO +0,062 · AMO +0,057 |
| SON | R4 (495 px) | ONI +0,089 · WP +0,089 · SOI +0,073 |

**O padrão dominante é um eixo, não quatro regimes distintos.** Em SON, R1 é +MEI e R3 é
−MEI; em JJA, R2 é +NIN04 e R1 é −NIN04; em MAM, R1 é +NIN03 e R2 é −NIN03. O SOM separa
principalmente alto contra baixo ao longo de **um** índice por estação, e os outros dois
regimes preenchem o meio. Isso é consistente com a predominância ter escolhido um único
candidato por estação — e sugere que k=4 é generoso demais para a estrutura que existe.

**Coerência com a predominância:** SON escolhe MEI e o SOM separa por MEI; JJA escolhe NIN04
e o SOM separa por NIN04. MAM é a exceção — a predominância escolheu TSA e o SOM separa por
NIN03/AO, com TSA aparecendo só em R4 e com sinal negativo. MAM também é a estação de pior
ARI (0,242) e a que falha em todos os k.

**A amplitude é pequena.** As anomalias vão de −0,13 a +0,16 sobre um α médio de ~0,40 — os
regimes diferem em ~30% do valor típico, não em ordem de grandeza.

## 7. Como reproduzir

Os três experimentos reaproveitam as fatias já calculadas em `/slices18` no volume Modal
`predep-common-window` (4 estações × 7 lags × 7 deslocamentos, escritas pelo
`experiments/18_modal_mapa_excesso_nulo.py`). **Nenhum deles recomputa PREDEP** — cada um
roda em um container e leva menos de um minuto.

```bash
modal run experiments/20_modal_topk.py
modal run experiments/21_modal_mov_espacial.py
modal run experiments/22_modal_predep_max_novo.py
modal run experiments/23_modal_export_som_features.py

modal volume get predep-common-window /results/exp20_topk.nc          results/exp20_topk.nc
modal volume get predep-common-window /results/exp21_mov_espacial.nc  results/exp21_mov_espacial.nc
modal volume get predep-common-window /results/exp22_predep_max_novo.nc results/exp22_predep_max_novo.nc
modal volume get predep-common-window /results/exp23_som_features.nc     results/exp23_som_features.nc

.venv/bin/python experiments/25_som_por_trimestre.py         # ~15 min, 192 SOMs
.venv/bin/python experiments/26_som_completo_assinaturas.py  # ~15 min

.venv/bin/python tools/gen_topk_figure.py
.venv/bin/python tools/gen_mov_espacial_figure.py
.venv/bin/python tools/gen_som_trimestre_figure.py

# as duas figuras finais saem do módulo em src/ — os tools/ acima são invólucros
.venv/bin/python -m src.visualization.predominancia_plots todas --k 4
```

As figuras `exp22_predep_max_novo.png` e `exp26_som_completo.png` são geradas por
`src/visualization/predominancia_plots.py`, não por scripts em `tools/`. As funções
`plot_predep_max`, `plot_som_regimes` e `gerar_todas` aceitam um `Dataset` já aberto ou um
caminho, devolvem o `Path` do PNG e não guardam estado — é a forma que os nós do Kedro em
`kedro/src/predep_viz/pipelines/analysis/nodes.py` consomem. `gerar_todas` não deixa a falta
de um `.nc` derrubar a outra figura.

Os dois `tools/gen_*_figure.py` correspondentes viraram invólucros finos sobre o módulo,
para não haver duas implementações divergindo. Foi verificado que o módulo reproduz as duas
figuras com **zero pixels de diferença** em relação às versões originais.

Use o `.venv` do repo para as figuras: o Python do usuário tem numpy 2.x incompatível com o
xarray instalado (`np.unicode_` foi removido). Os contornos dos países saem de
`data/reference/south_america_110m.geojson` — `geopandas.datasets` não existe mais no
GeoPandas 1.0.

## 8. O que fica em aberto

**Subir de 6 para ~50 surrogates.** É o único teste que decide DJF (20,2% contra 14,3% de
acaso) e SON. Com 6, a resolução por pixel é 1/7 e os pixels são espacialmente
correlacionados, então o n efetivo espacial é de dezenas, não de 1.570. Custa 4 estações ×
7 lags × 50 deslocamentos no Modal, na ordem de uma hora.

**Escolher um MoV por sub-região.** Meio-termo entre o pixel (quinta coluna do exp22, e o
exp21) e a bacia inteira, com 4 a 6 sub-regiões definidas pela climatologia da chuva — nunca pelo α, o que
seria seleção circular. Cada decisão passaria a usar centenas de pixels e o mapa sairia em
blocos genuínos, testáveis contra surrogate.

**As outras 11 bacias.** Não foram rodadas de propósito: com o critério antigo o
`experiments/14` reproduziria o falso positivo 12 vezes. Com o critério novo e a
calibração nula, faz sentido — mas só depois dos 50 surrogates.

## 9. Arquivos

| arquivo | papel |
|---|---|
| `experiments/18_modal_mapa_excesso_nulo.py` | produz `/slices18`, a base dos três |
| `experiments/19_modal_mov_predominante.py` | o critério de predominância, primeira versão |
| `experiments/20_modal_topk.py` | varredura de k |
| `experiments/21_modal_mov_espacial.py` | escolha por pixel + coerência espacial |
| `experiments/22_modal_predep_max_novo.py` | α por fonte no candidato escolhido |
| `tools/gen_alpha_cru_figure.py` | `plots/exp19_alpha_cru.png` |
| `tools/gen_topk_figure.py` | `plots/exp20_topk.png` |
| `tools/gen_mov_espacial_figure.py` | `plots/exp21_mov_espacial.png` |
| `tools/gen_predep_max_novo_figure.py` | `plots/exp22_predep_max_novo.png` |
| `experiments/23_modal_export_som_features.py` | features do SOM, sem seleção por pixel |
| `experiments/25_som_por_trimestre.py` | 8 SOMs (trimestre × fonte) sob a máscara |
| `tools/gen_som_trimestre_figure.py` | `plots/exp25_som_trimestre.png` |
| `experiments/26_som_completo_assinaturas.py` | SOM em todos os pixels + assinaturas |
| `tools/gen_som_completo_figure.py` | invólucro sobre o módulo |
| `src/visualization/predominancia_plots.py` | **as duas figuras finais**, para a pipeline |
| `src/visualization/predominancia_base.py` | paleta, mapa-base e grade compartilhados |
| `src/predep_calculation/core.py` | `build_partition` / `predep_with_partition`, usados por todos |

Contexto anterior: `results/auditoria_metodologica.md` (os 4 desvios corrigidos),
`results/conformidade_final.md` (Table 1 do artigo reproduzida, desvio médio 0,007),
`results/exp09_bootstrap_defeito.md` (os pares `i == j`).
