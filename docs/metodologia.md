# Metodologia

## 1. Aquisição e Ingestão de Dados

Os dados de precipitação utilizados neste estudo provêm do conjunto de dados diários de alta resolução espacial produzido por Xavier et al., distribuído em formato NetCDF. A série temporal abrange o período de 1961 a 2019 e é organizada em quatro arquivos particionados cronologicamente: 1961–1980, 1981–2000 e dois arquivos cobrindo 2001–2019. Cada arquivo contém a variável de precipitação diária (`pr`) georreferenciada em uma grade regular com resolução de aproximadamente 0,1° × 0,1°, totalizando uma grade de 393 pontos de latitude por 391 pontos de longitude sobre o território brasileiro.

O índice de variabilidade oceânica utilizado como preditor climático é o índice do Atlântico Norte Tropical (TNA — *Tropical North Atlantic Index*), fornecido em formato de tabela delimitada por vírgulas (CSV). A série é de natureza mensal e compreende registros de valor médio por mês e ano, representando anomalias de temperatura da superfície do mar na região tropical do Atlântico Norte.

Toda a infraestrutura de armazenamento e processamento remoto foi gerenciada por meio de volumes persistentes na plataforma Modal, para os quais os arquivos de entrada foram transferidos previamente à execução dos experimentos.

---

## 2. Procedimentos de Pré-processamento

### 2.1 Agregação Temporal da Precipitação

Os dados diários de precipitação foram convertidos para escala mensal mediante a operação de soma acumulada ao longo de todos os dias de cada mês, resultando em totais mensais expressos em milímetros por mês (mm/mês). Em seguida, os quatro arquivos diários foram concatenados ao longo da dimensão temporal e ordenados cronologicamente. Registros duplicados em datas coincidentes — resultantes da sobreposição entre arquivos — foram identificados e eliminados, preservando-se apenas a primeira ocorrência de cada instante temporal.

### 2.2 Agregação Sazonal e Cálculo de Anomalias

A partir da série mensal, procedeu-se à seleção dos meses correspondentes a cada uma das quatro estações climáticas convencionais: dezembro–fevereiro (DJF), março–maio (MAM), junho–agosto (JJA) e setembro–novembro (SON). Para cada estação, calculou-se a média aritmética dos três meses componentes em cada ano, produzindo uma série anual de médias sazonais por ponto de grade. As anomalias foram obtidas subtraindo-se a climatologia — definida como a média da série sazonal ao longo de todo o período — do valor de cada ano, de modo a remover o sinal de longo prazo e isolar a variabilidade interanual.

### 2.3 Alinhamento Temporal com Defasagem (Lag)

Para investigar relações com defasagem temporal entre o índice TNA e a precipitação, procedeu-se ao alinhamento das séries com valores de lag variando de 0 a 11 meses. Para cada par (ano, lag), identificou-se o mês de referência da estação sazonal — janeiro para DJF, abril para MAM, julho para JJA e outubro para SON — e subtraiu-se o valor do lag em meses. Quando o resultado excedia os limites do calendário (i.e., resultava em mês menor ou igual a zero), a operação foi corrigida com decremento do ano e adição de 12 ao mês, garantindo consistência na navegação retroativa no tempo. O valor mensal do índice TNA correspondente ao mês e ano resultantes foi então associado ao respectivo ano de precipitação. Anos para os quais não havia registro disponível do índice TNA foram marcados como ausentes (`NaN`).

### 2.4 Controle de Qualidade e Filtragem por Ponto de Grade

Para cada ponto de grade e cada combinação de estação e lag, aplicou-se um critério mínimo de qualidade: exigiram-se ao menos dez observações válidas após a remoção de valores ausentes em ambas as séries (precipitação e TNA). Adicionalmente, impôs-se a condição de que cada série apresentasse no mínimo três valores distintos, evitando casos degenerados em que a distribuição empírica fosse inadequada para estimação estatística. Pontos de grade que não atenderam a esses critérios tiveram seus resultados definidos como `NaN` no mapa final.

---

## 3. Pipeline de Dependência Preditiva (PREDEP)

### 3.1 Fundamentação Matemática

O índice PREDEP (*Predictive Dependence*) quantifica o quanto o conhecimento de uma variável $Y$ reduz a incerteza sobre uma variável $X$, com base em estimativas de densidade de probabilidade. Seja $s(X)$ uma medida de concentração da distribuição marginal de $X$ e $s(X \mid Y)$ a medida equivalente da distribuição condicional de $X$ dado $Y$, o coeficiente PREDEP $\alpha$ é definido como:

$$\alpha = \frac{s(X \mid Y) - s(X)}{s(X \mid Y)}$$

onde $\alpha \in [0, 1]$, com $\alpha = 0$ indicando independência entre as variáveis e $\alpha = 1$ indicando que $Y$ determina perfeitamente $X$.

### 3.2 Estimação de $s(X)$

A medida de concentração marginal $s(X)$ foi estimada a partir de amostras bootstrap. Foram sorteadas, com reposição, duas amostras independentes de tamanho $n_{\text{boot}} = 2000$ da série de precipitação $X$, denotadas $\{BX_1^{(k)}\}$ e $\{BX_2^{(k)}\}$. Calculou-se a distribuição das diferenças $DX = BX_1 - BX_2$ e estimou-se a densidade desta distribuição em zero por meio de um estimador de densidade de núcleo gaussiano (*Gaussian Kernel Density Estimation* — KDE), avaliado no ponto $DX = 0$. Este procedimento equivale a estimar a densidade da distribuição de $X$ em sua média, sem requerer suposições paramétricas sobre a forma da distribuição.

### 3.3 Particionamento Adaptativo de $Y$ via Blocos Bayesianos

A distribuição condicional de $X$ dado $Y$ foi estimada mediante a partição do espaço de $Y$ em intervalos adaptativos. Para tanto, aplicou-se o algoritmo de Blocos Bayesianos (*Bayesian Blocks*), que determina automaticamente o número e a posição ótima das fronteiras de partição em função dos dados observados, maximizando a evidência bayesiana do histograma resultante. Este método é não paramétrico e sensível à estrutura local da distribuição, sendo particularmente adequado para séries climáticas com comportamento irregular.

### 3.4 Estimação de $s(X \mid Y)$

Para cada intervalo $[b_{k-1}, b_k]$ definido pelo particionamento de $Y$, extraíram-se os valores de $X$ cujos pares $(X_i, Y_i)$ satisfaziam $b_{k-1} \leq Y_i < b_k$. Sobre esse subconjunto condicional $X \mid Y$, aplicou-se o mesmo procedimento bootstrap e KDE descrito para $s(X)$, obtendo-se a densidade condicional local $\hat{s}(X \mid Y \in [b_{k-1}, b_k])$. A estimativa agregada $s(X \mid Y)$ foi obtida pela soma ponderada dessas densidades condicionais, com pesos dados pela probabilidade empírica de $Y$ recair em cada intervalo, calculada a partir da função de distribuição acumulada empírica (ECDF):

$$s(X \mid Y) = \sum_{k=1}^{K} \left[ F_Y(b_k) - F_Y(b_{k-1}) \right] \cdot \hat{s}(X \mid Y \in [b_{k-1}, b_k])$$

onde $F_Y$ denota a ECDF de $Y$ e $K$ é o número de partições resultante do algoritmo de Blocos Bayesianos.

---

## 4. Modelagem de Regressão Linear

### 4.1 Especificação do Modelo

Como método alternativo e de referência para comparação com o PREDEP, aplicou-se um modelo de regressão linear simples (MQO — Mínimos Quadrados Ordinários) entre a série de precipitação sazonal $X$ e o índice TNA alinhado temporalmente $Y$. O modelo assume a forma:

$$X = \beta_0 + \beta_1 Y + \varepsilon$$

onde $\beta_0$ é o intercepto, $\beta_1$ é o coeficiente angular e $\varepsilon$ é o termo de erro com média zero.

### 4.2 Estimação e Métrica de Ajuste

Os parâmetros foram estimados pelo método dos mínimos quadrados ordinários, implementado por meio da função `scipy.stats.linregress`, que minimiza a soma dos quadrados dos resíduos. A métrica de desempenho reportada é o coeficiente de determinação $r^2$, calculado como o quadrado do coeficiente de correlação de Pearson $r$ entre $X$ e $Y$:

$$r^2 = \left( \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i - \bar{Y})^2}} \right)^2$$

O coeficiente $r^2$ pertence ao intervalo $[0, 1]$, onde zero indica ausência de relação linear e um indica ajuste perfeito. Por ser simétrico em relação à troca entre $X$ e $Y$, o $r^2$ é diretamente comparável ao $\alpha$ do PREDEP como medida de associação estatística entre precipitação e índice TNA, embora capture apenas componentes lineares da dependência.

### 4.3 Critérios de Validade

A regressão foi condicionada aos mesmos critérios de controle de qualidade descritos na seção de pré-processamento: mínimo de dez observações válidas emparelhadas e mínimo de três valores distintos em cada série. Estes critérios garantem que a estimativa de $r^2$ seja estatisticamente significativa e não resulte de ajuste trivial a conjuntos de dados degenerados.

---

## 5. Projeção em Escala Nacional

### 5.1 Estrutura Espacial de Cômputo

O cálculo das métricas (PREDEP $\alpha$ ou $r^2$) foi realizado de forma independente para cada um dos aproximadamente 153.663 pontos de grade da malha espacial brasileira (393 latitudes × 391 longitudes). Para cada ponto, a série temporal de precipitação sazonal foi emparelhada com a série correspondente do índice TNA alinhado ao lag analisado, e a métrica foi computada conforme descrita nas seções anteriores.

### 5.2 Estrutura de Paralelismo

O processamento foi organizado em dois níveis de paralelismo. No primeiro nível, cada uma das quatro estações climáticas foi processada em um contêiner de computação independente na plataforma Modal, com execução simultânea. Cada contêiner dispunha de 8 CPUs virtuais e 4 GB de memória. No segundo nível, dentro de cada contêiner, os 12 lags foram distribuídos entre 8 processos paralelos gerenciados por um *pool* de multiprocessamento (`multiprocessing.Pool`). O array de precipitação foi mantido em memória compartilhada entre os processos por meio do mecanismo de herança por *fork* do sistema operacional Linux (*copy-on-write*), evitando a replicação de dados de grande volume entre os workers.

### 5.3 Estrutura dos Resultados

Os resultados foram armazenados como mapas bidimensionais de formato (latitude × longitude), um por combinação de estação e lag. Para cada estação, os 12 mapas de lag foram consolidados em um único arquivo NetCDF, totalizando 4 arquivos por experimento. Um arquivo adicional combinado foi gerado pela concatenação dos quatro arquivos sazonais ao longo de uma nova dimensão `season`. O conjunto final de resultados compreende 48 mapas por experimento (12 lags × 4 estações), permitindo a identificação do lag ótimo — definido como aquele com maior valor médio da métrica sobre o território — para cada estação climática e para cada método comparado.
