# Análise do Artigo: *Previsão de Preços de Ações e ETF na Bolsa de Valores B3 Aplicando Técnicas de Machine Learning*
**Zanotto & Hölbig (2026) — Revista Sítio Novo, Vol. 10, e1879**

---

## 1. Introdução e Objetivo do Trabalho

O número de investidores na bolsa de valores brasileira cresceu 23% em 2023, segundo dados da própria B3, abrangendo tanto pessoas físicas quanto jurídicas. Com esse crescimento, cresce também a demanda por ferramentas de suporte à decisão, uma vez que muitos investidores ingressam no mercado sem o conhecimento técnico necessário para analisar oportunidades e riscos.

O comportamento dos preços dos ativos é influenciado por múltiplos fatores: indicadores fundamentalistas (Dividend Yield, P/VP, P/L, Dívida Líquida, Margem Líquida), variáveis macroeconômicas (taxa de juros, inflação, crescimento econômico), sentimento de mercado e eventos geopolíticos. Navegar por esse volume de informações exige ferramentas sofisticadas que a maioria dos investidores individuais não possui.

### Questão de pesquisa

> É possível prever o preço de ações e ETF acompanhando o desempenho de um índice ou de uma cesta de ativos na bolsa de valores utilizando dados históricos e técnicas de aprendizado de máquina?

### Objetivo principal

Desenvolver e validar um modelo **LSTM (Long Short-Term Memory)** capaz de prever o **preço de fechamento do próximo dia** de ações e ETF negociados na B3, utilizando dados históricos de preços e indicadores técnicos.

### Ativos analisados

A escolha dos ativos foi orientada pelo alto volume de transações na bolsa e pela diversidade de natureza jurídica entre as empresas, permitindo uma análise mais abrangente de variáveis que afetam o mercado.

| Ativo | Empresa / Fundo | Tipo |
|---|---|---|
| BBAS3 | Banco do Brasil S.A. | Ação — estatal |
| ITUB4 | Itaú Unibanco S.A. | Ação — privada |
| VALE3 | Vale S.A. | Ação — commodity |
| PETR4 | Petrobras S.A. | Ação — estatal/commodity |
| CXSE3 | Caixa Seguridade S.A. | Ação — estatal (IPO em 2021) |
| BOVA11 | ETF que replica o Ibovespa | ETF — índice amplo |
| FIND11 | ETF que replica o IFNC | ETF — setor financeiro |

### Contribuições do trabalho

1. Aplicação de LSTM ao contexto específico da B3, mercado com menor cobertura na literatura do que bolsas internacionais.
2. Avaliação do modelo sobre ativos de setores e naturezas jurídicas distintas, revelando como o comportamento de cada tipo de empresa afeta a qualidade das previsões.
3. Entrega de uma ferramenta versátil — o modelo aceita qualquer ticker como entrada, não ficando restrito aos ativos de validação.
4. Análise dos impactos do histórico curto de dados (caso CXSE3) sobre o desempenho preditivo.

---

## 2. Referencial Teórico

### 2.1 Inteligência Artificial e Machine Learning em finanças

O avanço da capacidade computacional e o crescimento do volume de dados disponíveis impulsionaram o uso de IA nas mais diversas áreas. No contexto financeiro, redes neurais artificiais se destacam pela capacidade de modelar padrões não-lineares nos dados, capturando relações complexas nos históricos do mercado de ações que modelos lineares clássicos não conseguem representar adequadamente.

A análise de séries temporais, campo fundamental da estatística, lida com a modelagem e previsão de dados que evoluem ao longo do tempo. A combinação de algoritmos de *machine learning* e *deep learning* com técnicas de séries temporais permite aos investidores tomadas de decisão mais embasadas e estratégicas.

### 2.2 Modelos e abordagens exploradas na literatura

| Autores | Modelo(s) | Contexto | Principais achados |
|---|---|---|---|
| Silveira (2021) | LSTM, Random Forest | B3 — PETR4 | Ambos eficazes na B3; escassez de estudos focados no mercado brasileiro |
| Alves e Prado (2022) | Regressão linear, RNN | NYSE | Regressão linear apresentou desempenho competitivo com RNN |
| Santos (2022) | Random Forest, LightGBM | B3 | LightGBM superior mesmo sem ajuste de hiperparâmetros |
| Nascimento, Santos e Ferreira (2022) | ARIMA, Prophet, LSTM | B3 — PETR4, ITUB4, BOVA11 | LSTM mais preciso para até 90 dias; ARIMA e Prophet melhores no curto prazo |
| Lin et al. (2021) | Random Forest, GBDT, LSTM | Bolsa chinesa | Precisão >60% em padrões de alta volatilidade com candlestick + *feature engineering* |
| Mintarya et al. (2023) | Revisão sistemática | Global | Redes neurais são os modelos mais usados; adoção de LSTM cresceu significativamente a partir de 2015 |
| Jiang, Ma e Zhu (2024) | Redes neurais, Random Forest | Bolsa chinesa | ML eficaz para prever risco de quedas; métricas de lucratividade são determinantes |
| Ren, Li e Zhang (2024) | AdaBoost com pesos de classe/tempo | S&P 500 | Aprendizado ativo com anotação automatizada melhorou classificação de riscos extremos |

### 2.3 Definição de ETF

ETF (*Exchange-Traded Fund*) são fundos negociados em bolsa que representam uma carteira de ativos e buscam replicar a performance de um índice de referência. O BOVA11 replica o Ibovespa (índice amplo do mercado brasileiro), enquanto o FIND11 replica o IFNC (Índice Financeiro, focado em bancos, fintechs e seguradoras).

---

## 3. Sobre o LSTM

### 3.1 Contexto e motivação

O LSTM (*Long Short-Term Memory*) é uma arquitetura de rede neural recorrente (RNN) proposta por Hochreiter e Schmidhuber (1997). Ele foi desenvolvido para superar o principal problema das RNNs convencionais: o **gradiente vanescente**, que impede o aprendizado de dependências de longo prazo em sequências temporais.

O LSTM destaca-se pela capacidade de **"lembrar" valores em intervalos arbitrários de tempo**, sendo especialmente eficaz em séries temporais com intervalos de duração desconhecida e alta variação — características presentes nos dados de preços de ativos financeiros.

### 3.2 Mecanismo de funcionamento

A célula LSTM mantém dois estados ao longo do tempo:

- **Estado da célula ($c_t$):** memória de longo prazo — fluxo linear que pode ser preservado ou modificado pelos portões.
- **Estado oculto ($h_t$):** memória de curto prazo — saída que será passada para a próxima célula e para a camada seguinte.

A cada instante $t$, a célula recebe três entradas: a sequência atual $x_t$, o estado oculto anterior $h_{t-1}$ e o estado da célula anterior $c_{t-1}$.

### 3.3 Os portões (gates)

| Portão | Função |
|---|---|
| **Forget gate** | Decide quais informações do estado anterior $c_{t-1}$ devem ser descartadas (saída entre 0 e 1 via sigmoid) |
| **Input gate** | Decide quais novas informações serão adicionadas ao estado da célula |
| **Change gate** | Gera candidatos a novos valores para o estado da célula via $\tanh$ |
| **Output gate** | Decide quais informações do estado atual serão passadas como saída $h_t$ |

### 3.4 Vantagem para séries temporais financeiras

Por aprender automaticamente quais padrões temporais são relevantes e quais devem ser esquecidos, o LSTM é especialmente adequado para dados de mercado financeiro, onde tendências de médio prazo coexistem com ruídos de curtíssimo prazo e choques externos. A revisão sistemática de Mintarya et al. (2023) confirma a tendência crescente de adoção do LSTM em aplicações financeiras desde 2015, superando modelos como SVM e KNN.

---

## 4. Metodologia

### 4.1 Ambiente e ferramentas

| Componente | Descrição |
|---|---|
| Linguagem | Python |
| Ambiente | Google Colab (GPU NVIDIA) |
| Framework de modelagem | TensorFlow / Keras |
| Fonte de dados | Yahoo Finance API (yfinance) |
| Manipulação de dados | Pandas |
| Normalização e métricas | scikit-learn |
| Visualização | Matplotlib |
| Monitoramento de experimentos | Neptune.ai |

### 4.2 Coleta de dados

Os dados históricos diários foram obtidos via API yfinance para os 7 ativos selecionados. O período cobre os **últimos 10 anos** contados a partir da data de execução do modelo. Caso um ativo não possua 10 anos de histórico (como a CXSE3, com IPO em 2021), o código ajusta automaticamente a data de início para o primeiro registro disponível.

Os dados coletados para cada ativo incluem: preço de abertura, máxima, mínima, fechamento, **fechamento ajustado** e volume de negociações.

### 4.3 Preparação dos dados

#### Ajuste de preços por eventos corporativos

A variável-alvo e os dados de entrada utilizam o **preço de fechamento ajustado**, que incorpora eventos societários — desdobramentos, grupamentos, bonificações, subscrições e pagamento de dividendos. Como o yfinance fornece apenas o fechamento ajustado diretamente, os preços de abertura, máxima e mínima foram ajustados manualmente através do seguinte fator:

$$\text{fator de ajuste} = \frac{\text{valor de fechamento} - \text{valor do evento por ação}}{\text{valor de fechamento}}$$

Todos os preços históricos foram recalculados com esse fator, garantindo consistência temporal dos dados.

#### Tratamento de dados faltantes

- **Preços ausentes** (finais de semana, feriados, falhas técnicas): interpolação linear, preservando a tendência dos dados sem introduzir vieses.
- **Volume zerado**: substituição pelo último valor conhecido (*forward fill*), evitando falsas avaliações do modelo.

#### Indicador técnico: EMA de 60 dias

A **Média Móvel Exponencial (EMA)** com período de 60 dias foi calculada e adicionada como feature adicional. Diferentemente da média móvel simples (SMA), a EMA atribui pesos maiores aos preços mais recentes, tornando-a mais sensível às mudanças recentes no mercado.

O período de 60 dias foi escolhido por identificar tendências de médio prazo, equilibrando sensibilidade a novas informações com estabilidade suficiente para evitar ruídos de curto prazo. A inclusão da EMA fornece ao modelo uma representação das tendências subjacentes nos preços, melhorando sua capacidade de prever movimentos futuros.

> **Nota:** O *EMA span* (60 dias) é distinto do *window size* (50 dias). O primeiro é o período de cálculo da EMA como feature; o segundo é a janela de observação usada pelo LSTM como contexto temporal.

### 4.4 Pré-processamento

#### Normalização (Z-score)

Optou-se pela padronização Z-score via `StandardScaler` (scikit-learn), que transforma cada variável para ter **média zero e desvio padrão 1**:

$$z = \frac{x - \mu}{\sigma}$$

Cada variável foi normalizada individualmente (abertura, máxima, mínima, fechamento, volume e EMA), preservando suas características específicas. Os parâmetros ($\mu$ e $\sigma$) foram calculados **exclusivamente no conjunto de treinamento**, evitando *data leakage* — os mesmos parâmetros foram aplicados ao conjunto de teste para garantir consistência.

Os escaladores foram armazenados para permitir a inversão da transformação (desnormalização) das previsões de volta à escala original em reais.

#### Divisão temporal dos dados

| Conjunto | Proporção | Conteúdo |
|---|---|---|
| Treinamento | 80% | Primeiros 80% dos dados em ordem cronológica |
| Teste | 20% | Últimos 20% dos dados — períodos mais recentes |

A ordem temporal foi rigorosamente preservada para refletir o cenário real de previsão no mercado financeiro.

#### Janela de observação (window size)

O *window size* de **50 dias** (aproximadamente 2 meses de pregões) foi definido com base na literatura (Gülmez, 2023; Li, 2024) e em testes empíricos durante o desenvolvimento. Esse valor captura movimentos e tendências de curto a médio prazo, mantendo equilíbrio entre desempenho preditivo e eficiência computacional.

### 4.5 Arquitetura do modelo

O modelo é uma RNN do tipo LSTM implementada com TensorFlow/Keras, com a seguinte estrutura:

| Camada | Configuração | Função |
|---|---|---|
| **Input** | Shape: (window_size, n_features) | Recebe as sequências históricas |
| **LSTM 1** | 500 unidades, `return_sequences=True` | Processa a sequência completa e passa para a próxima camada |
| **Dropout 1** | Taxa = 30% | Reduz overfitting desativando aleatoriamente unidades |
| **LSTM 2** | 500 unidades, `return_sequences=False` | Produz uma única saída — estado final da sequência |
| **Dropout 2** | Taxa = 30% | Regularização adicional |
| **Dense** | 1 neurônio, ativação linear | Gera a previsão do preço para o próximo dia |

O uso de `return_sequences=True` na primeira camada LSTM garante que a segunda camada receba a sequência completa de estados ocultos, não apenas o estado final — padrão necessário para empilhamento de LSTMs.

### 4.6 Treinamento

| Hiperparâmetro | Configuração |
|---|---|
| Função de perda | MSE (*Mean Squared Error*) |
| Otimizador | Adam (*Adaptive Moment Estimation*) |
| Divisão treino/teste | 80% / 20% |
| Window size | 50 dias |
| EMA span | 60 dias |
| Máximo de épocas | 50 (com early stopping) |
| Early stopping (patience) | 10 épocas consecutivas sem melhora na loss de validação |

**Callbacks implementados:**

- **Early Stopping:** interrompe o treinamento quando a perda de validação não melhora por 10 épocas consecutivas, prevenindo overfitting e treinamento desnecessário.
- **ModelCheckpoint:** salva automaticamente os pesos do modelo sempre que há melhora na perda de validação, garantindo que a melhor versão seja usada nas previsões.
- **NeptuneCallback:** integra o Neptune.ai para monitoramento em tempo real das métricas, registro de hiperparâmetros, visualização de gráficos e manutenção de histórico detalhado dos experimentos.

O método `fit` do TensorFlow foi executado **sem embaralhamento dos dados**, preservando a ordem temporal das sequências.

### 4.7 Métricas de avaliação

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | Erro na mesma unidade dos dados (R$); menor = melhor |
| **MAE** | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | Erro médio absoluto; menos sensível a outliers que o RMSE; menor = melhor |
| **MAPE** | $\frac{1}{N}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$ | Erro relativo percentual; permite comparar ativos com escalas de preço diferentes; menor = melhor |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proporção da variância explicada pelo modelo; valores próximos de 1 = melhor ajuste |

---

## 5. Resultados Obtidos

### 5.1 Desempenho geral do modelo

| Ativo | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| PETR4 | 1,3247 | 1,0350 | 3,4042 | 0,9692 |
| BBAS3 | 0,5293 | 0,4360 | 1,8372 | 0,9752 |
| VALE3 | 1,3175 | 1,0259 | 1,6758 | 0,8564 |
| ITUB4 | 0,4073 | 0,3246 | 1,1167 | 0,9917 |
| CXSE3 | 0,3077 | 0,2347 | 1,6045 | 0,7713 |
| BOVA11 | 1,4713 | 1,1625 | 0,9839 | 0,9738 |
| FIND11 | 1,9832 | 1,5468 | 1,3176 | 0,9705 |
| **Faixa geral** | **0,30 – 1,98** | **0,23 – 1,54** | **0,98 – 3,40** | **0,77 – 0,99** |

### 5.2 Análise por ativo

**ITUB4 — melhor desempenho geral (R² = 0,9917)**
O Itaú Unibanco, como maior banco privado da América Latina, apresenta comportamento de preços mais previsível por estar menos sujeito a interferências governamentais diretas do que empresas estatais. O modelo capturou 99,17% da variabilidade dos preços, com MAPE de apenas 1,12%. A previsão para 07/11/2024 foi de R$ 36,41, contra o preço real de R$ 35,64 — desvio de R$ 0,77.

**BBAS3 — alto desempenho, com ressalvas (R² = 0,9752)**
O Banco do Brasil apresentou erros médios muito baixos (RMSE = 0,53, MAE = 0,44). Contudo, sendo uma empresa estatal, seu preço está sujeito a influências políticas e decisões governamentais que o modelo não contempla. A previsão para 07/11/2024 foi de R$ 26,66, contra R$ 26,19 real — desvio de R$ 0,47.

**BOVA11 — MAPE mais baixo (0,98%), ETF de índice amplo (R² = 0,9738)**
Por replicar o Ibovespa, o BOVA11 está exposto a todas as variáveis macroeconômicas e eventos que influenciam o mercado brasileiro como um todo. O modelo obteve o menor MAPE entre todos os ativos, indicando alta precisão relativa. A previsão para 07/11/2024 foi de R$ 125,18, contra R$ 126,00 real — desvio de R$ 0,82.

**PETR4 — maior MAPE (3,40%), commodity volátil (R² = 0,9692)**
A Petrobras é uma empresa de commodities estatal, altamente influenciada por fatores externos como o preço internacional do petróleo e decisões de política de preços do governo. O modelo ainda capturou 96,92% da variabilidade, mas o MAPE mais elevado reflete a dificuldade de prever movimentos originados por especulação e sentimento de mercado. A previsão para 07/11/2024 foi de R$ 33,38, contra R$ 35,51 real — desvio de R$ 2,13.

**VALE3 — R² mais baixo entre commodities (R² = 0,8564)**
A Vale, empresa de mineração exposta às flutuações internacionais do preço do minério de ferro, apresentou o segundo pior R² do estudo. O modelo capturou apenas 85,64% da variabilidade dos preços, com desvios mais perceptíveis em períodos de alta oscilação. A previsão para 07/11/2024 foi de R$ 62,00, contra R$ 63,00 real — desvio de R$ 1,00.

**CXSE3 — pior R² (0,7713), impacto do histórico curto**
A Caixa Seguridade realizou seu IPO em 2021, resultando em apenas ~3,5 anos de dados históricos disponíveis — bem abaixo do padrão de 10 anos adotado para os demais ativos. Isso revela uma **limitação estrutural** do modelo: menos dados históricos comprometem diretamente a qualidade das previsões, independentemente da qualidade da arquitetura. Apesar de RMSE e MAE baixos em termos absolutos (preço em escala menor), o R² de 0,77 indica que o modelo explica apenas 77,13% da variabilidade — resultado claramente inferior ao dos demais ativos.

**FIND11 — maior RMSE absoluto (1,98), bom R² (0,9705)**
O ETF do setor financeiro apresentou o maior erro absoluto, possivelmente porque o preço do FIND11 opera em faixa mais elevada (~R$ 90–140) e está exposto a fatores que afetam todo o mercado financeiro brasileiro. Ainda assim, capturou 97,05% da variabilidade. A previsão para 07/11/2024 foi de R$ 127,25, contra R$ 126,23 real — desvio de R$ 1,02.

### 5.3 Comparação com trabalhos anteriores

| Trabalho | Ativo | Modelo | RMSE | MAPE (%) |
|---|---|---|---|---|
| Silveira (2021) | PETR4 | LSTM | 1,7194 | — |
| **Este trabalho** | **PETR4** | **LSTM** | **1,3247** | **3,40** |
| Nascimento et al. (2022) | PETR4, ITUB4, BOVA11 | LSTM | 0,54 – 4,07 | 1,69 – 2,75 |
| **Este trabalho** | **7 ativos** | **LSTM** | **0,30 – 1,98** | **0,98 – 3,40** |

Em comparação com Silveira (2021), o modelo deste estudo reduziu o RMSE do PETR4 em **aproximadamente 23%** (de 1,72 para 1,32). Frente a Nascimento et al. (2022), os resultados são competitivos ou superiores para os ativos em comum, com RMSE máximo menor (1,98 vs. 4,07) e faixa de MAPE comparável.

### 5.4 Principais conclusões

1. **O modelo LSTM é eficaz para a previsão de ativos na B3**, com resultados sólidos para a maioria dos ativos avaliados (R² ≥ 0,96 para 5 dos 7 ativos).
2. **A natureza do ativo influencia o desempenho:** bancos privados (ITUB4) são mais previsíveis do que empresas de commodities (VALE3, PETR4), que estão mais expostas a fatores externos não modelados.
3. **O volume de histórico é determinante:** a CXSE3, com apenas ~3,5 anos de dados, apresentou R² de 0,77 — evidência direta de que a escassez de dados históricos limita a qualidade das previsões, independentemente da qualidade do modelo.
4. **O modelo não captura sentimento de mercado, especulação ou fatores políticos**, o que explica os desvios pontuais observados especialmente em empresas estatais e de commodities.
5. **A ferramenta é versátil e adaptável**, aceitando qualquer ticker como entrada e podendo ser estendida a outros ativos da B3.

### 5.5 Trabalhos futuros indicados pelos autores

- Incorporação de **variáveis macroeconômicas e dados textuais** (notícias, sentimento do mercado, mudanças políticas) como features adicionais.
- Combinação do LSTM com outras arquiteturas, como **CNN-LSTM** ou **Transformers**, para capturar padrões mais complexos.
- Implementação de **otimização automatizada de hiperparâmetros** via *grid search*, *random search* ou otimização bayesiana.
- Exploração de **janelas de observação adaptativas** e ajuste dinâmico dos hiperparâmetros por ativo.

---

*Referência completa: Zanotto, E. L.; Hölbig, C. A. (2026). Previsão de preços de ações e ETF na bolsa de valores B3 aplicando técnicas de machine learning. Revista Sítio Novo, Palmas, v. 10, e1879. https://doi.org/10.47236/2594-7036.2026.v10.1879*