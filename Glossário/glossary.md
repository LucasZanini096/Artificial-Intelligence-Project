# Glossário de Termos Técnicos

Termos levantados a partir da revisão dos artigos de referência do projeto:

- **Bhandari et al. (2022)** — *Predicting Stock Market Index Using LSTM*
- **Zanotto & Hölbig (2026)** — *Previsão de preços de ações e ETF na B3 aplicando técnicas de machine learning*

---

## A

**ARIMA** (*Auto-Regressive Integrated Moving Average*)
: Modelo clássico de séries temporais proposto por Box e Jenkins (1970) que combina autorregressão (AR), diferenciação para estacionariedade (I) e médias móveis dos resíduos (MA). Suas principais limitações em dados financeiros são a exigência de estacionariedade e a incapacidade de capturar relações não-lineares.

**ATR** (*Average True Range*)
: Indicador técnico que mede a volatilidade média de um ativo ao longo de um período (tipicamente 14 dias), calculado como a média dos maiores valores entre: (i) máxima menos mínima do dia, (ii) módulo da máxima menos fechamento anterior, e (iii) módulo da mínima menos fechamento anterior.

**Adam** (*Adaptive Moment Estimation*)
: Otimizador de gradiente estocástico que combina os benefícios do RMSProp e do Momentum, ajustando adaptativamente a taxa de aprendizado para cada parâmetro. Amplamente utilizado no treinamento de redes neurais por sua eficiência e robustez.

---

## B

**B3** (*Brasil, Bolsa e Balcão*)
: Principal bolsa de valores do Brasil, formada pela fusão da BM&FBovespa com a Cetip em 2017. Opera o mercado de ações, derivativos, câmbio e renda fixa privada no país.

**Backpropagation Through Time (BPTT)**
: Algoritmo de treinamento para redes neurais recorrentes (RNNs). O erro é propagado de volta ao longo dos *timesteps* da sequência para calcular os gradientes em relação a todos os parâmetros da rede.

**Baseline**
: Modelo de referência simples utilizado como piso mínimo de comparação. Neste projeto, o *baseline* adotado é o modelo de **persistência**: o preço previsto para o dia $t+1$ é simplesmente o preço observado no dia $t$ ($\hat{y}_{t+1} = y_t$).

**Batch Size**
: Número de amostras processadas pelo modelo antes de cada atualização dos pesos durante o treinamento. Valores menores introduzem mais ruído no gradiente, enquanto valores maiores produzem estimativas mais estáveis.

---

## C

**Cell State** ($c_t$)
: Componente de memória de longo prazo da célula LSTM. Percorre a cadeia temporal com apenas operações lineares (somas e multiplicações por escalares), o que preserva o fluxo de gradientes e resolve o problema do gradiente vanescente.

**Close Ajustado** (*Adjusted Close*)
: Preço de fechamento corrigido para incorporar eventos corporativos como desdobramentos, grupamentos, bonificações e pagamento de dividendos. Garante consistência temporal da série histórica.

---

## D

**Data Leakage** (vazamento de dados)
: Situação em que informações do conjunto de teste (futuro) contaminam o treinamento do modelo, produzindo avaliações artificialmente otimistas. Evitado ao estimar parâmetros de normalização exclusivamente no conjunto de treino.

**Deep Learning**
: Subcampo do aprendizado de máquina baseado em redes neurais com múltiplas camadas (profundas), capaz de aprender representações hierárquicas dos dados diretamente a partir de exemplos brutos.

**Dropout**
: Técnica de regularização que desativa aleatoriamente uma fração das unidades da rede durante o treinamento, forçando o modelo a aprender representações mais robustas e reduzindo o overfitting.

---

## E

**Early Stopping**
: Callback de treinamento que interrompe o processo quando uma métrica monitorada (tipicamente a perda de validação) não apresenta melhora por um número determinado de épocas consecutivas (*patience*). Previne overfitting e treinamento desnecessário.

**EMA** (*Exponential Moving Average* / Média Móvel Exponencial)
: Indicador técnico de tendência que pondera as observações históricas de forma exponencialmente decrescente, atribuindo maior peso aos preços mais recentes. Calculada recursivamente por $\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}$, onde $\alpha = 2/(n+1)$. Diferentemente da SMA, responde mais rapidamente a mudanças recentes.

**ETF** (*Exchange-Traded Fund*)
: Fundo de investimento negociado em bolsa que replica a performance de um índice de referência. Exemplos: BOVA11 (replica o Ibovespa) e FIND11 (replica o IFNC — Índice Financeiro).

---

## F

**Feature Engineering**
: Processo de criação ou transformação de variáveis de entrada (*features*) para melhorar a capacidade preditiva do modelo. Neste projeto inclui o cálculo da EMA-60 a partir do preço de fechamento.

**Feedforward Neural Network (FNN)**
: Rede neural em que o fluxo de informação ocorre apenas no sentido direto (entrada → saída), sem conexões cíclicas. Cada entrada é processada independentemente, tornando-a inadequada para dados sequenciais.

**Forget Gate**
: Portão da célula LSTM que decide quais informações do estado da célula anterior ($c_{t-1}$) devem ser descartadas. Produz valores entre 0 (esquecer completamente) e 1 (manter completamente) via função sigmoide.

**Forward Fill**
: Técnica de preenchimento de dados ausentes que propaga o último valor conhecido para as posições faltantes. Utilizada neste projeto para tratar registros de volume zerado.

---

## G

**Gradiente Vanescente** (*Vanishing Gradient*)
: Problema que ocorre no treinamento de RNNs convencionais para sequências longas: os gradientes decaem exponencialmente à medida que são propagados de volta no tempo, impedindo o aprendizado de dependências de longo prazo. O LSTM foi projetado especificamente para resolver este problema.

---

## H

**Hidden State** ($h_t$)
: Componente de memória de curto prazo da célula LSTM, correspondente ao estado oculto. É passado para a próxima célula e para a camada de saída, representando a "memória de trabalho" do modelo no instante $t$.

---

## I

**Input Gate**
: Portão da célula LSTM que decide quais novas informações serão armazenadas no estado da célula. Combina uma camada sigmoide (que decide o que atualizar) com uma camada tanh (que gera os valores candidatos).

**Interpolação Linear**
: Técnica de preenchimento de dados ausentes que estima valores intermediários assumindo variação linear entre dois pontos conhecidos. Utilizada para preencher lacunas nos preços causadas por feriados ou falhas técnicas.

**ITUB4**
: Código de negociação das ações preferenciais do Itaú Unibanco S.A. na B3. Ativo selecionado como objeto de estudo por apresentar o melhor desempenho preditivo dentre os ativos analisados por Zanotto & Hölbig (2026) ($R^2 = 0{,}9917$, MAPE = 1,12%) e menor influência de fatores políticos externos.

---

## J

**Janela Temporal** (*Window Size*)
: Número de instantes de tempo consecutivos usados como contexto para cada previsão. Uma janela de 50 dias significa que o modelo observa os últimos 50 pregões para prever o fechamento do dia seguinte.

---

## L

**LSTM** (*Long Short-Term Memory*)
: Arquitetura de rede neural recorrente proposta por Hochreiter & Schmidhuber (1997). Resolve o problema do gradiente vanescente por meio de um mecanismo de portões (forget, input e output gates) e um estado de célula que mantém um caminho linear ao longo do tempo. É o padrão dominante em aplicações de previsão de séries temporais financeiras desde 2015.

---

## M

**MAE** (*Mean Absolute Error*)
: Métrica de avaliação calculada como a média dos erros absolutos: $\text{MAE} = \frac{1}{N}\sum|y_i - \hat{y}_i|$. Menos sensível a valores extremos (*outliers*) que o RMSE.

**MACD** (*Moving Average Convergence Divergence*)
: Indicador técnico de momento calculado como a diferença entre duas médias móveis exponenciais (geralmente EMA-12 e EMA-26). Utilizado para identificar mudanças de tendência, força e direção de um ativo.

**MAPE** (*Mean Absolute Percentage Error*)
: Métrica de avaliação que expressa o erro como percentual relativo ao valor real: $\text{MAPE} = \frac{1}{N}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100$. Permite comparar o desempenho do modelo independentemente da escala de preços do ativo.

**Min-Max Scaling**
: Técnica de normalização que transforma os dados para o intervalo $[0, 1]$: $z = (x - x_{\min})/(x_{\max} - x_{\min})$. Alternativa ao Z-score; utilizada por Bhandari et al. (2022).

**ModelCheckpoint**
: Callback que salva automaticamente os pesos do modelo sempre que uma métrica monitorada melhora. Garante que a melhor versão do modelo durante o treinamento seja preservada.

**MSE** (*Mean Squared Error*)
: Função de perda utilizada no treinamento do modelo, calculada como a média dos quadrados dos erros: $\text{MSE} = \frac{1}{N}\sum(y_i - \hat{y}_i)^2$. Penaliza erros maiores de forma desproporcional.

---

## O

**Output Gate**
: Portão da célula LSTM que decide quais informações do estado da célula atual serão passadas como saída ($h_t$). Controla o que a célula "revela" ao restante da rede no instante $t$.

**Overfitting**
: Fenômeno em que o modelo aprende os dados de treinamento com precisão excessiva, incluindo o ruído, perdendo a capacidade de generalizar para dados novos. Mitigado por técnicas como Dropout, Early Stopping e redução da complexidade da arquitetura.

---

## R

**$R^2$** (Coeficiente de Determinação)
: Métrica que indica a proporção da variância dos valores reais explicada pelo modelo: $R^2 = 1 - \sum(y_i - \hat{y}_i)^2 / \sum(y_i - \bar{y})^2$. Valores próximos de 1 indicam excelente ajuste.

**RNN** (*Recurrent Neural Network*)
: Classe de redes neurais com conexões cíclicas que permitem a persistência de informação ao longo do tempo, tornando-as adequadas para dados sequenciais. Sofrem do problema do gradiente vanescente em sequências longas, limitação superada pelo LSTM.

**RMSE** (*Root Mean Square Error*)
: Raiz quadrada do MSE: $\text{RMSE} = \sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$. Expressa o erro na mesma unidade dos dados (R\$) e penaliza erros grandes mais severamente que o MAE.

**RSI** (*Relative Strength Index*)
: Indicador técnico de momentum que mede a velocidade e a magnitude das variações recentes de preço em uma escala de 0 a 100. Valores acima de 70 indicam sobrecompra; abaixo de 30, sobrevenda.

---

## S

**S&P 500**
: Índice de mercado que acompanha o desempenho das 500 maiores empresas listadas nas bolsas americanas (NYSE e NASDAQ). Utilizado como ativo de estudo por Bhandari et al. (2022).

**SMA** (*Simple Moving Average* / Média Móvel Simples)
: Média aritmética dos preços de fechamento dos últimos $n$ dias, com peso igual para todas as observações. Mais lenta para refletir mudanças recentes do que a EMA.

**StandardScaler**
: Implementação do Z-score no scikit-learn. Transforma cada variável para média zero e desvio padrão 1 com base nos parâmetros estimados no conjunto de treinamento.

---

## T

**Teste t de Welch**
: Teste estatístico para comparar as médias de duas amostras independentes sem assumir igualdade de variâncias. Utilizado por Bhandari et al. (2022) para confirmar que a diferença de desempenho entre modelos single-layer e multilayer é estatisticamente significativa ($p = 7{,}08 \times 10^{-29}$).

**Timestep**
: Cada instante de tempo em uma sequência de entrada do modelo. Em uma janela de 50 dias, há 50 *timesteps*, cada um correspondendo a um pregão.

---

## V

**VIX** (*CBOE Volatility Index*)
: Índice que mede a expectativa de volatilidade do S&P 500 para os próximos 30 dias, calculado a partir dos preços das opções. Conhecido como o "índice do medo" do mercado financeiro americano.

---

## W

**Wavelet de Haar**
: Transformada matemática utilizada para decomposição e filtragem de sinais. Aplicada por Bhandari et al. (2022) para remover ruído (*denoising*) do preço de fechamento antes do treinamento do LSTM.

**Window Size**
: Ver *Janela Temporal*.

---

## Y

**yfinance**
: Biblioteca Python de código aberto que permite coleta de dados históricos de ativos financeiros diretamente da API do Yahoo Finance, sem necessidade de autenticação. Utilizada para obter os dados do ITUB4.SA.

---

## Z

**Z-score** (Padronização)
: Técnica de normalização que transforma cada variável para média zero e desvio padrão unitário: $z = (x - \mu) / \sigma$. Os parâmetros $\mu$ e $\sigma$ são estimados exclusivamente no conjunto de treinamento para evitar *data leakage*. Garante que variáveis com escalas muito distintas (preços em R\$ vs. volume em milhões) contribuam de forma equilibrada para o aprendizado da rede neural.
