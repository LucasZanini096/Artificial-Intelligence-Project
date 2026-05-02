# Análise do Artigo: *Predicting Stock Market Index Using LSTM*
**Bhandari et al. (2022) — Machine Learning with Applications, Vol. 9**

---

## 1. Introdução e Objetivo do Trabalho

O mercado de ações é inerentemente ruidoso, não-paramétrico, não-linear e caracterizado por comportamento caótico determinístico. A volatilidade dos preços é influenciada por uma vasta gama de fatores interconectados: dados econômicos globais, taxas de desemprego, políticas monetárias, políticas de imigração, desastres naturais e condições de saúde pública, entre outros.

Diante dessa complexidade, o problema central abordado pelo artigo é: **como construir um modelo preditivo confiável e preciso que capture o comportamento não-linear e volátil do mercado de ações de forma holística?**

### Objetivo principal

Desenvolver um modelo baseado em **LSTM (Long Short-Term Memory)** para prever o **preço de fechamento do próximo dia** do índice **S&P 500**, utilizando uma combinação balanceada de variáveis preditoras que abranjam múltiplas dimensões da economia.

### Lacunas que o trabalho busca preencher

- Trabalhos anteriores frequentemente usam apenas indicadores técnicos ou apenas dados históricos (abordagem limitada).
- Incluir todas as variáveis disponíveis pode levar à colinearidade e dificultar a interpretabilidade.
- Há necessidade de um modelo com **boa combinação de features** e **arquitetura simples**, capaz de capturar o comportamento do mercado de múltiplas perspectivas.

### Contribuições declaradas

1. Seleção criteriosa de variáveis com justificativa econômica formal, cobrindo dados fundamentais, macroeconômicos e indicadores técnicos.
2. Comparação sistemática entre arquiteturas LSTM de camada única e múltiplas camadas.
3. Demonstração de que maior complexidade arquitetural **não** implica melhor desempenho preditivo.

---

## 2. Referencial Teórico

### 2.1 Modelos clássicos de séries temporais

A análise quantitativa em finanças tem longa história. O primeiro modelo de destaque foi o **ARIMA** (*Auto-Regressive Integrated Moving Average*), desenvolvido na década de 1970 por Box e Jenkins. Suas principais limitações são:

- Adequado apenas para dados **estacionários** — é necessário transformar a série antes do uso, o que pode comprometer a estrutura e interpretabilidade original dos dados.
- Assume **relacionamento linear** entre as variáveis, o que frequentemente não reflete a realidade dos mercados financeiros.

### 2.2 Redes Neurais e Deep Learning

A partir dos anos 1980, com o crescimento do volume de dados e do poder computacional, pesquisadores passaram a explorar **redes neurais artificiais** e técnicas de **deep learning** para modelagem de dados sequenciais, por serem capazes de capturar relações complexas e não-lineares.

#### Feedforward Neural Networks (FNN)
- O fluxo de informação ocorre apenas no sentido direto (forward).
- Cada entrada é processada de forma independente — **não há memória de passos anteriores**.
- Inadequadas para dados sequenciais onde eventos passados são relevantes para a predição futura.

#### Recurrent Neural Networks (RNN)
- Arquitetura projetada para **dados sequenciais**: possui loops internos que permitem que informações persistam ao longo do tempo.
- A informação é passada de um *timestep* para o próximo.
- Treinamento via **backpropagation through time (BPTT)**: o erro é propagado de volta pela rede, calculando derivadas parciais em relação a todos os parâmetros.
- **Problema do gradiente vanescente (*vanishing gradient*)**: para sequências longas, os gradientes podem decair a valores próximos de zero nas camadas anteriores, impedindo o aprendizado de dependências de longo prazo.

#### LSTM (Long Short-Term Memory)
- Arquitetura recorrente específica desenvolvida para **superar o problema do gradiente vanescente**.
- Memorizar informações por longos períodos é o comportamento padrão do modelo.
- Amplamente aplicado em: previsão de mercado financeiro, tradução de linguagem natural, previsão de tráfego móvel, detecção de anomalias, análise de dados fMRI, modelagem chuva-vazão, entre outros.

### 2.3 Trabalhos relacionados (seleção)

| Autores | Contribuição |
|---|---|
| Chen et al. (2015) | LSTM para retornos do mercado chinês; melhoria de 14,3% para 27,2% vs. predição aleatória |
| Bao et al. (2017) | Wavelet Haar para denoising + autoencoders + LSTM; R médio < 88% no S&P 500 |
| Fischer & Krauss (2018) | LSTM para classificação de direção de preços; superou Random Forest, redes profundas e regressão logística |
| Yadav et al. (2020) | LSTM no mercado indiano; concluiu que LSTM de camada única supera múltiplas camadas |
| Gao et al. (2020) | Comparação de MLP, LSTM, CNN e Uncertainty-Aware Attention; sugeriu que VIX e desemprego podem melhorar a predição |
| Karmiani et al. (2019) | Comparação LSTM vs. Backpropagation, SVM, Kalman Filter; LSTM obteve menor variância e maior acurácia |

---

## 3. Sobre o LSTM

### 3.1 Arquitetura geral

A célula LSTM é composta por:

- **Camada de entrada**
- **Camada oculta** (estado escondido $h_t$)
- **Estado da célula** ($c_t$): componente principal que percorre a cadeia com apenas interação linear, mantendo o fluxo de informação estável ao longo do tempo
- **Camada de saída**

O mecanismo de **gates (portões)** controla seletivamente quais informações são mantidas, atualizadas ou descartadas no estado da célula.

### 3.2 Equações dos portões

Para uma sequência de entrada $\{x_1, x_2, \ldots, x_n\}$, sendo $x_t \in \mathbb{R}^{k \times 1}$ a entrada no instante $t$:

| Portão | Equação | Função |
|---|---|---|
| **Input gate** | $i_t = \sigma(W_i x_t + W_{hi} h_{t-1} + b_i)$ | Decide quais informações novas serão armazenadas |
| **Forget gate** | $f_t = \sigma(W_f x_t + W_{hf} h_{t-1} + b_f)$ | Decide quais informações do estado anterior serão descartadas |
| **Output gate** | $o_t = \sigma(W_o x_t + W_{ho} h_{t-1} + b_o)$ | Decide quais informações serão passadas como saída |
| **Change gate** | $\tilde{c}_t = \tanh(W_c x_t + W_{hc} h_{t-1} + b_c)$ | Candidatos a novos valores no estado da célula |

**Atualização do estado da célula e do estado oculto:**

$$c_t = f_t \otimes c_{t-1} + i_t \otimes \tilde{c}_t$$

$$h_t = o_t \otimes \tanh(c_t)$$

Onde $\sigma$ é a função sigmoid, $\tanh$ é a tangente hiperbólica e $\otimes$ é o produto elemento a elemento.

### 3.3 Entradas da célula LSTM

A cada instante $t$, a célula recebe **3 informações**:

1. **$x_t$** — sequência de entrada atual
2. **$h_{t-1}$** — memória de curto prazo da célula anterior (estado oculto)
3. **$c_{t-1}$** — memória de longo prazo da célula anterior (estado da célula)

### 3.4 Por que o LSTM resolve o gradiente vanescente?

O estado da célula ($c_t$) mantém um **caminho linear** ao longo do tempo, com apenas multiplicações e somas simples, sem ativações não-lineares que causariam o decaimento dos gradientes. Os portões aprendem, de forma adaptativa, quando preservar ou descartar informações.

---

## 4. Metodologia

### 4.1 Dados utilizados

- **Índice:** S&P 500
- **Período:** 2006–2020 (15 anos de dados completos)
- **Justificativa do período:** abrange dois grandes mercados de baixa — a crise financeira de 2008 e a pandemia de COVID-19 em 2020, tornando o modelo mais robusto a cenários extremos.

### 4.2 Variáveis preditoras (features)

| Categoria | Variável | Fonte | Frequência |
|---|---|---|---|
| **Fundamental** | Preço de abertura | Yahoo Finance | Diária |
| **Fundamental** | Preço de fechamento | Yahoo Finance | Diária |
| **Macroeconômica** | Índice de volatilidade Cboe (VIX) | Yahoo Finance | Diária |
| **Macroeconômica** | Taxa de juros federal (EFFR) | FRED | Diária |
| **Macroeconômica** | Taxa de desemprego civil (UNRATE) | FRED | Mensal* |
| **Macroeconômica** | Índice de sentimento do consumidor (UMCSENT) | FRED | Mensal* |
| **Macroeconômica** | Índice do dólar americano (USDX) | Yahoo Finance | Diária |
| **Técnica** | MACD | Calculado | Diária |
| **Técnica** | ATR (período de 14 dias) | Calculado | Diária |
| **Técnica** | RSI | Calculado | Diária |

> *Dados mensais foram convertidos para diários via **forward filling**.

**Seleção final:** O preço de abertura foi removido por apresentar correlação de 1,0 com o preço de fechamento (features duplicadas). O limiar de corte para remoção por colinearidade foi fixado em **0,80**. Após esse processo, **9 variáveis** foram mantidas como input (mais 2 derivadas = 11 features no total do modelo).

### 4.3 Pré-processamento

1. **Denoising:** Transformada Wavelet de Haar (modo *soft*) aplicada ao preço de fechamento via biblioteca `scikit-image`.
2. **Normalização:** Min-max scaling aplicado a todas as features:

$$z = \frac{x - x_{min}}{x_{max} - x_{min}}$$

3. **Reshape do input:** O LSTM exige entrada 3D — *(n_observações, time_step, n_features)*. Os dados originais são 2D.

### 4.4 Divisão dos dados

| Conjunto | Proporção | Uso |
|---|---|---|
| Treino | 80% do total | Ajuste do modelo |
| Validação | 20% do treino (16% do total) | Tuning de hiperparâmetros |
| Teste | 20% do total | Avaliação final (out-of-sample) |

> A ordem temporal é sempre preservada — sem embaralhamento aleatório.

### 4.5 Métricas de avaliação

| Métrica | Fórmula | Interpretação |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$ | Menor = melhor |
| **MAPE** | $\frac{1}{N}\sum_{i=1}^{N}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | Menor = melhor |
| **R (Correlação)** | $\frac{\sum(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum(y_i-\bar{y})^2 \cdot \sum(\hat{y}_i-\bar{\hat{y}})^2}}$ | Maior = melhor |

### 4.6 Tuning de hiperparâmetros

**Hiperparâmetros explorados:**

- **Otimizadores:** Adam, Adagrad, Nadam
- **Learning rates:** 0,1 / 0,01 / 0,001
- **Batch sizes:** 4 / 8 / 16
- **Total de combinações por modelo:** 27

Cada combinação foi executada **10 vezes** e o critério de seleção foi o **menor RMSE médio** no conjunto de validação. O critério de parada antecipada (*early stopping*) foi configurado com `patience = 5` épocas sem melhora na loss de treino.

**Modelos avaliados:**

| Arquitetura | Configurações de neurônios |
|---|---|
| Single layer LSTM | 10, 30, 50, 100, 150, 200 |
| Multilayer LSTM (2 camadas) | (10,5), (20,10), (50,20), (100,50), (150,100) |
| Multilayer LSTM (3 camadas) | (100,50,20) |

Cada modelo foi replicado **30 vezes** para a avaliação final no conjunto de teste, tratando a variabilidade estocástica do treinamento.

### 4.7 Ambiente computacional

| Item | Especificação |
|---|---|
| Hardware | Google Colab com GPU NVIDIA-SMI 495.44 |
| Linguagem | Python 3.6.0 |
| Frameworks | TensorFlow + Keras |

---

## 5. Resultados Obtidos

### 5.1 Melhor modelo — Single Layer LSTM (150 neurônios)

| Métrica | Mínimo | Médio | Máximo | Desvio Padrão |
|---|---|---|---|---|
| RMSE | 37,94 | **40,46** | 43,40 | 1,40 |
| MAPE | 0,7008 | **0,7989** | 0,9768 | 0,0584 |
| R | 0,9974 | **0,9976** | 0,9979 | 0,0001 |

> O desvio padrão muito baixo indica **alta consistência** entre as 30 replicações.

### 5.2 Melhor modelo — Multilayer LSTM (150, 100 neurônios)

| Métrica | Mínimo | Médio | Máximo | Desvio Padrão |
|---|---|---|---|---|
| RMSE | 46,50 | **49,84** | 52,62 | 1,81 |
| MAPE | 0,9108 | **1,0269** | 1,1351 | 0,0580 |
| R | 0,9959 | **0,9964** | 0,9967 | 0,0001 |

### 5.3 Comparação Single Layer vs. Multilayer

- O **pior resultado** entre as 30 replicações do modelo single-layer (150 neurônios) apresenta RMSE menor do que o **melhor resultado** do modelo multilayer (150, 100).
- Em todas as métricas (RMSE, MAPE e R), os modelos de camada única superam consistentemente os modelos multicamadas.
- A adição de **dropout (10%)** nas camadas ocultas dos modelos multicamadas **não melhorou** o desempenho.

### 5.4 Validação estatística

Para confirmar que a diferença de desempenho não é resultado de variação aleatória, foi aplicado o **teste t de Welch** (duas amostras independentes, normalidade verificada via QQ-plots):

| Estatística de teste | p-valor |
|---|---|
| $t = -22,2387$ | $p = 7,08 \times 10^{-29}$ |

> **Conclusão:** rejeita-se a hipótese nula de igualdade de médias com altíssima significância estatística. O modelo single-layer com 150 neurônios é **significativamente melhor** que o multilayer (150, 100).

### 5.5 Comportamento por número de neurônios (single layer)

- RMSE e MAPE **aumentam** de 10 para 30 neurônios, depois **diminuem** progressivamente até 150 neurônios.
- A partir de 200 neurônios, há **aumento abrupto** de RMSE e MAPE — indicativo de *overfitting*.
- O valor de R segue o padrão inverso: cai de 10 para 30 neurônios, sobe até 150 e cai em 200.

### 5.6 Desempenho durante a pandemia (COVID-19)

O modelo demonstrou **robustez** mesmo durante o período de maior volatilidade da série (início de 2020):
- Capturou corretamente a queda acentuada do mercado.
- Capturou o rápido movimento de recuperação em forma de "V".
- Isso valida a generalização do modelo para dados *out-of-sample* em cenários extremos.

### 5.7 Principais conclusões

1. **Simplicidade supera complexidade:** o modelo de camada única com 150 neurônios supera todas as arquiteturas multicamadas testadas.
2. **A combinação de features é determinante:** a inclusão de dados fundamentais, macroeconômicos e indicadores técnicos captura o comportamento do mercado de forma mais completa do que abordagens com variáveis isoladas.
3. **Alta acurácia preditiva:** correlação média de R = 0,9976 no conjunto de teste demonstra forte aderência entre valores reais e preditos.
4. **O modelo é facilmente adaptável** a outros índices de mercado amplo com comportamento similar.

### 5.8 Trabalhos futuros indicados pelos autores

- Incorporação de **dados textuais não estruturados**: sentimento de investidores em redes sociais, relatórios de resultados, notícias de política monetária.
- Desenvolvimento de **modelos híbridos** combinando LSTM com outras arquiteturas (ex.: CNN + LSTM).
- Uso de **algoritmos de otimização globais** (algoritmos genéticos, PSO — *Particle Swarm Optimization*) combinados com otimizadores locais para o ajuste de parâmetros do modelo.

---

*Referência completa: Bhandari, H.N., Rimal, B., Pokhrel, N.R., Rimal, R., Dahal, K.R., & Khatri, R.K.C. (2022). Predicting stock market index using LSTM. Machine Learning with Applications, 9, 100320. https://doi.org/10.1016/j.mlwa.2022.100320*