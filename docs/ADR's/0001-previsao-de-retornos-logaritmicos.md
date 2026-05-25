# ADR 0001: Previsão de Retornos Logarítmicos em vez de Preços Absolutos

* **Status:** Aceito
* **Data:** 24 de Maio de 2026
* **Autores:** Gabriel Alves de F Spinola Sucupira, Henrique Pena Ribeiro, Lucas Zanini da Silva, Tiago Teraoka e Sá

---

## 1. Contexto e Problema

No desenvolvimento do modelo de inteligência artificial para previsão do preço de fechamento do ativo **ITUB4** (Fase 3 do projeto), a rede neural recorrente **LSTM** foi treinada utilizando o preço de fechamento absoluto normalizado globalmente via Z-score. 

Durante a avaliação no conjunto de teste, observou-se uma discrepância severa de desempenho, na qual a **Baseline de Persistência** ($\hat{y}_{t+1} = y_t$) superou amplamente a LSTM em todas as métricas:

*   **Baseline:** RMSE: 0.5995 | MAE: 0.4447 | MAPE: 1.1196%
*   **LSTM:** RMSE: 2.2660 | MAE: 1.6483 | MAPE: 3.8887%

### Causa Raiz Identificada
A análise estatística dos dados revelou o fenômeno de **Concept Drift / Ruptura de Regime (Regime Shift)**:
1.  **Limitação de Treino:** O período de treinamento (2021–2025) registrou preços do ITUB4 na faixa de R$ 14,00 a R$ 31,00 ($Z$-scores de $-1.6\sigma$ a $+2.7\sigma$). Apenas 0,6% das amostras de treino superaram o patamar de R$ 30,00.
2.  **Extrapolação Inédita no Teste:** O período de teste (2025–2026) coincidiu com um ciclo de valorização histórico, empurrando o preço do ativo para patamares entre R$ 31,00 e R$ 49,00 ($Z$-scores entre $+2.3\sigma$ e $+6.7\sigma$).
3.  **Saturação do Modelo:** Por lidar com variáveis absolutas não-estacionárias, a LSTM foi forçada a extrapolar além do domínio conhecido. Suas ativações internas saturaram, limitando as previsões a um teto artificial de ~R$ 42,62, subestimando sistematicamente a tendência real.
4.  **Resiliência da Baseline:** O modelo de persistência, por meramente repetir o último valor conhecido, manteve-se imune às mudanças de escala, beneficiando-se da altíssima autocorrelação inerente à série financeira de curtíssimo prazo (Passeio Aleatório).

---

## 2. Decisão Arquitetural

Para mitigar a vulnerabilidade do modelo à extrapolação e à não-estacionaridade dos preços absolutos, **decidimos adotar a previsão de retornos logarítmicos diários em vez de preços de fechamento nominais.**

O alvo da rede neural passa a ser o retorno logarítmico $r_t$:
$$r_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$

### Fluxo de Implementação:

1.  **Pré-processamento:**
    *   Substituir a variável-alvo nominal (`Close`) pelo cálculo de seu retorno logarítmico.
    *   Remover a primeira linha resultante da diferenciação (que se tornará `NaN`).
    *   Ajustar os scalers Z-score para operar sobre o retorno logarítmico (que agora flutua em uma distribuição estável em torno de zero, tipicamente entre $-4\%$ e $+4\%$).
2.  **Arquitetura do Modelo:**
    *   A camada final de saída linear da LSTM prevê o retorno logarítmico esperado para o dia $t+1$ ($\hat{r}_{t+1}$).
3.  **Pós-processamento e Reconstrução de Preço:**
    *   A avaliação do modelo para cálculo das métricas de negócios (RMSE, MAE, MAPE) continuará ocorrendo no domínio de Reais (R$).
    *   O preço absoluto previsto ($\hat{P}_{t+1}$) será reconstruído de forma recursiva a partir do último preço real conhecido no instante $t$ ($P_t$), aplicando a equação:
        $$\hat{P}_{t+1} = P_t \cdot e^{\hat{r}_{t+1}}$$

---

## 3. Alternativas Consideradas

*   **Alternativa A: Normalização Local por Janela (Min-Max por Janela de 50 Dias)**
    *   *Descrição:* Normalizar a janela temporal local dividindo todos os preços pelo valor do último dia da janela.
    *   *Rejeição:* Embora resolva a dependência de escala absoluta, a previsão direta de preços relativos ainda pode apresentar menor estabilidade matemática em longos períodos de teste em comparação aos retornos logarítmicos, que possuem melhores propriedades estatísticas para modelagem financeira (como aditividade temporal).
*   **Alternativa B: Retreinamento Contínuo (Rolling Walk-Forward)**
    *   *Descrição:* Retreinar periodicamente a rede neural à medida que novos preços de teste são coletados.
    *   *Rejeição:* Complexidade operacional elevada e alto custo computacional no ambiente de produção (Streamlit). Além disso, não resolve na raiz o problema conceitual de tentar modelar dados não-estacionários em escala global.
*   **Alternativa C: Manutenção do Modelo de Preço Nominal e Discussão Teórica**
    *   *Descrição:* Não alterar a engenharia de dados do modelo e defender a incapacidade da LSTM como uma limitação natural no artigo científico.
    *   *Rejeição:* Embora academicamente aceitável, limita o valor prático da ferramenta interativa de Streamlit, reduzindo sua utilidade como simulador analítico viável em ambiente de produção real.

---

## 4. Consequências da Decisão

### Consequências Positivas (Benefícios):

*   **Estacionaridade dos Dados:** O retorno logarítmico possui média estável e variância controlada ao longo do tempo, garantindo que o modelo nunca enfrente Z-scores extremos (como $+6.7\sigma$) causados por novos patamares nominais de preço.
*   **Imunidade a Escalas Absolutas:** O modelo aprenderá a dinâmica de oscilação do mercado (percentual), operando de forma idêntica independentemente do ativo valer R$ 20,00 ou R$ 100,00.
*   **Alinhamento com Padrões de Mercado:** A previsão de retornos logarítmicos é o padrão-ouro e a prática mais robusta na literatura de finanças quantitativas de alta performance.
*   **Superação Justa do Baseline:** Ao focar no comportamento dinâmico e na estacionaridade, o modelo LSTM ganha condições estatísticas reais de capturar padrões não-lineares de tendência diária, superando consistentemente a persistência estática.

### Consequências Negativas (Esforços & Riscos):

*   **Refatoração do Código:** Exige alteração na função de engenharia de dados no notebook (limpeza de NaNs gerados pela defasagem e redefinição das janelas de treino/teste).
*   **Reconstrução de Preço no Pipeline:** Adiciona complexidade na reconstrução de preços nominal no Streamlit, pois a fórmula de desnormalização agora necessita de um valor âncora real do pregão anterior para deduzir o preço absoluto de amanhã.
*   **Acúmulo de Erro Recursivo:** A inferência continuada para múltiplos passos à frente (se necessária) acumula incertezas exponencialmente através dos retornos, embora o escopo do projeto limite-se estritamente à previsão de 1 passo ($t+1$).
