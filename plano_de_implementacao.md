# Plano de Implementação Faseado: Previsão de Preços B3 com LSTM

Este documento detalha o plano de ação para a construção do projeto de IA de previsão da B3, baseado no *feedback* do professor e nas metodologias do artigo de Zanotto & Hölbig (2026). O projeto foi desenhado para ser executado em **4 semanas (1 mês)**.

---

## 👥 Estratégia Geral e Divisão de Papéis

Para otimizar o tempo e atingir a restrição *"atribua a tarefa ao componente com maior habilidade"*, sugere-se a seguinte divisão (ajuste conforme a afinidade do grupo):
- **Cientista/Engenheiro de Dados:** Responsável pelas Fases 1 e 2 (coleta `yfinance`, manipulação `pandas`, gráficos de EDA e construção das Baselines).
- **Engenheiro de Machine Learning:** Responsável pela Fase 3 (arquitetura LSTM com `keras/tensorflow`, janelamento temporal, treinamento e extração das métricas).
- **Desenvolvedor Frontend / Analista:** Responsável pela Fase 4 (interface `Streamlit`, integração visual do modelo e documentação detalhada das limitações).
- **Todos os Membros:** Redação do relatório metodológico (Artigo) paralelamente durante as 4 semanas.

---

## 🗓️ Fase 1: Coleta de Dados e Análise Exploratória (EDA)
**Prazo Sugerido: Semana 1**
**Objetivo:** Obter a matéria-prima do modelo, entender os dados visualmente e criar os indicadores base.

* **Tarefa 1.1: Configuração do Ambiente e Coleta (`yfinance`)**
  * Definir o ativo alvo único. Recomendação: `ITUB4.SA` (por ter dados com menos "choques políticos", gerando curvas de aprendizado mais fáceis) ou `PETR4.SA` (para ter rico material qualitativo na seção de limitações).
  * Criar script para extrair de 3 a 5 anos de dados diários (Open, High, Low, Close, Volume). 
  * Focar a variável alvo no *Close* (ou Fechamento Ajustado).
* **Tarefa 1.2: Criação do Dataset Enriquecido**
  * Incluir variáveis auxiliares no *dataframe* Pandas (Dica: não exagere no número de colunas para facilitar o treinamento):
    * **Retornos diários** (variação percentual).
    * **Médias Móveis** (ex.: SMA de 15 dias ou EMA de 60 dias conforme o artigo de referência).
    * **RSI** (Índice de Força Relativa) e Medida de Volatilidade (desvio padrão móvel).
  * Limpar dados nulos (`dropna`) ou preencher lacunas de feriados (*forward fill*).
* **Tarefa 1.3: Análise Exploratória de Dados (EDA)**
  * Gerar gráficos (usando `matplotlib` ou `seaborn`):
    * Histórico de Preço vs. Tempo.
    * Gráficos sobrepostos: Preço Real + Média Móvel.
    * Mapa de calor (Correlation Matrix) entre as variáveis recolhidas.
  * Identificar visualmente tendências, sazonalidades e períodos atípicos (como quebras abruptas de preço).

---

## 🗓️ Fase 2: Baseline e Preparação para a LSTM
**Prazo Sugerido: Semana 2**
**Objetivo:** Criar um ponto de comparação justo e formatar os dados matematicamente para o algoritmo da LSTM.

* **Tarefa 2.1: Implementação dos Modelos Baseline (Simples)**
  * Programar o modelo de Persistência (*"Amanhã = Hoje"*): Shift do preço de fechamento em 1 dia para frente.
  * *Opcional:* Programar uma Baseline baseada em Média Móvel (prever o próximo dia usando a média dos últimos X dias).
  * Calcular as métricas dessa Baseline (RMSE, MAE, MAPE) — este será o "piso" mínimo que a LSTM precisa superar.
* **Tarefa 2.2: Escalonamento (Normalização)**
  * Dividir cronologicamente os dados (sem embaralhar): **80% Treino e Validação, 20% Teste**.
  * Aplicar escala (`MinMaxScaler` ou `StandardScaler` / Z-score) **apenas no conjunto de treino** e, em seguida, aplicar a transformação salva no conjunto de teste.
* **Tarefa 2.3: Janelamento temporal (Windowing)**
  * Criar função para transformar o formato tabular do Pandas no *shape* 3D da LSTM: `[amostras, passos_de_tempo, features]`.
  * Definir a janela de observação (*loopback*) pedida pelo professor: algo entre **30 e 60 dias** (Recomendação: usar 50 dias, seguindo o artigo referência).

---

## 🗓️ Fase 3: Modelagem, Treinamento e Avaliação da LSTM
**Prazo Sugerido: Semana 3**
**Objetivo:** Construir, treinar a rede neural e comprovar seu ganho real frente à baseline.

* **Tarefa 3.1: Construção da Arquitetura da Rede Neural**
  * Montar a estrutura da rede com `tensorflow.keras`.
  * *Insight do Artigo de Referência:* Projetar uma rede enxuta (ex: 1 camada LSTM com 50-150 neurônios, um limitador Dropout de 20% a 30% contra overfitting, e 1 camada Densa final para a predição).
  * Compilar o modelo usando o otimizador `Adam` e a função de perda de Erro Quadrático Médio (`MSE`).
* **Tarefa 3.2: Treinamento e Validação (Fit)**
  * Separar 10-20% dos dados da parcela de treino como validação (*validation_split*).
  * Implementar as funções de retorno (*Callbacks*): *Early Stopping* (paciência = ~10 épocas) para parar se o erro não melhorar, e *ModelCheckpoint* para salvar o melhor arquivo `.h5`.
  * Treinar o modelo definindo o *batch size*.
* **Tarefa 3.3: Avaliação e Comparação de Métricas**
  * Realizar predições no conjunto de Teste e **desnormalizar** (inverter o *scaler*) os valores de volta para Reais (R$).
  * Calcular as métricas exigidas (RMSE, MAE, MAPE).
  * Comparar os resultados absolutos com a Baseline da Fase 2.
  * Gerar o gráfico de avaliação: *Preço Real vs. Predição LSTM vs. Baseline* na mesma tela do período de Teste.

---

## 🗓️ Fase 4: Protótipo Web (Streamlit) e Escrita Final
**Prazo Sugerido: Semana 4**
**Objetivo:** Consolidar a entrega visual (UI) e a documentação final do N1/Projeto.

* **Tarefa 4.1: Desenvolvimento do Protótipo (Streamlit)**
  * Estruturar um projeto local simples no `streamlit` importando `.h5` treinado ou apenas os gráficos/dataframes estáticos já gerados.
  * **Tela Principal:**
    1. Gráfico dinâmico (com `plotly` ou `st.line_chart`) exibindo *Preço Real vs Previsto*.
    2. *Cards* exibindo as métricas finais obtidas: "RMSE: X.XX | MAPE: Y.YY%".
  * **Seção de Limitações e Disclaimer (OBRIGATÓRIO):**
    1. Painel de alerta: *"Mero apoio analítico. Padrões passados não garantem retornos futuros. Não é uma recomendação de investimento."*
    2. Texto de Limitações: Explicar que a LSTM baseada apenas em histórico de preços tem "memória curta", não consegue prever "cisnes negros" (ex: Pandemia) e ignora análises de balanço, sentimento de mercado em redes sociais e decisões cruciais de conselho/governo.
* **Tarefa 4.2: Finalização do Relatório (Artigo SBC)**
  * Transcrever as seções para o *template* LaTeX (`sbc-template.sty`).
  * Descrever rigorosamente a Metodologia: desde de onde o dado veio (`yfinance`), técnicas de limpeza, tamanho da janela usada (ex: 50 dias), e hiperparâmetros (Dropout, neurônios, Épocas).
  * Discutir os resultados usando a comparação base x LSTM.
  * Formatar referências (lembrar de citar *Zanotto & Hölbig (2026)* como justificativa da escolha da B3, de métricas e de janelamento).

---
*Dica Técnica:* Vocês podem centralizar a coleta e manipulação numa biblioteca Python própria da equipe (ex: arquivos `coleta.py`, `modelo.py`) e usá-los tanto no *Jupyter Notebook* (para EDA e treinamento) quanto no script principal do `app.py` do Streamlit, mantendo o repositório limpo e profissional.
