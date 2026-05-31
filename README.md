# Previsão do Preço Futuro da Ação ITUB4 com LSTM

Projeto da disciplina **Inteligência Artificial** — 7º semestre (Noturno) — Ciência da Computação
**Universidade Presbiteriana Mackenzie**
Prof. Dr. Ivan Carlos Alcântara de Oliveira

---

## Descrição

Desenvolvimento e validação de uma rede neural recorrente do tipo **LSTM (Long Short-Term Memory)** de camada única para prever o **preço de fechamento do dia seguinte** do ativo **ITUB4** (Itaú Unibanco S.A.) na B3, utilizando 5 anos de dados históricos diários (Open, High, Low, Close, Volume) e a Média Móvel Exponencial de 60 dias (EMA-60) como feature adicional.

O modelo é comparado a um *baseline* de persistência ("amanhã = hoje") e os resultados são apresentados em uma aplicação web interativa construída com **Streamlit**.

> Este projeto é um apoio analítico baseado em padrões históricos. Não constitui recomendação de investimento.

---

## Integrantes

| Nome | RA |
| --- | --- |
| Gabriel Alves de F. Spinola Sucupira | 10418133 |
| Henrique Pena Ribeiro | 10417975 |
| Lucas Zanini da Silva | 10417361 |
| Tiago Teraoka e Sá | 10418485 |

---

## Estrutura de Pastas

```
PROJETO/
│
├── README.md                              # Este arquivo
├── .gitignore
├── requirements.txt                       # Dependências Python
├── 01_IA_EAD_projeto_20261_7N.pdf         # Enunciado oficial do projeto
├── Entrega-Final.zip                      # Pacote de entrega final
│
├── Artigo Projeto/                        # Relatório (LaTeX / SBC)
│   ├── relatorio_n1.tex                   # Código-fonte do artigo
│   ├── relatorio_n1.pdf                   # PDF compilado
│   ├── referencias_n1.bib                 # Referências bibliográficas (BibTeX)
│   ├── sbc-template.sty                   # Estilo SBC
│   ├── sbc.bst                            # Estilo bibliográfico SBC
│   └── figuras/                           # Gráficos gerados para o artigo
│       ├── 01_preco_volume.png
│       ├── 02_retornos.png
│       ├── 03_medias_moveis.png
│       ├── 04_correlacao.png
│       ├── 05_baseline_vs_real.png
│       ├── 06_baseline_residuos.png
│       ├── 07_loss.png
│       ├── 08_lstm_vs_real.png
│       ├── 09_lstm_vs_baseline.png
│       ├── 10_lstm_residuos.png
│       └── 11_streamlit_app.png
│
├── docs/
│   ├── ADR's/
│   │   └── 0001-previsao-de-retornos-logaritmicos.md  # Decisão de arquitetura
│   ├── Articles/                          # Resumos dos artigos de referência
│   │   ├── Predicting stock market index using LSTM.md
│   │   ├── Previsão de preços de ações e ETF na bolsa de valores B3 (...).md
│   │   └── pdf´s/                         # PDFs originais dos artigos
│   ├── Deliverys/                         # Relatórios entregues nas avaliações
│   │   ├── Relatorio_parte_1.pdf
│   │   ├── Relatorio_parte_2.pdf
│   │   └── Relatorio_parte_3.pdf
│   ├── Glossary/
│   │   └── glossary.md                    # Glossário de termos técnicos
│   └── Plan-Implementation/               # Planos de execução por fase
│       ├── phase1/                        # Coleta de dados e EDA
│       ├── phase2/                        # Pré-processamento e baseline
│       ├── phase3/                        # Construção e avaliação do LSTM
│       └── phase4/                        # Streamlit e entrega final
│
└── src/
    ├── app/                               # Aplicação Streamlit
    │   ├── app.py                         # Ponto de entrada da aplicação
    │   ├── utils.py                       # Funções auxiliares
    │   └── .streamlit/
    │       └── config.toml                # Configurações de tema e layout
    │
    ├── database/
    │   ├── raw/
    │   │   └── itub4_raw.csv              # Dados brutos baixados via yfinance
    │   └── processed/
    │       └── ITUB4_processed.csv        # Dados com EMA-60 e retornos log
    │
    ├── models/
    │   ├── baseline/
    │   │   ├── baseline_metrics.json      # RMSE / MAE / MAPE do baseline
    │   │   ├── y_pred_baseline.npy        # Predições do modelo de persistência
    │   │   └── y_test_real.npy            # Valores reais do conjunto de teste
    │   ├── lstm/
    │   │   ├── lstm_itub4.keras           # Modelo LSTM treinado (Keras)
    │   │   ├── lstm_metrics.json          # RMSE / MAE / MAPE do LSTM
    │   │   ├── y_pred_lstm.npy            # Predições do LSTM
    │   │   └── test_dates.npy             # Datas correspondentes ao teste
    │   └── scalers/                       # StandardScalers serializados (joblib)
    │       ├── scaler_close.pkl
    │       ├── scaler_ema_60.pkl
    │       ├── scaler_high.pkl
    │       ├── scaler_log_return_close.pkl
    │       ├── scaler_low.pkl
    │       ├── scaler_open.pkl
    │       └── scaler_volume.pkl
    │
    └── notebooks/
        ├── 1-colect_data.ipynb            # Coleta via yfinance e salva CSV bruto
        ├── 2-exploration_analysis.ipynb   # EDA — preços, retornos, correlações
        └── 3-data_preparation_baseline.ipynb  # Pré-processamento, LSTM e baseline
```

---

## Metodologia (Resumo)

| Etapa | Descrição |
| --- | --- |
| **Coleta** | API `yfinance` — ITUB4.SA, últimos 5 anos, dados diários |
| **EDA** | Série de preços, retornos, médias móveis, correlações, períodos atípicos |
| **Feature engineering** | EMA-60 calculada sobre o fechamento |
| **Pré-processamento** | Z-score por variável (parâmetros somente do treino); janelas de 50 dias |
| **Divisão** | 80% treino / 20% teste (ordem cronológica preservada) |
| **Arquitetura** | LSTM 1 camada — 150 neurônios — Dropout 20% — Dense 1 neurônio |
| **Treinamento** | Otimizador Adam, perda MSE, Early Stopping (patience=10), ModelCheckpoint |
| **Baseline** | Persistência: $\hat{y}_{t+1} = y_t$ |
| **Métricas** | RMSE, MAE, MAPE (calculadas na escala original em R$) |
| **Interface** | Streamlit com gráfico real vs. previsto, painel de métricas e aviso legal |

---

## Como Compilar o Artigo

Requer uma instalação de LaTeX (TeX Live ou MiKTeX) com BibTeX.

```bash
cd "Artigo Projeto"

# 1. Primeira passagem — gera o .aux
pdflatex relatorio_n1.tex

# 2. BibTeX — processa as referências e gera o .bbl
bibtex relatorio_n1

# 3 e 4. Duas passagens finais — resolve referências cruzadas
pdflatex relatorio_n1.tex
pdflatex relatorio_n1.tex
```

O PDF final será gerado em `Artigo Projeto/relatorio_n1.pdf`.

---

## Referências Principais

- **Bhandari et al. (2022)** — *Predicting stock market index using LSTM.* Machine Learning with Applications, 9, 100320. DOI: 10.1016/j.mlwa.2022.100320
- **Zanotto & Hölbig (2026)** — *Previsão de preços de ações e ETF na bolsa de valores B3 aplicando técnicas de machine learning.* Revista Sítio Novo, v.10, e1879. DOI: 10.47236/2594-7036.2026.v10.1879
- **Hochreiter & Schmidhuber (1997)** — *Long short-term memory.* Neural Computation, 9(8), 1735–1780.
