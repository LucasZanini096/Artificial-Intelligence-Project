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
| Gabriel Alves de F. Spinola Sucupira | — |
| Henrique Pena Ribeiro | — |
| Lucas Zanini da Silva | — |
| Tiago Teraoka e Sá | — |

---

## Estrutura de Pastas

```
PROJETO/
│
├── README.md                          # Este arquivo
├── .gitignore
├── 01_IA_EAD_projeto_20261_7N.pdf     # Enunciado oficial do projeto
│
├── Artigo Projeto/                    # Relatório N1 (LaTeX / SBC)
│   ├── relatorio_n1.tex               # Código-fonte do artigo
│   ├── relatorio_n1.pdf               # PDF compilado
│   ├── referencias_n1.bib             # Referências bibliográficas (BibTeX)
│   ├── sbc-template.sty               # Estilo SBC
│   └── sbc.bst                        # Estilo bibliográfico SBC
│
├── Artigos/                           # Material bibliográfico
│   ├── Predicting stock market index using LSTM.md
│   ├── Previsão de preços de ações e ETF na bolsa de valores B3 (...).md
│   └── pdf´s/
│       ├── Predicting stock market index using LSTM.pdf
│       └── Previsão de preços de ações e ETF na bolsa de valores B3 (...).pdf
│
├── Entregas/                          # Arquivos entregues nas avaliações
│
└── Glossário/
    └── glossary.md                    # Glossário de termos técnicos
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
