# Fase 1 — Coleta de Dados e Análise Exploratória (EDA)

## Objetivo

Coletar a série histórica de 5 anos do ativo ITUB4.SA via API yfinance, calcular a feature derivada EMA-60, realizar análise exploratória completa e executar o pré-processamento dos dados, produzindo os artefatos necessários para as fases seguintes.

## Escopo

| Etapa | Arquivo do plano | Produto gerado |
|---|---|---|
| 1. Coleta de dados | [01_coleta_dados.md](01_coleta_dados.md) | `data/raw/ITUB4_raw.csv` |
| 2. Análise exploratória | [02_analise_exploratoria.md](02_analise_exploratoria.md) | `notebooks/01_eda.ipynb` + figuras em `data/figures/` |
| 3. Pré-processamento | [03_preprocessamento.md](../phase2/01_preprocessamento.md) | `data/processed/ITUB4_processed.csv` + scalers em `models/scalers/` |

## Estrutura de pastas esperada ao final da fase

```
src/
  notebooks/
    01_eda.ipynb
data/
  raw/
    ITUB4_raw.csv
  processed/
    ITUB4_processed.csv
  figures/
    01_preco_volume.png
    02_retornos_diarios.png
    03_medias_moveis.png
    04_correlacao_heatmap.png
    05_periodos_atipicos.png
models/
  scalers/
    scaler_open.pkl
    scaler_high.pkl
    scaler_low.pkl
    scaler_close.pkl
    scaler_volume.pkl
    scaler_ema60.pkl
```

## Dependências Python

```
yfinance>=0.2.36
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
joblib>=1.3
jupyter
```

Instalar com:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## Checklist de conclusão da fase

- [ ] `ITUB4_raw.csv` gerado com ~1.250 linhas e 6 colunas (OHLCV + EMA-60)
- [ ] Zero valores ausentes no dataset processado
- [ ] 5 figuras de EDA salvas em `data/figures/`
- [ ] Divisão 80/20 cronológica confirmada
- [ ] 6 scalers Z-score serializados em `models/scalers/`
- [ ] Notebook `01_eda.ipynb` executado sem erros do início ao fim
