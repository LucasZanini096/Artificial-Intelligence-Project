# Fase 2 — Preparação dos Dados e Baseline

## Objetivo

Carregar e validar os artefatos produzidos na Fase 1 (tensores, scalers e dataset processado), implementar o modelo de persistência como *baseline* de comparação e calcular as métricas RMSE, MAE e MAPE sobre os valores desnormalizados do conjunto de teste, gerando os artefatos necessários para a comparação com a LSTM na Fase 3.

## Escopo

| Etapa | Arquivo do plano | Produto gerado |
|---|---|---|
| 1. Carregamento e validação dos tensores | [01_carregamento_tensores.md](01_carregamento_tensores.md) | Validação dos shapes e integridade dos artefatos da Fase 1 |
| 2. Modelo de persistência (baseline) | [02_modelo_baseline.md](02_modelo_baseline.md) | `models/baseline/y_pred_baseline.npy` + `data/figures/06_baseline_vs_real.png` |
| 3. Avaliação e métricas do baseline | [03_avaliacao_metricas.md](03_avaliacao_metricas.md) | `models/baseline/baseline_metrics.json` + `data/figures/07_baseline_residuos.png` |

## Estrutura de pastas esperada ao final da fase

```
src/
  notebooks/
    02_baseline.ipynb
models/
  baseline/
    y_pred_baseline.npy
    y_test_real.npy
    baseline_metrics.json
data/
  figures/
    06_baseline_vs_real.png
    07_baseline_residuos.png
```

> Os artefatos da Fase 1 (`data/processed/`, `models/scalers/`, tensores em memória) são pré-requisitos obrigatórios desta fase.

## Dependências Python

```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.4
matplotlib>=3.8
joblib>=1.3
jupyter
```

As mesmas dependências instaladas na Fase 1 cobrem esta fase integralmente. Não há novas instalações necessárias.

## Checklist de conclusão da fase

- [ ] `X_train`, `y_train`, `X_test`, `y_test` carregados e shapes confirmados
- [ ] `scaler_close.pkl` carregado e desnormalização validada
- [ ] `y_pred_baseline` gerado com o modelo de persistência (shift de 1 dia)
- [ ] `y_pred_baseline.npy` e `y_test_real.npy` serializados em `models/baseline/`
- [ ] RMSE, MAE e MAPE calculados sobre valores em reais (desnormalizados)
- [ ] `baseline_metrics.json` salvo em `models/baseline/`
- [ ] Gráfico `06_baseline_vs_real.png` gerado (preço real vs. baseline no período de teste)
- [ ] Gráfico `07_baseline_residuos.png` gerado (distribuição dos resíduos do baseline)
- [ ] Notebook `02_baseline.ipynb` executado sem erros do início ao fim
