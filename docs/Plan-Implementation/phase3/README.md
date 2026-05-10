# Fase 3 — Modelagem e Treinamento da LSTM

## Objetivo

Construir a arquitetura LSTM de camada única conforme especificada no artigo (150 neurônios, Dropout 20%), treinar o modelo sobre os tensores produzidos na Fase 2 com os callbacks Early Stopping e ModelCheckpoint, e avaliar seu desempenho comparativamente ao baseline de persistência pelas métricas RMSE, MAE e MAPE, gerando os artefatos e visualizações necessários para a Fase 4.

## Escopo

| Etapa | Arquivo do plano | Produto gerado |
|---|---|---|
| 1. Construção da arquitetura | [01_construcao_modelo.md](01_construcao_modelo.md) | Definição e compilação do modelo Keras |
| 2. Treinamento com callbacks | [02_treinamento.md](02_treinamento.md) | `models/lstm/lstm_itub4.h5` + `data/figures/08_history_loss.png` |
| 3. Avaliação comparativa | [03_avaliacao_comparativa.md](03_avaliacao_comparativa.md) | `models/lstm/lstm_metrics.json` + figuras `09`, `10`, `11` |

## Estrutura de pastas esperada ao final da fase

```
src/
  notebooks/
    03_lstm_training.ipynb
models/
  lstm/
    lstm_itub4.h5
    lstm_metrics.json
data/
  figures/
    08_history_loss.png
    09_lstm_vs_real.png
    10_lstm_vs_baseline.png
    11_lstm_residuos.png
```

> Os artefatos das Fases 1 e 2 (`data/processed/`, `models/scalers/`, `models/baseline/`, tensores em memória) são pré-requisitos obrigatórios desta fase.

## Dependências Python

```
tensorflow>=2.15
numpy>=1.26
pandas>=2.0
scikit-learn>=1.4
matplotlib>=3.8
joblib>=1.3
jupyter
```

Instalar a nova dependência com:

```bash
pip install tensorflow
```

As demais já estão disponíveis do ambiente das fases anteriores.

## Checklist de conclusão da fase

- [ ] Modelo Keras construído com Input(50,6) → LSTM(150) → Dropout(0.2) → Dense(1)
- [ ] Compilado com `loss="mse"` e `optimizer="adam"`
- [ ] Split de validação interno (10% do treino) criado **sem tocar** o conjunto de teste
- [ ] Early Stopping com `patience=10` sobre `val_loss`
- [ ] ModelCheckpoint salvando `lstm_itub4.h5` apenas quando `val_loss` melhora
- [ ] Treinamento concluído e melhor modelo carregado via `load_model`
- [ ] Gráfico `08_history_loss.png` gerado (curvas de loss treino e validação por época)
- [ ] `y_pred_lstm_real` gerado com shape idêntico a `y_test_real.npy`
- [ ] RMSE, MAE e MAPE calculados sobre valores desnormalizados (em R$)
- [ ] `lstm_metrics.json` salvo em `models/lstm/`
- [ ] LSTM supera baseline nas três métricas
- [ ] Gráfico `09_lstm_vs_real.png` gerado (preço real vs. LSTM no período de teste)
- [ ] Gráfico `10_lstm_vs_baseline.png` gerado (comparação LSTM vs. baseline vs. real)
- [ ] Gráfico `11_lstm_residuos.png` gerado (série temporal e histograma dos resíduos da LSTM)
- [ ] Notebook `03_lstm_training.ipynb` executado sem erros do início ao fim
