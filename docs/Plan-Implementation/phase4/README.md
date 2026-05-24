# Fase 4 — Protótipo Web e Relatório Final

## Objetivo

Completar os artefatos de avaliação da Fase 3 que ainda não foram gerados (métricas, previsões serializadas e figuras 08–11), integrar o modelo LSTM treinado em uma aplicação web interativa desenvolvida com Streamlit e produzir os materiais finais do projeto.

## Pré-requisitos obrigatórios

| Artefato | Localização | Status |
|---|---|---|
| `lstm_itub4.keras` | `src/models/lstm/` | ✅ Gerado na Fase 3 |
| `scaler_close.pkl` … `scaler_volume.pkl` | `src/models/scalers/` | ✅ Gerados na Fase 2 |
| `y_test_real.npy` | `src/models/baseline/` | ✅ Gerado na Fase 2 |
| `y_pred_baseline.npy` | `src/models/baseline/` | ✅ Gerado na Fase 2 |
| `baseline_metrics.json` | `src/models/baseline/` | ✅ Gerado na Fase 2 |
| `ITUB4_processed.csv` | `src/database/processed/` | ✅ Gerado na Fase 1 |
| `lstm_metrics.json` | `src/models/lstm/` | ❌ Pendente |
| `y_pred_lstm.npy` | `src/models/lstm/` | ❌ Pendente |
| Figuras 08–11 | `src/data/figures/` | ❌ Pendentes |

## Escopo

| Etapa | Arquivo do plano | Produto gerado |
|---|---|---|
| 1. Completar artefatos da Fase 3 | [01_completar_fase3.md](01_completar_fase3.md) | `lstm_metrics.json`, `y_pred_lstm.npy`, figuras `08`–`11` |
| 2. Estrutura e configuração do Streamlit | [02_estrutura_streamlit.md](02_estrutura_streamlit.md) | `src/app/app.py`, `src/app/utils.py`, `.streamlit/config.toml` |
| 3. Página principal: gráfico e métricas | [03_pagina_principal.md](03_pagina_principal.md) | Interface funcional com gráfico Plotly e painel de métricas |
| 4. Aviso legal e limitações | [04_aviso_legal.md](04_aviso_legal.md) | Seção obrigatória de disclaimer integrada ao app |

## Estrutura de pastas esperada ao final da fase

```
src/
  app/
    app.py                   ← aplicação Streamlit principal
    utils.py                 ← funções auxiliares (carregamento, inferência)
    .streamlit/
      config.toml            ← tema e configuração da UI
  models/
    lstm/
      lstm_itub4.keras
      lstm_metrics.json      ← RMSE, MAE, MAPE + melhoria vs. baseline
      y_pred_lstm.npy        ← previsões em R$, shape (200,)
  data/
    figures/
      08_history_loss.png
      09_lstm_vs_real.png
      10_lstm_vs_baseline.png
      11_lstm_residuos.png
```

## Dependências Python novas nesta fase

```
streamlit>=1.35
plotly>=5.22
```

Instalar com:

```bash
pip install streamlit plotly
```

As demais dependências (tensorflow, numpy, pandas, joblib, scikit-learn, matplotlib) já estão disponíveis das fases anteriores.

## Checklist de conclusão da fase

- [ ] `lstm_metrics.json` salvo em `models/lstm/`
- [ ] `y_pred_lstm.npy` serializado em `models/lstm/` com shape `(200,)`
- [ ] LSTM supera baseline nas três métricas (RMSE, MAE, MAPE)
- [ ] Figuras `08` a `11` geradas em `data/figures/`
- [ ] `src/app/app.py` executa sem erros com `streamlit run src/app/app.py`
- [ ] Gráfico interativo Plotly exibe preços reais vs. LSTM vs. baseline no período de teste
- [ ] Painel de métricas exibe RMSE, MAE, MAPE da LSTM e do baseline lado a lado
- [ ] Aviso legal obrigatório exibido em destaque na interface
- [ ] Seção de limitações do modelo documentada no app
- [ ] App funciona a partir do diretório raiz do projeto (`PROJETO/`)
