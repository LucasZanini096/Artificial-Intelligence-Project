# Etapa 3 — Página Principal: Gráfico Interativo e Painel de Métricas

## Objetivo

Implementar o arquivo `app.py` com o conteúdo central da aplicação: cabeçalho institucional, gráfico interativo Plotly com preços reais vs. LSTM vs. baseline no período de teste, e painel de métricas comparativo (RMSE, MAE, MAPE).

---

## 3.1 Estrutura geral do `app.py`

```python
# src/app/app.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from utils import load_predictions, load_metrics, load_full_series

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITUB4 — Previsão LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("📈 ITUB4 — Previsão do Preço de Fechamento com LSTM")
st.caption(
    "Universidade Presbiteriana Mackenzie · Faculdade de Computação e Informática · IA 2026.1"
)
st.markdown("---")

# ── Carregamento dos artefatos ────────────────────────────────────────────────
y_test_real, y_pred_lstm, y_pred_baseline, test_dates = load_predictions()
lstm_metrics, baseline_metrics = load_metrics()
```

---

## 3.2 Gráfico interativo Plotly

O gráfico principal exibe as três séries no período de teste com interatividade (zoom, pan, hover com valores em R$).

```python
# ── Gráfico interativo ────────────────────────────────────────────────────────
st.subheader("Preço Real vs. Previsão LSTM vs. Baseline (Período de Teste)")

fig = go.Figure()

# Preço real
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_test_real,
    name="Preço Real",
    line=dict(color="#1f77b4", width=2),
    hovertemplate="<b>Real</b><br>Data: %{x|%d/%m/%Y}<br>Preço: R$ %{y:.2f}<extra></extra>",
))

# Previsão LSTM
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_pred_lstm,
    name=f"LSTM (MAPE={lstm_metrics['MAPE']:.2f}%)",
    line=dict(color="#d62728", width=1.5),
    hovertemplate="<b>LSTM</b><br>Data: %{x|%d/%m/%Y}<br>Previsão: R$ %{y:.2f}<extra></extra>",
))

# Baseline
fig.add_trace(go.Scatter(
    x=test_dates,
    y=y_pred_baseline,
    name=f"Baseline (MAPE={baseline_metrics['MAPE']:.2f}%)",
    line=dict(color="#ff7f0e", width=1.2, dash="dash"),
    opacity=0.75,
    hovertemplate="<b>Baseline</b><br>Data: %{x|%d/%m/%Y}<br>Previsão: R$ %{y:.2f}<extra></extra>",
))

fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço de Fechamento (R$)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480,
    margin=dict(l=0, r=0, t=40, b=0),
)
fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")

st.plotly_chart(fig, use_container_width=True)
```

---

## 3.3 Painel de métricas comparativo

Exibe RMSE, MAE e MAPE lado a lado para LSTM e baseline, com delta colorido indicando a melhoria.

```python
st.markdown("---")
st.subheader("Métricas de Avaliação — Conjunto de Teste")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="RMSE (R$)",
        value=f"R$ {lstm_metrics['RMSE']:.4f}",
        delta=f"{lstm_metrics['melhoria_vs_baseline']['RMSE_pct']:+.1f}% vs. baseline",
        delta_color="normal",    # verde = melhoria (delta negativo = menor erro)
        help="Root Mean Squared Error. Penaliza desvios grandes. Menor é melhor.",
    )
    st.caption(f"Baseline: R$ {baseline_metrics['RMSE']:.4f}")

with col2:
    st.metric(
        label="MAE (R$)",
        value=f"R$ {lstm_metrics['MAE']:.4f}",
        delta=f"{lstm_metrics['melhoria_vs_baseline']['MAE_pct']:+.1f}% vs. baseline",
        delta_color="normal",
        help="Mean Absolute Error. Erro médio absoluto em reais. Menor é melhor.",
    )
    st.caption(f"Baseline: R$ {baseline_metrics['MAE']:.4f}")

with col3:
    st.metric(
        label="MAPE (%)",
        value=f"{lstm_metrics['MAPE']:.2f}%",
        delta=f"{lstm_metrics['melhoria_vs_baseline']['MAPE_pct']:+.1f}% vs. baseline",
        delta_color="normal",
        help="Mean Absolute Percentage Error. Erro relativo percentual. Menor é melhor.",
    )
    st.caption(f"Baseline: {baseline_metrics['MAPE']:.2f}%")
```

> **Comportamento do `delta_color`:** o Streamlit interpreta delta positivo (melhoria %) como verde e delta negativo como vermelho por padrão. Como a métrica de melhoria é calculada como `(baseline - lstm) / baseline * 100`, um valor positivo indica que a LSTM tem **menor** erro. Isso produz cor verde corretamente.

---

## 3.4 Tabela resumo

Complementa o painel de métricas com uma tabela formatada, útil para comparação direta.

```python
import pandas as pd

st.markdown("#### Tabela Comparativa")

df_metrics = pd.DataFrame({
    "Modelo":   ["Baseline (persistência)", "LSTM (camada única, 150 neur.)"],
    "RMSE (R$)": [
        f"{baseline_metrics['RMSE']:.4f}",
        f"{lstm_metrics['RMSE']:.4f}",
    ],
    "MAE (R$)": [
        f"{baseline_metrics['MAE']:.4f}",
        f"{lstm_metrics['MAE']:.4f}",
    ],
    "MAPE (%)": [
        f"{baseline_metrics['MAPE']:.2f}",
        f"{lstm_metrics['MAPE']:.2f}",
    ],
})

st.dataframe(df_metrics, hide_index=True, use_container_width=True)
```

---

## 3.5 Informações do modelo

Expander colapsável com detalhes técnicos da arquitetura, para usuários interessados.

```python
with st.expander("ℹ️ Detalhes do Modelo"):
    st.markdown(f"""
    | Parâmetro | Valor |
    |---|---|
    | Ativo | ITUB4.SA (Itaú Unibanco) |
    | Período de treino | {lstm_metrics.get('test_start', '—')} (80% dos dados) |
    | Período de teste | {lstm_metrics.get('test_start', '—')} → {lstm_metrics.get('test_end', '—')} |
    | Amostras de teste | {lstm_metrics['n_samples']} pregões |
    | Arquitetura | {lstm_metrics['architecture']} |
    | Features | Open, High, Low, Close, Volume, EMA-60 |
    | Janela temporal | 50 dias |
    | Normalização | Z-score (µ e σ do treino) |
    | Otimizador | Adam · Loss: MSE |
    """)
```

---

## Checklist da Etapa 3

- [ ] `app.py` criado em `src/app/`
- [ ] Cabeçalho institucional exibido corretamente
- [ ] Gráfico Plotly renderiza as três séries com hover funcional
- [ ] Painel de métricas exibe RMSE, MAE, MAPE com delta vs. baseline
- [ ] Tabela comparativa renderiza sem erros
- [ ] Expander de detalhes do modelo funciona
- [ ] App não exibe erros no terminal ao carregar os artefatos
