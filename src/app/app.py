# src/app/app.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

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

# ── Painel de métricas comparativo ────────────────────────────────────────────
st.markdown("---")
st.subheader("Métricas de Avaliação — Conjunto de Teste")

col1, col2, col3 = st.columns(3)

# Note: delta_color is normal (positive metric = green, negative = red).
# Standard Streamlit metrics will display green for positive delta and red for negative.
# We present the percentage difference: positive indicates error reduction (better), negative indicates error increase (worse).
with col1:
    st.metric(
        label="RMSE (R$)",
        value=f"R$ {lstm_metrics['RMSE']:.4f}",
        delta=f"{lstm_metrics['melhoria_vs_baseline']['RMSE_pct']:+.1f}% vs. baseline",
        delta_color="normal",
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

st.markdown("#### Tabela Comparativa")

df_metrics = pd.DataFrame({
    "Modelo":   ["Baseline (persistência)", "LSTM (camada única, 150 neur.)"],
    "RMSE (R$)": [
        f"R$ {baseline_metrics['RMSE']:.4f}",
        f"R$ {lstm_metrics['RMSE']:.4f}",
    ],
    "MAE (R$)": [
        f"R$ {baseline_metrics['MAE']:.4f}",
        f"R$ {lstm_metrics['MAE']:.4f}",
    ],
    "MAPE (%)": [
        f"{baseline_metrics['MAPE']:.2f}%",
        f"{lstm_metrics['MAPE']:.2f}%",
    ],
})

st.dataframe(df_metrics, hide_index=True, use_container_width=True)

# ── Informações do modelo ─────────────────────────────────────────────────────
with st.expander("ℹ️ Detalhes do Modelo"):
    st.markdown(f"""
    | Parâmetro | Valor |
    |---|---|
    | Ativo | ITUB4.SA (Itaú Unibanco) |
    | Período de treino | 03/05/2021 a 30/04/2025 (80% dos dados) |
    | Período de teste | {lstm_metrics.get('test_start', '14/07/2025')} a {lstm_metrics.get('test_end', '30/04/2026')} |
    | Amostras de teste | {lstm_metrics['n_samples']} pregões |
    | Arquitetura | {lstm_metrics['architecture']} |
    | Features | Open, High, Low, Close, Volume, EMA-60 |
    | Janela temporal | 50 dias |
    | Normalização | Z-score (µ e σ do treino) |
    | Otimizador | Adam · Loss: MSE |
    """)

# ── Aviso legal ───────────────────────────────────────────────────────────────
st.markdown("---")
st.warning(
    "⚠️ **Aviso Legal**\n\n"
    "Esta ferramenta é um apoio analítico baseado em padrões históricos. "
    "O desempenho passado não garante resultados futuros. "
    "**Não constitui recomendação de investimento.**",
    icon="⚠️",
)

# ── Seção de limitações técnicas ──────────────────────────────────────────────
st.subheader("Limitações do Modelo")

st.markdown("""
O modelo LSTM prevê o preço de fechamento do ITUB4 com base exclusivamente
em **padrões históricos de preço e volume**. Ele não possui acesso a:

- **Eventos exógenos não modelados**: pandemias, mudanças abruptas de política
  monetária, intervenções governamentais ou escândalos corporativos.
  [Zanotto & Hölbig (2026)](https://doi.org/10.17648/sitio-novo-v10n1-1879)
  demonstrou empiricamente que esses fatores elevam significativamente o
  erro de predição.
- **Análise fundamentalista**: demonstrativos financeiros, dividendos, guidance
  de resultados e valuation não são considerados.
- **Sentimento de mercado**: notícias, redes sociais e fluxo institucional
  estão fora do escopo do modelo.
- **Horizonte de previsão**: o modelo produz previsões de **apenas 1 dia útil**
  à frente. Extrapolações para múltiplos dias compõem os erros de forma
  exponencial e não devem ser realizadas.

A escolha do ITUB4 como ativo de estudo mitiga parte das limitações acima
(empresa privada com menor interferência governamental), mas não as elimina.
""")

# ── Rodapé ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Desenvolvido por Gabriel Sucupira · Henrique Ribeiro · Lucas Zanini · Tiago Teraoka — "
    "Universidade Presbiteriana Mackenzie, 2026.1 · "
    "[Repositório GitHub](https://github.com/LucasZanini096/Artificial-Intelligence-Project)"
)
