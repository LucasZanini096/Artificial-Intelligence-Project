# Etapa 2 — Modelo de Persistência (Baseline)

## Objetivo

Implementar o modelo de persistência ("amanhã = hoje"), que prevê o preço de fechamento do dia `t+1` usando simplesmente o preço observado no dia `t`. Este modelo trivial serve como **piso mínimo de comparação**: a LSTM só será considerada válida se superar o baseline em todas as métricas (RMSE, MAE, MAPE) sobre o conjunto de teste.

---

## 2.1 Lógica do modelo de persistência

O modelo de persistência não possui parâmetros treináveis. Sua definição formal é:

```
ŷ(t+1) = y(t)
```

Ou seja, a previsão para o próximo pregão é igual ao valor de fechamento do pregão atual. Apesar de trivial, este modelo é difícil de superar em séries financeiras de curto prazo, pois captura a autocorrelação de lag-1 que domina os preços diários.

**Por que usar persistência e não média ou ARIMA?**  
O modelo de persistência é o *baseline* canônico em previsão de séries financeiras de um passo à frente. Ele não requer ajuste de hiperparâmetros, não vaza dados e é reprodutível por qualquer leitor do artigo. ARIMA e outros modelos estatísticos introduziriam hiperparâmetros adicionais sem acrescentar clareza à comparação principal (LSTM vs. naive).

---

## 2.2 Alinhamento entre previsão e realidade

No contexto do janelamento criado na Fase 1, cada `y[i]` corresponde ao fechamento do pregão imediatamente após a janela de 50 dias. O baseline no espaço normalizado é:

```python
# y_test[i]   = fechamento normalizado do pregão i  (valor real)
# y_test[i-1] = fechamento normalizado do pregão i-1 (previsão do baseline)
y_pred_baseline_norm = np.roll(y_test, shift=1)
y_pred_baseline_norm[0] = y_test[0]   # sem previsão para o primeiro ponto; repete o próprio valor
```

**Por que usar `y_test` e não `X_test`?**  
`y_test[i]` é o fechamento do dia `t`, e o modelo prevê o fechamento do dia `t+1` como sendo `y_test[i]`. O `np.roll` desloca o vetor um passo para frente, simulando exatamente essa lógica. O primeiro elemento é inicializado com o próprio valor real (o modelo não tem histórico para esse ponto).

---

## 2.3 Implementação

```python
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs("models/baseline", exist_ok=True)

# Carrega o scaler de Close (fit apenas no treino — sem data leakage)
scaler_close = joblib.load("models/scalers/scaler_close.pkl")

# Desnormaliza y_test para comparação em R$ (conforme padrão da Etapa 1, seção 3.7)
y_test_real = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()

# --- Baseline no espaço normalizado ---
y_pred_baseline_norm = np.roll(y_test, shift=1)
y_pred_baseline_norm[0] = y_test[0]

# --- Desnormalização ---
y_pred_baseline_real = scaler_close.inverse_transform(
    y_pred_baseline_norm.reshape(-1, 1)
).flatten()

print(f"y_pred_baseline_real — min: R${y_pred_baseline_real.min():.2f} | max: R${y_pred_baseline_real.max():.2f}")
print(f"Amostras no conjunto de teste: {len(y_test_real)}")
```

---

## 2.4 Serialização dos arrays

```python
np.save("models/baseline/y_pred_baseline.npy", y_pred_baseline_real)
np.save("models/baseline/y_test_real.npy",     y_test_real)

print("Arrays serializados em models/baseline/")
```

Esses arquivos serão reutilizados na Fase 3 para gerar o gráfico comparativo LSTM vs. baseline e calcular a melhoria relativa entre os modelos.

---

## 2.5 Recuperação do índice temporal do conjunto de teste

Para plotar o gráfico com eixo x em datas reais (não índices inteiros), é necessário recuperar as datas correspondentes ao `y_test`:

```python
n = len(df_processed)
train_size = int(n * TRAIN_RATIO)

# As primeiras WINDOW_SIZE linhas do split de teste são consumidas pela janela
# A primeira previsão corresponde ao índice train_size + WINDOW_SIZE
test_dates = df_processed.index[train_size + WINDOW_SIZE:]

assert len(test_dates) == len(y_test_real), (
    f"Incompatibilidade de tamanhos: datas={len(test_dates)}, y_test={len(y_test_real)}"
)

print(f"Período de teste: {test_dates[0].date()} → {test_dates[-1].date()}")
print(f"Total de pregões no teste: {len(test_dates)}")
```

---

## 2.6 Gráfico: Preço real vs. baseline

```python
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test_dates, y_test_real,        color="#1f77b4", linewidth=1.2,
        label="Preço Real (Fechamento)")
ax.plot(test_dates, y_pred_baseline_real, color="#ff7f0e", linewidth=1.0,
        linestyle="--", alpha=0.85, label="Baseline — Persistência (t+1 = t)")

ax.set_title("ITUB4 — Preço Real vs. Modelo de Persistência (Conjunto de Teste)")
ax.set_ylabel("Preço de Fechamento (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/figures/06_baseline_vs_real.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Critério de aceite:** as duas séries devem ser visualmente próximas (o baseline de persistência acompanha o preço com 1 dia de atraso), mas com desvios visíveis em reversões de tendência. Isso demonstra que o baseline captura parte do padrão, mas não antecipa mudanças.

---

## Resultado esperado

| Artefato | Localização | Descrição |
|---|---|---|
| `y_pred_baseline.npy` | `models/baseline/` | Previsões do baseline em reais (R$), shape `(n_test,)` |
| `y_test_real.npy` | `models/baseline/` | Valores reais do conjunto de teste em reais (R$), shape `(n_test,)` |
| `06_baseline_vs_real.png` | `data/figures/` | Gráfico comparativo preço real vs. persistência |

---

## Próximo passo

Com as previsões do baseline geradas e salvas, prosseguir para o [cálculo das métricas de avaliação](03_avaliacao_metricas.md).
