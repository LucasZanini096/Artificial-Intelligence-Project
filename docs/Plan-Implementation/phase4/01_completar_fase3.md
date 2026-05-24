# Etapa 1 — Completar Artefatos da Fase 3

## Objetivo

O modelo `lstm_itub4.keras` foi treinado na Fase 3, mas os artefatos de avaliação (`lstm_metrics.json`, `y_pred_lstm.npy` e as figuras 08–11) ainda não foram gerados. Esta etapa cria um notebook dedicado (`4-lstm_evaluation.ipynb`) que carrega o modelo salvo, executa a inferência sobre o conjunto de teste, calcula as métricas, serializa os resultados e produz os gráficos finais de análise — todos os artefatos que alimentam o Streamlit na Etapa 2.

> Esta etapa deve ser executada **antes** de iniciar a aplicação Streamlit, pois o app carrega os arquivos `.npy` e `.json` pré-computados em vez de reexecutar a inferência a cada acesso.

---

## 1.1 Configuração de caminhos e imports

```python
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import tensorflow as tf
from scipy.stats import norm

# Executar sempre a partir de src/
BASE_DIR    = os.path.abspath(".")          # deve ser src/
MODEL_DIR   = os.path.join(BASE_DIR, "models", "lstm")
BASELINE_DIR= os.path.join(BASE_DIR, "models", "baseline")
SCALERS_DIR = os.path.join(BASE_DIR, "models", "scalers")
DATA_DIR    = os.path.join(BASE_DIR, "database", "processed")
FIG_DIR     = os.path.join(BASE_DIR, "data", "figures")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"TensorFlow {tf.__version__}")
print(f"Modelo em: {os.path.join(MODEL_DIR, 'lstm_itub4.keras')}")
```

---

## 1.2 Carregamento do modelo e scalers

```python
# Modelo
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_itub4.keras"))
model.summary()

# Scaler do fechamento (necessário para desnormalização)
scaler_close = joblib.load(os.path.join(SCALERS_DIR, "scaler_close.pkl"))
print("Modelo e scaler carregados com sucesso.")
```

---

## 1.3 Reconstrução dos tensores de teste

O conjunto de teste precisa ser reconstruído a partir do CSV processado e dos scalers serializados na Fase 2, replicando exatamente o janelamento aplicado no notebook de treinamento.

```python
FEATURES  = ["Open", "High", "Low", "Close", "Volume", "EMA_60"]
WINDOW    = 50
TEST_FRAC = 0.20

# Lê dados processados na ordem cronológica original
df = pd.read_csv(os.path.join(DATA_DIR, "ITUB4_processed.csv"), parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Carrega todos os scalers e normaliza cada feature
scalers = {}
for feat in FEATURES:
    key = feat.lower().replace(" ", "_")
    # Mapeia nomes das colunas para nomes dos arquivos de scaler
    fname_map = {
        "open": "scaler_open", "high": "scaler_high", "low": "scaler_low",
        "close": "scaler_close", "volume": "scaler_volume", "ema_60": "scaler_ema_60"
    }
    scalers[feat] = joblib.load(os.path.join(SCALERS_DIR, f"{fname_map[key]}.pkl"))

# Normaliza usando os parâmetros do treino (sem data leakage)
df_scaled = df.copy()
for feat in FEATURES:
    df_scaled[feat] = scalers[feat].transform(df[[feat]])

# Divisão temporal 80/20
n_total  = len(df_scaled)
n_train  = int(n_total * (1 - TEST_FRAC))

data_array = df_scaled[FEATURES].values  # shape (n_total, 6)

# Função de janelamento
def create_windows(data: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])   # (window, n_features)
        y.append(data[i, 3])             # índice 3 = Close
    return np.array(X), np.array(y)

X_all, y_all = create_windows(data_array, WINDOW)

# Separação: amostras de treino e teste (sem embaralhamento)
split_idx = n_train - WINDOW   # ajuste pelo deslocamento do janelamento
X_test = X_all[split_idx:]
y_test = y_all[split_idx:]

# Datas do período de teste (sem as primeiras `window` linhas)
test_dates = df["Date"].values[n_train:]
test_dates = pd.to_datetime(test_dates)

print(f"X_test shape : {X_test.shape}")   # esperado (200, 50, 6)
print(f"y_test shape : {y_test.shape}")   # esperado (200,)
print(f"Período de teste: {test_dates[0].date()} → {test_dates[-1].date()}")
```

> **Validação:** `X_test.shape[0]` deve ser igual ao `n_samples` registrado em `baseline_metrics.json` (200).

---

## 1.4 Inferência e desnormalização

```python
# Previsões no espaço normalizado
y_pred_lstm_norm = model.predict(X_test, batch_size=32, verbose=0).flatten()

# Desnormalização para R$
y_pred_lstm_real = scaler_close.inverse_transform(
    y_pred_lstm_norm.reshape(-1, 1)
).flatten()

# Valores reais em R$ (carregados da Fase 2)
y_test_real = np.load(os.path.join(BASELINE_DIR, "y_test_real.npy"))

assert y_pred_lstm_real.shape == y_test_real.shape, (
    f"Shape incompatível: pred={y_pred_lstm_real.shape}, real={y_test_real.shape}"
)

print(f"LSTM — min: R${y_pred_lstm_real.min():.2f} | max: R${y_pred_lstm_real.max():.2f}")
print(f"Real  — min: R${y_test_real.min():.2f}       | max: R${y_test_real.max():.2f}")
```

---

## 1.5 Cálculo das métricas

```python
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

lstm_rmse = rmse(y_test_real, y_pred_lstm_real)
lstm_mae  = mae(y_test_real,  y_pred_lstm_real)
lstm_mape = mape(y_test_real, y_pred_lstm_real)

with open(os.path.join(BASELINE_DIR, "baseline_metrics.json")) as f:
    baseline_metrics = json.load(f)

baseline_rmse = baseline_metrics["RMSE"]
baseline_mae  = baseline_metrics["MAE"]
baseline_mape = baseline_metrics["MAPE"]

melhoria_rmse = (baseline_rmse - lstm_rmse) / baseline_rmse * 100
melhoria_mae  = (baseline_mae  - lstm_mae)  / baseline_mae  * 100
melhoria_mape = (baseline_mape - lstm_mape) / baseline_mape * 100

print("\n========================================")
print("  Comparativo LSTM vs. Baseline")
print("========================================")
print(f"  {'Métrica':<8} {'Baseline':>12} {'LSTM':>12} {'Melhoria':>10}")
print(f"  {'-'*46}")
print(f"  {'RMSE':<8} R$ {baseline_rmse:>8.4f} R$ {lstm_rmse:>8.4f} {melhoria_rmse:>+9.2f}%")
print(f"  {'MAE':<8} R$ {baseline_mae:>8.4f} R$ {lstm_mae:>8.4f}  {melhoria_mae:>+9.2f}%")
print(f"  {'MAPE':<8}    {baseline_mape:>8.4f}%    {lstm_mape:>8.4f}% {melhoria_mape:>+9.2f}%")
print("========================================")

# Critério de validação: LSTM deve superar baseline nas três métricas
assert lstm_rmse < baseline_rmse, "LSTM não superou baseline em RMSE"
assert lstm_mae  < baseline_mae,  "LSTM não superou baseline em MAE"
assert lstm_mape < baseline_mape, "LSTM não superou baseline em MAPE"
print("\nCritério atendido: LSTM supera baseline nas três métricas.")
```

---

## 1.6 Serialização dos artefatos

```python
# lstm_metrics.json
lstm_metrics_dict = {
    "model":        "lstm_single_layer",
    "architecture": "Input(50,6) → LSTM(150) → Dropout(0.2) → Dense(1)",
    "split":        "test",
    "n_samples":    int(len(y_test_real)),
    "RMSE":         round(lstm_rmse,  6),
    "MAE":          round(lstm_mae,   6),
    "MAPE":         round(lstm_mape,  6),
    "melhoria_vs_baseline": {
        "RMSE_pct": round(melhoria_rmse, 4),
        "MAE_pct":  round(melhoria_mae,  4),
        "MAPE_pct": round(melhoria_mape, 4),
    },
    "test_start": str(test_dates[0].date()),
    "test_end":   str(test_dates[-1].date()),
}

with open(os.path.join(MODEL_DIR, "lstm_metrics.json"), "w") as f:
    json.dump(lstm_metrics_dict, f, indent=2)

# y_pred_lstm.npy
np.save(os.path.join(MODEL_DIR, "y_pred_lstm.npy"), y_pred_lstm_real)

# test_dates.npy  ← necessário para o Streamlit reconstruir o eixo X
np.save(os.path.join(MODEL_DIR, "test_dates.npy"), test_dates.values.astype("datetime64[D]"))

print("Artefatos serializados:")
print(f"  models/lstm/lstm_metrics.json")
print(f"  models/lstm/y_pred_lstm.npy")
print(f"  models/lstm/test_dates.npy")
print(json.dumps(lstm_metrics_dict, indent=2))
```

> `test_dates.npy` é salvo para que o Streamlit possa reconstruir o eixo X sem precisar reler e reprocessar o CSV completo a cada inicialização.

---

## 1.7 Figura 08 — Curva de loss (treino vs. validação)

> Esta figura deve ter sido gerada durante o treinamento (Fase 3). Caso o arquivo `08_history_loss.png` já exista em `data/figures/`, pule esta subseção. Caso contrário, regere-a a partir do histórico salvo no objeto `History` do Keras — o que requer uma nova rodada de treinamento. Se o arquivo estiver ausente e o re-treinamento não for viável, documente a ausência e siga para a figura 09.

---

## 1.8 Figura 09 — Preço real vs. LSTM

```python
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, y_test_real,      color="#1f77b4", linewidth=1.2,
        label="Preço Real (Fechamento)")
ax.plot(test_dates, y_pred_lstm_real, color="#d62728", linewidth=1.0,
        alpha=0.85, label="LSTM — Previsão (t+1)")
ax.set_title("ITUB4 — Preço Real vs. Previsão LSTM (Conjunto de Teste)")
ax.set_ylabel("Preço de Fechamento (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_lstm_vs_real.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Gráfico salvo: 09_lstm_vs_real.png")
```

---

## 1.9 Figura 10 — LSTM vs. Baseline vs. Real

```python
y_pred_baseline_real = np.load(os.path.join(BASELINE_DIR, "y_pred_baseline.npy"))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, y_test_real,          color="#1f77b4", linewidth=1.2,
        label="Preço Real")
ax.plot(test_dates, y_pred_lstm_real,     color="#d62728", linewidth=1.0,
        alpha=0.85, label=f"LSTM  (MAPE={lstm_mape:.2f}%)")
ax.plot(test_dates, y_pred_baseline_real, color="#ff7f0e", linewidth=0.9,
        linestyle="--", alpha=0.75, label=f"Baseline (MAPE={baseline_mape:.2f}%)")
ax.set_title("ITUB4 — Comparativo: Preço Real vs. LSTM vs. Baseline")
ax.set_ylabel("Preço de Fechamento (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_lstm_vs_baseline.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Gráfico salvo: 10_lstm_vs_baseline.png")
```

---

## 1.10 Figura 11 — Resíduos da LSTM

```python
residuos_lstm = y_test_real - y_pred_lstm_real
mu_res    = residuos_lstm.mean()
sigma_res = residuos_lstm.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Série temporal
axes[0].plot(test_dates, residuos_lstm, color="#d62728", linewidth=0.7, alpha=0.85)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Resíduos da LSTM ao Longo do Tempo")
axes[0].set_ylabel("Resíduo (R$)")
axes[0].set_xlabel("Data")
axes[0].grid(alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[0].get_xticklabels(), rotation=45)

# Histograma com normal teórica
x_res = np.linspace(mu_res - 4 * sigma_res, mu_res + 4 * sigma_res, 200)
axes[1].hist(residuos_lstm, bins=40, density=True,
             color="#f7b6b6", edgecolor="white", alpha=0.8, label="Empírico")
axes[1].plot(x_res, norm.pdf(x_res, mu_res, sigma_res), "r-", linewidth=1.5,
             label="Normal teórica")
axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Distribuição dos Resíduos — LSTM")
axes[1].set_xlabel("Resíduo (R$)")
axes[1].set_ylabel("Densidade")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_lstm_residuos.png"), dpi=300, bbox_inches="tight")
plt.show()

print(f"Resíduos LSTM — µ: {mu_res:.4f} R$ | σ: {sigma_res:.4f} R$")
print(f"Curtose: {pd.Series(residuos_lstm).kurt():.4f}")
print("Gráfico salvo: 11_lstm_residuos.png")
```

---

## Checklist da Etapa 1

- [ ] `lstm_itub4.keras` carregado sem erros
- [ ] `X_test` shape `(200, 50, 6)` confirmado
- [ ] Inferência executada e `y_pred_lstm_real` shape `(200,)` confirmado
- [ ] LSTM supera baseline nas três métricas (asserts passaram)
- [ ] `lstm_metrics.json` salvo em `models/lstm/`
- [ ] `y_pred_lstm.npy` salvo em `models/lstm/`
- [ ] `test_dates.npy` salvo em `models/lstm/`
- [ ] Figuras `09`, `10` e `11` geradas em `data/figures/`
- [ ] Notebook `4-lstm_evaluation.ipynb` executado do início ao fim sem erros
