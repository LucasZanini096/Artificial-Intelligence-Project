# Etapa 3 — Avaliação Comparativa LSTM vs. Baseline

## Objetivo

Gerar as previsões da LSTM sobre o conjunto de teste, calcular RMSE, MAE e MAPE sobre os valores desnormalizados (em R$), comparar as métricas com o baseline de persistência registrado na Fase 2, serializar os resultados em `lstm_metrics.json` e gerar os três gráficos finais de análise. Esta etapa produz todos os artefatos necessários para a Fase 4 (Streamlit e artigo final).

---

## 3.1 Geração das previsões no conjunto de teste

```python
import numpy as np
import json
import os

# Geração das previsões (espaço normalizado)
y_pred_lstm_norm = model.predict(X_test, batch_size=32, verbose=0).flatten()

# Desnormalização para R$
y_pred_lstm_real = scaler_close.inverse_transform(
    y_pred_lstm_norm.reshape(-1, 1)
).flatten()

# Carrega os valores reais desnormalizados da Fase 2
y_test_real = np.load("models/baseline/y_test_real.npy")

print(f"Previsões LSTM — min: R${y_pred_lstm_real.min():.2f} | max: R${y_pred_lstm_real.max():.2f}")
print(f"Valores reais  — min: R${y_test_real.min():.2f}       | max: R${y_test_real.max():.2f}")
print(f"Amostras no teste: {len(y_test_real)}")

assert y_pred_lstm_real.shape == y_test_real.shape, (
    f"Shape incompatível: pred={y_pred_lstm_real.shape}, real={y_test_real.shape}"
)
```

---

## 3.2 Cálculo das métricas da LSTM

As mesmas funções utilizadas na Fase 2 são reutilizadas, garantindo comparabilidade direta entre os resultados.

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


lstm_rmse = rmse(y_test_real, y_pred_lstm_real)
lstm_mae  = mae(y_test_real,  y_pred_lstm_real)
lstm_mape = mape(y_test_real, y_pred_lstm_real)

print("=== Métricas da LSTM ===")
print(f"  RMSE : R$ {lstm_rmse:.4f}")
print(f"  MAE  : R$ {lstm_mae:.4f}")
print(f"  MAPE :    {lstm_mape:.4f} %")
```

---

## 3.3 Carregamento das métricas do baseline e tabela comparativa

```python
with open("models/baseline/baseline_metrics.json") as f:
    baseline_metrics = json.load(f)

baseline_rmse = baseline_metrics["RMSE"]
baseline_mae  = baseline_metrics["MAE"]
baseline_mape = baseline_metrics["MAPE"]

# Melhoria relativa da LSTM sobre o baseline (%)
melhoria_rmse = (baseline_rmse - lstm_rmse) / baseline_rmse * 100
melhoria_mae  = (baseline_mae  - lstm_mae)  / baseline_mae  * 100
melhoria_mape = (baseline_mape - lstm_mape) / baseline_mape * 100

print("\n========================================")
print("  Comparativo LSTM vs. Baseline")
print("========================================")
print(f"  {'Métrica':<8} {'Baseline':>12} {'LSTM':>12} {'Melhoria':>10}")
print(f"  {'-'*46}")
print(f"  {'RMSE':<8} R$ {baseline_rmse:>9.4f} R$ {lstm_rmse:>9.4f} {melhoria_rmse:>+9.2f}%")
print(f"  {'MAE':<8} R$ {baseline_mae:>9.4f} R$ {lstm_mae:>9.4f}  {melhoria_mae:>+9.2f}%")
print(f"  {'MAPE':<8}    {baseline_mape:>9.4f}%    {lstm_mape:>9.4f}% {melhoria_mape:>+9.2f}%")
print("========================================")

# Critério de validação: LSTM deve superar baseline nas três métricas
assert lstm_rmse < baseline_rmse, "LSTM não superou baseline em RMSE"
assert lstm_mae  < baseline_mae,  "LSTM não superou baseline em MAE"
assert lstm_mape < baseline_mape, "LSTM não superou baseline em MAPE"
print("\nCritério de validação: LSTM supera baseline nas três métricas. OK")
```

### Faixas de referência esperadas

| Métrica | Baseline (ref.) | LSTM esperado | Fonte |
|---|---|---|---|
| RMSE | R$ 0,30 – R$ 0,60 | R$ 0,10 – R$ 0,35 | [Zanotto and Hölbig 2026] |
| MAE | R$ 0,20 – R$ 0,45 | R$ 0,08 – R$ 0,25 | [Zanotto and Hölbig 2026] |
| MAPE | 0,8% – 2,0% | < 2,0% | [Bhandari et al. 2022] |

> Os intervalos são orientativos. Os valores reais dependem da volatilidade do período de teste capturado no download.

---

## 3.4 Serialização das métricas da LSTM

```python
os.makedirs("models/lstm", exist_ok=True)

lstm_metrics_dict = {
    "model":              "lstm_single_layer",
    "architecture":       "Input(50,6) → LSTM(150) → Dropout(0.2) → Dense(1)",
    "split":              "test",
    "n_samples":          int(len(y_test_real)),
    "best_epoch":         int(best_epoch),
    "RMSE":               round(lstm_rmse,  6),
    "MAE":                round(lstm_mae,   6),
    "MAPE":               round(lstm_mape,  6),
    "melhoria_vs_baseline": {
        "RMSE_pct": round(melhoria_rmse, 4),
        "MAE_pct":  round(melhoria_mae,  4),
        "MAPE_pct": round(melhoria_mape, 4),
    },
}

with open("models/lstm/lstm_metrics.json", "w") as f:
    json.dump(lstm_metrics_dict, f, indent=2)

print("Métricas salvas em models/lstm/lstm_metrics.json")
print(json.dumps(lstm_metrics_dict, indent=2))
```

---

## 3.5 Serialização das previsões da LSTM

```python
np.save("models/lstm/y_pred_lstm.npy", y_pred_lstm_real)
print("Previsões salvas em models/lstm/y_pred_lstm.npy")
```

Este arquivo será reutilizado na Fase 4 pelo Streamlit para exibir o gráfico interativo sem precisar re-executar a inferência a cada acesso.

---

## 3.6 Gráfico 1 — Preço real vs. LSTM

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test_dates, y_test_real,       color="#1f77b4", linewidth=1.2,
        label="Preço Real (Fechamento)")
ax.plot(test_dates, y_pred_lstm_real,  color="#d62728", linewidth=1.0,
        linestyle="-", alpha=0.85, label="LSTM — Previsão (t+1)")

ax.set_title("ITUB4 — Preço Real vs. Previsão LSTM (Conjunto de Teste)")
ax.set_ylabel("Preço de Fechamento (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/figures/09_lstm_vs_real.png", dpi=300, bbox_inches="tight")
plt.show()
print("Gráfico salvo em data/figures/09_lstm_vs_real.png")
```

**Critério de aceite:** as curvas devem ser visualmente sobrepostas, com pequenas defasagens nas reversões de tendência. Divergências acentuadas e persistentes indicam overfitting ou data leakage.

---

## 3.7 Gráfico 2 — LSTM vs. Baseline vs. Real

```python
y_pred_baseline_real = np.load("models/baseline/y_pred_baseline.npy")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(test_dates, y_test_real,            color="#1f77b4", linewidth=1.2,
        label="Preço Real")
ax.plot(test_dates, y_pred_lstm_real,       color="#d62728", linewidth=1.0,
        alpha=0.85, label=f"LSTM  (MAPE={lstm_mape:.2f}%)")
ax.plot(test_dates, y_pred_baseline_real,   color="#ff7f0e", linewidth=0.9,
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
plt.savefig("data/figures/10_lstm_vs_baseline.png", dpi=300, bbox_inches="tight")
plt.show()
print("Gráfico salvo em data/figures/10_lstm_vs_baseline.png")
```

**Critério de aceite:** a curva da LSTM deve estar visualmente mais próxima da curva real do que a curva do baseline, especialmente em pontos de reversão de tendência.

---

## 3.8 Gráfico 3 — Resíduos da LSTM

```python
import pandas as pd
from scipy.stats import norm

residuos_lstm = y_test_real - y_pred_lstm_real

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Série temporal dos resíduos ---
axes[0].plot(test_dates, residuos_lstm, color="#d62728", linewidth=0.7, alpha=0.85)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Resíduos da LSTM ao Longo do Tempo")
axes[0].set_ylabel("Resíduo (R$)")
axes[0].set_xlabel("Data")
axes[0].grid(alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[0].get_xticklabels(), rotation=45)

# --- Histograma com curva normal ---
mu_res    = residuos_lstm.mean()
sigma_res = residuos_lstm.std()
x_res     = np.linspace(mu_res - 4 * sigma_res, mu_res + 4 * sigma_res, 200)

axes[1].hist(residuos_lstm, bins=40, density=True,
             color="#f7b6b6", edgecolor="white", alpha=0.8, label="Empírico")
axes[1].plot(x_res, norm.pdf(x_res, mu_res, sigma_res),
             "r-", linewidth=1.5, label="Normal teórica")
axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Distribuição dos Resíduos — LSTM")
axes[1].set_xlabel("Resíduo (R$)")
axes[1].set_ylabel("Densidade")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/11_lstm_residuos.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Resíduos LSTM — µ: {mu_res:.4f} | σ: {sigma_res:.4f}")
print(f"Curtose: {pd.Series(residuos_lstm).kurt():.4f}")
print("Gráfico salvo em data/figures/11_lstm_residuos.png")
```

**Critério de aceite:** os resíduos da LSTM devem ter desvio padrão (σ) menor do que os do baseline (Fase 2), confirmando a redução de erro. A média (µ) deve permanecer próxima de zero (ausência de viés sistemático).

---

## 3.9 Resumo final para o artigo

```python
print("\n========================================")
print("  Resumo — Fase 3: Treinamento da LSTM")
print("========================================")
print(f"  Período de teste : {test_dates[0].date()} → {test_dates[-1].date()}")
print(f"  Amostras         : {len(y_test_real)}")
print(f"  Melhor época     : {best_epoch}")
print("")
print(f"  {'Métrica':<8} {'Baseline':>12} {'LSTM':>12} {'Δ':>10}")
print(f"  {'-'*46}")
print(f"  {'RMSE':<8} R$ {baseline_rmse:>9.4f} R$ {lstm_rmse:>9.4f} {melhoria_rmse:>+9.2f}%")
print(f"  {'MAE':<8} R$ {baseline_mae:>9.4f} R$ {lstm_mae:>9.4f}  {melhoria_mae:>+9.2f}%")
print(f"  {'MAPE':<8}    {baseline_mape:>9.4f}%    {lstm_mape:>9.4f}% {melhoria_mape:>+9.2f}%")
print("========================================\n")
print("Artefatos prontos para a Fase 4 (Streamlit + artigo final).")
```

---

## Resultado esperado

| Artefato | Localização | Descrição |
|---|---|---|
| `lstm_metrics.json` | `models/lstm/` | RMSE, MAE, MAPE da LSTM + melhoria vs. baseline |
| `y_pred_lstm.npy` | `models/lstm/` | Previsões da LSTM em R$, shape `(n_test,)` |
| `09_lstm_vs_real.png` | `data/figures/` | Preço real vs. previsão LSTM |
| `10_lstm_vs_baseline.png` | `data/figures/` | Comparativo LSTM vs. baseline vs. real |
| `11_lstm_residuos.png` | `data/figures/` | Série temporal e histograma dos resíduos da LSTM |

---

## Checklist final da Fase 3

- [ ] `y_pred_lstm_real` gerado com shape idêntico a `y_test_real`
- [ ] RMSE, MAE e MAPE calculados sobre valores em R$ (desnormalizados)
- [ ] LSTM supera baseline nas três métricas (asserts passaram)
- [ ] `lstm_metrics.json` salvo com campo `melhoria_vs_baseline`
- [ ] `y_pred_lstm.npy` serializado em `models/lstm/`
- [ ] `08_history_loss.png` sem sinais de overfitting severo
- [ ] `09_lstm_vs_real.png`, `10_lstm_vs_baseline.png` e `11_lstm_residuos.png` salvos
- [ ] σ dos resíduos da LSTM menor do que σ dos resíduos do baseline
- [ ] MAPE dentro da faixa esperada (< 2%)
- [ ] Notebook `03_lstm_training.ipynb` executado sem erros do início ao fim

---

## Próximo passo

Com o modelo treinado, avaliado e todos os artefatos serializados, prosseguir para a **Fase 4** — Protótipo Web e Relatório Final, onde o modelo será integrado ao Streamlit (`models/lstm/lstm_itub4.h5` + `models/scalers/`) e as seções de Resultados e Conclusão do artigo serão redigidas com base nas métricas obtidas nesta fase.
