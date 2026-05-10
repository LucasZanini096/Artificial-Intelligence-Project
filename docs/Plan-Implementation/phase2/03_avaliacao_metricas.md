# Etapa 3 — Avaliação e Métricas do Baseline

## Objetivo

Calcular as três métricas de avaliação (RMSE, MAE, MAPE) sobre os valores desnormalizados do conjunto de teste, gerar o gráfico de distribuição dos resíduos e serializar os resultados em `baseline_metrics.json`. Esses números estabelecem o piso que a LSTM deverá superar na Fase 3.

---

## 3.1 Cálculo das métricas

Todas as métricas são calculadas sobre os valores em reais (R$), **não** sobre os valores normalizados. Isso permite interpretação direta do erro e comparação com a escala real do ativo.

```python
import numpy as np
import json

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Exclui pontos onde y_true == 0 para evitar divisão por zero
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


baseline_rmse = rmse(y_test_real, y_pred_baseline_real)
baseline_mae  = mae(y_test_real,  y_pred_baseline_real)
baseline_mape = mape(y_test_real, y_pred_baseline_real)

print("=== Métricas do Baseline (Persistência) ===")
print(f"  RMSE : R$ {baseline_rmse:.4f}")
print(f"  MAE  : R$ {baseline_mae:.4f}")
print(f"  MAPE :    {baseline_mape:.4f} %")
```

### Interpretação esperada

| Métrica | Interpretação | Valor típico para ITUB4 |
|---|---|---|
| RMSE | Erro quadrático médio em R$; penaliza erros grandes | R$ 0,30 – R$ 0,60 |
| MAE | Erro absoluto médio em R$; mais robusto a outliers | R$ 0,20 – R$ 0,45 |
| MAPE | Erro percentual médio; independente da escala de preços | 0,8% – 2,0% |

> Os intervalos acima são referências orientativas baseadas em [Zanotto and Hölbig 2026]. Os valores reais dependerão da volatilidade do período de teste capturado pelo download.

---

## 3.2 Serialização das métricas

```python
import os

os.makedirs("models/baseline", exist_ok=True)

metrics = {
    "model":      "persistence_baseline",
    "split":      "test",
    "n_samples":  int(len(y_test_real)),
    "RMSE":       round(baseline_rmse, 6),
    "MAE":        round(baseline_mae,  6),
    "MAPE":       round(baseline_mape, 6),
}

with open("models/baseline/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Métricas salvas em models/baseline/baseline_metrics.json")
print(json.dumps(metrics, indent=2))
```

O arquivo JSON será lido na Fase 3 para compor a tabela comparativa LSTM vs. baseline no notebook e no artigo.

---

## 3.3 Gráfico de resíduos do baseline

Os resíduos (`y_real - y_pred`) revelam o padrão de erro do modelo de persistência: em tendência de alta, os resíduos serão sistematicamente positivos; em queda, negativos.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

residuos = y_test_real - y_pred_baseline_real

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Série temporal dos resíduos ---
axes[0].plot(test_dates, residuos, color="#2ca02c", linewidth=0.7, alpha=0.85)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Resíduos do Baseline ao Longo do Tempo")
axes[0].set_ylabel("Resíduo (R$)")
axes[0].set_xlabel("Data")
axes[0].grid(alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[0].get_xticklabels(), rotation=45)

# --- Histograma dos resíduos com curva normal ---
mu_res, sigma_res = residuos.mean(), residuos.std()
x_res = np.linspace(mu_res - 4 * sigma_res, mu_res + 4 * sigma_res, 200)

axes[1].hist(residuos, bins=40, density=True,
             color="#aec7e8", edgecolor="white", alpha=0.8, label="Empírico")
axes[1].plot(x_res, norm.pdf(x_res, mu_res, sigma_res),
             "r-", linewidth=1.5, label="Normal teórica")
axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Distribuição dos Resíduos — Baseline")
axes[1].set_xlabel("Resíduo (R$)")
axes[1].set_ylabel("Densidade")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/07_baseline_residuos.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Resíduos — µ: {mu_res:.4f} | σ: {sigma_res:.4f}")
print(f"Curtose: {pd.Series(residuos).kurt():.4f}")
```

**Critério de aceite:**
- Resíduos centrados próximos de zero (o modelo não possui viés sistemático de longo prazo).
- Distribuição aproximadamente normal com caudas pesadas (leptocúrtica), característica de séries financeiras.
- Série temporal dos resíduos sem padrão estrutural claro (ausência de autocorrelação visível indicaria que o baseline não deixou informação aproveitável).

---

## 3.4 Resumo final para o artigo

```python
print("\n========================================")
print("  Resumo — Fase 2: Baseline de Persistência")
print("========================================")
print(f"  Período de teste : {test_dates[0].date()} → {test_dates[-1].date()}")
print(f"  Amostras         : {len(y_test_real)}")
print(f"  RMSE             : R$ {baseline_rmse:.4f}")
print(f"  MAE              : R$ {baseline_mae:.4f}")
print(f"  MAPE             : {baseline_mape:.4f} %")
print("========================================\n")
print("Estes valores são o piso mínimo que a LSTM deverá superar na Fase 3.")
```

---

## Resultado esperado

| Artefato | Localização | Descrição |
|---|---|---|
| `baseline_metrics.json` | `models/baseline/` | RMSE, MAE, MAPE do baseline em formato estruturado |
| `07_baseline_residuos.png` | `data/figures/` | Série temporal e histograma dos resíduos |

---

## Checklist final da Fase 2

- [ ] `y_pred_baseline.npy` gerado com shape idêntico a `y_test_real.npy`
- [ ] RMSE, MAE, MAPE calculados **exclusivamente** sobre valores desnormalizados (em R$)
- [ ] `baseline_metrics.json` salvo e legível por `json.load()`
- [ ] `06_baseline_vs_real.png` e `07_baseline_residuos.png` salvos em `data/figures/`
- [ ] Notebook `3-data_preparation_baseline.ipynb` executado sem erros do início ao fim
- [ ] Valores de MAPE dentro da faixa plausível (< 5% para dados sem eventos exógenos extremos)

---

## Próximo passo

Com o baseline avaliado e as métricas registradas, prosseguir para a **Fase 3** — Modelagem e Treinamento da LSTM, onde a arquitetura de camada única (150 neurônios, Dropout 20%) será construída, treinada com *Early Stopping* e *ModelCheckpoint*, e comparada diretamente com os resultados desta fase.
