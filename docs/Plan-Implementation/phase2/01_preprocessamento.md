# Etapa 3 — Pré-processamento dos Dados

## Objetivo

Transformar o dataset bruto em tensores prontos para alimentar o modelo LSTM, executando: tratamento de ausentes, divisão cronológica 80/20, normalização Z-score (sem data leakage) e janelamento temporal de 50 dias. Serializar os scalers para uso na inferência do Streamlit.

---

## 3.1 Carregamento e seleção de features

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

df = pd.read_csv("data/raw/ITUB4_raw.csv", index_col="Date", parse_dates=True)

FEATURES = ["Open", "High", "Low", "Close", "Volume", "EMA_60"]
TARGET = "Close"
WINDOW_SIZE = 50
TRAIN_RATIO = 0.80

df = df[FEATURES].copy()
print(f"Shape após seleção: {df.shape}")
```

---

## 3.2 Tratamento de valores ausentes

Dois casos distintos exigem tratamentos diferentes:

```python
# Preços ausentes (NaN): interpolação linear preserva a continuidade da série
price_cols = ["Open", "High", "Low", "Close", "EMA_60"]
df[price_cols] = df[price_cols].interpolate(method="linear", limit_direction="both")

# Volume zerado: substituir pelo último valor válido (forward fill)
df["Volume"] = df["Volume"].replace(0, np.nan)
df["Volume"] = df["Volume"].fillna(method="ffill")

# Verificação final — não deve sobrar nenhum NaN
assert df.isnull().sum().sum() == 0, "Ainda há valores ausentes após tratamento!"
print("Tratamento de ausentes concluído. Nenhum NaN restante.")
print(df.shape)
```

**Por que não usar `dropna()`?** Remover linhas quebraria a continuidade temporal da série, o que invalidaria o janelamento subsequente.

---

## 3.3 Divisão cronológica treino / teste

A divisão **nunca embaralha** os dados — a ordem temporal deve ser preservada para simular o cenário real de previsão.

```python
n = len(df)
train_size = int(n * TRAIN_RATIO)

df_train = df.iloc[:train_size]
df_test  = df.iloc[train_size:]

print(f"Total: {n} dias")
print(f"Treino: {len(df_train)} dias  ({df_train.index[0].date()} → {df_train.index[-1].date()})")
print(f"Teste:  {len(df_test)}  dias  ({df_test.index[0].date()} → {df_test.index[-1].date()})")
```

> **Importante:** os parâmetros dos scalers são estimados **exclusivamente** no conjunto de treino e depois aplicados ao teste. Usar o conjunto completo para calcular `mean` e `std` vaza informação do futuro (data leakage).

---

## 3.4 Normalização Z-score por feature

Cada uma das 6 features é normalizada independentemente para respeitar as diferentes escalas (preços em R$ 20–40 vs. volume em dezenas de milhões).

```python
os.makedirs("models/scalers", exist_ok=True)

scalers = {}
df_train_scaled = df_train.copy()
df_test_scaled  = df_test.copy()

for col in FEATURES:
    scaler = StandardScaler()
    # Fit apenas no treino
    df_train_scaled[col] = scaler.fit_transform(df_train[[col]])
    # Transform no teste com os parâmetros do treino
    df_test_scaled[col]  = scaler.transform(df_test[[col]])

    scalers[col] = scaler
    joblib.dump(scaler, f"models/scalers/scaler_{col.lower()}.pkl")
    print(f"  {col}: µ={scaler.mean_[0]:.4f}, σ={scaler.scale_[0]:.4f}")

print("\nScalers serializados em models/scalers/")
```

### Verificação da normalização

```python
print("=== Treino (deve ter µ≈0, σ≈1) ===")
print(df_train_scaled.describe().loc[["mean", "std"]].round(4))

print("\n=== Teste (µ e σ podem diferir ligeiramente) ===")
print(df_test_scaled.describe().loc[["mean", "std"]].round(4))
```

---

## 3.5 Janelamento temporal

Converte o DataFrame 2D em tensores 3D `[amostras, passos_de_tempo, features]` exigidos pela LSTM.

```python
def create_windows(data: pd.DataFrame, window_size: int, target_col: str):
    """
    Retorna X com shape (n_samples, window_size, n_features)
    e y com shape (n_samples,) contendo o valor de fechamento do dia seguinte.
    """
    X, y = [], []
    values = data.values
    target_idx = data.columns.get_loc(target_col)

    for i in range(window_size, len(data)):
        X.append(values[i - window_size : i, :])   # janela de 50 dias
        y.append(values[i, target_idx])             # fechamento do dia seguinte

    return np.array(X), np.array(y)


X_train, y_train = create_windows(df_train_scaled, WINDOW_SIZE, TARGET)
X_test,  y_test  = create_windows(df_test_scaled,  WINDOW_SIZE, TARGET)

print(f"X_train: {X_train.shape}  →  (amostras, janela, features)")
print(f"y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")
```

**Shapes esperados (aproximados para ~1.250 dias):**

| Tensor | Shape esperado |
|---|---|
| `X_train` | `(~950, 50, 6)` |
| `y_train` | `(~950,)` |
| `X_test` | `(~200, 50, 6)` |
| `y_test` | `(~200,)` |

> O número de amostras é `total_dias_na_split - WINDOW_SIZE`, porque as primeiras 50 linhas de cada split são consumidas pela janela inicial.

---

## 3.6 Exportação do dataset processado

Salvar o DataFrame normalizado (sem janelamento) para referência e debug futuro:

```python
df_processed = pd.concat([df_train_scaled, df_test_scaled])
df_processed.to_csv("data/processed/ITUB4_processed.csv", index=True)
print(f"Dataset processado salvo: data/processed/ITUB4_processed.csv ({len(df_processed)} linhas)")
```

---

## 3.7 Como reverter a normalização (desnormalização)

As métricas RMSE, MAE e MAPE serão calculadas sobre os valores em reais (não normalizados). Para isso, use o scaler de `Close` guardado:

```python
scaler_close = joblib.load("models/scalers/scaler_close.pkl")

y_test_real  = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()
# y_pred_real = scaler_close.inverse_transform(y_pred.reshape(-1, 1)).flatten()
```

Esse padrão será reutilizado na Fase 3 durante a avaliação do modelo.

---

## Resultado esperado

| Artefato | Localização | Descrição |
|---|---|---|
| `ITUB4_processed.csv` | `data/processed/` | Dataset Z-score normalizado sem NaN |
| `scaler_open.pkl` | `models/scalers/` | StandardScaler da feature Open |
| `scaler_high.pkl` | `models/scalers/` | StandardScaler da feature High |
| `scaler_low.pkl` | `models/scalers/` | StandardScaler da feature Low |
| `scaler_close.pkl` | `models/scalers/` | StandardScaler da feature Close (usado na desnorm.) |
| `scaler_volume.pkl` | `models/scalers/` | StandardScaler da feature Volume |
| `scaler_ema_60.pkl` | `models/scalers/` | StandardScaler da feature EMA_60 |
| `X_train`, `y_train` | Memória / notebook | Tensores prontos para treino da LSTM |
| `X_test`, `y_test` | Memória / notebook | Tensores para avaliação da LSTM |

---

## Checklist final da Fase 1

- [ ] Nenhum NaN em `ITUB4_processed.csv`
- [ ] Divisão 80/20 cronológica confirmada (treino antes, teste depois)
- [ ] Scalers fitted **apenas** no treino (`scaler.fit_transform` no treino, `scaler.transform` no teste)
- [ ] 6 arquivos `.pkl` em `models/scalers/`
- [ ] `X_train.shape[1] == 50` e `X_train.shape[2] == 6`
- [ ] `y_train` e `y_test` contêm valores normalizados do fechamento

---

## Próximo passo

Com os tensores `X_train`, `y_train`, `X_test`, `y_test` e os scalers disponíveis, prosseguir para a **Fase 2** — Preparação do Baseline e implementação do modelo de persistência.
