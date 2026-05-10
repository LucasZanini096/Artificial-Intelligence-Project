# Etapa 1 — Construção da Arquitetura LSTM

## Objetivo

Definir e compilar o modelo LSTM de camada única conforme a arquitetura especificada em [Bhandari et al. 2022] e adotada no artigo do projeto: Input(50, 6) → LSTM(150, return_sequences=False) → Dropout(0.20) → Dense(1, linear). Esta etapa não realiza treinamento; ela apenas instancia e valida o grafo computacional.

---

## 1.1 Pré-requisitos: carregamento dos tensores da Fase 2

O notebook desta fase deve recarregar os tensores e o scaler produzidos nas fases anteriores antes de construir o modelo.

```python
import numpy as np
import pandas as pd
import joblib
import os

# Constantes herdadas das fases anteriores
WINDOW_SIZE  = 50
N_FEATURES   = 6        # Open, High, Low, Close, Volume, EMA_60
TRAIN_RATIO  = 0.80

# Carrega o dataset processado para recuperar as datas do teste
df_processed = pd.read_csv(
    "data/processed/ITUB4_processed.csv",
    index_col="Date",
    parse_dates=True
)

# Recria os tensores a partir do dataset processado
FEATURES = ["Open", "High", "Low", "Close", "Volume", "EMA_60"]
TARGET   = "Close"

n          = len(df_processed)
train_size = int(n * TRAIN_RATIO)

df_train_scaled = df_processed.iloc[:train_size]
df_test_scaled  = df_processed.iloc[train_size:]

def create_windows(data: pd.DataFrame, window_size: int, target_col: str):
    X, y = [], []
    values     = data.values
    target_idx = data.columns.get_loc(target_col)
    for i in range(window_size, len(data)):
        X.append(values[i - window_size : i, :])
        y.append(values[i, target_idx])
    return np.array(X), np.array(y)

X_train, y_train = create_windows(df_train_scaled, WINDOW_SIZE, TARGET)
X_test,  y_test  = create_windows(df_test_scaled,  WINDOW_SIZE, TARGET)

# Carrega o scaler de Close para desnormalização posterior
scaler_close = joblib.load("models/scalers/scaler_close.pkl")

# Recupera as datas do conjunto de teste
test_dates = df_processed.index[train_size + WINDOW_SIZE:]

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")
print(f"Período de teste: {test_dates[0].date()} → {test_dates[-1].date()}")

assert X_train.shape[1] == WINDOW_SIZE,  "window_size incompatível"
assert X_train.shape[2] == N_FEATURES,   "n_features incompatível"
assert len(test_dates)  == len(y_test),  "incompatibilidade entre datas e y_test"
```

---

## 1.2 Definição da arquitetura

A arquitetura segue exatamente a Tabela 2 do artigo. O modelo é sequencial com quatro camadas:

| Camada | Configuração | Parâmetros treináveis |
|---|---|---|
| Input | shape=(50, 6) | 0 |
| LSTM | 150 unidades, `return_sequences=False` | 4 × 150 × (6 + 150 + 1) = 93.600 |
| Dropout | taxa=0.20 | 0 |
| Dense | 1 neurônio, ativação linear | 151 |

> **Por que `return_sequences=False`?** Com apenas uma camada LSTM, apenas o estado oculto final (`h_T`) é passado à camada Dense, produzindo um único valor de saída — o preço de fechamento previsto para t+1. O valor `True` seria necessário apenas se houvesse uma segunda camada LSTM empilhada acima.

> **Por que ativação linear na Dense?** A tarefa é regressão: a saída deve ser um valor contínuo no espaço Z-score normalizado, sem limitação de faixa. Ativações como sigmoid ou tanh restringiriam o intervalo de saída artificialmente.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

tf.random.set_seed(42)   # reprodutibilidade

model = Sequential([
    Input(shape=(WINDOW_SIZE, N_FEATURES)),
    LSTM(150, return_sequences=False),
    Dropout(0.20),
    Dense(1, activation="linear"),
], name="LSTM_ITUB4")
```

---

## 1.3 Compilação do modelo

```python
model.compile(
    loss="mse",        # Mean Squared Error — função de perda padrão para regressão
    optimizer="adam",  # Adam com lr=0.001 (padrão Keras)
    metrics=["mae"],   # MAE exibido durante o treino para monitoramento visual
)

model.summary()
```

**Saída esperada do `summary()`:**

```
Model: "LSTM_ITUB4"
_________________________________________________________________
 Layer (type)                Output Shape         Param #
=================================================================
 lstm (LSTM)                 (None, 150)          93,600
 dropout (Dropout)           (None, 150)          0
 dense (Dense)               (None, 1)            151
=================================================================
Total params: 93,751
Trainable params: 93,751
Non-trainable params: 0
_________________________________________________________________
```

**Por que MSE e Adam?**

- **MSE** penaliza erros grandes quadraticamente, alinhando a função de perda do treinamento com a métrica RMSE de avaliação. É o padrão para regressão em séries temporais financeiras [Bhandari et al. 2022].
- **Adam** adapta a taxa de aprendizado individualmente por parâmetro, convergindo mais rapidamente do que SGD em dados com gradientes esparsos ou de magnitude variável, característica comum em séries financeiras.

---

## 1.4 Verificação do grafo

```python
# Confirma que o modelo aceita o shape esperado sem erros
dummy_input = np.zeros((1, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
dummy_output = model.predict(dummy_input, verbose=0)

assert dummy_output.shape == (1, 1), (
    f"Shape de saída inesperado: {dummy_output.shape}"
)
print(f"Verificação de shape: OK — saída com shape {dummy_output.shape}")
print(f"Total de parâmetros treináveis: {model.count_params():,}")
```

---

## Resultado esperado

| Verificação | Critério de aceite |
|---|---|
| `model.summary()` exibe 3 camadas | LSTM, Dropout, Dense |
| Total de parâmetros | 93.751 |
| Shape de saída em inferência | `(batch_size, 1)` |
| Compilação sem erros | `model.optimizer` é instância de `Adam` |

---

## Próximo passo

Com o modelo construído e validado, prosseguir para o [treinamento com Early Stopping e ModelCheckpoint](02_treinamento.md).
