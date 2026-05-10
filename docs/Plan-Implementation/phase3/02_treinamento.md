# Etapa 2 — Treinamento com Early Stopping e ModelCheckpoint

## Objetivo

Treinar o modelo LSTM utilizando os tensores `X_train` / `y_train` com um split interno de validação (10% do treino), os callbacks Early Stopping e ModelCheckpoint, e serializar o melhor modelo em `models/lstm/lstm_itub4.h5`. Ao final, gerar o gráfico de curvas de loss para diagnóstico de overfitting.

---

## 2.1 Criação do diretório de saída

```python
import os

os.makedirs("models/lstm", exist_ok=True)
```

---

## 2.2 Hiperparâmetros de treinamento

```python
EPOCHS      = 150    # teto; Early Stopping interromperá antes se convergir
BATCH_SIZE  = 32     # padrão amplamente utilizado em séries temporais com LSTM
VAL_SPLIT   = 0.10   # 10% do treino reservado para validação interna
```

**Justificativas:**

- **`BATCH_SIZE = 32`**: lotes menores produzem gradientes mais ruidosos mas tendem a generalizar melhor; 32 é o valor padrão em estudos semelhantes [Bhandari et al. 2022].
- **`VAL_SPLIT = 0.10`**: a validação é extraída das últimas 10% amostras do treino (Keras preserva a ordem cronológica ao usar `validation_split`, sem embaralhamento). Isso garante que o Early Stopping monitore o desempenho em dados cronologicamente posteriores ao treino efetivo — sem contaminar o conjunto de teste.
- **`EPOCHS = 150`**: teto suficientemente alto para que o Early Stopping atue antes do limite; com paciência 10, o treinamento máximo real é de ~60–90 épocas em séries desta escala.

> **Importante:** `validation_split` no Keras usa as últimas `k%` amostras **na ordem em que aparecem no array**. Como `X_train` preserva a ordem cronológica, essa fração corresponde ao período mais recente do treino — comportamento desejável para validação de séries temporais.

---

## 2.3 Configuração dos callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor="val_loss",      # métrica monitorada: perda no conjunto de validação
    patience=10,             # encerra após 10 épocas sem melhora
    restore_best_weights=True,  # restaura automaticamente os pesos da melhor época
    verbose=1,
)

checkpoint = ModelCheckpoint(
    filepath="models/lstm/lstm_itub4.h5",
    monitor="val_loss",
    save_best_only=True,     # salva apenas quando val_loss melhora
    verbose=1,
)
```

**Por que `restore_best_weights=True`?**  
Sem essa opção, o modelo ao final do treinamento possui os pesos da última época — que pode ter `val_loss` maior do que o mínimo atingido. Com `restore_best_weights=True`, o estado ótimo é restaurado automaticamente, dispensando um `load_model` adicional imediato após o `fit`.

**Por que `patience=10`?**  
Conforme o artigo (seção 3.5), a paciência de 10 épocas é o valor adotado. Esse valor evita paradas prematuras por flutuações normais de `val_loss` em séries financeiras ruidosas, ao mesmo tempo que previne overfitting prolongado.

---

## 2.4 Execução do treinamento

```python
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[early_stop, checkpoint],
    shuffle=False,   # preserva a ordem temporal — NÃO embaralhar séries temporais
    verbose=1,
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"\nMelhor época: {best_epoch}")
print(f"val_loss mínimo : {min(history.history['val_loss']):.6f}")
print(f"train_loss correspondente: {history.history['loss'][best_epoch - 1]:.6f}")
```

> **`shuffle=False` é obrigatório.** Embaralhar os lotes quebraria a dependência temporal entre as janelas, invalidando o aprendizado de padrões sequenciais pela LSTM. O Keras embaralha por padrão (`shuffle=True`) — é necessário sobrescrever explicitamente.

---

## 2.5 Carregamento do melhor modelo

Mesmo com `restore_best_weights=True` no EarlyStopping, é boa prática carregar o arquivo `.h5` salvo pelo ModelCheckpoint para garantir reprodutibilidade entre sessões:

```python
from tensorflow.keras.models import load_model

model = load_model("models/lstm/lstm_itub4.h5")
print("Melhor modelo carregado de models/lstm/lstm_itub4.h5")
```

---

## 2.6 Gráfico de curvas de loss

O gráfico de histórico de treinamento é o principal diagnóstico de overfitting/underfitting. Ele deve ser gerado imediatamente após o `fit` enquanto o objeto `history` está disponível.

```python
import matplotlib.pyplot as plt

epochs_range = range(1, len(history.history["loss"]) + 1)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(epochs_range, history.history["loss"],
        color="#1f77b4", linewidth=1.2, label="Loss — Treino")
ax.plot(epochs_range, history.history["val_loss"],
        color="#ff7f0e", linewidth=1.2, linestyle="--", label="Loss — Validação")

ax.axvline(best_epoch, color="gray", linewidth=0.8, linestyle=":",
           label=f"Melhor época ({best_epoch})")

ax.set_title("Curvas de Loss — Treinamento LSTM ITUB4")
ax.set_xlabel("Época")
ax.set_ylabel("MSE (espaço normalizado)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/08_history_loss.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico salvo em data/figures/08_history_loss.png")
```

**Critério de aceite do gráfico:**

| Padrão observado | Diagnóstico | Ação |
|---|---|---|
| `val_loss` converge junto com `train_loss` e estabiliza | Ajuste saudável | Prosseguir |
| `val_loss` cai e depois sobe enquanto `train_loss` continua caindo | Overfitting | EarlyStopping atuou corretamente; verificar a época de parada |
| Ambas as curvas estabilizam em valor alto | Underfitting | Aumentar neurônios ou épocas (fora do escopo deste projeto) |
| `val_loss` oscila fortemente | Instabilidade | Reduzir `learning_rate` do Adam (não esperado para esta escala de dados) |

---

## 2.7 Resumo do treinamento

```python
n_epochs_executadas = len(history.history["loss"])

print("\n========================================")
print("  Resumo do Treinamento LSTM")
print("========================================")
print(f"  Épocas executadas    : {n_epochs_executadas} / {EPOCHS}")
print(f"  Melhor época         : {best_epoch}")
print(f"  train_loss (melhor)  : {history.history['loss'][best_epoch - 1]:.6f}")
print(f"  val_loss   (melhor)  : {min(history.history['val_loss']):.6f}")
print(f"  Modelo salvo em      : models/lstm/lstm_itub4.h5")
print("========================================\n")
```

---

## Resultado esperado

| Artefato | Localização | Descrição |
|---|---|---|
| `lstm_itub4.h5` | `models/lstm/` | Pesos da melhor época segundo `val_loss` |
| `08_history_loss.png` | `data/figures/` | Curvas MSE de treino e validação por época |

---

## Próximo passo

Com o modelo treinado e serializado, prosseguir para a [avaliação comparativa LSTM vs. baseline](03_avaliacao_comparativa.md).
