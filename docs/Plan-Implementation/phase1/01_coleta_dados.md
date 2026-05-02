# Etapa 1 — Coleta de Dados

## Objetivo

Baixar via API yfinance os dados históricos de 5 anos do ativo ITUB4.SA e calcular a EMA-60 como variável derivada, gerando o arquivo `data/raw/ITUB4_raw.csv`.

---

## 1.1 Configuração do ambiente

Crie a estrutura de pastas antes de executar qualquer código:

```python
import os

dirs = [
    "data/raw",
    "data/processed",
    "data/figures",
    "models/scalers",
    "src/notebooks",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
```

---

## 1.2 Coleta via yfinance

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

TICKER = "ITUB4.SA"
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
```

**Parâmetros importantes:**
- `auto_adjust=True` — usa preços de fechamento ajustados por dividendos e splits, garantindo continuidade da série histórica.
- O yfinance retorna um `DataFrame` com colunas: `Open`, `High`, `Low`, `Close`, `Volume`.
- Espera-se aproximadamente **1.250 linhas** (dias de pregão em 5 anos).

### Verificação imediata

```python
print(f"Shape: {df.shape}")
print(f"Período: {df.index[0].date()} → {df.index[-1].date()}")
print(df.dtypes)
print(df.head())
```

Resultado esperado:
- Shape: `(~1250, 5)`
- Todas as colunas do tipo `float64`, `Volume` do tipo `int64`
- Sem linhas com índice duplicado

---

## 1.3 Cálculo da EMA-60

A EMA-60 é calculada sobre o preço de fechamento ajustado. A fórmula recursiva é:

```
EMA_t = α · Close_t + (1 − α) · EMA_(t-1)    onde α = 2 / (60 + 1)
```

O pandas já implementa isso nativamente:

```python
df["EMA_60"] = df["Close"].ewm(span=60, adjust=False).mean()
```

**Por que `adjust=False`?**  
Usa a fórmula recursiva exata (igual ao artigo), em vez da forma ponderada que o pandas usa por padrão.

### Verificação

```python
# As primeiras 59 linhas terão EMA calculada mas com menos dados (bootstrap)
# A partir da linha 60 a EMA já é estável
print(df[["Close", "EMA_60"]].head(65))
print(f"Valores nulos na EMA_60: {df['EMA_60'].isna().sum()}")
```

---

## 1.4 Inspeção de qualidade dos dados brutos

```python
print("=== Valores ausentes ===")
print(df.isnull().sum())

print("\n=== Estatísticas descritivas ===")
print(df.describe())

print("\n=== Linhas com volume zerado ===")
zero_vol = df[df["Volume"] == 0]
print(f"Total: {len(zero_vol)}")
print(zero_vol)
```

**O que verificar:**
- Preços ausentes (`NaN`): podem ocorrer em feriados nacionais ou falhas da API.
- Volume zerado: indica pregão suspenso ou dado corrompido.
- Preços negativos ou absurdos: sinal de erro no download — reexecutar.

---

## 1.5 Exportação do arquivo bruto

```python
df.to_csv("data/raw/ITUB4_raw.csv", index=True)
print(f"Arquivo salvo: data/raw/ITUB4_raw.csv ({len(df)} linhas)")
```

O arquivo deve conter 7 colunas: `Date` (índice), `Open`, `High`, `Low`, `Close`, `Volume`, `EMA_60`.

---

## Resultado esperado

| Verificação | Critério de aceite |
|---|---|
| Número de linhas | ≥ 1.200 (anos com menos pregões podem ter ~248 dias) |
| Colunas presentes | Open, High, Low, Close, Volume, EMA_60 |
| Tipo do índice | `DatetimeIndex` com frequência de dias úteis |
| Valores nulos | Permitidos apenas para EMA_60 nas primeiras linhas do bootstrap |
| Volume zerado | Registrado; será tratado na etapa de pré-processamento |

---

## Próximo passo

Com o arquivo `data/raw/ITUB4_raw.csv` gerado, prosseguir para a [Análise Exploratória](02_analise_exploratoria.md).
