# Etapa 2 — Análise Exploratória dos Dados (EDA)

## Objetivo

Compreender a estrutura da série histórica do ITUB4, identificar tendências, sazonalidades, outliers e períodos atípicos, gerando 5 visualizações que fundamentarão as decisões de modelagem.

Todas as figuras são salvas em `data/figures/` em formato PNG (300 dpi) para uso no artigo.

---

## Carregamento dos dados

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

df = pd.read_csv("data/raw/ITUB4_raw.csv", index_col="Date", parse_dates=True)
print(df.shape)
```

---

## 2.1 Série histórica de preços e volume

**O que observar:** tendência de longo prazo, rupturas estruturais (quedas abruptas), correlação visual entre picos de volume e movimentos de preço.

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.0, label="Fechamento")
ax1.set_ylabel("Preço de Fechamento (R$)")
ax1.set_title("ITUB4 — Preço de Fechamento e Volume Diário (últimos 5 anos)")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)

ax2.bar(df.index, df["Volume"] / 1e6, color="#aec7e8", width=1.0)
ax2.set_ylabel("Volume (milhões)")
ax2.set_xlabel("Data")
ax2.grid(alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/figures/01_preco_volume.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Critério de aceite:** gráfico com dois painéis alinhados no eixo x; série de preços legível; eixo y do volume em milhões.

---

## 2.2 Retornos diários logarítmicos

**O que observar:** distribuição aproximadamente normal com caudas pesadas (leptocúrtica), assimetria, outliers extremos em períodos de crise.

```python
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Série temporal dos retornos
axes[0].plot(df.index, df["log_return"], color="#2ca02c", linewidth=0.6, alpha=0.8)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Retornos Logarítmicos Diários — ITUB4")
axes[0].set_ylabel("ln(Pt / Pt-1)")
axes[0].set_xlabel("Data")
axes[0].grid(alpha=0.3)

# Histograma com curva normal sobreposta
from scipy.stats import norm

mu, sigma = df["log_return"].dropna().mean(), df["log_return"].dropna().std()
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
axes[1].hist(df["log_return"].dropna(), bins=60, density=True,
             color="#aec7e8", edgecolor="white", alpha=0.8, label="Empírico")
axes[1].plot(x, norm.pdf(x, mu, sigma), "r-", linewidth=1.5, label="Normal teórica")
axes[1].set_title("Distribuição dos Retornos Diários")
axes[1].set_xlabel("Retorno logarítmico")
axes[1].set_ylabel("Densidade")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/02_retornos_diarios.png", dpi=300, bbox_inches="tight")
plt.show()

# Estatísticas resumidas
print(df["log_return"].describe())
print(f"Curtose: {df['log_return'].dropna().kurt():.4f}")
print(f"Assimetria: {df['log_return'].dropna().skew():.4f}")
```

**Critério de aceite:** curtose > 3 (distribuição leptocúrtica, como esperado em ativos financeiros); presença de outliers visíveis na série temporal.

---

## 2.3 Médias móveis sobrepostas ao preço

**O que observar:** tendências de curto prazo (SMA-20), médio prazo (SMA-50, EMA-60), cruzamentos de médias como sinais técnicos.

```python
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()
# EMA_60 já foi calculada na coleta

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df["Close"],  color="#1f77b4", linewidth=0.8, alpha=0.7, label="Fechamento")
ax.plot(df.index, df["SMA_20"], color="#ff7f0e", linewidth=1.2, label="SMA-20")
ax.plot(df.index, df["SMA_50"], color="#2ca02c", linewidth=1.2, label="SMA-50")
ax.plot(df.index, df["EMA_60"], color="#d62728", linewidth=1.4, linestyle="--", label="EMA-60")

ax.set_title("ITUB4 — Preço de Fechamento com Médias Móveis")
ax.set_ylabel("Preço (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/figures/03_medias_moveis.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Critério de aceite:** 4 séries distintas e legíveis; EMA-60 com linha tracejada para diferenciá-la das SMAs.

---

## 2.4 Mapa de calor de correlações (Pearson)

**O que observar:** alta correlação esperada entre Open/High/Low/Close (> 0,95); correlação da EMA-60 com Close próxima de 1; Volume tipicamente com correlação mais baixa em relação aos preços.

```python
features = ["Open", "High", "Low", "Close", "Volume", "EMA_60"]
corr_matrix = df[features].corr(method="pearson")

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # mostra triângulo inferior

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Matriz de Correlação de Pearson — ITUB4 (5 anos)")
plt.tight_layout()
plt.savefig("data/figures/04_correlacao_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print(corr_matrix)
```

**Critério de aceite:** mapa com anotações numéricas em cada célula; escala de cores de -1 (azul) a +1 (vermelho); imagem quadrada.

---

## 2.5 Identificação visual de períodos atípicos

**O que observar:** picos de volatilidade que o modelo não conseguirá prever (eventos exógenos). Marcar ao menos dois períodos relevantes no recorte de 5 anos.

```python
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.0, label="Fechamento")

# Definir períodos atípicos dentro do recorte de 5 anos
# (ajuste as datas conforme o período real do download)
atypical_periods = [
    ("2022-01-01", "2022-12-31", "#ff7f0e", "2022 — Alta Selic (13,75% a.a.)"),
    ("2023-08-01", "2023-10-31", "#d62728", "2023 — Volatilidade fiscal"),
]

for start, end, color, label in atypical_periods:
    # Verificar se o período está dentro do range do df
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if start_dt >= df.index[0] and end_dt <= df.index[-1]:
        ax.axvspan(start_dt, end_dt, alpha=0.15, color=color, label=label)

ax.set_title("ITUB4 — Períodos Atípicos de Alta Volatilidade")
ax.set_ylabel("Preço de Fechamento (R$)")
ax.set_xlabel("Data")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("data/figures/05_periodos_atipicos.png", dpi=300, bbox_inches="tight")
plt.show()
```

> **Atenção:** ajuste os pares `(start, end)` do dicionário `atypical_periods` com base nos dados efetivamente baixados. Se o período coberto começar em 2021, inclua o pico de volatilidade de 2021 (recuperação pós-pandemia e juros em alta).

---

## 2.6 Resumo estatístico para o artigo

```python
summary = df[["Open", "High", "Low", "Close", "Volume", "EMA_60"]].describe()
print(summary.to_string())
```

Registrar no notebook o período exato, número de pregões, média, desvio padrão, mínimo e máximo do preço de fechamento para incluir na seção 3.2 (Dataset) do artigo.

---

## Resultado esperado

| Figura | Arquivo | Conteúdo |
|---|---|---|
| 1 | `01_preco_volume.png` | Preço de fechamento + volume em dois painéis |
| 2 | `02_retornos_diarios.png` | Série de retornos log + histograma com normal |
| 3 | `03_medias_moveis.png` | Preço com SMA-20, SMA-50 e EMA-60 |
| 4 | `04_correlacao_heatmap.png` | Heatmap Pearson 6×6 anotado |
| 5 | `05_periodos_atipicos.png` | Preço com regiões sombreadas de eventos |

---

## Próximo passo

Com as visualizações geradas e a série compreendida, prosseguir para o [Pré-processamento](03_preprocessamento.md).
