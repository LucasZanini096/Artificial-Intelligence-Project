# Etapa 2 — Estrutura e Configuração do Streamlit

## Objetivo

Criar a estrutura de arquivos da aplicação Streamlit, configurar o tema visual e definir o módulo de funções auxiliares (`utils.py`) que abstrai o carregamento dos artefatos do modelo, garantindo que o app funcione a partir do diretório raiz do projeto (`PROJETO/`).

---

## 2.1 Estrutura de arquivos

```
PROJETO/
  src/
    app/
      app.py              ← ponto de entrada do Streamlit
      utils.py            ← carregamento de modelos, dados e previsões
      .streamlit/
        config.toml       ← tema e layout
```

Criar o diretório com:

```bash
mkdir -p src/app/.streamlit
```

---

## 2.2 Configuração do tema (`.streamlit/config.toml`)

```toml
[theme]
primaryColor   = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f4f8"
textColor      = "#1a1a2e"
font           = "sans serif"

[server]
headless = true
port     = 8501
```

Salvar em `src/app/.streamlit/config.toml`.

---

## 2.3 Módulo auxiliar (`utils.py`)

O `utils.py` encapsula todo o I/O de artefatos com cache do Streamlit, evitando releituras a cada interação do usuário.

```python
# src/app/utils.py

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Caminhos relativos ao diretório raiz do projeto (PROJETO/)
BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC       = os.path.join(BASE, "src")
MODEL_DIR = os.path.join(SRC, "models", "lstm")
BASE_DIR  = os.path.join(SRC, "models", "baseline")
SCAL_DIR  = os.path.join(SRC, "models", "scalers")
DATA_DIR  = os.path.join(SRC, "database", "processed")


@st.cache_data
def load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Carrega previsões pré-computadas, valores reais e datas do período de teste.
    Retorna (y_test_real, y_pred_lstm, y_pred_baseline, test_dates).
    """
    y_test_real      = np.load(os.path.join(BASE_DIR,  "y_test_real.npy"))
    y_pred_lstm      = np.load(os.path.join(MODEL_DIR, "y_pred_lstm.npy"))
    y_pred_baseline  = np.load(os.path.join(BASE_DIR,  "y_pred_baseline.npy"))
    test_dates_raw   = np.load(os.path.join(MODEL_DIR, "test_dates.npy"),
                                allow_pickle=True)
    test_dates = pd.to_datetime(test_dates_raw)
    return y_test_real, y_pred_lstm, y_pred_baseline, test_dates


@st.cache_data
def load_metrics() -> tuple[dict, dict]:
    """
    Carrega métricas da LSTM e do baseline.
    Retorna (lstm_metrics, baseline_metrics).
    """
    with open(os.path.join(MODEL_DIR, "lstm_metrics.json")) as f:
        lstm_m = json.load(f)
    with open(os.path.join(BASE_DIR,  "baseline_metrics.json")) as f:
        base_m = json.load(f)
    return lstm_m, base_m


@st.cache_data
def load_full_series() -> pd.DataFrame:
    """
    Carrega a série histórica completa para exibição contextual.
    """
    df = pd.read_csv(
        os.path.join(DATA_DIR, "ITUB4_processed.csv"),
        parse_dates=["Date"]
    ).sort_values("Date").reset_index(drop=True)
    return df
```

> **Decisão de design:** `@st.cache_data` serializa os objetos numpy/pandas em cache após a primeira leitura. Artefatos numpy são leves (~1,6 KB cada), portanto o cache é mantido por toda a sessão sem custo de memória relevante.

---

## 2.4 Dependências da fase

Adicionar ao `requirements.txt` do projeto:

```
streamlit>=1.35
plotly>=5.22
```

Instalar:

```bash
pip install streamlit plotly
```

---

## 2.5 Execução do app

O app deve ser iniciado **a partir do diretório raiz do projeto** (`PROJETO/`):

```bash
cd /caminho/para/PROJETO
streamlit run src/app/app.py
```

O uso de caminhos absolutos em `utils.py` garante que o app funcione independentemente do diretório de trabalho atual, sem necessidade de manipular `sys.path` ou `PYTHONPATH`.

---

## Checklist da Etapa 2

- [ ] Diretório `src/app/` criado
- [ ] `src/app/.streamlit/config.toml` criado com tema configurado
- [ ] `src/app/utils.py` criado e funções testadas em ambiente Python puro (sem Streamlit)
- [ ] `requirements.txt` atualizado com `streamlit` e `plotly`
- [ ] `streamlit run src/app/app.py` inicia sem erros de importação
