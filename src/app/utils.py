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
