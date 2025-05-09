import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

WINDOW = 14         
UNITS  = 32         

st.set_page_config(page_title="Прогноз цены золота", layout="wide")
st.title("📉 Прогноз цены золотых слитков (LSTM)")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_model():
    m = Sequential([LSTM(UNITS, input_shape=(WINDOW, 1)), Dense(1)])
    m.load_weights("lstm_model.keras")
    return m

def create_dataset(arr, window=WINDOW):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
    return np.array(X), np.array(y)

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mape, r2

def plot_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label="Истинные значения", linewidth=2)
    ax.plot(y_pred, label="Прогноз (LSTM)", linestyle="--")
    ax.set_title("Прогноз LSTM vs Реальные данные")
    ax.set_xlabel("Дни")
    ax.set_ylabel("Цена")
    ax.legend()
    ax.grid(True)
    return fig

scaler = load_scaler()
model  = load_model()

file = st.file_uploader("📂 Загрузите CSV или Excel с колонками date,price", type=["csv", "xlsx"])

if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    if df.shape[1] != 2:
        st.error("Файл должен иметь два столбца: date, price")
        st.stop()

    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    st.write("Последние строки датасета:", df.tail())

    try:
        scaled = scaler.transform(df[["price"]])
    except ValueError as e:
        st.error(f"Scaler не смог трансформировать данные: {e}")
        st.stop()

    if scaled.min() < 0 or scaled.max() > 1:
        st.warning("⚠️ Загруженные цены выходят за диапазон, на котором обучался scaler. "
                   "Прогноз может быть менее точным.")

    X, y = create_dataset(scaled, WINDOW)
    if len(X) == 0:
        st.warning("Недостаточно строк для окна размера 14 дней.")
        st.stop()

    X = X.reshape((X.shape[0], X.shape[1], 1))
    st.markdown("""
    Если у вас нет данных, вы можете скачать их из моего репозитория  
    [AMIROLIMI/AMIR_OLIMI](https://github.com/AMIROLIMI/AMIR_OLIMI).
    """
    )
    if st.button("🔮 Сделать прогноз"):
        y_pred = model.predict(X, verbose=0)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))

        rmse, mape, r2 = calc_metrics(y_true_inv, y_pred_inv)
        st.success(f"RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.3f}")

        st.pyplot(plot_pred(y_true_inv, y_pred_inv))
else:
    st.info("Загрузите файл для получения прогноза.")
