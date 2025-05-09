
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

WINDOW = 14          # длина окна, как в ноутбуке

st.set_page_config(page_title="Прогноз цены золота", layout="wide")
st.title("📉 Прогноз цены золотых слитков (LSTM)")

# ---------- кешируем загрузку артефактов ----------
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_lstm():
    return load_model("lstm_model.keras", compile=False)   # для predict компиляция не нужна

scaler = load_scaler()
model  = load_lstm()

# ---------- утилиты ----------
def create_dataset(arr, win=WINDOW):
    X, y = [], []
    for i in range(len(arr) - win):
        X.append(arr[i:i + win])
        y.append(arr[i + win])
    return np.array(X), np.array(y)

def metrics(y_true, y_pred):
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
    ax.legend(); ax.grid(True)
    return fig

# ---------- интерфейс ----------
file = st.file_uploader("📂 Загрузите CSV или Excel с колонками date,price", type=("csv", "xlsx"))

if file:
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    if df.shape[1] != 2:
        st.error("Файл должен содержать два столбца: date, price"); st.stop()

    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    st.write("📄 Предпросмотр данных:", df.tail())

    # --- масштабирование тем же scaler ---
    try:
        scaled = scaler.transform(df[["price"]])
    except ValueError as e:
        st.error(f"Ошибка масштабирования: {e}"); st.stop()

    if scaled.min() < 0 or scaled.max() > 1:
        st.warning("⚠️ Цены выходят за диапазон train-масштабирования. "
                   "Результаты могут быть недостоверны."); st.stop()

    X, y = create_dataset(scaled, WINDOW)
    if len(X) == 0:
        st.warning("Недостаточно строк для окна 14 дней."); st.stop()

    X = X.reshape((X.shape[0], X.shape[1], 1))

    if st.button("🔮 Сделать прогноз"):
        y_pred = model.predict(X, verbose=0)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))

        rmse, mape, r2 = metrics(y_true_inv, y_pred_inv)
        st.success(f"RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.3f}")
        st.pyplot(plot_pred(y_true_inv, y_pred_inv))
else:
    st.info("Загрузите файл для получения прогноза.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
# import os
# import joblib

# st.set_page_config(page_title="Прогноз цены золота", layout="wide")
# st.title("Прогноз цены золотых слитков с помощью LSTM")

# st.markdown("""
# Модель предсказывает цену золота с использованием нейросети LSTM.  
# Обучена на 2861 точке с окном в 14 дней.  
# Метрики:  
# - **RMSE**: 47.33  
# - **MAPE**: 0.007  
# - **R²**: 0.99  
# Кросс-валидация:  
# - **RMSE**: 53.05  
# - **MAPE**: 0.01  
# - **R²**: 0.90  
# """)

# scaler = joblib.load("scaler.pkl")

# def load_lstm_model():
#     model = Sequential([LSTM(32, input_shape=(14, 1)), Dense(1)])
#     model.load_weights("lstm_model.weights.h5") 
#     return model

# def create_dataset(data, window=14):
#     X, y = [], []
#     for i in range(len(data) - window):
#         X.append(data[i:i+window])
#         y.append(data[i + window])
#     return np.array(X), np.array(y)

# def calculate_metrics(y_true, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_true, (y_pred)))
#     mape = mean_absolute_percentage_error(y_true, (y_pred))
#     r2 = r2_score(y_true, (y_pred))
#     return rmse, mape, r2

# def plot_predictions(y_true, y_pred):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     ax.plot(y_true, label="Истинные значения", linewidth=2)
#     ax.plot(y_pred, label="Прогноз (LSTM)", linestyle='--')
#     ax.set_title("Прогноз LSTM vs Реальные данные")
#     ax.set_xlabel("Дни")
#     ax.set_ylabel("Цена")
#     ax.legend()
#     ax.grid(True)
#     return fig


# model = load_lstm_model()
# uploaded_file = st.file_uploader("📂 Загрузите CSV или Excel с ценами", type=["csv", "xlsx"])
# st.markdown("""
# Если у вас нет данных, вы можете скачать их из моего репозитория  
# [AMIROLIMI/AMIR_OLIMI](https://github.com/AMIROLIMI/AMIR_OLIMI).
# """
# )

# st.subheader("📈 Прогноз по данным:")

# if uploaded_file:
#     df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
#     df.columns = ['date', 'price']
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.set_index('date').sort_index()
#     st.write(df.tail())

#     try:
#         scaled = scaler.transform(df[['price']])
#     except:
#         st.error("❌ Ошибка при масштабировании. Убедитесь, что формат 'price' правильный.")
#         st.stop()

#     X, y = create_dataset(scaled)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     if model:
#         if st.button("Сделать прогноз"):
#             y_pred = model.predict(X)
#             y_pred_inv = scaler.inverse_transform(y_pred)
#             y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))
#             rmse, mape, r2 = calculate_metrics(y_true_inv, y_pred_inv)
#             st.success(f"📌 RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.2f}")
#             st.pyplot(plot_predictions(y_true_inv, y_pred_inv))
# else:
#     st.info("Загрузите файл для прогноза.")
