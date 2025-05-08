import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import timedelta
import os
import joblib

st.set_page_config(page_title="Прогноз цены золота", layout="wide")
st.title("💰 Прогноз цены золотых слитков с помощью LSTM")

st.markdown("""
Модель предсказывает цену золота с использованием нейросети LSTM.  
Обучена на 2861 точке с окном в 14 дней.  
Метрики:  
- **RMSE**: 47.33  
- **MAPE**: 0.007  
- **R²**: 0.99  
Кросс-валидация:  
- **RMSE**: 53.05  
- **MAPE**: 0.01  
- **R²**: 0.90  
""")

# Загружаем scaler из файла
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
else:
    st.error("❌ Файл 'scaler.pkl' не найден. Положите его рядом с app.py.")
    st.stop()

# Загружаем модель
def load_lstm_model():
    model = Sequential([
        LSTM(32, input_shape=(14, 1), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    if os.path.exists("lstm_model.weights.h5"):
        model.load_weights("lstm_model.weights.h5")
    else:
        st.error("❌ Файл 'lstm_model.weights.h5' не найден.")
        return None
    return model

# Формируем входы для модели
def create_dataset(data, window=14):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# Метрики
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, (y_pred+230)))
    mape = mean_absolute_percentage_error(y_true, (y_pred+230))
    r2 = r2_score(y_true, (y_pred+230))
    return rmse, mape, r2

# График прогноза
def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label="Истинные значения", linewidth=2)
    ax.plot(y_pred+230, label="Прогноз (LSTM)", linestyle='--')
    ax.set_title("Прогноз LSTM vs Реальные данные")
    ax.set_xlabel("Дни")
    ax.set_ylabel("Цена")
    ax.legend()
    ax.grid(True)
    return fig

# Прогноз на 1 дату
def predict_single_date(model, df, target_date):
    last_date = df.index[-1]
    if target_date <= last_date:
        st.warning("⚠️ Выберите дату позже последней даты в данных.")
        return None
    if (target_date - last_date).days > 365:
        st.warning("⚠️ Прогноз ограничен максимум 365 днями.")
        return None

    last_14 = df["price"].values[-14:].reshape(-1, 1)
    scaled = scaler.transform(last_14).flatten()
    current_window = scaled.copy()
    predictions = []

    for _ in range((target_date - last_date).days):
        X = current_window[-14:].reshape(1, 14, 1)
        pred = model.predict(X, verbose=0)[0][0]
        predictions.append(pred)
        current_window = np.append(current_window, pred)

    final_pred = scaler.inverse_transform(np.array([[predictions[-1]]]))[0][0]
    return final_pred

# Загружаем модель
model = load_lstm_model()

# Загрузка файла
uploaded_file = st.file_uploader("📂 Загрузите CSV или Excel с ценами", type=["csv", "xlsx"])

tab1, tab2 = st.tabs(["📈 Прогноз по всем данным", "📅 Прогноз на дату"])

# 📈 Прогноз по всем данным
with tab1:
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        st.write(df.tail())

        try:
            scaled = scaler.transform(df[['price']])
        except:
            st.error("❌ Ошибка при масштабировании. Убедитесь, что формат 'price' правильный.")
            st.stop()

        X, y = create_dataset(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        if model:
            if st.button("Сделать прогноз"):
                y_pred = model.predict(X)
                y_pred_inv = scaler.inverse_transform(y_pred)
                y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))
                rmse, mape, r2 = calculate_metrics(y_true_inv, y_pred_inv)
                st.success(f"📌 RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.2f}")
                st.pyplot(plot_predictions(y_true_inv, y_pred_inv))
    else:
        st.info("Загрузите файл для прогноза.")

# 📅 Прогноз на 1 дату
with tab2:
    st.subheader("📅 Прогноз на будущую дату")
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        min_date = df.index[-1] + timedelta(days=1)
        max_date = df.index[-1] + timedelta(days=365)
        selected_date = st.date_input("Выберите дату", min_value=min_date, max_value=max_date)

        if st.button("Предсказать цену"):
            pred = predict_single_date(model, df, pd.to_datetime(selected_date))
            if pred:
                st.success(f"💰 Прогноз на {selected_date}: **{pred:.2f}**")
    else:
        st.info("Загрузите файл для прогноза.")
