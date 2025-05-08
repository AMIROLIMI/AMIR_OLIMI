import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 📌 Настройки страницы
st.set_page_config(page_title="Прогноз цены золота", layout="wide")

st.title("💰 Прогноз цены золота на основе LSTM-модели")

st.markdown("""
Модель предсказывает цену золота, используя нейросеть LSTM.  
Она обучена на 2861 точке и использует окно в 14 дней.  
Метрики:
- **RMSE:** 47.33
- **MAPE:** 0.007
- **R²:** 0.99  
Кросс-валидация:
- **RMSE:** 53.05
- **MAPE:** 0.01
- **R²:** 0.90
""")

uploaded_file = st.file_uploader("Загрузите файл с ценами (Excel)", type=["xlsx", "csv"])
if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    st.write("📊 Загруженные данные:", df.head())

    if st.button("📈 Построить прогноз"):
        # Подготовка данных
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['price']])

        # Создание выборки
        def create_dataset(data, window=14):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i+window])
                y.append(data[i+window])
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Архитектура и загрузка весов
        model = Sequential([LSTM(32, input_shape=(14, 1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.load_weights("lstm_model.weights.h5")

        # Прогноз
        y_pred = model.predict(X)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))

        # Метрики
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)

        st.success(f"📌 RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.2f}")

        # График
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true_inv, label='Истинные значения')
        ax.plot(y_pred_inv, label='Прогноз (LSTM)', linestyle='--')
        ax.set_title("Сравнение прогноза и реальности")
        ax.legend()
        st.pyplot(fig)
