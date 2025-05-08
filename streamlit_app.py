import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
from datetime import datetime, timedelta

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

Загрузите файл с ценами, чтобы получить прогноз или предсказать цену на конкретную дату.
""")

def load_and_preprocess_data(file):
    try:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def create_dataset(data, window=14):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def load_lstm_model():
    try:
        model = Sequential([
            LSTM(32, input_shape=(14, 1), activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        if os.path.exists("lstm_model.weights.h5"):
            model.load_weights("lstm_model.weights.h5")
        else:
            st.error("Файл весов модели 'lstm_model.weights.h5' не найден.")
            return None
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2

def plot_predictions(y_true, y_pred, title="Прогноз LSTM vs Реальные данные"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label='Реальные значения', linewidth=2)
    ax.plot(y_pred, label='Прогноз LSTM', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Дни")
    ax.set_ylabel("Цена")
    ax.legend()
    ax.grid(True)
    return fig

def predict_single_date(model, scaler, df, target_date):
    try:
        last_date = df.index[-1]
        if target_date <= last_date:
            st.error("Выберите дату после последней даты в данных.")
            return None
        last_14_days = df['price'].values[-14:]
        scaled_data = scaler.transform(last_14_days.reshape(-1, 1))
        X = scaled_data.reshape(1, 14, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return pred_price
    except Exception as e:
        st.error(f"Ошибка прогноза на дату: {str(e)}")
        return None

st.subheader("📂 Загрузка данных")
uploaded_file = st.file_uploader("Загрузите файл Excel или CSV с ценами", type=["xlsx", "csv"])

if uploaded_file:
    with st.spinner("Загрузка данных..."):
        df = load_and_preprocess_data(uploaded_file)
    
    if df is not None:
        st.write("📊 Просмотр данных:", df.head())
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['price']])
        X, y = create_dataset(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = load_lstm_model()

        if model:
            tab1, tab2 = st.tabs(["📈 Прогноз по всем данным", "📅 Прогноз на дату"])

            with tab1:
                st.subheader("Прогноз по всем данным")
                if st.button("Сделать прогноз", key="full_predict"):
                    with st.spinner("Создание прогноза..."):
                        y_pred = model.predict(X, verbose=0)
                        y_pred_inv = scaler.inverse_transform(y_pred)
                        y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))
                        rmse, mape, r2 = calculate_metrics(y_true_inv, y_pred_inv)
                        st.success(f"📌 **Метрики**: RMSE: {rmse:.2f} | MAPE: {mape:.3f} | R²: {r2:.2f}")
                        fig = plot_predictions(y_true_inv, y_pred_inv)
                        st.pyplot(fig)

            with tab2:
                st.subheader("Прогноз на одну дату")
                max_date = df.index[-1] + timedelta(days=365)
                selected_date = st.date_input(
                    "Выберите дату для прогноза",
                    min_value=df.index[-1] + timedelta(days=1),
                    max_value=max_date
                )
                if st.button("Прогнозировать", key="single_predict"):
                    with st.spinner("Прогнозирование..."):
                        pred_price = predict_single_date(model, scaler, df, pd.to_datetime(selected_date))
                        if pred_price is not None:
                            st.success(f"💰 Прогноз цены золота на {selected_date}: **{pred_price:.2f}**")

st.markdown("---")
st.markdown("Разработано с ❤️ с использованием Streamlit | Источник данных: Национальный банк Таджикистана")
