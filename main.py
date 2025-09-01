# ------------------------------
# Multi-Model Stock Forecasting Dashboard (Streamlit)
# ------------------------------
# Save as app.py and run: streamlit run app.py
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import gdown
import os

# ------------------------------
# Load Data
# ------------------------------
def load_data():
    file_id = "1RnAnC6hzF7pPynaRmnGRQ5aqP-EQhhBe"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "all_stocks_5yr.csv"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()
st.title("📈 Stock Market Forecasting Dashboard (Prophet, ARIMA, SARIMA, LSTM)")

# ------------------------------
# Sidebar - Stock & Model Selection
# ------------------------------
companies = df['name'].unique()
company = st.sidebar.selectbox("Select a Stock", companies)

models_selected = st.sidebar.multiselect(
    "Select Models to Run",
    ["Prophet", "ARIMA", "SARIMA", "LSTM"],
    default=["Prophet"]
)

periods = st.sidebar.slider("Days to Forecast", 30, 365, 90)

# ------------------------------
# Prepare Data
# ------------------------------
data = df[df['name'] == company].copy()
data = data.rename(columns={'date': 'ds', 'close': 'y'})  # Prophet requirement
data = data[['ds', 'y']].dropna()

# ------------------------------
# Results Placeholder
# ------------------------------
results = {}

# ------------------------------
# Prophet Model
# ------------------------------
if "Prophet" in models_selected:
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(data)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # Plot
    st.subheader(f"📊 Prophet Forecast for {company}")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    # Metrics
    df_merged = data.merge(forecast[['ds','yhat']], on='ds', how='left')
    mae = mean_absolute_error(df_merged['y'], df_merged['yhat'])
    mse = mean_squared_error(df_merged['y'], df_merged['yhat'])
    rmse = np.sqrt(mse)
    results['Prophet'] = (mae, mse, rmse)

# ------------------------------
# ARIMA Model
# ------------------------------
if "ARIMA" in models_selected:
    ts = data.set_index('ds')['y']
    model = ARIMA(ts, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)

    # Plot
    st.subheader(f"📊 ARIMA Forecast for {company}")
    plt.figure(figsize=(10,5))
    plt.plot(ts, label="Actual")
    plt.plot(pd.date_range(ts.index[-1], periods=periods+1, freq="D")[1:], forecast, label="Forecast")
    plt.legend()
    st.pyplot(plt)

    # Metrics (on train)
    fitted = model_fit.fittedvalues
    mae = mean_absolute_error(ts[1:], fitted)
    mse = mean_squared_error(ts[1:], fitted)
    rmse = np.sqrt(mse)
    results['ARIMA'] = (mae, mse, rmse)

# ------------------------------
# SARIMA Model
# ------------------------------
if "SARIMA" in models_selected:
    ts = data.set_index('ds')['y']
    model = SARIMAX(ts, order=(2,1,2), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=periods)

    # Plot
    st.subheader(f"📊 SARIMA Forecast for {company}")
    plt.figure(figsize=(10,5))
    plt.plot(ts, label="Actual")
    plt.plot(pd.date_range(ts.index[-1], periods=periods+1, freq="D")[1:], forecast, label="Forecast")
    plt.legend()
    st.pyplot(plt)

    # Metrics
    fitted = model_fit.fittedvalues
    mae = mean_absolute_error(ts[1:], fitted[1:])
    mse = mean_squared_error(ts[1:], fitted[1:])
    rmse = np.sqrt(mse)
    results['SARIMA'] = (mae, mse, rmse)
# ------------------------------
# LSTM Model
# ------------------------------
if "LSTM" in models_selected:
    from sklearn.preprocessing import MinMaxScaler

    # Scale data
    ts = data.set_index('ds')['y'].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    ts_scaled = scaler.fit_transform(ts)

    # prepare sequences (60 timesteps)
    X, y = [], []
    for i in range(60, len(ts_scaled)):
        X.append(ts_scaled[i-60:i,0])
        y.append(ts_scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Forecast future values
    last_60 = ts_scaled[-60:]
    predictions = []
    current_batch = last_60.reshape((1,60,1))
    for i in range(periods):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    # Plot
    st.subheader(f"📊 LSTM Forecast for {company}")
    plt.figure(figsize=(10,5))
    plt.plot(data['ds'], data['y'], label="Actual")
    future_dates = pd.date_range(data['ds'].iloc[-1], periods=periods+1, freq="D")[1:]
    plt.plot(future_dates, predictions, label="Forecast", color="red")
    plt.legend()
    st.pyplot(plt)

    # Metrics on test set
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_test = scaler.inverse_transform(y_pred_test)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

    mae = mean_absolute_error(y_test_rescaled, y_pred_test)
    mse = mean_squared_error(y_test_rescaled, y_pred_test)
    rmse = np.sqrt(mse)
    results['LSTM'] = (mae, mse, rmse)

# ------------------------------
# Final Comparison Table
# ------------------------------
if results:
    st.subheader("📊 Model Performance Comparison")
    results_df = pd.DataFrame(results, index=["MAE","MSE","RMSE"]).T
    st.dataframe(results_df.style.format("{:.2f}"))
