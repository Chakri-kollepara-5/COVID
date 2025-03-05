import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.graph_objs as go

# Function to create dataset for prediction
def create_dataset(data, look_back=100):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Streamlit app
def main():
    st.title("Stock Price Prediction App")
    st.sidebar.title('Stock Price Forecasting App')

    # User input for stock ticker symbol
    stock_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., MSFT):')

    # Date range input
    start_date = st.sidebar.date_input('Select Start Date:', pd.to_datetime('2000-01-01'))
    end_date = st.sidebar.date_input('Select End Date:', pd.to_datetime('today'))

    if stock_symbol:
        # Load stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        st.subheader('Stock Data')
        st.write(stock_data)

        # Data preprocessing
        data = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create training and testing datasets
        look_back = 100
        X, y = create_dataset(scaled_data, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train the model
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=32)

        # Predict future prices
        last_100_days = scaled_data[-look_back:]
        last_100_days = last_100_days.reshape((1, look_back, 1))
        predicted_price = model.predict(last_100_days)
        predicted_price = scaler.inverse_transform(predicted_price)

        st.write(f"Predicted closing price for the next day: ${predicted_price[0][0]:.2f}")

        # Plotting
        st.subheader('Price vs Predicted Price')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=1), 
                                  y=predicted_price.flatten(), mode='markers', name='Predicted Price'))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
