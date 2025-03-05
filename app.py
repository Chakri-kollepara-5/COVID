import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Function to load and preprocess data
def load_data(stock, start, end):
    df = yf.download(stock, start=start, end=end)
    return df

# Function to create LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit app layout
st.title('Stock Price Prediction App')

# User inputs
stock = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

# Load data
if st.button('Predict'):
    data = load_data(stock, start_date, end_date)
    st.write(data)

    # Data preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare training data
    train_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:train_data_len]

    x_train, y_train = [], []
    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Reshape for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create and train model
    model = create_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Testing the model
    test_data = scaled_data[train_data_len - 100:]
    x_test, y_test = [], data['Close'][train_data_len:].values
    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plotting results
    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['Predictions'] = predictions

    st.subheader('Training Data')
    st.line_chart(train['Close'])
    st.subheader('Validation Data')
    st.line_chart(valid[['Close', 'Predictions']])
