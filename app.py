import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Fetch stock data
stock_symbol = "AAPL"  # You can change this to any stock symbol
data = yf.download(stock_symbol, start="2020-01-01", end="2023-12-31")

# Display the data
st.title(f"{stock_symbol} Stock Price Prediction")
st.write("Fetching historical stock data...")

# Show the last few rows of the data
st.write(data.tail())

# Prepare the data for prediction
data['Date'] = data.index
data['Day'] = np.arange(len(data))  # Create a day column for regression

# Select relevant columns
df_historical = data[['Day', 'Close']].reset_index(drop=True)

# Split the data into training and testing sets
X = df_historical[['Day']]
y = df_historical['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict next day's price
next_day = np.array([[len(df_historical)]])  # Predict for the next day
predicted_price = model.predict(next_day)
st.write(f"Predicted closing price for Day {len(df_historical) + 1}: ${predicted_price[0]:.2f}")

# User Input for prediction
day_input = st.number_input("Enter day number for prediction (e.g., 1 for the first day)", min_value=1, max_value=len(df_historical) + 30)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted closing price for Day {day_input}: ${prediction[0]:.2f}")

# Plotting the historical data and predictions
plt.figure(figsize=(10, 5))
plt.plot(df_historical['Day'], df_historical['Close'], label='Historical Prices', color='blue')
plt.scatter(day_input, prediction, color='red', label='Predicted Price', zorder=5)
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.title(f"{stock_symbol} Stock Price History and Prediction")
plt.legend()
plt.grid()
plt.show()

# Display the plot in Streamlit
st.pyplot(plt)
