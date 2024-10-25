import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load your saved model
model = load_model('.\models\model2.h5')

# Define the stock options
stock_options = ['AAPL', 'GOOGL', 'MSFT']

# Streamlit App Layout
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")

# Sidebar for user inputs
st.sidebar.header("User Input")
st.sidebar.write("Select a stock to predict the closing prices for the next 14 days.")

stock_choice = st.sidebar.selectbox("Choose a stock", stock_options)
predict_button = st.sidebar.button("Predict")

# Information Box
st.sidebar.info("This app uses a trained model to predict stock prices based on historical data. "
                "Please select a stock from the dropdown menu and click on 'Predict'.")

# Fetch stock data from Yahoo Finance
@st.cache_data
def fetch_stock_data(stock):
    data = yf.download(stock, period='1y')  # Fetch the last year of data
    data['Stock'] = stock  # Add a column for stock name
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Data preprocessing for the model
def preprocess_data(stock_choice):
    stock_df = fetch_stock_data(stock_choice)
    
    # Extract relevant features (e.g., Close prices) for prediction
    close_prices = stock_df[['Close']].values
    
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices)
    
    # Ensure there are enough data points to create the sequence
    if len(close_prices_scaled) < 60:  # We need at least 60 data points for a proper prediction
        st.error("Not enough data points available for the selected stock.")
        return None, None, None
    
    # Select the last 60 days for prediction
    X_input = close_prices_scaled[-60:].reshape(1, 60, 1)  # Reshaping to (1, 60, 1) to match the model input shape
    
    # Prepare stock index input
    stock_idx_input = np.array([stock_options.index(stock_choice)]).reshape(1, 1)
    
    return X_input, stock_idx_input, scaler, close_prices, stock_df

# Predict the closing price for the next 14 days
def predict_stock_prices(stock_choice):
    X_input, stock_idx_input, scaler, close_prices, stock_df = preprocess_data(stock_choice)
    
    if X_input is None or stock_idx_input is None or scaler is None:
        return None, None, None, None
    
    # Predict the next 14 days
    predicted_prices = []
    for _ in range(14):
        predicted_price = model.predict([X_input, stock_idx_input])
        predicted_prices.append(predicted_price[0][0])
        
        # Update input data for the next prediction
        new_data_point = np.array([predicted_price[0][0]]).reshape(1, 1, 1)
        X_input = np.concatenate([X_input[:, 1:, :], new_data_point], axis=1)
    
    # Inverse transform the prediction to get the actual price
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    
    return predicted_prices, close_prices[-14:], stock_df

# Display prediction button and plot
if predict_button:
    try:
        predicted_prices, past_prices, stock_df = predict_stock_prices(stock_choice)
        
        if predicted_prices is not None and past_prices is not None:
            # Prepare date range for past and future predictions
            last_14_days = stock_df.index[-14:]
            future_dates = [last_14_days[-1] + timedelta(days=i) for i in range(1, 15)]
            
            # Create the Plotly figure
            fig = go.Figure()
            
            # Plot past prices (last 14 days)
            fig.add_trace(go.Scatter(x=last_14_days, y=past_prices.flatten(), mode='lines', name='Past Prices', line=dict(color='blue')))
            
            # Prepare color list for predicted prices
            predicted_colors = ['green' if predicted_prices[i] > past_prices[-1] else 'red' for i in range(14)]

            # Plot predicted prices (next 14 days) as a continuous line
            fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Prices',
                                     line=dict(color='red' if predicted_prices[-1] < past_prices[-1] else 'green')))

            # Update the layout
            fig.update_layout(title=f'{stock_choice} Stock Price Prediction',
                              xaxis_title='Date',
                              yaxis_title='Price',
                              xaxis_rangeslider_visible=True,
                              plot_bgcolor='rgba(0,0,0,0)',
                              template='plotly_white')
            
            # Show the plot
            st.plotly_chart(fig)
            
            # Prepare the DataFrame for predicted prices
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predicted_prices.flatten()
            })
            
            # Display the prediction table
            st.subheader("Predicted Prices for the Next 14 Days")
            st.table(prediction_df)
            
            st.write(f"Predicted closing prices for the next 14 days are displayed in the graph and the table.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
