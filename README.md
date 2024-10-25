# Stock Price Prediction

## Overview

The **Stock Price Prediction** project is a web application that utilizes machine learning to forecast future stock prices for selected companies. Users can easily select a stock, view historical prices, and see predictions for the next 14 days, along with a visual representation of the expected trends.

## Approach

The project employs a machine learning model trained on historical stock price data to predict future prices. The main steps involved in the approach are:

1. **Data Acquisition**: Historical stock prices are fetched using the Yahoo Finance API. This data forms the basis for training our predictive model.

2. **Data Preprocessing**: The historical data is cleaned and processed using Pandas and NumPy. This involves handling missing values, scaling features, and preparing the dataset for training.

3. **Model Training**: A deep learning model LSTM for multi-stock prediction is trained on the processed data to learn patterns in stock price movements.

4. **Prediction Generation**: After training, the model is used to predict stock prices for the next 14 days based on the historical data.

5. **Visualization**: The past and predicted stock prices are visualized using Plotly. Interactive graphs display trends in the data, color-coded to indicate increases (green) and decreases (red).

6. **User Interface**: The application uses Streamlit, allowing users to select a stock, view historical trends, and see future predictions in an intuitive and user-friendly interface.

## Features

- **Stock Selection**: Choose from a predefined list of stocks.
- **Future Predictions**: Get predictions for the next 14 days based on historical data.
- **Dynamic Visualization**: Interactive graphs displaying past and predicted stock prices, with color-coded trends (red for declines, green for increases).
- **Data Table**: A comprehensive table at the bottom presenting the predicted stock prices for the next 14 days.

## Tech Stack

- **Frontend**: Streamlit for user interface
- **Backend**: TensorFlow for predictive modeling
- **Data Source**: Yahoo Finance API for historical stock data
- **Visualization**: Plotly for interactive graphs
- **Data Processing**: Pandas and NumPy for data manipulation

## Getting Started

To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2.Navigate to the project directory:
  ```bash
   cd stock-price-prediction
   ```
3.Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4.Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
