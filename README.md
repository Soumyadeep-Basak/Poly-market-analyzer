# Stock-price-prediction

## Overview

The **Stock Price Prediction** project is a web application that leverages machine learning to predict future stock prices for selected companies. Users can select a stock, view past prices, and see predictions for the next 14 days, along with a visual representation of the predicted trends.

## Features

- **Stock Selection**: Choose from a predefined list of stocks .
- **Future Predictions**: Get predictions for the next 14 days based on historical data.
- **Dynamic Visualization**: Interactive graphs displaying past and predicted stock prices, with color-coded trends (red for declines, green for increases).
- **Data Table**: A comprehensive table at the bottom presenting the predicted stock prices for the next 14 days.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow for predictive modeling
- **Data Source**: Yahoo Finance API for historical stock data
- **Visualization**: Plotly for interactive graphs
- **Data Processing**: Pandas and NumPy for data manipulation

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
2. pip install -r requirements.txt
3. streamlit run app.py
