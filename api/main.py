from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI()

# Stock clusters
banking_stocks = [
    'TECHM.NS', 'INFY.NS', 'TCS.NS', 'HCLTECH.NS', 'WIPRO.NS',
    'CIPLA.NS', 'SUNPHARMA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'M&M.NS', 'MARUTI.NS',
    'BRITANNIA.NS', 'HINDUNILVR.NS', 'BAJFINANCE.NS', 'TITAN.NS', 'ASIANPAINT.NS', 'TATACONSUM.NS',
    'BPCL.NS', 'HINDPETRO.NS', 'NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'GAIL.NS', 'ONGC.NS',
    'HINDALCO.NS', 'JSWSTEEL.NS', 'SHREECEM.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'TATAMOTORS.NS', 'GRASIM.NS', 'LT.NS', 'ADANIPORTS.NS', 'TATAPOWER.NS',
    'AXISBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'HDFCBANK.NS', 'INDUSINDBK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'RELIANCE.NS'
]

# Create cluster mapping
cluster_mapping = {}
for i, stock in enumerate(banking_stocks):
    cluster_mapping[stock] = i // 5 + 1  # 5 stocks per cluster

class PredictionRequest(BaseModel):
    stock_name: str
    seq_length: int = 30
    model_type: int  # 0 for dummy, 1 for cluster-based, 2 for single stock

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

def get_stock_data(stock_name: str, days: int = 30) -> np.ndarray:
    """Fetch stock data for the last n days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days*2)  # Fetch extra days for safety
    
    stock = yf.Ticker(stock_name)
    hist = stock.history(start=start_date, end=end_date)
    
    # Get the last n days of data
    data = hist[['Open', 'Close', 'Volume']].values[-days:]
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def get_cluster_data(cluster_no: int, days: int = 30) -> np.ndarray:
    """Fetch data for all stocks in a cluster"""
    cluster_stocks = [stock for stock, cluster in cluster_mapping.items() if cluster == cluster_no]
    cluster_data = []
    
    for stock in cluster_stocks:
        stock_data = get_stock_data(stock, days)
        cluster_data.append(stock_data)
    
    return np.array(cluster_data)

def load_model(model_type: int, seq_length: int, cluster_no: int) -> tf.keras.Model:
    """Load the appropriate model based on type and parameters"""
    if model_type == 0:
        return None  # Dummy model
    
    model_path = f"models/model_{model_type}_{seq_length}_{cluster_no}.h5"
    try:
        return tf.keras.models.load_model(model_path)
    except:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Validate stock name
    if request.stock_name not in cluster_mapping:
        raise HTTPException(status_code=400, detail="Invalid stock name")
    
    # Get cluster number
    cluster_no = cluster_mapping[request.stock_name]
    
    # Handle dummy model
    if request.model_type == 0:
        return PredictionResponse(prediction=1, confidence=1.0)
    
    # Load model
    model = load_model(request.model_type, request.seq_length, cluster_no)
    
    # Prepare input data based on model type
    if request.model_type == 1:
        # Cluster-based model
        input_data = get_cluster_data(cluster_no, request.seq_length)
        input_data = input_data.reshape(1, *input_data.shape)
    else:
        # Single stock model
        input_data = get_stock_data(request.stock_name, request.seq_length)
        input_data = input_data.reshape(1, *input_data.shape)
    
    # Make prediction
    prediction = model.predict(input_data)
    binary_prediction = int(prediction[0][0] > 0.5)
    confidence = float(prediction[0][0] if binary_prediction else 1 - prediction[0][0])
    
    return PredictionResponse(prediction=binary_prediction, confidence=confidence)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 