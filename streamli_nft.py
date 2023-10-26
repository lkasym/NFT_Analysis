import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Load the models and data
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))
lstm_model = load_model("lstm_model.h5")
with open("lstm_scaler.pkl", "rb") as f:
    lstm_scaler = pickle.load(f)

data_path = "Processed_OpenSea_NFT_1_Sales.csv"
nft_data = pd.read_csv(data_path)
nft_data['price_in_ether'] = nft_data['total_price'] / 1e18

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

st.title("NFT Explorer and Price Predictor")

# NFT Lookup
nft_name = st.text_input("Enter NFT name:")

if nft_name:
    selected_nft = nft_data[nft_data['asset.name'] == nft_name]
    
    if not selected_nft.empty:
        # Display NFT details
        st.write(f"Name: {selected_nft['asset.name'].iloc[0]}")
        st.write(f"Collection: {selected_nft['asset.collection.name'].iloc[0]}")
        st.write(f"Category: {selected_nft['Category'].iloc[0]}")
        st.write(f"Number of Sales: {selected_nft['asset.num_sales'].iloc[0]}")
        st.write(f"Last Sale Price in Ether: {selected_nft['price_in_ether'].iloc[0]}")
        
        # Linear Regression Prediction
        if st.button("Predict Price with Linear Regression"):
            price = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
            st.write(f"Predicted Price in Ether (using Linear Regression): {price[0]}")
        
        # LSTM Prediction
        if st.button("Predict Price with LSTM"):
            selected_prices = selected_nft.sort_values(by='sales_datetime')['price_in_ether'].values
            if len(selected_prices) > 2:
                selected_prices = selected_prices.reshape(-1, 1)
                selected_prices_scaled = lstm_scaler.transform(selected_prices)
                X_lstm, _ = create_dataset(selected_prices_scaled)
                X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], 1, X_lstm.shape[1]))
                predicted_price_scaled = lstm_model.predict(X_lstm)
                predicted_price = lstm_scaler.inverse_transform(predicted_price_scaled)
                st.write(f"Predicted Price in Ether (using LSTM): {predicted_price[-1][0]}")
            else:
                st.write("LSTM isn't applicable for this NFT due to insufficient transaction data.")
        
        # Plotting the historical prices using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=selected_nft['sales_datetime'], y=selected_nft['price_in_ether'], mode='lines', name='Price in Ether'))
        fig.update_layout(title='Historical Prices of the NFT in Ether', xaxis_title='Date', yaxis_title='Price in Ether')
        st.plotly_chart(fig)

    else:
        st.write("NFT not found in the dataset.")
