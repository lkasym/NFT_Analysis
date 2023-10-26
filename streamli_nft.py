import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt




def predict_price_lstm(nft_name, data):
    if nft_name not in data['asset.name'].values:
        return None

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 1
    nft_data = data[data['asset.name'] == nft_name].sort_values(by='sales_datetime')['price_in_ether'].values
    nft_data = nft_data.reshape(-1, 1)
    
    if len(nft_data) <= look_back + 2:
        return {"Predicted Prices": "Insufficient data for LSTM prediction"}

    train_size = int(len(nft_data) * 0.7)
    train, test = nft_data[0:train_size,:], nft_data[train_size:len(nft_data),:]

    if len(train) <= look_back + 2 or len(test) <= look_back + 2:
        return {"Predicted Prices": "Insufficient data for LSTM prediction after splitting"}

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)
    
    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    
    return {"Predicted Prices": testPredict[:,0].tolist()}

# Load the models and data
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))
data_path = "Processed_OpenSea_NFT_1_Sales.csv"
nft_data = pd.read_csv(data_path)
nft_data['price_in_ether'] = nft_data['total_price'] / 1e18

st.title("NFT Explorer and Price Predictor")

nft_name = st.text_input("Enter NFT name:")

if nft_name:
    selected_nft = nft_data[nft_data['asset.name'] == nft_name]
    
    if not selected_nft.empty:
        
        
        st.write(f"Name: {selected_nft['asset.name'].iloc[0]}")
        st.write(f"Collection: {selected_nft['asset.collection.name'].iloc[0]}")
        st.write(f"Category: {selected_nft['Category'].iloc[0]}")
        st.write(f"Number of Sales: {selected_nft['asset.num_sales'].iloc[0]}")
        st.write(f"Last Sale Price in Ether: {selected_nft['price_in_ether'].iloc[0]}")

        plt.figure(figsize=(10, 6))
        plt.plot(selected_nft['sales_datetime'], selected_nft['price_in_ether'])
        plt.xlabel('Date')
        plt.ylabel('Price in Ether')
        plt.title('Historical Prices')
        st.pyplot(plt)

        price = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
        st.write(f"Predicted Price in Ether (using Linear Regression): {price[0][0]}")
        
        lstm_result = predict_price_lstm(nft_name, nft_data)
        if lstm_result:
            st.write(f"Predicted Price in Ether (using LSTM): {lstm_result['Predicted Prices'][-1]}")
        else:
            st.write("LSTM isn't applicable for this NFT due to insufficient transaction data.")
    else:
        st.write("NFT not found in the dataset.")
