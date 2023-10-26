import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pickle

# Load the models
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

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

def predict_price_lstm(nft_name, data):
    if nft_name not in data['asset.name'].values:
        return None

    look_back = 1
    nft_data = data[data['asset.name'] == nft_name].sort_values(by='sales_datetime')['price_in_ether'].values
    nft_data = nft_data.reshape(-1, 1)
    
    if len(nft_data) <= look_back + 2:
        return "Insufficient data for LSTM prediction"

    train_size = int(len(nft_data) * 0.7)
    train, test = nft_data[0:train_size,:], nft_data[train_size:len(nft_data),:]

    if len(train) <= look_back + 2 or len(test) <= look_back + 2:
        return "Insufficient data for LSTM prediction after splitting"

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
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0)
    
    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    
    return testPredict[-1][0]

st.title("NFT Explorer and Price Predictor")

# Sidebar Selection
st.sidebar.header("NFT Data Analysis")
options = ["NFT Lookup", "Market Analysis", "User/Trader Analysis", "NFT Categories"]
selection = st.sidebar.selectbox("Choose an option:", options)

if selection == "NFT Lookup":
    # NFT Lookup Code
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
            if st.button("Rough Price Prediction (Linear Regression)"):
                price = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
                st.write(f"Predicted Price in Ether: {price[0]}")
            
            # KNN Prediction
            if st.button("Rough Price Prediction (KNN)"):
                price_knn = knn_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
                st.write(f"Predicted Price in Ether: {price_knn[0]}")

            # LSTM Prediction
            if st.button("Better Price Prediction (LSTM)"):
                predicted_price = predict_price_lstm(nft_name, nft_data)
                if isinstance(predicted_price, str):
                    st.write(predicted_price)
                else:
                    st.write(f"Predicted Price in Ether (using LSTM): {predicted_price}")

            # Previous Owners
            previous_owners = selected_nft[['sales_datetime', 'seller.user.username']].sort_values(by='sales_datetime')
            st.table(previous_owners)

elif selection == "Market Analysis":
    # Market Analysis Code
    st.write("Market Analysis")
    collections = nft_data['asset.collection.name'].value_counts().reset_index()
    collections.columns = ['Collection Name', 'Number of Sales']
    collections = collections[~collections['Collection Name'].isin(['uncategorized', 'unknown'])]
    st.table(collections)

elif selection == "User/Trader Analysis":
    # User/Trader Analysis Code
    st.write("User/Trader Analysis")
    # Placeholder for User/Trader Analysis code

elif selection == "NFT Categories":
    # NFT Categories Code
    st.write("NFT Categories Analysis")
    categories = nft_data['Category'].value_counts().reset_index()
    categories.columns = ['Category Name', 'Number of Sales']
    categories = categories[~categories['Category Name'].isin(['uncategorized', 'unknown'])]
    st.table(categories)
