import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go
import pickle

# Load the models and data
data_path = "Processed_OpenSea_NFT_1_Sales.csv"
nft_data = pd.read_csv(data_path)
nft_data['price_in_ether'] = nft_data['total_price'] / 1e18
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

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

selection = st.sidebar.selectbox("Choose an Analysis Option:", ["NFT Lookup", "Market Analysis", "User/Trader Analysis", "NFT Categories"])

if selection == "NFT Lookup":
    # NFT Lookup Code
    nft_name = st.text_input("Enter NFT name:")
    if nft_name:
        selected_nft = nft_data[nft_data['asset.name'] == nft_name]
        if not selected_nft.empty:
            # Display NFT details and previous owners
            st.write("NFT Details")
            st.table(selected_nft[['asset.name', 'asset.collection.name', 'Category', 'asset.num_sales', 'price_in_ether']])
            # Price prediction using Linear Regression
            price_lr = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
            st.write(f"Rough Price Prediction (using Linear Regression): {price_lr[0][0]}")
            # Price prediction using KNN
            price_knn = knn_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
            st.write(f"Rough Price Prediction (using KNN): {price_knn[0][0]}")
            # Price prediction using LSTM
            price_lstm = predict_price_lstm(nft_name, nft_data)
            if isinstance(price_lstm, str):
                st.write(price_lstm)
            else:
                st.write(f"Better Price Prediction (using LSTM): {price_lstm}")

elif selection == "Market Analysis":
    # Market Analysis Code
    st.write("Market Analysis")
    top_categories = nft_data['Category'].value_counts().drop(['uncategorized', 'unknown'])
    st.write("Most Popular Categories")
    fig = go.Figure(data=[go.Bar(x=top_categories.index, y=top_categories.values)])
    st.plotly_chart(fig)

elif selection == "User/Trader Analysis":
    st.write("User/Trader Analysis")
    # Most selling collections
    top_collections = nft_data['asset.collection.name'].value_counts().drop(['uncategorized', 'unknown']).head(10)
    st.write("Most Selling Collections:")
    fig = go.Figure(data=[go.Bar(x=top_collections.index, y=top_collections.values)])
    st.plotly_chart(fig)
    # Most active sellers
    top_sellers = nft_data['seller.user.username'].value_counts().head(10)
    st.write("Most Active Sellers:")
    fig = go.Figure(data=[go.Bar(x=top_sellers.index, y=top_sellers.values)])
    st.plotly_chart(fig)
    # Most active buyers
    top_buyers = nft_data[nft_data['winner_account.address'].isin(nft_data['seller.address'])]['winner_account.address'].value_counts().head(10)
    buyer_names = []
    for address in top_buyers.index:
        name = nft_data[nft_data['seller.address'] == address]['seller.user.username'].iloc[0]
        if pd.isna(name):
            buyer_names.append(address)
        else:
            buyer_names.append(name)
    st.write("Most Active Buyers:")
    fig = go.Figure(data=[go.Bar(x=buyer_names, y=top_buyers.values)])
    st.plotly_chart(fig)

elif selection == "NFT Categories":
    # NFT Categories Code
    st.write("NFT Categories")
    categories = nft_data['Category'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=categories.index, values=categories.values)])
    st.plotly_chart(fig)
