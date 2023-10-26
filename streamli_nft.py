import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Load the Linear Regression model
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))

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

option = st.sidebar.selectbox(
    "Choose Analysis",
    ["NFT Lookup", "Trends and Analysis", "Market Analysis", "User/Trader Analysis", "NFT Categories"]
)

if option == "NFT Lookup":
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
            
            # Display previous owners
            previous_owners = selected_nft['seller.user.username'].unique()
            st.write("Previous Owners:", ", ".join(previous_owners))
            
            # Linear Regression Prediction
            if st.button("Predict Price with Linear Regression"):
                price = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
                st.write(f"Predicted Price in Ether (using Linear Regression): {price[0][0]}")
            
            # LSTM Prediction
            if st.button("Predict Price with LSTM"):
                predicted_price = predict_price_lstm(nft_name, nft_data)
                if isinstance(predicted_price, str):
                    st.write(predicted_price)
                else:
                    st.write(f"Predicted Price in Ether (using LSTM): {predicted_price}")

            # Plotting the historical prices using Plotly
            daily_avg_prices = selected_nft.groupby('sales_datetime')['price_in_ether'].mean().reset_index()
            fig = px.line(daily_avg_prices, x='sales_datetime', y='price_in_ether', title='Average Daily NFT Sales Prices in Ether')
            st.plotly_chart(fig)

        else:
            st.write("NFT not found in the dataset.")

# ... [Rest of the code for other options like "Trends and Analysis", "Market Analysis", etc.]
# ... [The code from before, up to the "NFT Lookup" section]

elif option == "Trends and Analysis":
    st.header("Time Series Analysis of NFT Sales Prices")
    
    # Calculate daily average prices
    daily_avg_prices = nft_data.groupby('sales_datetime')['price_in_ether'].mean().reset_index()
    fig = px.line(daily_avg_prices, x='sales_datetime', y='price_in_ether', title='Average Daily NFT Sales Prices in Ether')
    st.plotly_chart(fig)

elif option == "Market Analysis":
    st.header("Market Analysis: Collections in Demand")
    
    # Count sales by collection
    collections = nft_data[nft_data['asset.collection.name'] != 'unknown'].groupby('asset.collection.name').size().sort_values(ascending=False).head(10)
    fig = px.bar(collections, title='Top 10 Collections by Sales Volume')
    st.plotly_chart(fig)

elif option == "User/Trader Analysis":
    st.header("Top Traders in the NFT Market")
    
    # Identify top traders by volume
    top_traders = nft_data.groupby('seller.user.username').size().sort_values(ascending=False).head(10)
    fig = px.bar(top_traders, title='Top 10 Traders by Sales Volume')
    st.plotly_chart(fig)

elif option == "NFT Categories":
    st.header("NFT Sales by Category")
    
    # Exclude unknown and uncategorized entries
    categories = nft_data[~nft_data['Category'].isin(['unknown', 'uncategorized'])]
    cat_counts = categories.groupby('Category').size().sort_values(ascending=False)
    fig = px.pie(cat_counts, values=cat_counts.values, names=cat_counts.index, title='Distribution of Sales Across NFT Categories')
    st.plotly_chart(fig)

# ... [End of Streamlit code]

