import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import pickle

# Load the models and data
lr_model = pickle.load(open("linear_regression_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))

data_path = "Processed_OpenSea_NFT_1_Sales.csv"
nft_data = pd.read_csv(data_path)
nft_data['price_in_ether'] = nft_data['total_price'] / 1e18

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Function to predict price using LSTM
def predict_price_lstm(nft_name, data):
    if nft_name not in data['asset.name'].values:
        return None

    look_back = 1
    nft_data_selected = data[data['asset.name'] == nft_name].sort_values(by='sales_datetime')['price_in_ether'].values
    nft_data_selected = nft_data_selected.reshape(-1, 1)

    if len(nft_data_selected) <= look_back + 2:
        return "Insufficient data for LSTM prediction"

    train_size = int(len(nft_data_selected) * 0.7)
    train, test = nft_data_selected[0:train_size, :], nft_data_selected[train_size:len(nft_data_selected), :]

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

# Vibrant Styling
st.markdown("""
<style>
    body {
        color: #ffffff;
        background-color: #341f97;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1, h2 {
        color: #fbcf02;
    }
    .stButton>button {
        color: #4E8BF5;
        background-color: #fbcf02;
        border: none;
        border-radius: 5px;
    }
</style>
    """, unsafe_allow_html=True)
nft_image_paths = ["DALL·E 2023-10-27 03.41.57 - Photo of a dreamy universe where countless stars form the shape of two faces gazing into each other's eyes. Around this celestial portrait, nebulas an.png",
"DALL·E 2023-10-27 03.38.29 - Photo of a surreal library where books fly like birds, forming flocks that soar through an ethereal sky. Ladders twist and turn like DNA helixes, and .png",
"DALL·E 2023-10-27 03.38.26 - Illustration of a cosmic art studio where planets serve as paint palettes and stars as glittering brush strokes. A female artist of Hispanic descent s.png",
"DALL·E 2023-10-27 03.38.21 - Vector design of a digital time portal. Ancient structures like pyramids and coliseums merge with futuristic cities and spaceships. Diverse human silh.png",
"DALL·E 2023-10-27 03.36.20 - Illustration of a digital neural network branching out like a tree, with nodes glowing in vibrant colors. At the base is a human brain, symbolizing th.png",
"DALL·E 2023-10-27 03.35.50 - Oil painting of a serene cybernetic garden. Mechanical plants with LED flowers sway gently, and robotic butterflies flutter around. A male and female .png",
"DALL·E 2023-10-27 03.33.47 - Oil painting of an overhead view of a cricket ground in India during sunset. The stadium lights are on, casting a golden hue on the field. Diverse pla.png",
"DALL·E 2023-10-27 03.33.42 - Vector design of a cricket ball and bat intertwined with digital elements, symbolizing the fusion of traditional sport and modern technology. Surround.png",
"DALL·E 2023-10-27 03.31.47 - Vector design of two chat bubbles against a starry night. Inside one bubble is a picturesque scene of Pune with the silhouette of the 19-year-old male.png",
"DALL·E 2023-10-27 03.25.54 - Vector design of a broken digital clock, its numbers glitching and shifting erratically. The background is a chaotic swirl of dark colors, and fractur.png",
"DALL·E 2023-10-27 03.25.50 - Oil painting of a desolate urban landscape post-apocalypse. Buildings are in ruins, the sky is overcast with dark clouds, and the streets are flooded.png",
"DALL·E 2023-10-27 03.24.34 - Vector design of a steampunk-inspired mechanical heart, intricately detailed with gears, pipes, and valves. It pulsates with electric energy and is co.png",
"DALL·E 2023-10-27 03.24.30 - Watercolor painting of a serene beach where the sand is pixelated and the waves form a digital pattern. In the distance, a lighthouse beams a WiFi sig.png",
"DALL·E 2023-10-27 03.24.26 - Photo of a futuristic cityscape at dusk with neon lights reflecting on the water and flying cars zooming past skyscrapers. People of various descents .png",
]
logo_path = "DALL·E 2023-10-27 14.25.15 - Photo logo of a luxurious digital coin adorned with intricate patterns reminiscent of ancient Indian temple architecture. The coin's center features a.png"
# Main Function
def main():
    st.title("Welcome to NFT Explorer and Price Predictor!")
    st.write("""
    **Disclaimer:** This model is trained on data from 2019-2021 and might not be accurate for the current date.
    """)

    menu = ["Home", "Price Predictor", "Market Analysis", "User/Trader Analysis", "NFT Categories", "NFT Gallery"]
    choice = st.selectbox("Menu", menu)

    if choice == "Home":
        st.image(logo_path, use_column_width=True, caption="Company Logo")
        st.write("""
            Explore various functionalities. Choose from Price Predictor, Market Analysis, User/Trader Analysis, NFT Categories, and NFT Gallery.
                """)


    elif choice == "Price Predictor":
        st.subheader("Price Predictor")
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
                    price_lr = lr_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
                    st.write(f"Rough Price Prediction (using Linear Regression): {price_lr[0]}")

                # KNN Prediction
                if st.button("Predict Price with KNN"):
                    price_knn = knn_model.predict([[selected_nft['asset.num_sales'].iloc[0]]])
                    st.write(f"Rough Price Prediction (using KNN): {price_knn[0]}")

                # LSTM Prediction
                if st.button("Predict Price with LSTM"):
                    price_lstm = predict_price_lstm(nft_name, nft_data)
                    if isinstance(price_lstm, str):
                        st.write(price_lstm)
                    else:
                        st.write(f"Better Price Prediction (using LSTM): {price_lstm}")

                # Plotting the historical prices using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=selected_nft['sales_datetime'], y=selected_nft['price_in_ether'], mode='lines', name='Price in Ether'))
                fig.update_layout(title='Historical Prices of the NFT in Ether', xaxis_title='Date', yaxis_title='Price in Ether')
                st.plotly_chart(fig)

                # Previous owners
                st.write("Previous Owners")
                past_owners = nft_data[nft_data['asset.name'] == nft_name].sort_values(by='sales_datetime', ascending=False)['seller.user.username'].dropna().unique()
                if len(past_owners) > 0:
                    st.table(past_owners[:10])
                else:
                    st.write("No past owner data available.")

            else:
                st.write("NFT not found in the dataset.")

    elif choice == "Market Analysis":
        st.subheader("Market Analysis")
        collections = nft_data[nft_data['asset.collection.name'] != 'uncategorized'].groupby('asset.collection.name').size().sort_values(ascending=False).head(10)
        fig = px.bar(collections, title='Top 10 Collections by Sales Volume')
        st.plotly_chart(fig)

    elif choice == "User/Trader Analysis":
        st.subheader("User/Trader Analysis")
        
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

    elif choice == "NFT Categories":
        st.subheader("NFT Categories")
        categories = nft_data['Category'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=categories.index, values=categories.values)])
        st.plotly_chart(fig)

    elif choice == "NFT Gallery":
        st.subheader("NFT Gallery")
        st.write("Explore a curated selection of NFTs!")
         
        # Displaying the images in a grid format
        col1, col2, col3 = st.columns(3)
    
        for index, image_path in enumerate(nft_image_paths):
            with [col1, col2, col3][index % 3]:
                st.image(image_path, use_column_width=True, caption=f"NFT {index+1}")

if __name__ == "__main__":
    main()
