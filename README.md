

# NFT Explorer and Price Predictor



This application provides insights into the NFT market by allowing users to predict NFT prices, analyze market trends, view top sellers and buyers, explore NFT categories, and browse a curated gallery of NFTs.

## Features

- Price Prediction: Uses Linear Regression, KNN, and LSTM to predict the price of an NFT.
- Market Analysis: Displays the top NFT collections by sales volume.
- User/Trader Analysis: Showcases the most active sellers and buyers in the NFT market.
- NFT Categories:Visualizes the distribution of different categories of NFTs.
- NFT Gallery: Explore a curated selection of NFT images.

## Installation

1. Clone the repository:

git clone [https://github.com/lkasym/NFT_Analysis]


2. Navigate to the directory:

cd path_to_directory


3. Install the required libraries:

pip install -r requirements.txt




Run the Streamlit app with:

streamlit run streamli_nft.py


Open your browser and go to `http://localhost:8501` to view the app.

## Data

The data used in this application comes from `Processed_OpenSea_NFT_1_Sales.csv`. It contains detailed information about NFT sales, including asset names, sale dates, prices, sellers, and more.

## Disclaimer

The prediction models are trained on data from 2019-2021. Predictions might not be accurate for the current date. Always conduct your own research before making any investment decisions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

