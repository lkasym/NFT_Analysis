Title: Demystifying the NFT Marketplace: A Comprehensive Data-Driven Analysis of Sales, Trends, and Stakeholder Behavior

Authors: Lakshit Mundra & Parth Santosh Tripathi
PRN: 059 & 071

### Introduction
In the digital realm, Non-Fungible Tokens (NFTs) have ushered in a revolutionary era where technology, finance, and digital ownership converge. Unlike cryptocurrencies like Bitcoin and Ethereum, NFTs stand out due to their non-interchangeable nature, representing assets with unique values and information crucial for establishing and verifying digital item ownership. The NFT market has blossomed into a hub of creativity, investment, and innovation, offering a new platform for artists, musicians, and content creators to monetize their creations. However, with the rapid rise, several challenges and uncertainties have emerged. This exploration aims to shed light on the NFT marketplace, analyzing sales, trends, and stakeholder behavior through a data-driven lens.

### Problem Statement
The burgeoning NFT realm has introduced a fresh way of owning digital assets, leading to a marketplace teeming with innovation, creativity, and investment prospects. However, as it evolves, it presents a range of challenges and dynamics requiring a deeper examination. This analysis aims to delve into market dominance, factors affecting prices, market volatility, the speculative frenzy, and investor strategies that collectively shape the NFT marketplace.

### Justification
The soaring financial activity in the NFT market, reaching billions of dollars in sales in 2021, underscores the necessity for a comprehensive analysis. By examining NFT sales data, stakeholders can uncover patterns, trends, and factors influencing digital asset valuations. Furthermore, understanding stakeholder behavior and currency dynamics is vital for risk management, predictive analysis, and gaining a nuanced understanding of market preferences and trends.

### Dataset Description
The dataset utilized for this analysis is sourced from Kaggle and encompasses various aspects of NFT transactions, providing a window into the marketplace. It contains numerous variables such as sales_datetime, asset.id, asset.name, total_price, payment_token.name, and several others which are instrumental in dissecting the NFT market dynamics.

Dataset: [OpenSea NFT Sales 2019-2021 (Kaggle)](https://www.kaggle.com/datasets/bryanw26/opensea-nft-sales-2019-2021?resource=download)

### Data Preprocessing
In preparing the data for analysis, several steps were undertaken to address missing values and ensure data consistency. This included replacing missing asset.id values with a sequence, filling missing asset.name and asset.collection.name with 'Unknown', dropping the column 'asset.collection.short_description', among other preprocessing steps to ensure the integrity and accuracy of our analysis.

### Exploratory Data Analysis (EDA)
Our EDA unveiled a multitude of insights:

1. **Distribution of NFT Categories:** This graph delineated the various categories within the NFT marketplace, offering a glimpse into the diversity of assets available.
2. **Distribution of Payment Tokens:** A depiction of the different payment tokens used, illuminating the interplay between cryptocurrencies and NFT transactions.
3. **Price Distribution in Ether:** This graph showcased the range of prices NFTs command in the marketplace, highlighting the market's speculative nature.
4. **Number of Sales by Month:** A year-on-year comparison of sales volume, revealing the market's growth trajectory.
5. **Time Series of Total Sales:** This provided a chronological view of total sales, underscoring market trends over time.
6. **Top 10 NFT Collections by Sales Volume:** Highlighting the most sought-after collections, indicating market preferences.
7. **Price Over Time of Top 10 Traded NFT Assets:** Showcasing the price trends of the most traded assets, offering a glimpse into the market's valuation dynamics.
8. **Top 10 Most Active Creator/Traders:** Identifying the market's most active participants, shedding light on stakeholder behavior.
9. **Most Traded Collections and NFTs of Top 10 Trader/Creators:** Delving into the favorite collections and NFTs of the top traders, indicating market trends.
10. **Average Selling Price by Top 10 Active Users:** This graph provided insight into the pricing strategies of the most active sellers.
11. **Average Selling Price of NFTs by Top Sellers Over Time:** Displaying the evolving pricing trends among top sellers.
12. **Average ROI for Top Sellers:** Highlighting the return on investment for top sellers, indicating the profit potential within the market.
13. **Average Buying and Selling Price of Top 10 Most Active Buyers:** This comparison offered a glimpse into the buying and selling dynamics among active buyers.

### Conclusion
Our analysis peels back the layers of the NFT marketplace, offering a data-driven glimpse into its intricacies. From uncovering market dominance to understanding pricing dynamics, the exploration provides a roadmap for stakeholders to navigate this burgeoning digital economy. The journey toward unraveling the NFT marketplace's potential is fraught with both promise and peril. Continuous exploration, education, and ethical frameworks are pivotal in fostering a sustainable and responsible NFT ecosystem.

### GitHub Link
For a deeper dive into our analysis, the dataset, and Python code, feel free to explore our [GitHub repository](https://github.com/lkasym/NFT_Analysis).

We invite our readers to delve into the data, engage with our findings, and contribute to the ongoing discourse surrounding the NFT marketplace. Your insights and perspectives are invaluable in unraveling the complexities of this digital frontier.
