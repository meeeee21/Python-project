import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import yfinance as yf

# Load the stock data
ticker = 'AAPL'
stock_data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

print(stock_data.head())

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.title(f'{ticker} Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.plot(stock_data['50_MA'], label='50-day Moving Average')
plt.plot(stock_data['200_MA'], label='200-day Moving Average')
plt.title(f'{ticker} Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

features = stock_data[['Close', '50_MA', '200_MA', 'Volume']].dropna()

kmeans = KMeans(n_clusters=3)
features['Cluster'] = kmeans.fit_predict(features[['Close', '50_MA', '200_MA', 'Volume']])

plt.figure(figsize=(12, 6))

plt.scatter(features['Close'], features['Volume'], c=features['Cluster'], cmap="viridis")
plt.colorbar()
plt.title(f'KMeans Clustering of {ticker} Stock Data')
plt.xlabel('Closing Price')
plt.ylabel('Volume')

centroids = kmeans.cluster_centers_
for idx, centroid in enumerate(centroids):
    plt.scatter(
        centroid[0],
        centroid[3],
        color='red',
        marker='X',
        s=200,
        label=f"Centroid {idx + 1}"
    )

plt.legend()
plt.show()


print("Cluster Centroids (for 'Close', '50_MA', '200_MA', 'Volume'):")
print(centroids)
print("Centroids shape:", centroids.shape)
