📈 **Stock Market Prediction with Sentiment Analysis**
======================================================

📌 **Project Overview**
-----------------------

Stock market prediction is a challenging task due to the **high volatility** of stock prices. Traditional methods that rely solely on **historical price trends** fail to account for **external factors** such as economic conditions, geopolitical risks, and media influence.

This project enhances **time-series forecasting models** by integrating **sentiment analysis of market news data**. Using **Yahoo! Finance API** (`yfinance`) for market news and **TextBlob** for sentiment analysis, the model adjusts predictions based on news sentiment to improve stock price forecasts.

* * * * *

🚀 **Features**
---------------

✔ **Time-Series Forecasting**: Uses **GRU, ARIMA, and SARIMA** models for stock price prediction.\
✔ **Sentiment Analysis**: Applies **TextBlob** to analyze news headlines and compute sentiment polarity.\
✔ **Ensemble Approach**: Adjusts predicted stock prices using **sentiment polarity scores**.\
✔ **Market News Integration**: Extracts news data from **Yahoo! Finance** API.\
✔ **Comparison Testing**: Evaluates price predictions **with and without** sentiment analysis.

* * * * *

📊 **Dataset & APIs**
---------------------

### **Market Data Source**

-   **Yahoo! Finance API (`yfinance`)**: Retrieves stock price history and related news articles.

### **Market News Data Format**

Each article contains:

json

CopyEdit

`{
  "title": "Tesla Woes Bolster Appeal of Top China EV Maker BYD: Tech Watch",
  "publisher": "Bloomberg",
  "link": "https://finance.yahoo.com/news/tesla-woes-bolster-appeal-top-020000753.html",
  "providerPublishTime": 1698638003,
  "relatedTickers": ["TSLA", "BYDDY"]
}`

-   **We extract and analyze the `title` field** to determine sentiment.

* * * * *

⚙️ **Sentiment Analysis**
-------------------------

📌 **Library Used:** [`TextBlob`](https://textblob.readthedocs.io/en/dev/)\
📌 **Polarity Score:**

-   `+1.0` → Strongly Positive
-   `0.0` → Neutral
-   `-1.0` → Strongly Negative
