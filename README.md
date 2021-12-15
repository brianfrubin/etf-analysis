# Sentiment Matters
## Market Analysis with Python

By: Brian Rubin


This ReadMe contains:

* [Project Contents](#contents)
* [Problem Statement](#problemstatement)
* [Background](#background)
* [Data - APIs/Dictionaries](#data)
* [Feature Engineering](#features)
* [Modeling and Evaluation](#model)
* [Future Considerations](#conclusion)
* [Sources](#sources)

## <a name="contents"></a>Project Contents

|Name|File Type|Description|
|---|---|--|
|QuiverQuant_API|Jupyter Notebook|Initial overview of Quiver Quant API|
|StockNews_API|Jupyter Notebook|Initial overview of StockNews API|
|Nasdaq_API|Jupyter Notebook|Initial overview of Nasdaq Retail Tracker API|
|Viz_EDA_Setup|Jupyter Notebook|Initial EDA / Analysis / Graphs |
|Viz_EDA_Main|Jupyter Notebook| Main notebook for functions and visualizations used for Deployment|
|PricePredictor|Jupyter Notebook|Logistic Regression Binary Classification Model|
|LITQ _ETF|Jupyter Notebook|Work in Progress - ETF Construction Tool|
|ConfigAuthToken|Jupyter Notebook|Used to save API keys|
|streamlit|folder|Contains streamlit-finance.py file for Dashboard Deployment|

## <a name="problemstatement"></a>Problem Statement

Litquidity, a popular FinMeme Instagram page, has been gaining traction and expanding its industry presence with services like ExecSum & Lit Ventures. With the recent rise in retail trading and sentiment driven ETFs, they hired us to create Pulse Analytics - a financial dashboard their users can access to analyze markets and gain some alternative data driven insights.    

## <a name="background"></a>Background

### Fundamental Analysis

Tries to identify companies that are overvalued or undervalued based on an analyst’s derived measure of intrinsic or fair market value. If a company is trading above this fair value, then an analyst can issue an underweight recommendation and vice versa. 
Fundamental analysis can be done from a top to bottom or bottom up approach. Top to bottom would look at macro factors like the economy or industry specific metrics where a bottom to top approach would begin with company specific metrics like Earnings, Cash Flows, etc. 
This can be both qualitative (company’s key management and governance) and quantitative (balance sheet, income statement analysis, etc). Usually this analysis has more of a long-term focus.

### Technical Analysis
Discipline that studies price action, volume, chart pattern recognition, sentiment and other indicators to identify opportunities or signs in the market. Indicators like momentum determine how strong or weak a stock’s price is. They measure the rate of the rise or fall of prices. TA has a big use case in Commodities and Forex markets where there is a shorter trading time horizon. Quant Trading and High Frequency Trading shops also utilize TA as they focus on quicker short term transactions. It’s important to highlight that algorithmic trading accounts for about 60-73% of total US equity trading volumes. This Dashboard will mainly focus on TA based analysis. 

### Efficient Market Hypothesis
EMH is similar to technical analysis in that proponents of this theory believe that market prices also reflect all available data, but the key difference is they don’t believe there’s any way to find alpha in the market so there’s no need for fundamental or technical analysis. They are advocates of passive investing and low cost funds. There are also three forms of EMH. Strong believe prices reflect all public and even private information. Semi believe prices reflect all publicly avail information, no fundamental or tech analysis helpful, only information that is not publicly available. And weak believe forms of fundamental analysis can be used but prices reflect all data of past prices and tech analysis can’t be effectively used. 

There is a lot of constructive debate and pros/cons for each of these. Most of Wall Street does perform fundamental analysis or technical analysis or a combination of both. It is important to also note the performance of these actively managed funds. Morningstar Active/Passive Barometer measures US active funds vs their passive peers. Only 49% of the active funds were able to outperform their average passive counterpart in 2020. 

## <a name="dict"></a>Data - APIs/Dictionaries
### Quiver Quant
An lternative data provider that is trying to bridge the gap of accessibility between wall street and main street. With their API, you can gain access to company specific Datasets like Congress transactions, Reddit sentiment, Government Contracts, Twitter followers, Insider trading, and Patents.

### Alpha Vantage
Used to retrieve historical prices & volumes. They cover most markets like equities, forex, and crypto. They also provide economic, technical, and fundamental information. 

### Nasdaq Retail Activity Tracker
Dataset that tracks over 30bn  

### Stock News API
Provides market and company specific news headlines with sentiment scores for each article piece. 


## <a name="features"></a>Feature Engineering
Price / Volume : we used the open, high, low, and adjusted closing prices which reflect any corporate actions like dividend payments or stock splits. Volume just reflects the total number of daily traded shares. 

Sentiment : Quiver Quant’s data for Rank & Mentions which show how often a ticker was mentioned on the WSB subreddit along with a sentiment score. NASDAQ Retail was used to show total retail net buying/selling and DPI indicates the total % of OTC short positions on a given ticker. 

Moving Averages : 10 and 20 day Moving Averages. A simple rolling average of the previous days.

Momentum : Relative Strength Index. This is a popular indicator that measures the rate of price change for up days vs down days. This is used to evaluate if a stock is entering overbought or oversold territory. Simply put it takes the average sum of gains over a given period and divides that by the average sum of losses over the same period. This is then banded into a 1-100 range for analysis. 

## <a name="model"></a>Modeling and Evaluation

For modeling, I decided to use a simple Logistic Regression binary classification model to predict if tomorrow’s stock price would go up or down. This was done using Facebook’s historical daily prices from June 26th 2020 to December 10th 2021, accounting for 368 trading days. The purpose of this wasn’t to actually claim any form of success in predicting stock price movements, but to analyze how the model performs based on given inputs to see if there is any change by including / excluding some of these indicators. 

The first model only used price and volume resulting in a 53%/58% (training/testing) results for a 1 day horizon. This model was underfit, so I moved on to incorporating all the other technical and sentiment features previously discussed to see if it would perform better. Using all the features resulted in a much better model 58%/60% (training/testing), but still underfit. 
I also changed the time frame to 1 week predictions using all the features. This had 57%/54% (training/testing) scores - first overfit model. 

I do want to highlight that even though these scores seem pretty good for predicting price movements, this was a very basic construction that doesn’t reflect the magnitude of price movements on false positive or false negative predictions. This is strictly being used to show the effect of sentiment/technical indicators inclusion. 

## <a name="conclusion"></a>Future Considerations
Expand coverage of assets to include Foreign Exchange, Fixed Income & Crypto. Would incorporate a live pricing feature and Litquidity's Exec Sum Newsletter as well. 
A few reach goals are to have a Portfolio Construction Tool where users can create custom portfolios and track performance over time vs. a benchmark. Finally, I would incorporate a community aspect to it by building out a messaging system for users to connect with one another. 

## <a name="sources"></a>Sources

* Institutional Investor: https://www.institutionalinvestor.com/article/b1np5qvh2x11fc/It-s-Time-to-Cash-In-on-Big-Data
* Institutional Investor: https://www.institutionalinvestor.com/article/b1mf7j9918j9lg/How-Much-Are-Managers-Paying-for-Data 
* Quiver Quant / Alpaca / TradingView Presentation: https://www.youtube.com/watch?v=kRQ72kovnlY
* Evidence Investor: https://www.evidenceinvestor.com/morningstar-active-passive-barometer/ 
* Mordor Intelligence: https://www.mordorintelligence.com/industry-reports/algorithmic-trading-market 
