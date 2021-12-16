# IMPORTS/KEYS
import streamlit as st
import requests

nasdaq_auth_key = st.secrets['nasd_key']
quant_auth_key = st.secrets['quant_key']
alpha_auth_key = st.secrets['alpha_key']
snap_auth_key = st.secrets['snap_key']

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from io import BytesIO
from datetime import date
import time

# FUNCTIONS CODE
# creates dataframe for given ticker. 14yr historical daily price (OLHC), vol, cum return, RSI
def alpha_historical_daily(ticker, start_date, end_date):
    # Alpha vantage API request for daily adjusted prices
    token = alpha_auth_key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={ticker}&interval=5min&apikey={token}&datatype=csv'
    r = requests.get(url)

    # Converting from bytes to pandas dataframe : https://stackoverflow.com/questions/47379476/how-to-convert-bytes-data-into-a-python-pandas-dataframe
    df_daily = pd.read_csv(BytesIO(r.content))

    # Creating Sorted Datetime index based on inputted date range
    df_daily.set_index('timestamp', inplace=True)
    df_daily = df_daily[df_daily.index >= start_date]
    df_daily = df_daily[df_daily.index <= end_date]
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily.sort_index(inplace=True)
    df_daily.drop(columns=['dividend_amount', 'split_coefficient'], inplace=True)

    # Engineering Cumulative Return and Total Position Column for $100,000 Investment
    daily_returns = df_daily['adjusted_close'] / df_daily['adjusted_close'].iloc[0]
    df_daily['cum_return'] = daily_returns

    # Position column to track performane (Using $100,000 as initial outlay investment)
    allocation = 100_000
    df_daily['position'] = allocation * df_daily['cum_return']

    # Engineering Moving Average 10 & 20 days columns
    df_daily['MA10'] = df_daily['adjusted_close'].rolling(10).mean()
    df_daily['MA20'] = df_daily['adjusted_close'].rolling(20).mean()

    # Engineering RSI-14 Indicator and 70/30 Bands
    # RSI Calculation Formula : https://www.macroption.com/rsi-calculation/
    #RSI = 100 – 100 / ( 1 + RS )
    #RS = Relative Strength = AvgU / AvgD
    #AvgU = average of all up moves in the last N price bars
    #AvgD = average of all down moves in the last N price bars
    #N = the period of RSI
    # Manually calculating RSI-14 using SMA - code used from this post : https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    adj_close = df_daily['adjusted_close']
    # Getting price difference from previous day
    price_diff = adj_close.diff()

    # Make the positive gains (up) and negative gains (down) Series
    up, down = price_diff.clip(lower=0), price_diff.clip(upper=0).abs()

    # Calculate the RSI based on SMA
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    rsi_sma = 100.0 - (100.0 / (1.0 + rs))
    df_daily['RSI_14'] = rsi_sma

    # Finalizing with a new Cumulative Return % column
    df_daily['cum_return_percent'] = df_daily.cum_return -1

    return df_daily
# takes ticker & date range. spits out tech chart w/ price, vol, RSI, candlestick&line graph
def ticker_analysis(ticker, start_date, end_date):
    # Calling alpha_historical_daily function to get all ticker data
    ticker1_final = alpha_historical_daily(ticker, start_date, end_date)

    # Creating Plotly Subplot Using The Two Ticker DataFrames
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.5,0.15,0.2])

    #Ticker 1 details
    # Adding Line Graph
    fig.add_trace(go.Scatter(
        x=ticker1_final.index, y=ticker1_final['adjusted_close'],
        line=dict(width=2), name=f'{ticker}', visible='legendonly'), row=1, col=1)
    # Adding Candlestick Graph
    fig.add_trace(go.Candlestick(
    x=ticker1_final.index, open=ticker1_final['open'],
    high=ticker1_final['high'], low=ticker1_final['low'],
    close=ticker1_final['close'], name=f'{ticker}'), row=1, col=1)
    #adding Moving Average 10day line
    fig.add_trace(go.Scatter(
        x=ticker1_final.index, y=ticker1_final['MA10'],
        line=dict(color='white', width=1), name=f'{ticker} MA(10)'))
    #adding Moving Average 20day line
    fig.add_trace(go.Scatter(
        x=ticker1_final.index, y=ticker1_final['MA20'],
        line=dict(color='yellow', width=1), name=f'{ticker} MA(20)'))
    #adding volume indicator
    fig.add_trace(go.Bar(
        x=ticker1_final.index, y=ticker1_final['volume'],
        marker_color = 'LightSkyBlue', showlegend=True, name=f'{ticker} Vol.'), row=2, col=1)
    #adding RSI indicator
    fig.add_trace(go.Scatter(
        x=ticker1_final.index, y=ticker1_final['RSI_14'],
        line=dict(color='#EF8820', width=2), showlegend=True, name=f'{ticker} RSI'), row=3, col=1)

    #setting RSI range from 0-100 as standard practice for this indicator
    fig.update_yaxes(
        range=[-10, 110], row=3, col=1)
    fig.add_hline(
        y=0, col=1, row=3, line_color="#C3BEB9", line_width=1)
    fig.add_hline(
        y=100, col=1, row=3, line_color="#C3BEB9", line_width=1)
    #adding overbought/oversold lines for RSI 30/70 levels
    fig.add_hline(
        y=40, col=1, row=3, line_color='#93a1a1', line_width=1.5, line_dash='dash')
    fig.add_hline(
        y=80, col=1, row=3, line_color='#93a1a1', line_width=1.5, line_dash='dash')

    fig.update_layout(
        height=800, width=1075, showlegend=True, xaxis_rangeslider_visible=False,
        paper_bgcolor=None, template='plotly_dark', legend_orientation="v",
        title=dict(
            text=f"<b>{ticker}<b>",
            y=0.975, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(
            yanchor="top", y=0.975, xanchor="left", x=0.0225, font={'size':12}),
        font=dict(
            family="Gotham Narrow, monospace", size=16, color="#A4A4A4"),
        margin=go.layout.Margin(
            l=60, r=25, b=30, t=60))

    fig.update_layout(xaxis_range=[f'{start_date}',f'{end_date}'])
    fig.update_xaxes(tickformat="%b\n%Y", dtick='M1')

    #updating sublot yaxis labels, used as resource : https://stackoverflow.com/questions/58849925/plotly-how-to-apply-different-titles-for-each-different-subplots
    fig['layout']['yaxis']['title']='Price'
    fig['layout']['yaxis2']['title']='Volume'
    fig['layout']['yaxis3']['title']='RSI-14'

    return fig
# chart showing cumulative return % over given date range + indicator widget for performance
def ticker_cum_return(ticker, start_date, end_date):
    # Calling alpha_historical_daily function to get all ticker data
    ticker1_final = alpha_historical_daily(ticker, start_date, end_date)

    fig = px.line(
        ticker1_final, x=ticker1_final.index, y='cum_return_percent',
        labels={
            'index': "", 'cum_return': 'Cumulative Return (%)'})

    fig.update_layout(
        title={
            'text': f'<b>{ticker}<b>', 'y':0.95, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top'})

    fig.update_layout(
        height=600, width=1000, showlegend=True,
        xaxis_rangeslider_visible=False,
        font=dict(
            family="Gotham Narrow, monospace", size=16, color="#A4A4A4"))

    fig.update_layout(margin=go.layout.Margin(
        l=30,r=30, b=10, t=60),
        paper_bgcolor=None, template='plotly_dark')
    fig.update_layout(xaxis_range=[f'{start_date}',f'{end_date}'], xaxis_visible=True)
    fig.update_xaxes(tickformat="%b\n%Y", dtick='M1')
    fig.update_yaxes(tickformat=".2%")

    fig.add_hline(
        y=0, col=1, row=1, line_color='#6EB7F8', line_width=2, line_dash='dash')

    fig.add_trace(go.Indicator(
    align = 'center', mode = "number+delta", value = ticker1_final['position'][-1], name = 'Performance', visible = True,
    title = {
        "text": "$100k Investment<br><span style='font-size:0.8em;color:gray'></span><br><span style='font-size:0.8em;color:yellow'>",
        'font':{
        'size':20,
        'color':'yellow'}, 'align':'center'},
    delta = {
        'reference': 100_000, 'relative': True, 'valueformat':'.2%',
        'font':{
        'size':40}},
    number = {
        'prefix': "$",
        'font':{
        'size':40}}))
#    domain = {
#        'x': [1, 1], 'y': [1, 1]}))

    fig['layout']['yaxis']['title']='Cumulative Return %'
    fig['layout']['xaxis']['title']=''

    return fig
# compares % returns for two tickers + SPY as benchmark
def ticker_cum_return_comp(ticker1, ticker2, start_date, end_date):
    # Calling alpha_historical_daily function to get all ticker data
    ticker1_final = alpha_historical_daily(ticker1, start_date, end_date)
    ticker2_final = alpha_historical_daily(ticker2, start_date, end_date)
    ticker3_final = alpha_historical_daily('SPY', start_date, end_date)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=ticker1_final.index, y=ticker1_final['cum_return_percent'],
    line=dict(width=2), name=f'{ticker1}', visible=True))
    fig.add_trace(go.Scatter(
    x=ticker2_final.index, y=ticker2_final['cum_return_percent'],
    line=dict(width=2), name=f'{ticker2}', visible=True))
    fig.add_trace(go.Scatter(
    x=ticker3_final.index, y=ticker3_final['cum_return_percent'],
    line=dict(width=2), name='SPY', visible='legendonly'))

    fig.update_layout(
        title={
            'text': f"<b>{ticker1}<b> | <b>{ticker2}<b>", 'y':0.95, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top'})

    fig.update_layout(
        height=600, width=1000, showlegend=True,
        xaxis_rangeslider_visible=False,
        font=dict(
            family="Gotham Narrow, monospace", size=16, color="#A4A4A4"),
        legend=dict(
            yanchor="top", y=0.975, xanchor="left", x=0.0225, font={'size':12}))

    fig.update_layout(margin=go.layout.Margin(
        l=30,r=30, b=10, t=60),
        paper_bgcolor=None, template='plotly_dark')
    fig.update_layout(xaxis_range=[f'{start_date}',f'{end_date}'], xaxis_visible=True)
    fig.update_xaxes(tickformat="%b\n%Y", dtick='M1')
    fig.update_yaxes(tickformat=".2%")

    fig.add_hline(
        y=0, col=1, row=1, line_color='#6EB7F8', line_width=2, line_dash='dash')


    fig['layout']['yaxis']['title']='Cumulative Return %'
    fig['layout']['xaxis']['title']=''

    return fig
# Price action for two tickers, dual yaxis scaled and rangeslider on bottom. Both OLHC and Line
def ticker_comparison(ticker1, ticker2, start_date, end_date):
    # Pulling 14 year historical daily price data for ticker 1, saving to df_ticker1 DataFrame
    ticker1_final = alpha_historical_daily(ticker1, start_date, end_date)
    ticker2_final = alpha_historical_daily(ticker2, start_date, end_date)

    # Creating Plotly Graph Using The Two Ticker DataFrames
    fig = make_subplots(specs=[[{'secondary_y': True}]])


    #Ticker 1 details
    fig.add_trace(go.Scatter(
        x=ticker1_final.index, y=ticker1_final['adjusted_close'],
        line=dict(width=2), name=f'{ticker1}'))
    # Adding candlestick chart
    fig.add_trace(go.Candlestick(
        x=ticker1_final.index, open=ticker1_final['open'],
        high=ticker1_final['high'], low=ticker1_final['low'],
        close=ticker1_final['close'], name=f'{ticker1}', visible='legendonly'))

    #Ticker 2 details
    # Adding line graph
    fig.add_trace(go.Scatter(
        x=ticker2_final.index, y=ticker2_final['adjusted_close'],
        line=dict(width=2), name=f'{ticker2}'), secondary_y=True)
    # Adding candlestick graph
    fig.add_trace(go.Candlestick(
        x=ticker1_final.index, open=ticker2_final['open'],
        high=ticker2_final['high'], low=ticker2_final['low'],
        close=ticker2_final['close'], name=f'{ticker2}', visible='legendonly'), secondary_y=True)

    # Customizing Graph layout and axes
    fig.update_layout(
        height=700, width=1050, showlegend=True, xaxis_rangeslider_visible=True,
        paper_bgcolor=None, template='plotly_dark', legend_orientation="v",
        title=dict(
            text=f"<b>{ticker1}<b> | <b>{ticker2}<b>",
            y=0.975, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(
            yanchor="top", y=0.975, xanchor="left", x=0.0225, font={'size':12}),
        font=dict(
            family="Gotham Narrow, monospace", size=16, color="#A4A4A4"),
        margin=go.layout.Margin(
            l=60, r=25, b=30, t=60))

    fig.update_layout(xaxis_range=[f'{start_date}',f'{end_date}'])
    fig.update_xaxes(tickformat="%b\n%Y", dtick='M1')
    fig.update_yaxes(title_text=f"{ticker1} Price", secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker2} Price", secondary_y=True)

    return fig

# SIDEBAR CODE
##silence deprecationwarnings
st.set_option('deprecation.showPyplotGlobalUse', False)
##sidebar title and subheader
st.sidebar.title('Pulse Analytics')
st.sidebar.subheader('Dashboard')
##setting up all the pages
page = st.sidebar.selectbox( '', ('Home Page', 'One Ticker', 'Two Tickers', 'Performance', 'Alt Data'))
st.sidebar.write('--------')
st.sidebar.write('--------')
##listing sources of data
st.sidebar.subheader('Data Providers')
st.sidebar.info("[Quiver Quant](https://www.quiverquant.com/)")
st.sidebar.info("[Alpha Vantage](https://www.alphavantage.co/)")
st.sidebar.info("[Nasdaq Retail Tracker](https://data.nasdaq.com/databases/RTAT/data)")
st.sidebar.info("[Stock News API](https://stocknewsapi.com/)")
st.sidebar.write('--------')
st.sidebar.write('--------')
##adding personal contact info
st.sidebar.subheader('Contact')
st.sidebar.info('Brian Rubin')
cols1, cols2 = st.sidebar.columns(2)
cols1.warning("[![Foo](https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-48.png)](https://www.linkedin.com/in/brian-f-rubin/)")
cols2.warning("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/brianfrubin/etf-analysis)")

# HOMEPAGE CODE
if page == 'Home Page':
    max_width= 900
    padding_top= 5
    padding_right=1
    padding_left=1
    padding_bottom=10
    COLOR = '#151620'
    BACKGROUND_COLOR = '#151620'
    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )

    # Setting up API pull request
    url = f'https://stocknewsapi.com/api/v1/category?section=general&items=50&token={snap_auth_key}'
    r = requests.get(url)
    top_news = r.json()
    # Creating Sentiment Count for Top 50 Headlines
    sentiment = []
    for i in range(len(top_news['data'])):
        sentiment.append(top_news['data'][i]['sentiment'])
    df = pd.DataFrame(data=np.array(sentiment), index=None,  columns=['sentiment'])
    st.title('MARKET HEADLINES')
    # Displaying Sentiment Count for Top 50 Headlines
    neu, pos, neg = st.columns(3)
    neu.code(f'Neutral : {df.value_counts()[0]}')
    pos.success(f'Positive : {df.value_counts()[1]}')
    neg.error(f'Negative : {df.value_counts()[2]}')
    st.write('-----------')

    # API call to pull top 50 headlines with title, source, date, and sentiment score
    for i in range (50):
        response = requests.get(f"{top_news['data'][i]['image_url']}")
        picture, context = st.columns(2)
        img = Image.open(BytesIO(response.content))
    #    img = Image.open(f"[{top_news['data'][i]['title']}]")
        img = img.resize((400,300), Image.ANTIALIAS)
        #img.resize((500,500), Image.ANTIALIAS)
        picture.image(img)
        context.write(f"[{top_news['data'][i]['title']}]({top_news['data'][i]['news_url']})")
        context.caption(f"{top_news['data'][i]['text']}")
        context.caption(f"{top_news['data'][i]['source_name']} / {top_news['data'][i]['date']}")
        if top_news['data'][i]['sentiment'] == 'Neutral':
            context.code(top_news['data'][i]['sentiment'])
        elif top_news['data'][i]['sentiment'] == 'Positive':
            context.success(top_news['data'][i]['sentiment'])
        else:
            context.error(top_news['data'][i]['sentiment'])
        st.write('------------')

if page == 'One Ticker':
    # Adjusting layout, found this solution : https://discuss.streamlit.io/t/where-to-set-page-width-when-set-into-non-widescreeen-mode/959
    max_width= 1100
    padding_top= 5
    padding_right=1
    padding_left=1
    padding_bottom=10
    COLOR = '#151620'
    BACKGROUND_COLOR = '#151620'
    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )    

    st.title('SELECT YOUR TICKER')
    st.write('-----------')
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.0001)
        my_bar.progress(percent_complete + 1)
    tick, start, end = st.columns(3)
    tick = tick.text_input('Select Ticker', 'SPY')
#    start = start.date_input('Start Date')
    start = start.text_input('Start Date', '2021-01-01')
#    end = end.date_input('End Date')
    end = end.text_input('End Date', date.today())
    st.write('-----------')

    st.plotly_chart(ticker_analysis(tick, start, end))

    st.write('-----------')

    # Setting up API pull request for single ticker
    url2 = f'https://stocknewsapi.com/api/v1?tickers={tick}&items=50&token={snap_auth_key}'
    r2 = requests.get(url2)
    ticker_news = r2.json()
    # Creating Sentiment Count for Top 50 Headlines
    sentiment2 = []
    for i in range(len(ticker_news['data'])):
        sentiment2.append(ticker_news['data'][i]['sentiment'])
    df2 = pd.DataFrame(data=np.array(sentiment2), index=None,  columns=['sentiment'])
    st.title(f'{tick} HEADLINES')
    # Displaying Sentiment Count for Top 50 Headlines
    neu2, pos2, neg2 = st.columns(3)
    neu2.code(f'Neutral : {df2.value_counts()[0]}')
    pos2.success(f'Positive : {df2.value_counts()[1]}')
    neg2.error(f'Negative : {df2.value_counts()[2]}')
    st.write('-----------')

    # API call to pull top 50 headlines with title, source, date, and sentiment score
    for i in range (50):
        response2 = requests.get(f"{ticker_news['data'][i]['image_url']}")
        picture2, context2 = st.columns(2)
        img2 = Image.open(BytesIO(response2.content))
    #    img = Image.open(f"[{top_news['data'][i]['title']}]")
        img2 = img2.resize((400,300), Image.ANTIALIAS)
        #img.resize((500,500), Image.ANTIALIAS)
        picture2.image(img2)
        context2.write(f"[{ticker_news['data'][i]['title']}]({ticker_news['data'][i]['news_url']})")
        context2.caption(f"{ticker_news['data'][i]['text']}")
        context2.caption(f"{ticker_news['data'][i]['source_name']} / {ticker_news['data'][i]['date']}")
        if ticker_news['data'][i]['sentiment'] == 'Neutral':
            context2.code(ticker_news['data'][i]['sentiment'])
        elif ticker_news['data'][i]['sentiment'] == 'Positive':
            context2.success(ticker_news['data'][i]['sentiment'])
        else:
            context2.error(ticker_news['data'][i]['sentiment'])
        st.write('------------')

if page == 'Two Tickers':
        # Formatting layout
        max_width= 1100
        padding_top= 5
        padding_right=1
        padding_left=1
        padding_bottom=10
        COLOR = '#151620'
        BACKGROUND_COLOR = '#151620'
        st.markdown(
                f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {max_width}px;
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {COLOR};
                background-color: {BACKGROUND_COLOR};
            }}
        </style>
        """,
                unsafe_allow_html=True,
            )

        st.title('SELECT YOUR TICKERS')
        st.write('-----------')
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.0001)
            my_bar.progress(percent_complete + 1)

        start3, end3 = st.columns(2)
        start3 = start3.text_input('Start Date', '2021-01-01')
        end3 = end3.text_input('End Date', date.today())
        tick3, tick4 = st.columns(2)
        tick3 = tick3.text_input('Select Ticker 1', 'FB')
        tick4 = tick4.text_input('Select Ticker 2', 'BX')
        st.write('---------')
        st.markdown('<p style="text-align: left;">&nbsp; &nbsp;<span style="font-size: 22px;">👇️</span></p>', unsafe_allow_html=True)
        if st.button('Compare'):
            st.plotly_chart(ticker_cum_return_comp(tick3, tick4, start3, end3))
            st.write('---------')
            st.plotly_chart(ticker_comparison(tick3, tick4, start3, end3))
            st.write('---------')
            st.write(f'{tick3} / {tick4} HEADLINES')
            st.write('---------')
            # Setting up API pull request for multiple tickers
            url3 = f'https://stocknewsapi.com/api/v1?tickers={tick3},{tick4}&items=50&token={snap_auth_key}'
            r3 = requests.get(url3)
            tickers_news = r3.json()
            # Creating Sentiment Count for Top 50 Headlines
            sentiment3 = []
            for i in range(len(tickers_news['data'])):
                sentiment3.append(tickers_news['data'][i]['sentiment'])
            df3 = pd.DataFrame(data=np.array(sentiment3), index=None,  columns=['sentiment'])
            st.title(f'{tick3}/{tick4} Headlines')
            # Displaying Sentiment Count for Top 50 Headlines
            neu3, pos3, neg3 = st.columns(3)
            neu3.code(f'Neutral : {df3.value_counts()[0]}')
            pos3.success(f'Negative : {df3.value_counts()[1]}')
            neg3.error(f'Positive : {df3.value_counts()[2]}')
            st.write('-----------')

            # API call to pull top 50 headlines with title, source, date, and sentiment score
            for i in range (50):
                response3 = requests.get(f"{tickers_news['data'][i]['image_url']}")
                picture3, context3 = st.columns(2)
                img3 = Image.open(BytesIO(response3.content))
            #    img = Image.open(f"[{top_news['data'][i]['title']}]")
                img3 = img3.resize((400,300), Image.ANTIALIAS)
                #img.resize((500,500), Image.ANTIALIAS)
                picture3.image(img3)
                context3.write(f"[{tickers_news['data'][i]['title']}]({tickers_news['data'][i]['news_url']})")
                context3.caption(f"{tickers_news['data'][i]['text']}")
                context3.caption(f"{tickers_news['data'][i]['source_name']} / {tickers_news['data'][i]['date']}")
                if tickers_news['data'][i]['sentiment'] == 'Neutral':
                    context3.code(tickers_news['data'][i]['sentiment'])
                elif tickers_news['data'][i]['sentiment'] == 'Positive':
                    context3.success(tickers_news['data'][i]['sentiment'])
                else:
                    context3.error(tickers_news['data'][i]['sentiment'])
                st.write('------------')

if page == 'Performance':
        # Formatting layout
        max_width= 1100
        padding_top= 5
        padding_right=1
        padding_left=1
        padding_bottom=10
        COLOR = '#151620'
        BACKGROUND_COLOR = '#151620'
        st.markdown(
                f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {max_width}px;
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {COLOR};
                background-color: {BACKGROUND_COLOR};
            }}
        </style>
        """,
                unsafe_allow_html=True,
            )

        st.title('RETURN CALCULATOR')
        st.write('-----------')
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.0001)
            my_bar.progress(percent_complete + 1)
        tick2, start2, end2 = st.columns(3)
        tick2 = tick2.text_input('Select Ticker', 'SPY')
        start2 = start2.text_input('Start Date', '2021-01-01')
        end2 = end2.text_input('End Date', date.today())
        st.write('-----------')

        st.plotly_chart(ticker_cum_return(tick2, start2, end2))

if page == 'Alt Data':
    st.title('ALTERNATIVE DATA')
    st.write('-----------')
    alt_tick = st.text_input('Select Ticker', 'GME')
    st.write('-----------')
    def get_wsb_stats(alt_tick):
        url = f"https://api.quiverquant.com/beta/historical/wallstreetbets/{alt_tick}"
        headers = {'accept': 'application/json',
        'X-CSRFToken': 'TyTJwjuEC7VV7mOqZ622haRaaUr0x0Ng4nrwSRFKQs7vdoBcJlK9qjAS69ghzhFu',
        'Authorization': f'Token {quant_auth_key}'}
        r = requests.get(url, headers=headers)
        WSB_quiver = df = pd.read_json(r.content)
        return WSB_quiver
    WSB_quiver = get_wsb_stats(alt_tick)
    uno, dos = st.columns(2)
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        align = 'center', mode = "number+delta", value = WSB_quiver[['Mentions', 'Rank']].iloc[-1][0] ,
        title = {
            "text": "WSB Mentions<br><span style='font-size:0.8em;color:gray'>+ Daily Change</span><br><span style='font-size:0.8em;color:gray'>",
            'font':{
            'size':20}},
        delta = {
            'reference': WSB_quiver[['Mentions', 'Rank']].iloc[-2][0], 'relative': True, 'valueformat':'.2%',
            'font':{
            'size':40}},
        number = {
            'font':{
            'size':40}},
        domain = {
            'x': [0, 0], 'y': [0, 0.001]}))

    fig.update_layout(
        paper_bgcolor=None, height=200, width=200)

    uno.plotly_chart(fig)

    fig2 = go.Figure()
    fig2.add_trace(go.Indicator(
        align = 'center', mode = "number+delta", value = WSB_quiver[['Mentions', 'Rank']].iloc[-1][1] ,
        title = {
            "text": "WSB Rank<br><span style='font-size:0.8em;color:gray'>+ Daily Change</span><br><span style='font-size:0.8em;color:gray'>",
            'font':{
            'size':20}},
        delta = {
            'reference': WSB_quiver[['Mentions', 'Rank']].iloc[-2][1], 'relative': True, 'valueformat':'.2%',
            'font':{
            'size':40}},
        number = {
            'font':{
            'size':40}},
        domain = {
            'x': [0, 0], 'y': [0, 0.001]}))

    fig2.update_layout(
        paper_bgcolor=None, height=200, width=200)

    dos.plotly_chart(fig2)

    tres, cuatro = st.columns(2)
    fig3 = go.Figure()

    fig3.add_trace(go.Indicator(
        align = 'center', mode = "number+delta", value = WSB_quiver[['Mentions', 'Rank']].iloc[-1][0] ,
        title = {
            "text": "WSB Mentions<br><span style='font-size:0.8em;color:gray'>+ Weekly Change</span><br><span style='font-size:0.8em;color:gray'>",
            'font':{
            'size':20}},
        delta = {
            'reference': WSB_quiver[['Mentions', 'Rank']].iloc[-6][0], 'relative': True, 'valueformat':'.2%',
            'font':{
            'size':40}},
        number = {
            'font':{
            'size':40}},
        domain = {
            'x': [0, 0], 'y': [0, 0.001]}))
    fig3.update_layout(
                paper_bgcolor=None, height=200, width=200)

    tres.plotly_chart(fig3)

    fig4 = go.Figure()
    fig4.add_trace(go.Indicator(
        align = 'center', mode = "number+delta", value = WSB_quiver[['Mentions', 'Rank']].iloc[-1][1] ,
        title = {
            "text": "WSB Rank<br><span style='font-size:0.8em;color:gray'>+ Weekly Change</span><br><span style='font-size:0.8em;color:gray'>",
            'font':{
            'size':20}},
        delta = {
            'reference': WSB_quiver[['Mentions', 'Rank']].iloc[-6][1], 'relative': True, 'valueformat':'.2%',
            'font':{
            'size':40}},
        number = {
            'font':{
            'size':40}},
        domain = {
            'x': [0, 0], 'y': [0, 0.001]}))

    fig4.update_layout(
        paper_bgcolor=None, height=200, width=200)

    cuatro.plotly_chart(fig4)
