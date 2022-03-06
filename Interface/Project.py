import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime as dt
import numpy as np
import random
import plotly.express as px
from streamlit_echarts import st_echarts
from matplotlib import pyplot as plt
import altair as alt
from scipy.stats import norm, t

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css(r"C:\Users\baren\PycharmProjects\pythonProject1\Interface\style.css")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color:  #86DAE4 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


scores = pd.read_excel(r"C:\Users\baren\PycharmProjects\pythonProject1\Interface\Model Score.xlsx")

# App title
st.markdown('''
# Cryptocurrency Price App
Shown are the cryptocurrency price data for query companies!
**Credits**
- App built by Chambers
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')
st.write('---')

st.sidebar.subheader('Portfolio')

# Retrieving tickers data
coins = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD', 'ATOM-USD', 'DOT-USD', 'NEAR-USD']

def png(name, col):
    if name == "DOGE":
        col.image("doge_logo.png")
    else:
        col.image("btc_logo.png")


def app():
    for i in coins:
        if i in coins_selected or select_all:
            tickerDf = yf.download([i], start=dt.date(2020, 1, 1), end=dt.date(2022, 3, 3))
            name = i.split("-")[0]
            col1, col2, col3, col4, col5, col6 = st.columns((1, 1, 1, 1, 1, 1))
            png(name, col1)
            lname = scores.loc[scores["Coin List"] == name, "Name"].iloc[0]
            col2.markdown(f"<h1 style='text-align: center; color: black;'>{lname}</h1>", unsafe_allow_html=True)
            col5.markdown(
                f"<h6 style='text-align: center; color: black;'>Model Score</h1>",
                unsafe_allow_html=True)
            score = scores.loc[scores["Coin List"] == name, "Model Score"].iloc[0]
            col5.progress(score)
            p_score = score*100
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-image: linear-gradient(to right, #FFF2D8 , #86DAE4);
                    }
                </style>""",
                unsafe_allow_html=True,
            )


            col6.markdown(
                f"<h2 style='text-align: left; color: black;'>{round(p_score)}%</h4>",
                unsafe_allow_html=True)
            #st.empty().text(" ")
            price = round(tickerDf["Close"].tail(1)[0], 2)
            volume = round(tickerDf["Volume"].tail(1)[0], 2)
            price_change = round((tickerDf["Close"].tail(2)[1]-tickerDf["Close"].tail(2)[0])/tickerDf["Close"].tail(2)[0], 3)
            volume_change = round((tickerDf["Volume"].tail(2)[1]-tickerDf["Volume"].tail(2)[0])/tickerDf["Volume"].tail(2)[0], 3)
            col1, col2 = st.columns((1, 3))
            col1.metric("Daily Price", f"{price:,}", price_change)
            col2.metric("Daily Volume", f"${volume:,}", volume_change)
            st.empty().text(" ")

            col1, col2 = st.columns((1.05, 1))
            qf = cf.QuantFig(tickerDf.tail(30), legend="right", name="")
            fig = qf.iplot(asFigure=True)
            col1.write("History")
            fig.update_layout(width=335, height=300)

            col1.plotly_chart(fig)

            name = "BTC"
            df2 = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\{name}_forecasts.csv")
            #df = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\{name}_Forecasts_VAR_Model.csv")
            df = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\forecast_{name}_with_PyCaret_Regression.csv")
            df["Open"][0] = tickerDf["Close"].tail(1)[0]
            st.write(r"C:\Users\baren\PycharmProjects\pythonProject1\work5\Graphs\m_btc.png")
            df.rename(columns={f"predicted_Close_{name}": "Close"}, inplace=True)

            for i in range(len(df["Open"])):
                df["Open"][i+1] = df["Close"][i]
                if df["Open"][i] > df["Close"][i]:
                    weights = np.random.RandomState(seed=i+1).random(2)
                    weights /= np.sum(weights)
                    w1 = weights[0]
                    w2 = weights[1]
                    weights = np.random.RandomState(seed=i+1).random(2)
                    weights /= np.sum(weights)
                    w3 = weights[1]
                    w4 = weights[0]
                    df["High"][i] = w1 * df["Open"][i] + w2 * df["High"][i]
                    df["Low"][i] = w3 * df["Close"][i] + w4 * df["Low"][i]
                else:
                    weights = np.random.RandomState(seed=i+1).random(2)
                    weights /= np.sum(weights)
                    w1 = weights[0]
                    w2 = weights[1]
                    weights = np.random.RandomState(seed=i+1).random(2)
                    weights /= np.sum(weights)
                    w3 = weights[1]
                    w4 = weights[0]
                    df["High"][i] = w1 * df["Close"][i] + w2 * df["High"][i]
                    df["Low"][i] = w3 * df["Open"][i] + w4 * df["Low"][i]

            #df["High"] = df["Close"]
            #df["Low"] = df["Close"]
            #df["Open"] = df["Close"]


            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index(df["Date"])
            #df["Date"]
            df.drop("Date", axis=1, inplace=True)
            qf = cf.QuantFig(df.tail(30), legend="true", name="")
            #qf.add_bollinger_bands(periods=1, boll_std=0.2, colors=['#A8F1F9'], fill=False, name="Close")
            fig = qf.iplot(asFigure=True)
            col2.write("Prediction")

            fig.update_layout(width=335, height=300)
            col2.plotly_chart(fig)

            st.line_chart(df["Close"])

            st.image("m_btc.png")

            #col2.write(df[["Close"]].tail(7))
            st.empty().text(" ")
            #col1, col2, col3 = st.columns(3)
            #col1.metric("Price Prediction", 57000, 0.05)
            #col2.metric("7d% Prediction", 57000, 0.67)
            #col3.metric("Volume Prediction", 100032139, -0.33)
            st.empty().text(" ")


            #st.markdown(f"<p style='text-align: center; color: black;'>Shown are the cryptocurrency price data for query companies!</h1>", unsafe_allow_html=True)
            st.write('---')
    return score


def pcr_return_mean_cov_matrix(dataframe, coins_selected):
    if len(coins_selected) == 1:
        print("HELLO")
        dataframe = dataframe.reset_index()
        dataframe["Adj Close1"] = dataframe["Adj Close"]
        dataframe.index = dataframe["Date"]
        dataframe.drop("Date", axis=1, inplace=True)
        print("HELLO1")
        returns = dataframe.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
    else:
        st.write("HELLO>")
        returns = dataframe.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()


    return returns, mean_returns, cov_matrix

def portfolioPerformance(weights, mean_returns, cov_matrix, Time):
    returns = np.sum(mean_returns*weights)*Time
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) ) * np.sqrt(Time)
    return returns, std

def cem(weights,coins_selected):
    start = dt.datetime(2021, 2, 1)
    end = dt.datetime(2021, 2, 28)
    df = yf.download(coins_selected, start, end)['Adj Close']
    if len(coins_selected)*2 == 8:
        df = df.reset_index()
        df = df.rename(columns = {"Adj Close":coins_selected[0]})

    col1, col2 = st.columns((2, 1))
    exp_returns, mean_returns, cov_matrix = pcr_return_mean_cov_matrix(df, coins_selected)
    exp_returns = exp_returns.dropna()

    exp_returns['portfolio'] = exp_returns.dot(weights)

    def historical_var(returns, alpha=5):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the percentile of the distribution at the given alpha confidence level
        """
        if isinstance(returns, pd.Series):
            return np.percentile(returns, alpha)

        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(historical_var, alpha=alpha)

        else:
            raise TypeError("Expected returns to be dataframe or series")

    def historical_con_var(returns, alpha=5):
        """
        Read in a pandas dataframe of returns / a pandas series of returns
        Output the con_var for dataframe / series

        :param returns:
        :param alpha:
        :return:
        """
        if isinstance(returns, pd.Series):
            below_var = returns <= historical_var(returns, alpha=alpha)
            return returns[below_var].mean()

        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(historical_con_var, alpha=5)

        else:
            raise TypeError("Expected returns to be dataframe or series")

    Time = 1
    var = historical_var(exp_returns['portfolio'], alpha=5) * np.sqrt(Time)
    conditional_var = historical_con_var(exp_returns['portfolio'], alpha=5) * np.sqrt(Time)
    p_return, p_std = portfolioPerformance(weights, mean_returns, cov_matrix, Time)

    def var_parametric(portfolio_return, portfolio_std, distribution="normal", alpha=5, dof=6):
        """
        Calculate the portfolio VAR given a distribution, with known parameters
        :param portfolio_return:
        :param portfolio_std:
        :param distribution:
        :param alpha:
        :param dof:
        :return:
        """
        if distribution == "normal":
            var = norm.ppf(1 - alpha / 100) * portfolio_std - portfolio_return
        elif distribution == "t-distribution":
            nu = dof
            var = np.sqrt((nu - 2) / nu) * t.ppf(1 - alpha / 100, nu) * portfolio_std - portfolio_return
        else:
            raise TypeError("Expected distribution to be 'normal' or 't-distribution'")
        return var

    def cvar_parametric(portfolio_return, portfolio_std, distribution="normal", alpha=5, dof=6):
        """
        Calculate the portfolio CVAR given a distribution, with known parameters

        :param portfolio_return:
        :param portfolio_std:
        :param distribution:
        :param alpha:
        :param dof:
        :return:
        """
        if distribution == "normal":
            conditional_var = (alpha / 100) ** -1 * norm.pdf(
                norm.ppf(alpha / 100)) * portfolio_std - portfolio_return

        elif distribution == "t-distribution":
            nu = dof
            x_anu = t.ppf(alpha / 100, nu)
            conditional_var = -1 / (alpha / 100) * (1 - nu) ** -1 * (nu - 2 + x_anu ** 2) * t.pdf(x_anu,
                                                                                                  nu) * portfolio_std - portfolio_return

        else:
            raise TypeError("Expected distribution to be 'normal' or 't-distribution'")
        return conditional_var

    norm_var = var_parametric(p_return, p_std)
    norm_cvar = cvar_parametric(p_return, p_std)

    t_var = var_parametric(p_return, p_std, distribution="t-distribution")
    t_cvar = cvar_parametric(p_return, p_std, distribution="t-distribution")

    return var, conditional_var, norm_var, norm_cvar, t_var, t_cvar

def part1(weights, returns,coins_selected):
    score = []
    Return1 = []
    Return2 = []
    Return3 = []
    var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

    for i in range(len(coins_selected)):
        coins_selected[i] = coins_selected[i].split("-")[0]
        score.append(scores.loc[scores["Coin List"] == coins_selected[i], "Model Score"].iloc[0])
        Return1.append(returns[f"Rate_{coins_selected[i]}"][0])
        Return2.append(returns[f"Rate_{coins_selected[i]}"][1])
        Return3.append(returns[f"Rate_{coins_selected[i]}"][2])

    x = []
    y1 = []
    y2 = []
    y3 = []
    for i in range(len(coins_selected)):
        x.append(score[i]*weights[i])
        y1.append(Return1[i] * weights[i])
        y2.append(Return2[i] * weights[i])
        y3.append(Return3[i] * weights[i])

    avg_score = round(sum(x) / len(x), 2)
    cc = round(sum(y1), 2)
    ch = round(sum(y2), 2)
    lh = round(sum(y3), 2)
    return cc, ch, lh, avg_score

def part2(avg_score, var,conditional_var, norm_var, norm_cvar, t_var, t_cvar,cc,lc,ch):
    col1, col2, col3 = st.columns(3)
    col1.metric("Close-Close", f"%{round(100 * avg_score, 2)}", cc)
    col2.metric("Close-High", f"%{round(100 * avg_score * 0.8, 2)}", lc)
    col3.metric("Low-High", f"%{round(100 * avg_score * 0.45, 2)}", ch)

    st.info(
        f"{round(var, 3)},{round(conditional_var, 3)},{round(norm_var, 3)},{round(norm_cvar, 3)},{round(t_var, 3)},{round(t_cvar, 3)}")

    st.markdown(f'''
    - Our value-at-risk for this portfolio: `{round(var, 3)}`
    - Our conditional value-at-risk for this portfolio: `{round(conditional_var, 3)}`
    - Our value-at-risk this portfolio according to the normal distribution: `{round(t_var, 3)}`
    - Our value-at-risk this portfolio according to the t-distribution: `{round(t_cvar, 3)}`
    ''')

    st.markdown(
        f"<div>Our value-at-risk for this portfolio with Close-Close: <span class='highlight red'>{cc}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our value-at-risk for this portfolio with Close-High: <span class='highlight red'>{lc}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our value-at-risk for this portfolio with Low-High: <span class='highlight red'>{ch}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our value-at-risk for this portfolio: <span class='highlight blue'>{-1 * round(var, 3)}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our conditional value-at-risk for this portfolio: <span class='highlight blue'>{-1 * round(conditional_var, 3)}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our value-at-risk this portfolio according to the normal distribution: <span class='highlight blue'>{round(t_var, 3)}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Our value-at-risk this portfolio according to the t-distribution: <span class='highlight blue'>{round(t_cvar, 3)}</span></div>",
        unsafe_allow_html=True)

    # st.markdown(f"<p style='color: red ;'>{round(var,3)}</p>", unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"ColorMeBlue text”"}</h1>', unsafe_allow_html=True)

    col2.markdown(
        f"<h1 style='text-align: center; color: white;'></h1>",
        unsafe_allow_html=True)

def portfolio(coins_selected):
    st.markdown(f"<h1 style='text-align: center; color: black;'>Portfolio</h1>", unsafe_allow_html=True)
    returns = pd.read_excel(r"C:\Users\baren\PycharmProjects\pythonProject1\Interface\coin_recommend.xlsx")
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD','NEAR-USD', 'ATOM-USD']

    if len(coins_selected) == 1:
        weights = np.random.random(1)
        weights /= np.sum(weights)
        rate1 = weights[0]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))

        option = {
                "series": [
                    {
                        "name": 'pp',
                        "type": 'pie',
                        "radius": ['40%', '75%'],
                        "avoidLabelOverlap": "true",
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"}
                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 2:

        weights = np.random.random(2)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))

        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                    "data": [
                        {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),1)}"},
                        {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'}
                    ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 3:
        weights = np.random.random(3)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))

        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),0)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100*rate3), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 4:
        weights = np.random.random(4)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))
        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),0)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'}


                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 5:
        weights = np.random.random(5)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))
        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 6:
        weights = np.random.random(6)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))
        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'},
                            {"value": rate6, "name": f'{coins_selected[5]} %{round(int(100 * rate6), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 7:
        weights = np.random.random(7)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))
        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'},
                            {"value": rate6, "name": f'{coins_selected[5]} %{round(int(100 * rate6), 0)}'},
                            {"value": rate7, "name": f'{coins_selected[6]} %{round(int(100 * rate7), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 8:
        weights = np.random.random(8)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        rate8 = weights[7]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))

        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'},
                            {"value": rate6, "name": f'{coins_selected[5]} %{round(int(100 * rate6), 0)}'},
                            {"value": rate7, "name": f'{coins_selected[6]} %{round(int(100 * rate7), 0)}'},
                            {"value": rate8, "name": f'{coins_selected[7]} %{round(int(100 * rate8), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 9:
        weights = np.random.random(9)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        rate8 = weights[7]
        rate9 = weights[8]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))

        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'},
                            {"value": rate6, "name": f'{coins_selected[5]} %{round(int(100 * rate6), 0)}'},
                            {"value": rate7, "name": f'{coins_selected[6]} %{round(int(100 * rate7), 0)}'},
                            {"value": rate8, "name": f'{coins_selected[7]} %{round(int(100 * rate8), 0)}'},
                            {"value": rate9, "name": f'{coins_selected[8]} %{round(int(100 * rate9), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)
    if len(coins_selected) == 10:
        weights = np.random.random(3)
        weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        rate8 = weights[7]
        rate9 = weights[8]
        rate10 = weights[9]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, lc, ch, avg_score = part1(weights, returns, coins_selected)
        col1, col2 = st.columns((4, 1))

        col2.markdown(
            f"<h1 style='text-align: center; color: white;'></h1>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h3 style='text-align: center; color: white;'></h3>",
            unsafe_allow_html=True)

        col2.markdown(
            f"<h2 style='text-align: center; color: white;'></h2>",
            unsafe_allow_html=True)

        genre = col2.radio(
            "Risk",
            ('Low', 'Medium', 'High'))
        option = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                        "data": [
                            {"value": rate1, "name": f"{coins_selected[0]} %{round(int(100*rate1),-1)}"},
                            {"value": rate2, "name": f'{coins_selected[1]} %{round(int(100*rate2),0)}'},
                            {"value": rate3, "name": f'{coins_selected[2]} %{round(int(100 * rate3), 0)}'},
                            {"value": rate4, "name": f'{coins_selected[3]} %{round(int(100 * rate4), 0)}'},
                            {"value": rate5, "name": f'{coins_selected[4]} %{round(int(100 * rate5), 0)}'},
                            {"value": rate6, "name": f'{coins_selected[5]} %{round(int(100 * rate6), 0)}'},
                            {"value": rate7, "name": f'{coins_selected[6]} %{round(int(100 * rate7), 0)}'},
                            {"value": rate8, "name": f'{coins_selected[7]} %{round(int(100 * rate8), 0)}'},
                            {"value": rate9, "name": f'{coins_selected[8]} %{round(int(100 * rate9), 0)}'},
                            {"value": rate10, "name": f'{coins_selected[9]} %{round(int(100 * rate10), 0)}'}

                        ]
                    }
                ]
            };

        with col1:
            st_echarts(options=option, key="4")

        part2(avg_score, var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, lc, ch)

    col1, col2, col3 = st.columns(3)

    if 1 == 2:
        option1 = {
        "tooltip": {
            "formatter": '{a} <br/>{b} : {c}%'
        },
        "series": [{
            "name": 'Low-Close',
            "type": 'gauge',
            "startAngle": 180,
            "endAngle": 0,
            "progress": {
                "show": "true"
            },
            "radius": '95%',

            "itemStyle": {
                "color": 'green',
                "shadowColor": 'green',
                "shadowBlur": 10,
                "shadowOffsetX": 2,
                "shadowOffsetY": 2,
                "radius": '55%',
            },
            "progress": {
                "show": "true",
                "roundCap": "false",
                "width": 10
            },
            "pointer": {
                "length": '60%',
                "width": 6,
                "offsetCenter": [-3, '0%']
            },
            "detail": {
                "valueAnimation": "true",
                "formatter": '{value}%',
                "backgroundColor": 'white',
                "borderColor": 'white',
                "borderWidth": 0,
                "width": '50%',
                "lineHeight": 40,
                "height": 48,
                "borderRadius": 40,
                "offsetCenter": [3, '45%'],
                "valueAnimation": "true",
            },
            "data": [{
                "value": 86.66,
                "name": 'Probability'
            }]
        }]
    };
    else:
        option1 = {
            "series": [{
                "name": 'Low-Close',
                "type": 'gauge',
                "startAngle": 180,
                "endAngle": 0,
                "progress": {
                    "show": "true"
                },
                "radius": '95%',

                "itemStyle": {
                    "color": 'red',
                    "shadowColor": 'red',
                    "shadowBlur": 10,
                    "shadowOffsetX": 2,
                    "shadowOffsetY": 2,
                    "radius": '120%',
                },
                "progress": {
                    "show": "true",
                    "roundCap": "false",
                    "width": 10
                },
                "pointer": {
                    "length": '50%',
                    "width": 3,
                    "offsetCenter": [-3, '0%']
                },
                "detail": {
                    "valueAnimation": "true",
                    "formatter": '{value}%',
                    "backgroundColor": 'white',
                    "borderColor": 'white',
                    "borderWidth": 0,
                    "width": '50%',
                    "lineHeight": 0,
                    "height": 0,
                    "borderRadius": 40,
                    "offsetCenter": [4, '50%'],
                    "valueAnimation": "true",
                },
                "data": [{
                    "value": 86.66,
                    "name": 'Low-Close'
                }]
            }]
        };

    with col1:
        st_echarts(options=option1, key="35")

    with col2:
        st_echarts(options=option1, key="36")

    with col3:
        st_echarts(options=option1, key="37")





    #col2.metric("Rate of Return"," ",delta = 0.05)
    #col3.metric("Cond. Value At Risk", " ", -0.33)

    st.markdown(
        f"<p style='text-align: center; color: black;'>Shown are the cryptocurrency price data for query companies!</h1>",
        unsafe_allow_html=True)

    #col2.empty().text(" ")
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write(f"Expected Portfolio Return: `{number}`")

    #col2.write("Expected Portfolio Return:        ", round(initial_investment*p_return, 2))


    st.write('---')

    st.markdown(f"<h1 style='text-align: center; color: black;'>Portfolio Recommendation</h1>", unsafe_allow_html=True)

    if genre == "Low":
        price1 = "%80"
        price2 = "%20"
    if genre == "Moderate":
        price1 = "%60"
        price2 = "%40"
    if genre == "High":
        price1 = "%30"
        price2 = "%70"


    option5 = {
            "series": [
                {
                    "name": 'pp',
                    "type": 'pie',
                    "radius": ['0%', '75%'],
                    "avoidLabelOverlap": "true",
                    "data": [
                        {"value": 0.5, "name": f"Bitcoin {price1}"},
                        {"value": 0.3, "name": f'Doge {price2}'},
                        {"value": 0.1, "name": f'BNB {price2}'},
                        {"value": 0.1, "name": f'AVAX {price2}'}

                    ]
                }
            ]
        };

    st_echarts(options=option5, key="3")
    col1, col2 = st.columns(2)


    col1, col2, col3, col4 = st.columns(4)


    initial_investment = 5
    p_return = 0.02

    number = 15
    col1.metric("Risk", "Low")
    col2.metric("Rate of Return", 57000, delta=0.05)
    col3.metric("Cond. Value At Risk", 1000, delta=-0.33)
    col4.metric("Avg Model Score", score)



    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        f"<p style='text-align: center; color: black;'>Our value-at-risk for Bitcoin:</h1>",
        unsafe_allow_html=True)

    col3.markdown(
        f"<p style='text-align: center; color: black;'>Our value-at-risk for Bitcoin:</h1>",
        unsafe_allow_html=True)

    """

    st.empty().text(" ")
    st.empty().text(" ")
    """

    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write(f"Expected Portfolio Return: `{number}`")

    #col2.write("Expected Portfolio Return:        ", round(initial_investment*p_return, 2))


    st.write('---')

select_all = st.sidebar.checkbox("Select All Coins")

if select_all:
    coins_selected = st.sidebar.multiselect('Select Partially', coins, default=['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD', 'ATOM-USD', 'DOT-USD'])
    app()
    st.balloons()
    portfolio(coins_selected)
else:
    coins_selected = st.sidebar.multiselect('Select Partially', coins)
    app()
    portfolio(coins_selected)


