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
from plotly import graph_objs as go

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

start_ = dt.date(2022, 2, 5)

end_ = dt.date(2022, 3, 5)

st.markdown(f"<h1 style='text-align: center; color: black;'>CHAMBERS APP</h1>", unsafe_allow_html=True)
st.write('---')

st.sidebar.image("charmbers123.png")
st.sidebar.markdown(f"<h1 style='text-align: center; color: black;'>Portfolio</h1>", unsafe_allow_html=True)


# Retrieving tickers data
coins = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD', 'ATOM-USD', 'DOT-USD', 'NEAR-USD']

def png(name, col):
    col.image(f"{name.lower()}_logo.png")

def app():
    for i in coins:
        if i in coins_selected:
            tickerDf = yf.download(i, start=start_, end=end_)
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
            col1, col2,col3 = st.columns((3, 3, 4.3))
            col2.metric("Daily Price", f"${price:,}", price_change)
            col3.metric("Daily Volume", f"${volume:,}", volume_change)
            st.empty().text(" ")

            col1, col2,col3 = st.columns((1, 5, 1))
            qf = cf.QuantFig(tickerDf.tail(120), legend="right", name="")
            fig = qf.iplot(asFigure=True)
            col1, col2, col3 = st.columns((0.65, 5, 1))
            col2.markdown(f"<p style=color: black;'>History</p>", unsafe_allow_html=True)
            fig.update_layout(width=600, height=300)
            col2.markdown(f"<p style=color: white;'></p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns((1, 5, 1))
            col2.plotly_chart(fig)

            #df = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\{name}_forecasts.csv")
            df = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast3\forecast_{name}_with_PyCaret_Regression.csv")
            #df = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\{name}_Forecasts_VAR_Model.csv")
            #df2 = pd.read_csv(rf"C:\Users\baren\PycharmProjects\pythonProject1\Forecast\forecast_{name}_with_PyCaret_Regression.csv")
            #df["Open"][0] = tickerDf["Close"].tail(1)[0]
            df.rename(columns={f"predicted_Close_{name}": "Close"}, inplace=True)

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
            #col2.write("Prediction")

            fig.update_layout(width=335, height=400)
            #col2.plotly_chart(fig)

            col1, col2, col3 = st.columns((0.2, 5, 2))

            tickerDf.reset_index(inplace=True)
            tickerDf["Close"][len(tickerDf["Close"]) - 1] = df["Close"][0]
            df.reset_index(inplace=True)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Close'], name="History", marker_color="#A8F1F9"))
            fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prediction", marker_color="pink"))
            fig1.layout.update(xaxis_rangeslider_visible=False, title_text='Prediction')
            fig1.update_layout(width=800, height=470)
            col2.plotly_chart(fig1)


            #col2.write(df[["Close"]].tail(7))
            st.empty().text(" ")
            #col1, col2, col3 = st.columns(3)
            #col1.metric("Price Prediction", 57000, 0.05)
            #col2.metric("7d% Prediction", 57000, 0.67)
            #col3.metric("Volume Prediction", 100032139, -0.33)
            st.empty().text(" ")


            #st.markdown(f"<p style='text-align: center; color: black;'>Shown are the cryptocurrency price data for query companies!</h1>", unsafe_allow_html=True)
            st.write('---')

def pcr_return_mean_cov_matrix(dataframe):
    try:
        returns = dataframe.pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
    except:
        returns = dataframe.pct_change()
        mean_returns = returns.mean()
        cov_matrix = 1
    return returns, mean_returns, cov_matrix

def portfolioPerformance(weights, mean_returns, cov_matrix, Time):
    returns = np.sum(mean_returns*weights)*Time
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) ) * np.sqrt(Time)
    return returns, std

def cem(weights,coins_selected):

    try:
        start = dt.datetime(2022, 1, 1)
        end = dt.datetime(2022, 6, 3)
        df = yf.download(coins_selected, start, end)['Adj Close']
        col1, col2 = st.columns((2, 1))

        exp_returns, mean_returns, cov_matrix = pcr_return_mean_cov_matrix(df)

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
    except:
        a = []
        for i in range(len(df)):
            a.append(df[i])
        b = a.copy()
        df2 = pd.DataFrame(list(zip(a, b)), columns=["1", "2"])
        weights = np.random.random(2)
        weights /= np.sum(weights)

        exp_returns, mean_returns, cov_matrix = pcr_return_mean_cov_matrix(df2)
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
    score_low = []
    score_high = []
    Return1 = []
    Return2 = []
    Return3 = []
    var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

    for i in range(len(coins_selected)):
        coins_selected[i] = coins_selected[i].split("-")[0]
        score.append(scores.loc[scores["Coin List"] == coins_selected[i], "Model Score"].iloc[0])
        score_low.append(scores.loc[scores["Coin List"] == coins_selected[i], "Model Score_Low"].iloc[0])
        score_high.append(scores.loc[scores["Coin List"] == coins_selected[i], "Model Score_High"].iloc[0])
        Return1.append(returns[f"Rate_{coins_selected[i]}"][0])
        Return2.append(returns[f"Rate_{coins_selected[i]}"][1])
        Return3.append(returns[f"Rate_{coins_selected[i]}"][2])

    cc = []
    lc = []
    lh = []
    for i in range(len(coins_selected)):
        cc.append((score[i] + score[i]) / 2)
        lc.append((score_low[i] + score[i])/2)
        lh.append((score_high[i] + score[i])/2)




    #cc: close close için tüm coinlerin model skoru lc lh...


    w_cc = []
    w_lc = []
    w_lh = []
    for i in range(len(coins_selected)):
        w_cc.append(cc[i] * weights[i])
        w_lc.append(lc[i] * weights[i])
        w_lh.append(lh[i] * weights[i])

    avg_score_cc = round(sum(w_cc), 2)
    avg_score_lc = round(sum(w_lc), 2)
    avg_score_lh = round(sum(w_lh), 2)

    #cc: close close weightlerle çarpılmış hali


    y1 = []
    y2 = []
    y3 = []
    for i in range(len(coins_selected)):
        y1.append(Return1[i] * weights[i])
        y2.append(Return2[i] * weights[i])
        y3.append(Return3[i] * weights[i])

    cc = round(sum(y1), 2)
    ch = round(sum(y2), 2)
    lh = round(sum(y3), 2)

    return cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh
def part2(var,conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre):
    col1, col2, col3 = st.columns(3)
    col1.metric("Close-Close", f"%{round(100 * avg_score_cc, 2)}", cc)
    col2.metric("Close-High", f"%{round(100 * avg_score_lc, 2)}", ch)
    col3.metric("Low-High", f"%{round(100 * avg_score_lh, 2)}", lh)
    #col1.metric("Close-Close", "", cc)
    #col2.metric("Close-High", "", ch)
    #col3.metric("Low-High", "", lh)

    #st.info(
    #    f"{round(var, 3)},{round(conditional_var, 3)},{round(norm_var, 3)},{round(norm_cvar, 3)},{round(t_var, 3)},{round(t_cvar, 3)}")

    #st.markdown(f'''
    #- Our value-at-risk for this portfolio: `{round(var, 3)}`
    #- Our conditional value-at-risk for this portfolio: `{round(conditional_var, 3)}`
    #- Our value-at-risk this portfolio according to the normal distribution: `{round(t_var, 3)}`
    #- Our value-at-risk this portfolio according to the t-distribution: `{round(t_cvar, 3)}`
    #''')

    st.markdown(
        f"<div>Here is the value-at-risk (VaR): <span class='highlight blue'>{-1 * round(var, 3)}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Here is the conditional value-at-risk (CVaR): <span class='highlight blue'>{-1 * round(conditional_var, 3)}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>According to the normal distribution the conditional value-at-risk (Normal - CVaR): <span class='highlight blue'>{round(t_var, 3)}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>According to the t distribution, the conditional value-at-risk (t - distribution - CVaR): <span class='highlight blue'>{round(t_cvar, 3)}</span></div>",
        unsafe_allow_html=True)

    # st.markdown(f"<p style='color: red ;'>{round(var,3)}</p>", unsafe_allow_html=True)
    # st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"ColorMeBlue text”"}</h1>', unsafe_allow_html=True)

    col2.markdown(
        f"<h1 style='text-align: center; color: white;'></h1>",
        unsafe_allow_html=True)
def weight_(df):
    num_ports = 5000
    all_weights = np.zeros((num_ports,len(df.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    df = df.dropna(axis=0)
    log_ret = np.log(df / df.shift(1))
    for x in range(num_ports):
        #weights
        weights = np.array(np.random.random(len(df.columns)))
        weights = weights / np.sum(weights)
        # Save weights
        all_weights[x, :] = weights
        # Expected return
        ret_arr[x] = np.sum((log_ret.mean() * weights * 256))
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 256, weights)))
        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x] / vol_arr[x]

    #Best Portföy
    p = sharpe_arr.argmax()
    # Best Weight
    w = all_weights[p, :]

    return w
def portfolio(coins_selected):
    #st.markdown(f"<h1 style='text-align: center; color: black;'>Portfolio</h1>", unsafe_allow_html=True)
    returns = pd.read_excel(r"C:\Users\baren\PycharmProjects\pythonProject1\Interface\coin_recommend.xlsx")
    tickerDf = yf.download(coins_selected, start=start_, end=end_)
    if len(coins_selected) == 1:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(1)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)
        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)

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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 2:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(2)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 3:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(3)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 4:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(4)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 5:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(5)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 6:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(6)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 7:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(7)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 8:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(8)
        #weights /= np.sum(weights)
        rate1 = weights[0]
        rate2 = weights[1]
        rate3 = weights[2]
        rate4 = weights[3]
        rate5 = weights[4]
        rate6 = weights[5]
        rate7 = weights[6]
        rate8 = weights[7]
        var, conditional_var, norm_var, norm_cvar, t_var, t_cvar = cem(weights, coins_selected)

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 9:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(9)
        #weights /= np.sum(weights)
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

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)
    if len(coins_selected) == 10:
        weights = weight_(tickerDf[["Adj Close"]])
        #weights = np.random.random(10)
        #weights /= np.sum(weights)
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

        cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh = part1(weights, returns, coins_selected)
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

        part2(var, conditional_var, norm_var, norm_cvar, t_var, t_cvar, cc, ch, lh, avg_score_cc, avg_score_lc, avg_score_lh,genre)

    col1, col2, col3 = st.columns(3)


    #col2.metric("Rate of Return"," ",delta = 0.05)
    #col3.metric("Cond. Value At Risk", " ", -0.33)

    #col2.empty().text(" ")
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write(f"Expected Portfolio Return: `{number}`")

    #col2.write("Expected Portfolio Return:        ", round(initial_investment*p_return, 2))


    st.write('---')

    st.markdown(f"<h1 style='text-align: center; color: black;'>Portfolio Recommendation</h1>", unsafe_allow_html=True)
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
        ('Low', 'Medium', 'High'), key=6)
    weights = np.random.random(3)
    weights /= np.sum(weights)

    if genre == "Low":
        rate1 = 0.45
        rate2 = 0.32
        rate3 = 0.24
        cc = 56.0
        ch = 39.0
        lh = 23.0
        ccr= 4.53
        chr= 8.54
        lhr= 14.43
    if genre == "Medium":
        rate1 = 0.36
        rate2 = 0.45
        rate3 = 0.14
        cc = 45.0
        ch = 22.0
        lh = 12.0
        ccr = 6.56
        chr = 12.11
        lhr = 22.35
    if genre == "High":
        rate1 = 0.29
        rate2 = 0.55
        rate3 = 0.10
        cc = 33.0
        ch = 21.0
        lh = 9.0
        ccr = 8.02
        chr = 15.34
        lhr = 29.97

    option5 = {
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
                    {"value": rate1, "name": f"BTC %{round(int(100 * rate1), 0)}"},
                    {"value": rate2, "name": f'ETH %{round(int(100 * rate2), 0)}'},
                    {"value": rate3, "name": f'ATOM %{round(int(100 * rate3), 0)}'}

                ]
            }
        ]
    };
    with col1:
        st_echarts(options=option5)



    col1, col2 = st.columns(2)


    col1, col2, col3, col4 = st.columns(4)


    col1, col2, col3 = st.columns(3)
    col1.metric("Close-Close", f"%{round(cc, 2)}", ccr)
    col2.metric("Close-High", f"%{round(ch, 2)}", chr)
    col3.metric("Low-High", f"%{round(lh, 2)}", lhr)
    st.markdown(
        f"<h2 style='text-align: center; color: white;'></h2>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div>Here is the value-at-risk (VaR): <span class='highlight blue'>{0.045}</span></span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>Here is the conditional value-at-risk (CVaR): <span class='highlight blue'>{0.023}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>According to the normal distribution the conditional value-at-risk (Normal - CVaR): <span class='highlight blue'>{0.015}</span></div>",
        unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown(
        f"<div>According to the t distribution, the conditional value-at-risk (t - distribution - CVaR): <span class='highlight blue'>{0.014}</span></div>",
        unsafe_allow_html=True)



    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write('''Expected Portfolio Return: `yfinance`''')
    #col2.write(f"Expected Portfolio Return: `{number}`")

    #col2.write("Expected Portfolio Return:        ", round(initial_investment*p_return, 2))


    st.write('---')

coins_selected = st.sidebar.multiselect('', coins)
app()
portfolio(coins_selected)


#if select_all:
#   coins_selected = st.sidebar.multiselect('Select Partially', coins, default=['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD', 'ATOM-USD', 'DOT-USD', 'NEAR-USD'])
#    app()
#    st.balloons()
#    portfolio(coins_selected)
#else:
#    coins_selected = st.sidebar.multiselect('Select Partially', coins)
#    app()
#    portfolio(coins_selected)

