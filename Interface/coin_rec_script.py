##################################
# Coin Recommend
##################################

####################
# Library
####################
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
warnings.filterwarnings('ignore')


####################
# Script Format
####################
def crypto_curr_extract(crypto_currency, against_currency, start_date, end_date):
    """

    Data extraction for 1 Coin from Yahoo Finance.

    Parameters
    ----------
        crypto_currency: Coin which will be extract
        against_currency: Parity coin
        start_date
        end_date

    Returns
    -------
        Dataset with daily values.

    Example
    -------
        crypto_curr_extract("BTC", "USD", start_date = [2021, 1, 1], end_date = [2022, 1, 1])

    """

    start = dt.datetime(start_date[0], start_date[1], start_date[2])
    end = dt.datetime(end_date[0], end_date[1], end_date[2])
    df = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)
    return df

def rec_coin(coin):
    dataframe = pd.read_csv("proje/actual/outputs/" + str(coin) + "_forecasts.csv")
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])

    df_actual = crypto_curr_extract(coin, "USD", [(dataframe["Date"][0] + dt.timedelta(days=-1)).year,
                                                   (dataframe["Date"][0] + dt.timedelta(days=-1)).month,
                                                   (dataframe["Date"][0] + dt.timedelta(days=-1)).day],
                                    [(dataframe["Date"][0] + dt.timedelta(days=-1)).year,
                                     (dataframe["Date"][0] + dt.timedelta(days=-1)).month,
                                     (dataframe["Date"][0] + dt.timedelta(days=-1)).day])

    df_actual.reset_index(inplace=True)
    df_actual = df_actual[0:1]
    df_actual = df_actual[["Date", "High", "Low", "Open", "Volume", "Close"]]

    # ROC1
    dataframe["ROC1-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(1)) - 1) * 100
    dataframe["ROC1-Close-Close"][0] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC1-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(1)) - 1) * 100
    dataframe["ROC1-Close-High"][0] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC1-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(1)) - 1) * 100
    dataframe["ROC1-Low-High"][0] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC2
    dataframe["ROC2-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(2)) - 1) * 100
    dataframe["ROC2-Close-Close"][1] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC2-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(2)) - 1) * 100
    dataframe["ROC2-Close-High"][1] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC2-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(2)) - 1) * 100
    dataframe["ROC2-Low-High"][1] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC3
    dataframe["ROC3-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(3)) - 1) * 100
    dataframe["ROC3-Close-Close"][2] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC3-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(3)) - 1) * 100
    dataframe["ROC3-Close-High"][2] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC3-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(3)) - 1) * 100
    dataframe["ROC3-Low-High"][2] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC4
    dataframe["ROC4-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(4)) - 1) * 100
    dataframe["ROC4-Close-Close"][3] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC4-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(4)) - 1) * 100
    dataframe["ROC4-Close-High"][3] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC4-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(4)) - 1) * 100
    dataframe["ROC4-Low-High"][3] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC5
    dataframe["ROC5-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(5)) - 1) * 100
    dataframe["ROC5-Close-Close"][4] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC5-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(5)) - 1) * 100
    dataframe["ROC5-Close-High"][4] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC5-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(5)) - 1) * 100
    dataframe["ROC5-Low-High"][4] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC6
    dataframe["ROC6-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(6)) - 1) * 100
    dataframe["ROC6-Close-Close"][5] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC6-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(6)) - 1) * 100
    dataframe["ROC6-Close-High"][5] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC6-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(6)) - 1) * 100
    dataframe["ROC6-Low-High"][5] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # ROC7
    dataframe["ROC7-Close-Close"] = ((dataframe[dataframe.columns[5]] /
                                       dataframe[dataframe.columns[5]].shift(7)) - 1) * 100
    dataframe["ROC7-Close-Close"][6] = ((dataframe[dataframe.columns[5]][0] /
                                         df_actual["Close"]) - 1) * 100
    dataframe["ROC7-Close-High"] = ((dataframe[dataframe.columns[1]] /
                                      dataframe[dataframe.columns[5]].shift(7)) - 1) * 100
    dataframe["ROC7-Close-High"][6] = ((dataframe[dataframe.columns[1]][0] /
                                        df_actual["Close"]) - 1) * 100
    dataframe["ROC7-Low-High"] = ((dataframe[dataframe.columns[1]] /
                                   dataframe[dataframe.columns[2]].shift(7)) - 1) * 100
    dataframe["ROC7-Low-High"][6] = ((dataframe[dataframe.columns[1]][0] /
                                      df_actual["Low"]) - 1) * 100

    # High Possibility (Close-Close)
    high_cols = [col for col in dataframe.columns[dataframe.columns.str.contains("Close-Close")]]

    type_high = abs(dataframe[high_cols]).max().sort_values(ascending=False).index[0]

    if abs(dataframe[abs(dataframe[high_cols]).max().sort_values(ascending=False).index[0]].max()) >= \
            abs(dataframe[abs(dataframe[high_cols]).max().sort_values(ascending=False).index[0]].min()):
        rate_high = dataframe[abs(dataframe[high_cols]).max().sort_values(ascending=False).index[0]].max()
    else:
        rate_high = dataframe[abs(dataframe[high_cols]).max().sort_values(ascending=False).index[0]].min()

    # Medium Possibility (Close-High)
    medium_cols = [col for col in dataframe.columns[dataframe.columns.str.contains("Close-High")]]

    type_medium = abs(dataframe[medium_cols]).max().sort_values(ascending=False).index[0]

    if abs(dataframe[abs(dataframe[medium_cols]).max().sort_values(ascending=False).index[0]].max()) >= \
            abs(dataframe[abs(dataframe[medium_cols]).max().sort_values(ascending=False).index[0]].min()):
        rate_medium = dataframe[abs(dataframe[medium_cols]).max().sort_values(ascending=False).index[0]].max()
    else:
        rate_medium = dataframe[abs(dataframe[medium_cols]).max().sort_values(ascending=False).index[0]].min()

    # Low Possibility (Low-High)
    low_cols = [col for col in dataframe.columns[dataframe.columns.str.contains("Low-High")]]

    type_low = abs(dataframe[low_cols]).max().sort_values(ascending=False).index[0]

    if abs(dataframe[abs(dataframe[low_cols]).max().sort_values(ascending=False).index[0]].max()) >= \
            abs(dataframe[abs(dataframe[low_cols]).max().sort_values(ascending=False).index[0]].min()):
        rate_low = dataframe[abs(dataframe[low_cols]).max().sort_values(ascending=False).index[0]].max()
    else:
        rate_low = dataframe[abs(dataframe[low_cols]).max().sort_values(ascending=False).index[0]].min()

    df_recommend = pd.DataFrame({"Type_"+str(coin): [type_high, type_medium, type_low], "Rate_"+str(coin): [rate_high, rate_medium, rate_low]}, index=[0, 1, 2])
    return df_recommend

def table_coins(coin, coins):
    dataframe = rec_coin(coin)
    for i in coins:
        dataframe_v2 = rec_coin(i)
        dataframe = dataframe.merge(dataframe_v2, left_index=True, right_index=True)

    # save to excel
    dataframe.to_excel("coin_reccommend.xlsx", index=False)

    return dataframe

# MATIC de open degeri yok!
# coins = ["ATOM", "AVAX", "BNB", "DOGE", "DOT", "ETH", "MATIC", "NEAR", "SOL"]

coins = ["ATOM", "AVAX", "BNB", "DOGE", "DOT", "ETH", "NEAR", "SOL"]
table_coins("BTC", coins)



