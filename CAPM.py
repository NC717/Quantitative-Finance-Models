import pandas_datareader as pdr
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

risk_free_rate = 0.05

def capm(start_date, end_date, ticker1, ticker2):
    stock1 = pdr.get_data_data_yahoo(ticker1, start_date, end_date)
    stock2 = pdr.get_data_data_yahoo(ticker2, start_date, end_date)

    #we prefer monthly returns instead of daily returns

    return_stock1 = stock1.resample('M').last()
    return_stock2 = stock2.resample('M').last()

    'Create Dataframe'
    data = pd.DataFrame({'S_adjclose' : return_stock1['Adj Close'], 'm_adjclose': return_stock2['Adj Close']})
    'natural logs of returns'
    data[['s_returns' 'm_retturns']] = np.log(data[['s_adjclose', 'm_adjclose']]/ data[['s_adjclose',
                                                                                        'm_adjclose']].shift(1))
    'remove the NA values'
    data = data.dropna()

    covmat = np.cov(data["s_returns"], data["m_returns"])
    'the matrix is symmetric'

    beta = covmat[0,1]/cov[1,1]
    print "Beta from formula:", beta

    'Using linear regression'

    beta, alpha = np.polyfit(data["m_returns"], data["s_returns"], deg=1)
    print "Beta from regression:", beta

    fig, axis = plt.subplots(1, figsize=(20,10))
    axis.scatter(data["m_returns"], data["s_returns"], label="Data points")
    axis.plot(data["m_returns"], beta*data["s_returns"] + alpha, color='red', label="CAPM LINE")
    plt.title("CAPM")
    plt.xlabel("Market return", fontsize=18)
    plt.ylabel("Stock return")
    plt.text(0.08, 0.05, r'$R_a = \beta*R_m + \alpha$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()
    expected_return = risk_free_rate + beta*(data["m_return"].mean()*12 - risk_free_rate)
    print expected_return


capm('2010-01-01', '2017-01-01', 'IBM', 'GSPC')
