import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data as web
from matplotlib import pyplot as plt
import datetime
from scipy import optimize as sco

stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
noa = len(stocks)

start_d = '01/01/2001'
end_d = '01/01/2017'


# downloading data from yahoo finance


def dwnld(stocks):
    data = web.DataReader(stocks, data_source='yahoo', start= start_d,end = end_d)['Adj Close']
    data.coloumns = stocks
    return data


def show_data(data):
    print data
    data.plot(figsize = (10,5))
    plt.show()


def calculate_returns(data):
    ret = np.log(data/data.shift(1))
    return ret


def plot_daily_returns(ret):
    ret.plot(figsize=(10, 5))
    plt.show()


def sho_statistics(ret):
    print ret.mean()*265
    print ret.cov()*265


def initialize_weights():
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights


dat = dwnld(stocks)
print show_data(dat)
values = calculate_returns(dat)
plot_daily_returns(values)
show_data(values)
wei = initialize_weights()


def calculate_portfolio_returns(returns, weights):
    portfolio_returns = np.sum(returns.mean()*weights)*252
    print "expected portfolio returns:", portfolio_returns


def generate_variance(returns, weights):
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(weights, returns.cov() * 252, weights)))
    print "Expected variance:", portfolio_variance


'MONTE CARLO SIMULATIONS FOR GENERATING PORTFOLIO AND HENCE OPTIMIZING USING THE SCIPY LIBRARY'


def generate_portfolio(weights, returns):

    preturns = []
    pvariance = []

    for i in range(20000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvariance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))))

    preturns = np.array(preturns)
    pvariance = np.array(pvariance)
    return preturns, pvariance


def plot_all(returns, variance):
    plt.figure(figsize=(10,5))
    plt.scatter(variance, returns,c=returns/variance)
    plt.grid = True
    plt.xlabel("Variance/volatality/risk")
    plt.ylabel("Expected returns")
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def statistics(weights, returns):
    portfolio_returns = np.sum(returns.mean()*weights)*252
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return np.array([portfolio_returns, portfolio_variance, portfolio_returns/portfolio_variance])


def min_func_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):
    constraints = ({'Type': 'eq','fun' : lambda x : np.sum(x)-1})
    bounds = tuple((0,1) for i in range(noa))
    optimum = optimization.minimize(fun=min_func_sharpe(), x0=weights, arg=returns, method='SLSQP', bounds=bounds,
                                    constraints=constraints)
    return optimum


def show_optimal(optimum, preturns, pvariance):
    plt.figure(figure=(10,5))
    plt.scatter(pvariance, preturns, c=preturns/pvariance)
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected return")
    plt.plot(statistics(optimum['x'], returns)[0], 'g*', markersize=20.0 )
    plt.show()

x=[]
calculate_portfolio_returns(values, wei)
generate_variance(values,wei)
x = generate_portfolio(wei, values)
plot_all(x[0], x[1])
sarray = statistics(wei, values)
print sarray
print min_func_sharpe(wei, values)
opp = optimize_portfolio(wei, values)
print opp
show_optimal(opp, generate_portfolio(wei, values))


'CAPITAL ASSET PRICE MODELLING'





























