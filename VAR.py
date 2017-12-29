from pandas_datareader import data as web
import datetime


def value_at_risk(position, sigma, mu, c):
    alpha = norm.ppf(1-c)
    var = position(mu - sigma*alpha)
    return var


def value_at_risk_long(S, c, mu, sigma, n):
    alpha = norm.ppf(1-c)
    var = S*(mu*n - sigma*sqrt(n)*alpha)
    return var


print value_at_risk(5000, 0.2, 5, 1)