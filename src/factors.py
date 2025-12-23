import pandas as pd
import numpy as np

# Momentum
def momentum(prices, lookback, skip):
    '''
    Momentum = Price(End) / Price(Start) of period
    prices:     stock prices df
    lookback:   number of months to cover
    skip:       number of (recent) months to exclude
    '''
    momentum_signal = prices.pct_change(lookback).shift(skip)

    return momentum_signal

# Volatility
def volatility(daily_returns, window=63):
    '''
    daily_returns:  daily returns df
    window:         num trading days
    '''
    # Rolling standard deviation of daily returns
    daily_vol = daily_returns.rolling(window=window).std()
    
    # Annualize: √252 trading days in a year
    annualized_vol = daily_vol * np.sqrt(252)
    
    return annualized_vol

