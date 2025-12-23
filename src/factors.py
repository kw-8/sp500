import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# MOMENTUM
def momentum(monthly_prices, lookback=12, skip=1):
    '''
    Momentum = Price(End) / Price(Start) of period
    prices:     stock prices df
    lookback:   number of months to cover
    skip:       number of (recent) months to exclude
    '''
    momentum_signal = monthly_prices.pct_change(lookback).shift(skip)
    momentum_norm = momentum_signal.apply(lambda row: (row - row.mean()) / row.std()
                                            if row.std() > 0 else 0, axis=1)
    return momentum_norm

# TOTAL VOLATILITY
def volatility(daily_returns, window=63):
    '''
    daily_returns:  daily returns df
    window:         num trading days
    '''
    daily_vol = daily_returns.rolling(window=window).std()  # rolling standard deviation of daily returns
    annualized_vol = daily_vol * np.sqrt(252)               # estimation: ~252 trading days / year
    
    return annualized_vol

# IDIOSYNCRATIC VOLATILITY
def volatility_idiosyncratic(daily_prices, market_ticker='SPY', window=63, min_obs=20):
    '''
    daily_returns:  daily returns df
    market_returns: daily market returns (SPY)
    window:         num trading days
    min_obs:        min obs required
    '''
    daily_returns = daily_prices.pct_change()
    
    market_data = yf.download(market_ticker, 
                            start=daily_prices.index[0].strftime('%Y-%m-%d'),
                            end=daily_prices.index[-1].strftime('%Y-%m-%d'),
                            progress=False)
    market_returns = market_data['Close'].pct_change()
    
    # Align dates
    common_idx = daily_returns.index.intersection(market_returns.index)
    daily_returns = daily_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]
    
    # Initialize output
    idio_vol = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns)
    
    print(f"Calculating idiosyncratic volatility (rolling {window} days)...")
    
    for ticker in daily_returns.columns:
        stock_returns = daily_returns[ticker].dropna()
        
        if len(stock_returns) < window + 20:
            continue
        
        # Rolling regression to estimate idiosyncratic volatility
        for i in range(window, len(stock_returns)):
            start_idx = i - window
            
            X = market_returns.iloc[start_idx:i].values.reshape(-1, 1)
            y = stock_returns.iloc[start_idx:i].values
            
            # Skip if insufficient data
            if len(y) < window * 0.8:
                continue
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                residuals = y - model.predict(X)
                
                # Daily idiosyncratic volatility
                daily_idio_vol = np.std(residuals)
                idio_vol.iloc[i, ticker] = daily_idio_vol
                
            except:
                continue
    
    # Estimate: ~252 trading days / year
    idio_vol_annual = idio_vol * np.sqrt(252)
    idio_vol_monthly = idio_vol_annual.resample('ME').last()
    idio_vol_monthly = idio_vol_monthly.reindex(daily_prices.resample('ME').last().index)
    # Z-score
    idio_norm = idio_vol_monthly.apply(lambda row: -(row - row.mean()) / row.std()
                                        if row.std() > 0 else 0, axis=1)
    return idio_norm

# VALUE: E/P (Earnings-to-Price) TTM with 2 month lag
def value_earnings_to_price(monthly_prices, earnings, lag_months=2):
    '''
    prices:     stock prices df w/ monthly prices (index: dates, columns: tickers)
    earnings:   quarterly earnings df with earnings data
    '''
    earnings_monthly = earnings.reindex(monthly_prices.index).ffill()   # forward fill quarterly -> monthly
    earnings_lagged = earnings_monthly.shift(lag_months)

    ep_ratios = earnings_lagged / monthly_prices
    ep_ratios = ep_ratios.replace([np.inf, -np.inf], np.nan)    # handle invalid values
    ep_norm = ep_ratios.apply(lambda row: (row - row.mean()) / row.std()
                                if row.std() > 0 else 0, axis=1)
    return ep_norm

# QUALITY: Gross Profitability
def quality_gross_profitability(monthly_prices, balance_sheets):
    '''
    Gross Profitability = Gross Profit / Total Assets
    prices:         monthly prices df (index: dates, columns: tickers)
    balance_sheets: dict of quarterly balance sheet DataFrames per ticker
                    Assumes all tickers in prices.columns exist in balance_sheet_dict
    '''
    gp_ratios = pd.DataFrame(index=monthly_prices.index, columns=monthly_prices.columns)
    
    for ticker in monthly_prices.columns:
        if ticker not in balance_sheets: continue # skip to next ticker
        bs_df = balance_sheets[ticker]
        
        # Find gross profit and total assets columns
        gp_col, ta_col = None, None

        for col in bs_df.columns:
            col_lower = col.lower()
            if 'gross profit' in col_lower or 'grossprofit' in col_lower.replace(' ', ''):
                gp_col = col
            elif 'total asset' in col_lower:
                ta_col = col
        
        if not gp_col or not ta_col: continue # skip to next ticker
        try:
            # Get quarterly series, forward fill and get monthly gp ratios
            gp_series = bs_df[gp_col]
            ta_series = bs_df[ta_col]

            quarterly_ratio = gp_series / ta_series
            monthly_ratio = quarterly_ratio.reindex(monthly_prices.index).ffill()
            gp_ratios[ticker] = monthly_ratio
            
        except Exception as e:
            print(f"Warning: Could not calculate GP for {ticker}: {e}")
            continue
    
    gp_norm = gp_ratios.apply(lambda row: (row - row.mean()) / row.std()
                                if row.std() > 0 else 0, axis=1)
    
    return gp_norm