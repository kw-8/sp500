import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# -------- -------- ------- -------- --------
#                   FACTORS
# -------- -------- ------- -------- --------
# MOMENTUM
def momentum(monthly_prices, lookback=12, skip=1):
    '''
    Momentum = Price(End) / Price(Start) of period
    prices:     stock prices df
    lookback:   number of months to cover
    skip:       number of (recent) months to exclude
    '''
    momentum_signal = monthly_prices.shift(skip).pct_change(lookback)
    return z_score_normalize(momentum_signal)

# TOTAL VOLATILITY
def volatility(daily_prices, window=63):
    '''
    daily_returns:  daily returns df
    window:         num trading days
    '''
    daily_returns = daily_prices.pct_change()
    daily_vol = daily_returns.rolling(window=window).std()  # rolling standard deviation of daily returns
    annualized_vol = daily_vol * np.sqrt(252)               # estimation: ~252 trading days / year
    monthly_vol = annualized_vol.resample('ME').last()
    return z_score_normalize(-monthly_vol)

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
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data = market_data.droplevel(1, axis=1)
    
    market_returns = market_data['Close'].pct_change().dropna()
    market_returns.index = pd.to_datetime(market_returns.index).tz_localize(None)

    
    # Align dates
    common_dates = daily_returns.index.intersection(market_returns.index)
    daily_returns = daily_returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]

    print(f"   Aligned data: {len(common_dates)} days")

    idio_vol_list = []
    for ticker in daily_returns.columns:
        r_stock = daily_returns[ticker].dropna()
        r_market = market_returns.loc[r_stock.index]
        
        # Rolling regression to estimate idiosyncratic volatility
        residuals = []
        for i in range(window, len(r_stock)):
            y = r_stock.iloc[i-window:i].values
            X = market_returns.loc[r_stock.index[i-window:i]].values.reshape(-1, 1)
            
            # Filter to get valid observations
            valid_mask = ~(np.isnan(y) | np.isnan(X.flatten()))
            if valid_mask.sum() < window * 0.8:
                residuals.append(np.nan)
                continue
            
            try:
                # R_stock = alpha + beta * R_market + epsilon
                model = LinearRegression()
                model.fit(X[valid_mask], y[valid_mask])
                resid = y[valid_mask] - model.predict(X[valid_mask]) # idiosyncratic returns
                residuals.append(np.std(resid))
                
            except:
                residuals.append(np.nan)
                continue

        full_residuals = [np.nan] * window + residuals
        idio_vol_series = pd.Series(full_residuals, index=r_stock.index)
        idio_vol_list.append(idio_vol_series)

    idio_vol_daily = pd.concat(idio_vol_list, axis=1)
    idio_vol_daily.columns = daily_returns.columns
    
    # Estimate: ~252 trading days / year
    idio_vol_annual = idio_vol_daily * np.sqrt(252)
    idio_vol_monthly = idio_vol_annual.resample('ME').last()
    print(f"   Monthly idio vol shape: {idio_vol_monthly.shape}")
    print(f"   Non-NaN values: {idio_vol_monthly.notna().sum().sum()}")
    return z_score_normalize(-idio_vol_monthly) # invert to minimize

# VALUE: E/P (Earnings-to-Price) TTM with 1 quarter lag
def value_earnings_to_price(monthly_prices, earnings, lag_quarters=2):
    '''
    prices:     stock prices df w/ monthly prices (index: dates, columns: tickers)
    earnings:   quarterly earnings df with earnings data
    '''
    if earnings.empty:
        return pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)
    
    earnings_clean = earnings.dropna(how='all') # quarter end data
    print(f"   Quarterly earnings: {len(earnings_clean)} quarters")
    ttm_earnings = earnings.rolling(window=4, min_periods=2).sum()
    ttm_lagged = ttm_earnings.shift(lag_quarters)
    ttm_monthly = ttm_lagged.reindex(monthly_prices.index, method='ffill')
    
    ep_ratios = ttm_monthly / monthly_prices
    ep_ratios = ep_ratios.replace([np.inf, -np.inf], np.nan)
    ep_ratios[ep_ratios <= 0] = np.nan  # unprofitable firms
    return z_score_normalize(ep_ratios)

# QUALITY: Gross Profitability
def quality_gross_profitability(monthly_prices, income_stmts, balance_sheets):
    '''
    Gross Profitability = Gross Profit / Total Assets
    prices:         monthly prices df (index: dates, columns: tickers)
    income_stmts:   dict of quarterly income statement Dataframes
    balance_sheets: dict of quarterly balance sheet DataFrames per ticker
                    Assumes all tickers in prices.columns exist in balance_sheet_dict
    '''
    gp_ratios = pd.DataFrame(index=monthly_prices.index, columns=monthly_prices.columns)
    
    for ticker in monthly_prices.columns:
        if ticker not in income_stmts or ticker not in balance_sheets:
            continue # skip to next ticker
        
        income_df = income_stmts[ticker]
        balance_df = balance_sheets[ticker]

        # Find columns (yf naming inconsistent)
        revenue_col = None
        cogs_col = None
        assets_col = None

        # Search income statement
        for col in income_df.columns:
            col_lower = col.lower().replace(' ', '')
            if 'totalrevenue' in col_lower or 'revenue' == col_lower:
                revenue_col = col
            elif 'costofrevenue' in col_lower or 'cogs' in col_lower:
                cogs_col = col
        
        # Search balance sheet
        for col in balance_df.columns:
            col_lower = col.lower().replace(' ', '')
            if 'totalassets' in col_lower:
                assets_col = col
                break
        
        # Calculate if we have all components
        if revenue_col and cogs_col and assets_col:
            try:
                revenue = income_df[revenue_col]
                cogs = income_df[cogs_col]
                assets = balance_df[assets_col]
                
                # Align dates (income and balance sheet may have different dates)
                common_dates = revenue.index.intersection(cogs.index).intersection(assets.index)
                
                if len(common_dates) > 0:
                    gross_profit = revenue.loc[common_dates] - cogs.loc[common_dates]
                    gp_ratio_quarterly = gross_profit / assets.loc[common_dates]
                    
                    # Convert to monthly, forward-fill
                    gp_ratio_monthly = gp_ratio_quarterly.reindex(monthly_prices.index, method='ffill')
                    gp_ratios[ticker] = gp_ratio_monthly
                    
            except Exception as e:
                print(f"  Warning: GP calculation failed for {ticker}: {e}")
                continue
        else:
            missing = []
            if not revenue_col: missing.append('Revenue')
            if not cogs_col: missing.append('COGS')
            if not assets_col: missing.append('Total Assets')
            print(f"  Warning: {ticker} missing: {', '.join(missing)}")
    
    return z_score_normalize(gp_ratios)

# -------- -------- ------- -------- --------
#      CALCULATE ALL THE FACTOR SCORES
# -------- -------- ------- -------- --------
def calculate_all_factors(daily_prices, monthly_prices, earnings, income_stmts, balance_sheets):
    mom_scores = momentum(monthly_prices, lookback=12, skip=1)
    val_scores = value_earnings_to_price(monthly_prices, earnings, lag_quarters=1)
    qual_scores = quality_gross_profitability(monthly_prices, income_stmts, balance_sheets)
    # vol_scores = volatility_idiosyncratic(daily_prices)
    vol_scores = volatility(daily_prices)

    factors = {
        'momentum': mom_scores,
        'value': val_scores,
        'quality': qual_scores,
        'volatility': vol_scores,
    }

    return factors

# -------- -------- ------- -------- --------
#               HELPER FUNCTIONS
# -------- -------- ------- -------- --------
def z_score_normalize(df):
    """
    Z-score normalization: each row (time period) normalized across columns (stocks)

    df: DataFrame with dates × tickers
    normalized: DataFrame with z-scores (mean=0, std=1 for each row)
    """
    # Row means and stds
    row_means = df.mean(axis=1)
    row_stds = df.std(axis=1)
    
    # Handle division by zero (replace 0 with NaN, then fill with 0)
    safe_stds = row_stds.replace(0, np.nan)
    
    # Z-score: (x - mean) / std
    normalized = df.sub(row_means, axis=0).div(safe_stds, axis=0)
    
    # Fill NaN (where std=0) with 0 (all values equal)
    return normalized.fillna(0)