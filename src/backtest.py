import yfinance as yf
import pandas as pd
import numpy as np

# -------- -------- -------- -------- -------- --------
#        Get Stock Data from Yahoo Finance
# -------- -------- -------- -------- -------- --------
def fetch_stock_data(tickers, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Fetch prices and fundamental data for factor analysis.
    
    Returns:
        prices: DataFrame of monthly closing prices (dates × tickers)
        fundamentals: dict with 'earnings' and 'balance_sheet' DataFrames
    """
    if isinstance(tickers, str): tickers = [tickers] # work with list even if input one ticker
    
    daily_data = yf.download(tickers, start=start_date, end=end_date,progress=False, interval="1d")
    
    # Get daily close prices (MultiIndex)
    if len(tickers) == 1: # transform for single ticker b/c yf returns simple columns vs multiindex
        daily_prices = daily_data[['Close']].copy()
        daily_prices.columns = tickers
    else:
        daily_prices = daily_data['Close']
    
    daily_prices.index = pd.to_datetime(daily_prices.index).tz_localize(None) # Standardize index (no timezone)
    monthly_prices = daily_prices.resample('ME').last()

    # Fetch fundamentals
    earnings_dict = {}
    balance_sheet_dict = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Quarterly financials (earnings)
            qtr_financials = stock.quarterly_financials
            if qtr_financials is not None and not qtr_financials.empty:
                qtr_financials_t = qtr_financials.T # transpose: dates index, financial items columns
                qtr_financials_t.index = pd.to_datetime(qtr_financials_t.index).tz_localize(None)
                
                # Net income -> series
                if 'Net Income' in qtr_financials_t.columns:
                    earnings_series = qtr_financials_t['Net Income']
                    earnings_series.name = ticker
                    earnings_dict[ticker] = earnings_series
            
            # Quarterly balance sheet
            qtr_balance = stock.quarterly_balance_sheet
            if qtr_balance is not None and not qtr_balance.empty:
                qtr_balance_t = qtr_balance.T
                qtr_balance_t.index = pd.to_datetime(qtr_balance_t.index).tz_localize(None)
                balance_sheet_dict[ticker] = qtr_balance_t
                
        except Exception as e:
            print(f"  Warning: Could not fetch fundamentals for {ticker}: {e}")
            continue
    
    # Earnings dict -> df (dates × tickers)
    earnings = pd.DataFrame(earnings_dict)
    earnings.index.name = 'Date'
    
    # Sort df by date
    daily_prices = daily_prices.sort_index()
    monthly_prices = monthly_prices.sort_index()
    earnings = earnings.sort_index()
    
    # Only keep tickers w/ both price and earnings data
    valid_tickers = monthly_prices.columns.intersection(earnings.columns)
    daily_prices = daily_prices[valid_tickers]
    monthly_prices = monthly_prices[valid_tickers]
    earnings = earnings[valid_tickers]
    
    # Filter balance sheets to only valid tickers
    balance_sheet_dict = {k: v for k, v in balance_sheet_dict.items() if k in valid_tickers}
    
    print(f"Successfully loaded {len(valid_tickers)} tickers with complete data")
    print(f"Daily prices:  {daily_prices.shape[0]} days × {daily_prices.shape[1]} stocks")
    print(f"Monthly prices: {monthly_prices.shape[0]} months × {monthly_prices.shape[1]} stocks")
    print(f"Earnings:      {earnings.shape[0]} quarters")
    print(f"Balance sheets: {len(balance_sheet_dict)} tickers")
    
    return daily_prices, monthly_prices, earnings, balance_sheet_dict

# -------- -------- -------- -------- -------- --------
#       S&P 500 Stocks (Tickers) Used
#           (top 100 by market cap)
# -------- -------- -------- -------- -------- --------
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'DHR',
    'LIN', 'VZ', 'NKE', 'ADBE', 'TXN', 'NEE', 'CRM', 'PM', 'DIS', 'CMCSA',
    'ORCL', 'WFC', 'HON', 'UPS', 'BMY', 'RTX', 'INTC', 'QCOM', 'UNP', 'AMD',
    'T', 'SPGI', 'LOW', 'INTU', 'BA', 'CAT', 'ELV', 'GE', 'SBUX', 'DE',
    'PFE', 'AXP', 'BKNG', 'BLK', 'MDT', 'ADP', 'TJX', 'GILD', 'AMGN', 'SYK',
    'CVS', 'MMC', 'ADI', 'CI', 'VRTX', 'AMT', 'C', 'MDLZ', 'ZTS', 'ISRG',
    'NOW', 'MO', 'PLD', 'REGN', 'DUK', 'SO', 'BDX', 'TGT', 'CB', 'SCHW',
    'ETN', 'ITW', 'BSX', 'USB', 'HCA', 'SLB', 'GD', 'MMM', 'EOG', 'NOC'
]
daily_prices, monthly_prices, earnings, balance_sheet_dict = fetch_stock_data(tickers)