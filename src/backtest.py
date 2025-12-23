import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(tickers, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Fetch prices and fundamental data for factor analysis.
    
    Returns:
        prices: DataFrame of monthly closing prices (dates × tickers)
        fundamentals: dict with 'earnings' and 'balance_sheet' DataFrames
    """
    if isinstance(tickers, str): tickers = [tickers] # work with list even if input one ticker
    
    data = yf.download(tickers, start=start_date, end=end_date,progress=False, interval="1mo")
    
    # Get close prices (MultiIndex)
    if len(tickers) == 1: # transform for single ticker b/c yf returns simple columns vs multiindex
        prices = data[['Close']].copy()
        prices.columns = tickers
    else:
        prices = data['Close']
    
    prices.index = pd.to_datetime(prices.index).tz_localize(None) # standardize datetime format (no timezone)
    
    # Fetch fundamentals for all tickers
    earnings_dict = {}
    balance_sheet_dict = {}
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        
        qtr_financials = stock.quarterly_financials
        if qtr_financials is not None and not qtr_financials.empty:
            if 'Net Income' in qtr_financials.index:
                earnings_dict[ticker] = qtr_financials.loc['Net Income']
        
        qtr_balance = stock.quarterly_balance_sheet
        if qtr_balance is not None and not qtr_balance.empty:
            balance_sheet_dict[ticker] = qtr_balance
    
    # Convert to DataFrames (dates × tickers)
    earnings = pd.DataFrame(earnings_dict)
    earnings.index = pd.to_datetime(earnings.index).tz_localize(None)
    earnings = earnings.sort_index()
    
    # Only keep tickers with both price and earnings data
    valid_tickers = prices.columns.intersection(earnings.columns)
    prices, earnings = prices[valid_tickers], earnings[valid_tickers]
    
    print(f"Successfully loaded {len(valid_tickers)} tickers with complete data")
    print(f"Price data: {prices.shape[0]} months × {prices.shape[1]} stocks")
    print(f"Earnings data: {len(earnings)} quarters")
    
    fundamentals = {
        'earnings': earnings,
        'balance_sheet': balance_sheet_dict
    }
    
    return prices, fundamentals

# -------- S&P 500 Stocks (Tickers) Used --------
# Top 100 S&P 500 stocks by market cap (manual list, or scrape)
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
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
prices, fundamentals = fetch_stock_data(tickers)


# # -------- Test same format style --------
# prices1, fundamentals1 = fetch_stock_data('AAPL')           # Returns DataFrame with 1 column
# prices2, fundamentals2 = fetch_stock_data(['AAPL', 'MSFT']) # Returns DataFrame with 2 columns

# prices1.shape, prices2.shape