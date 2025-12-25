import yfinance as yf
import pandas as pd

# -------- -------- -------- -------- -------- --------
#        Get Stock Data from Yahoo Finance
# -------- -------- -------- -------- -------- --------
def fetch_stock_data(tickers, start_date='2021-01-01', end_date='2024-12-31'):
    """
    Fetch prices and fundamental data for factor analysis.
    
    Returns:
        prices: DataFrame of monthly closing prices (dates × tickers)
        fundamentals: dict with 'earnings' and 'balance_sheet' DataFrames
    """
    if isinstance(tickers, str): tickers = [tickers] # work with list even if input one ticker
    
    daily_data = yf.download(tickers, start=start_date, end=end_date,progress=False, interval="1d")
    
    if len(tickers) == 1:
        daily_prices = daily_data[['Close']].copy()
        daily_prices.columns = tickers
    else:
        daily_prices = daily_data['Close'].copy()
    
    daily_prices.index = pd.to_datetime(daily_prices.index).tz_localize(None) # Standardize index (no timezone)
    monthly_prices = daily_prices.resample('ME').last()

    # Fetch fundamentals
    earnings_dict = {}
    income_stmt_dict = {}
    balance_sheet_dict = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Quarterly income statement
            qtr_income = stock.quarterly_income_stmt
            if qtr_income is not None and not qtr_income.empty:
                qtr_income_t = qtr_income.T
                qtr_income_t.index = pd.to_datetime(qtr_income_t.index).tz_localize(None)
                income_stmt_dict[ticker] = qtr_income_t
                
                # Extract Net Income
                if 'Net Income' in qtr_income_t.columns:
                    earnings_dict[ticker] = qtr_income_t['Net Income']
            
            # Quarterly balance sheet
            qtr_balance = stock.quarterly_balance_sheet
            if qtr_balance is not None and not qtr_balance.empty:
                qtr_balance_t = qtr_balance.T
                qtr_balance_t.index = pd.to_datetime(qtr_balance_t.index).tz_localize(None)
                balance_sheet_dict[ticker] = qtr_balance_t
                
        except Exception as e:
            print(f"  Warning: Could not fetch fundamentals for {ticker}: {e}")
            continue
    
    # Earnings (dict) -> df (dates × tickers)
    earnings = pd.DataFrame(earnings_dict)
    if not earnings.empty:
        earnings.index = pd.to_datetime(earnings.index).tz_localize(None)
        earnings = earnings.sort_index()
    
    # Sort df by date
    daily_prices = daily_prices.sort_index()
    monthly_prices = monthly_prices.sort_index()
    earnings = earnings.sort_index()
    
    # Only keep tickers w/ both price and earnings data
    valid_tickers = monthly_prices.columns.intersection(earnings.columns)
    if len(valid_tickers) == 0:
        print("⚠️  WARNING: No tickers have both price and earnings data!")
        return daily_prices, monthly_prices, pd.DataFrame(), {}
    daily_prices = daily_prices[valid_tickers]
    monthly_prices = monthly_prices[valid_tickers]
    earnings = earnings[valid_tickers]
    
    # Filter other dicts
    income_stmt_dict = {k: v for k, v in income_stmt_dict.items() if k in valid_tickers}
    balance_sheet_dict = {k: v for k, v in balance_sheet_dict.items() if k in valid_tickers}
    
    print(f"✓ Loaded {len(valid_tickers)} tickers with complete data")
    print(f"  Daily prices:   {daily_prices.shape[0]} days × {daily_prices.shape[1]} stocks")
    print(f"  Monthly prices: {monthly_prices.shape[0]} months × {monthly_prices.shape[1]} stocks")
    print(f"  Earnings data:  {len(earnings)} quarters")
    print(f"  Income stmts:   {len(income_stmt_dict)} tickers")
    print(f"  Balance sheets: {len(balance_sheet_dict)} tickers")
    
    return daily_prices, monthly_prices, earnings, income_stmt_dict, balance_sheet_dict

# # -------- -------- -------- -------- -------- --------
# #       S&P 500 Stocks (Tickers) Used
# #           (top 100 by market cap)
# # -------- -------- -------- -------- -------- --------
# if __name__ == "__main__":
#     tickers = [
#         'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
#         'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
#         'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'DHR',
#         'LIN', 'VZ', 'NKE', 'ADBE', 'TXN', 'NEE', 'CRM', 'PM', 'DIS', 'CMCSA',
#         'ORCL', 'WFC', 'HON', 'UPS', 'BMY', 'RTX', 'INTC', 'QCOM', 'UNP', 'AMD',
#         'T', 'SPGI', 'LOW', 'INTU', 'BA', 'CAT', 'ELV', 'GE', 'SBUX', 'DE',
#         'PFE', 'AXP', 'BKNG', 'BLK', 'MDT', 'ADP', 'TJX', 'GILD', 'AMGN', 'SYK',
#         'CVS', 'MMC', 'ADI', 'CI', 'VRTX', 'AMT', 'C', 'MDLZ', 'ZTS', 'ISRG',
#         'NOW', 'MO', 'PLD', 'REGN', 'DUK', 'SO', 'BDX', 'TGT', 'CB', 'SCHW',
#         'ETN', 'ITW', 'BSX', 'USB', 'HCA', 'SLB', 'GD', 'MMM', 'EOG', 'NOC'
#     ]
#     daily_prices, monthly_prices, earnings, balance_sheet_dict = fetch_stock_data(tickers)