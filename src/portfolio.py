from matplotlib.style import available
import pandas as pd
import numpy as np
import yfinance as yf

# CONSTRUCT QUANTILE PORTFOLIO based on single factor z-scores
def construct_quantile_portfolio(prices, factor_scores, long_pct=0.2, short_pct=0.0):
    """
    Args:
        prices:         DataFrame of monthly prices (dates × tickers)
        factor_scores:  DataFrame of factor scores (dates × tickers)
        long_pct:       percentage of stocks to go long (0.2 = top 20%)
        short_pct:      percentage of stocks to short (0.0 = bottom 0%)
    
    Returns:
        portfolio_returns: Series of portfolio returns
        portfolio_weights: DataFrame of portfolio weights each period
    """
    forward_returns = prices.pct_change() # monthly returns
    portfolio_returns = []
    all_weights = []
    
    # Align dates (factor_scores and prices)
    common_dates = factor_scores.index.intersection(prices.index)
    factor_scores = factor_scores.loc[common_dates]
    prices = prices.loc[common_dates]

    # Loop through months: rebalance, get returns + weights
    for i in range(len(factor_scores) - 1):
        current_date = factor_scores.index[i]
        next_date = factor_scores.index[i + 1]

        scores = factor_scores.loc[current_date].dropna()
        
        port_return = np.nan
        weights = pd.Series()
        
        if len(scores) >= 20:
            weights = pd.Series(0.0, index=scores.index)
            
            # Select top stocks
            threshold_long = scores.quantile(1 - long_pct)
            long_stocks = scores[scores >= threshold_long].index
            
            if len(long_stocks) > 0:
                weight_long = 1.0 / len(long_stocks)
                weights[long_stocks] = weight_long
                
                if next_date in forward_returns.index: # have next period data
                    stock_returns = forward_returns.loc[next_date, weights.index]
                    
                    # Calculate if there is at least one valid return
                    if not stock_returns.empty:
                        valid_returns = stock_returns.dropna() # handle NaN
                        if not valid_returns.empty:
                            valid_weights = weights[valid_returns.index]
                            port_return = (valid_weights * valid_returns).sum()
        
        portfolio_returns.append(port_return)
        all_weights.append(weights)

    return_series = pd.Series(portfolio_returns, index=factor_scores.index[:-1])
    
    # Filter out empty Series
    weights_data = []
    weights_index = []
    for date, weights in zip(return_series.index, all_weights):
        if not weights.empty:
            weights_data.append(weights)
            weights_index.append(date)
    weights_df = pd.DataFrame(weights_data, index=weights_index) if weights_data else pd.DataFrame()
    
    return return_series, weights_df

# RETURNS FOR MULTIPLE FACTORS: returns dictionary w/ factors as keys
def factor_portfolios(prices, factors_dict, long_pct=0.2):
    """
    Args:
        prices:         monthly prices DataFrame
        factors_dict:   {factor_name: factor_scores_df}
        long_pct:       percentage to long (top)
    
    Returns:    dict of {factor_name: returns_series}
    """
    factor_returns = {}
    
    for factor_name, factor_scores in factors_dict.items():
        print(f"  Constructing {factor_name} portfolio...")
        returns, _ = construct_quantile_portfolio(prices, factor_scores, long_pct=long_pct)
        factor_returns[factor_name] = returns
    
    return factor_returns


# COMBINED PORTFOLIO created from multiple factors
def combined_portfolio(prices, factors_dict, method='equal_weight', weights=None):
    """
    Args:
        prices: monthly prices DataFrame
        factors_dict: {factor_name: factor_scores_df}
        method: 'equal_weight', 'rank_sum', 'optimized'
        weights: dict of custom weights if method='custom'
    
    Returns:
        combined returns Series
    """
    if method == 'equal_weight':
        combined_scores = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Align df dates and columns, combine z-scores
        for factor_name, factor_scores in factors_dict.items():
            factor_aligned = factor_scores.reindex(combined_scores.index)[combined_scores.columns]
            combined_scores += (1/len(factors_dict)) * factor_aligned
        
        combined_returns, _ = construct_quantile_portfolio(prices, combined_scores)
        return combined_returns
    
    elif method == 'rank_sum': # Combine ranks instead of scores
        combined_ranks = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Convert to ranks (1 = best, N = worst)
        for factor_name, factor_scores in factors_dict.items():
            ranks = factor_scores.rank(axis=1, ascending=False)
            factor_aligned = ranks.reindex(combined_ranks.index)[combined_ranks.columns]
            combined_ranks += factor_aligned
        
        combined_scores = -combined_ranks  # negative b/c lower sum of ranks is better
        combined_returns, _ = construct_quantile_portfolio(prices, combined_scores)
        return combined_returns
    
    elif method == 'optimized' and weights: # Custom weights
        combined_scores = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        for factor_name, factor_scores in factors_dict.items():
            weight = weights.get(factor_name, 0)
            factor_aligned = factor_scores.reindex(combined_scores.index)[combined_scores.columns]
            combined_scores += weight * factor_aligned
        
        combined_returns, _ = construct_quantile_portfolio(prices, combined_scores)
        return combined_returns
    
    else:
        raise ValueError(f"Unknown method: {method}")

def benchmark_returns(benchmark_ticker='SPY', start_date=None, end_date=None):
    """
    Get benchmark returns (e.g., SPY).
    """
    if start_date and end_date:
        spy = yf.download(benchmark_ticker, start=start_date, end=end_date, interval='1mo')
    else:
        spy = yf.download(benchmark_ticker, period='max', interval='1mo')
    
    spy_returns = spy['Adj Close'].pct_change()
    spy_returns.name = 'SPY'
    
    return spy_returns