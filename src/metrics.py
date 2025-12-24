import numpy as np
import pandas as pd

# -------- Standard Metrics --------
# ANNUALIZED RETURN w/ geometric mean
def annualized_return(returns, periods_per_year=12):
    '''returns are a series, df'''

    if len(returns) == 0: return np.nan
    return (1 + returns.mean()) ** periods_per_year - 1

# ANNUALIZED VOLATILITY
def annualized_volatility(returns, periods_per_year=12):
    '''12 mo/yr; ~252 trading days/yr'''

    if len(returns) < 2: return np.nan
    return returns.std() * np.sqrt(periods_per_year)

# SHARPE RATIO
def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=12):
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    return (ann_return - risk_free_rate) / ann_vol if (ann_vol > 0) else np.nan

# SORTINO RATIO: downside risk-adjusted return, similar to sharpe
def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=12):
    """
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: 12 for monthly
    
    Returns:
        float: Sortino ratio
    """
    ann_return = annualized_return(returns, periods_per_year)
    downside_returns = returns[returns < 0] # downside deviation
    
    if len(downside_returns) > 1:
        downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
        if downside_vol > 0:
            return (ann_return - risk_free_rate) / downside_vol
    return np.nan


# -------- Behavioral Metrics --------
# MAX DRAWDOWN, most severe decline (negative)
def max_drawdown(returns):
    if len(returns) == 0: return np.nan
    
    cumulative = (1 + returns).cumprod()    # time series of returns
    peak = cumulative.expanding().max()     # time series of max return
    drawdown = (cumulative - peak) / peak   # time series of drawdowns
    
    return drawdown.min()

# WIN RATE
def win_rate(returns):
    '''
    Args:   series of returns, period-agnostic
    Return: value from [0,1]
    '''
    if len(returns) == 0: return np.nan
    return (returns > 0).mean()


# -------- Visualization --------
def create_summary_table(returns_df):
    summary_data = []
    
    for strategy in returns_df.columns:
        returns = returns_df[strategy].dropna()
        
        if len(returns) < 12: continue  # Skip if less than 1 year of data    
            
        summary_data.append({
            'Strategy': strategy,
            'Ann Return': f"{(1 + returns.mean())**12 - 1:.1%}",
            'Ann Vol': f"{returns.std() * np.sqrt(12):.1%}",
            'Sharpe': f"{sharpe_ratio(returns):.2f}",
            'Max DD': f"{max_drawdown(returns):.1%}",
            'Win Rate': f"{win_rate(returns):.1%}",
            'Months': len(returns)
        })
    
    return pd.DataFrame(summary_data).set_index('Strategy')