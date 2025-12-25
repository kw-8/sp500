import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_data import fetch_stock_data
from portfolio import factor_portfolios, combined_portfolio, benchmark_returns
from factors import calculate_all_factors
from metrics import annualized_return, annualized_volatility, create_summary_table

def analyze_factor_strategies(daily_prices, monthly_prices, earnings, income_stmts, balance_sheets):
    """Analysis pipeline with visualizations"""
    print("=" * 60)
    print("FACTOR PORTFOLIO ANALYSIS")
    print("=" * 60)

    # 1. Calculate factors
    print("\nCalculating factor signals...")
    factors = calculate_all_factors(daily_prices, monthly_prices, earnings, income_stmts, balance_sheets)
    
    # 2. Construct individual factor portfolios
    print("\nConstructing individual factor portfolios...")
    factor_returns = factor_portfolios(monthly_prices, factors, long_pct=0.2)
    
    # 3. Create combined portfolio
    print("\nConstructing combined portfolio (equal-weight)...")
    combined = combined_portfolio(monthly_prices, factors, method='equal_weight')
    factor_returns['combined'] = combined
    
    # 4. Get benchmark
    print("\nGetting benchmark (SPY)...")
    start_date = monthly_prices.index[0].strftime('%Y-%m-%d')
    end_date = monthly_prices.index[-1].strftime('%Y-%m-%d')
    spy = benchmark_returns('SPY', start_date, end_date)
    factor_returns['benchmark'] = spy
    
    # 5. Combine into DataFrame
    returns_df = pd.DataFrame(factor_returns)
    returns_df = returns_df.dropna()

    # DEBUG CHCECK CORR
    print("\n" + "="*60)
    print("FACTOR RETURN CORRELATION CHECK")
    print("="*60)

    # Get just the factor returns (exclude benchmark/combined)
    factor_return_cols = [col for col in returns_df.columns 
                        if col not in ['benchmark', 'combined']]

    if len(factor_return_cols) > 1:
        # Calculate correlation of RETURNS (not signals)
        corr_matrix = returns_df[factor_return_cols].corr()
        
        print("Correlation of Factor PORTFOLIO RETURNS:")
        print(corr_matrix)
        
        # Check if correlations are suspiciously high
        for i, col1 in enumerate(factor_return_cols):
            for j, col2 in enumerate(factor_return_cols):
                if i < j:
                    corr_value = corr_matrix.loc[col1, col2]
                    print(f"{col1} vs {col2}: {corr_value:.3f}", 
                        end="")
                    if abs(corr_value) > 0.7:
                        print("  ⚠️ TOO HIGH - factors are identical")
                    elif abs(corr_value) > 0.3:
                        print("  ⚠️ Moderately high")
                    else:
                        print("  ✓ Good")
    else:
        print("Not enough factor returns to calculate correlations")

    

    # Align SPY to factor returns index
    spy_aligned = spy.reindex(returns_df.index)
    returns_df['benchmark'] = spy_aligned
    
    # Drop rows where ALL strategies are NaN
    returns_df = returns_df.dropna(how='all')
    
    # Drop rows where benchmark is NaN (but keep rows where only some factors are NaN)
    returns_df = returns_df[returns_df['benchmark'].notna()]
    
    print(f"\nAnalysis complete: {returns_df.shape[0]} months, {returns_df.shape[1]} strategies")
    
    return returns_df, factors

def create_visualizations(returns_df, factors, save_dir='./results'):
    """Create all visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. CUMULATIVE RETURNS
    print("\nCreating cumulative returns chart...")
    cumulative = (1 + returns_df).cumprod()
    
    plt.figure(figsize=(14, 7))
    for col in cumulative.columns:
        linewidth = 2.5 if col in ['combined', 'benchmark'] else 1.5
        alpha = 1.0 if col in ['combined', 'benchmark'] else 0.7
        plt.plot(cumulative.index, cumulative[col], label=col, linewidth=linewidth, alpha=alpha)
    
    plt.legend(loc='best', fontsize=10)
    plt.title('Cumulative Returns: Factor Strategies (2023-2024)', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_dir}/cumulative_returns.png")
    plt.close()
    
    # 2. CORRELATION MATRIX (of factor returns, not signals)
    print("\nCreating correlation matrix...")
    
    # Get just the factor returns (exclude benchmark and combined)
    factor_cols = [col for col in returns_df.columns if col not in ['benchmark', 'combined']]
    corr_matrix = returns_df[factor_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation'})
    plt.title('Factor Return Correlations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_dir}/correlation_matrix.png")
    plt.close()
    
    # 3. RISK-RETURN SCATTER
    print("\nCreating risk-return scatter...")
    
    metrics_data = []
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        metrics_data.append({
            'Strategy': col,
            'Return': annualized_return(returns),
            'Volatility': annualized_volatility(returns)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 7))
    colors = ['red' if s == 'benchmark' else 'green' if s == 'combined' else 'blue' 
              for s in metrics_df['Strategy']]
    sizes = [150 if s in ['benchmark', 'combined'] else 100 for s in metrics_df['Strategy']]
    
    plt.scatter(metrics_df['Volatility'], metrics_df['Return'], 
                c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    for i, row in metrics_df.iterrows():
        plt.annotate(row['Strategy'], 
                    (row['Volatility'], row['Return']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    plt.xlabel('Annualized Volatility', fontsize=12)
    plt.ylabel('Annualized Return', fontsize=12)
    plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/risk_return_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_dir}/risk_return_scatter.png")
    plt.close()
    
    # 4. DRAWDOWN CHART
    print("\nCreating drawdown chart...")
    
    cumulative = (1 + returns_df).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    plt.figure(figsize=(14, 7))
    
    # Plot combined and benchmark drawdowns
    plt.plot(drawdown.index, drawdown['combined'], label='Combined', linewidth=2, color='green')
    plt.plot(drawdown.index, drawdown['benchmark'], label='S&P 500', linewidth=2, color='red', linestyle='--')
    
    # Fill area
    plt.fill_between(drawdown.index, 0, drawdown['combined'], alpha=0.3, color='green')
    
    plt.legend(loc='lower left', fontsize=10)
    plt.title('Drawdowns Over Time', fontsize=14, fontweight='bold')
    plt.ylabel('Drawdown', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/drawdowns.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_dir}/drawdowns.png")
    plt.close()
    
    print("\nAll visualizations created!")




# -------- -------- ------- -------- --------
#        BACKTEST SAMPLE: RUN ANALYSIS
# -------- -------- ------- -------- --------
if __name__ == "__main__":
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
        'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'COST', 'KO', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN', 'ABT', 'DHR',
        'LIN', 'VZ', 'NKE', 'ADBE', 'TXN', 'NEE', 'CRM', 'PM', 'DIS', 'CMCSA',
        'ORCL', 'WFC', 'HON', 'UPS', 'BMY', 'RTX', 'INTC', 'QCOM', 'UNP', 'AMD'#,
        # 'T', 'SPGI', 'LOW', 'INTU', 'BA', 'CAT', 'ELV', 'GE', 'SBUX', 'DE',
        # 'PFE', 'AXP', 'BKNG', 'BLK', 'MDT', 'ADP', 'TJX', 'GILD', 'AMGN', 'SYK',
        # 'CVS', 'MMC', 'ADI', 'CI', 'VRTX', 'AMT', 'C', 'MDLZ', 'ZTS', 'ISRG',
        # 'NOW', 'MO', 'PLD', 'REGN', 'DUK', 'SO', 'BDX', 'TGT', 'CB', 'SCHW',
        # 'ETN', 'ITW', 'BSX', 'USB', 'HCA', 'SLB', 'GD', 'MMM', 'EOG', 'NOC'
    ]
    
    print("Fetching data...")
    daily_prices, monthly_prices, earnings, income_stmts, balance_sheets = fetch_stock_data(
        tickers, '2000-01-01', '2025-12-31'
    )
    
    # Run analysis
    returns_df, factors = analyze_factor_strategies(daily_prices, monthly_prices, earnings, income_stmts, balance_sheets)
    
    # Summary table
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    summary = create_summary_table(returns_df)
    print(summary)
    
    # Summary to CSV
    summary.to_csv('results/performance_summary.csv')
    print("\nSaved: results/performance_summary.csv")
    
    # Visualizations
    create_visualizations(returns_df, factors)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nResults saved to ./results/")
    print("  - cumulative_returns.png")
    print("  - correlation_matrix.png")
    print("  - risk_return_scatter.png")
    print("  - drawdowns.png")
    print("  - performance_summary.csv")