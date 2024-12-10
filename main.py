"""
Portfolio Optimization using Modern Portfolio Theory (MPT).

This script performs portfolio optimization using the Markowitz model with specific
allocation constraints for two groups of stocks ('on_list' and 'off_list').
"""

import logging
from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Configure rich console
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'ON_LIST_ALLOCATION': 0.7,  # 70% allocation to on_list stocks
    'START_DATE': '2020-01-01',
    'END_DATE': '2024-12-09',
    'TRADING_DAYS_PER_YEAR': 252
}

def create_portfolio_table(allocation: pd.DataFrame) -> Table:
    """
    Create a rich table for portfolio allocation.
    
    Args:
        allocation: DataFrame containing portfolio allocation
        
    Returns:
        Rich Table object with formatted portfolio data
    """
    table = Table(title="Optimized Portfolio Allocation", show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", justify="left")
    table.add_column("Weight", justify="right")
    table.add_column("Group", style="green")
    table.add_column("Allocation %", justify="right")
    
    for _, row in allocation.iterrows():
        table.add_row(
            row['Ticker'],
            f"{row['Weight']:.4f}",
            row['Group'],
            f"{row['Weight']*100:.2f}%"
        )
    
    return table

def display_portfolio_stats(portfolio_return: float, portfolio_risk: float, sharpe_ratio: float):
    """
    Display portfolio statistics in a formatted panel.
    
    Args:
        portfolio_return: Expected annual return
        portfolio_risk: Expected annual risk (volatility)
        sharpe_ratio: Portfolio Sharpe ratio
    """
    stats_text = Text()
    stats_text.append("\nPortfolio Statistics\n", style="bold underline")
    stats_text.append(f"\nExpected Annual Return: ", style="cyan")
    stats_text.append(f"{portfolio_return:.2%}", style="green bold")
    stats_text.append(f"\nExpected Annual Risk: ", style="cyan")
    stats_text.append(f"{portfolio_risk:.2%}", style="yellow bold")
    stats_text.append(f"\nSharpe Ratio (0% risk-free rate): ", style="cyan")
    stats_text.append(f"{sharpe_ratio:.2f}", style="green bold")
    
    console.print(Panel(stats_text, border_style="blue"))

def validate_tickers(tickers: List[str]) -> bool:
    """
    Validate if the provided stock tickers exist.
    
    Args:
        tickers: List of stock tickers to validate
        
    Returns:
        bool: True if all tickers are valid, False otherwise
    """
    try:
        with console.status("[bold green]Validating tickers..."):
            for ticker in tickers:
                if not yf.Ticker(ticker).info:
                    console.print(f"[red]Invalid ticker: {ticker}[/red]")
                    return False
        return True
    except Exception as e:
        console.print(f"[red]Error validating tickers: {e}[/red]")
        return False

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data for the given tickers.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame containing adjusted closing prices
    """
    try:
        with console.status(f"[bold green]Fetching data for {len(tickers)} stocks..."):
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
            if data.empty:
                raise ValueError("No data retrieved")
            return data
    except Exception as e:
        console.print(f"[red]Error fetching stock data: {e}[/red]")
        raise

def calculate_portfolio_metrics(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate expected returns and covariance matrix.
    
    Args:
        returns: DataFrame of historical returns
        
    Returns:
        Tuple of (expected returns, covariance matrix)
    """
    with console.status("[bold green]Calculating portfolio metrics..."):
        expected_returns = returns.mean() * CONFIG['TRADING_DAYS_PER_YEAR']
        cov_matrix = returns.cov() * CONFIG['TRADING_DAYS_PER_YEAR']
        return expected_returns, cov_matrix

def optimize_portfolio(
    cov_matrix: pd.DataFrame,
    num_assets: int,
    on_list_size: int
) -> np.ndarray:
    """
    Optimize portfolio weights using convex optimization.
    
    Args:
        cov_matrix: Covariance matrix of returns
        num_assets: Total number of assets
        on_list_size: Number of assets in the on_list
        
    Returns:
        Array of optimized weights
    """
    try:
        with console.status("[bold green]Optimizing portfolio..."):
            weights = cp.Variable(num_assets)
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            objective = cp.Minimize(portfolio_variance)
            
            constraints = [
                cp.sum(weights) == 1,  # Full investment constraint
                # weights >= 0,          # Non-negative weights`
                weights >= 0.02,       # Minimum 2% per stock
                weights <= 0.10,       # Maximum 10% per stock
                cp.sum(weights[:on_list_size]) == CONFIG['ON_LIST_ALLOCATION'],
                cp.sum(weights[on_list_size:]) == (1 - CONFIG['ON_LIST_ALLOCATION'])
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status != cp.OPTIMAL:
                raise ValueError(f"Optimization failed with status: {problem.status}")
                
            return weights.value
    except Exception as e:
        console.print(f"[red]Optimization error: {e}[/red]")
        raise

def main():
    """Main execution function."""
    try:
        # Print header
        console.print("\n[bold blue]Portfolio Optimization Tool[/bold blue]", justify="center")
        console.print("=" * 50, justify="center")
        console.print()
        
        # Define stock lists
        on_list = ['COST', 'GMAB', 'GSK', 'MA', 'TGT', 'TSLA', 'TSM', 'HINDALCO.NS', 'SPY', '^FVX']
        off_list = ['ASC', 'FDX', 'LZRFY', 'MTNOY', 'RIVN', 'SIEGY', 'TTE', 'CRPG5.SA', 'SEPL.L']
        all_tickers = on_list + off_list
        
        # Validate tickers
        if not validate_tickers(all_tickers):
            raise ValueError("Invalid tickers detected")
            
        # Fetch and process data
        data = fetch_stock_data(all_tickers, CONFIG['START_DATE'], CONFIG['END_DATE'])
        returns = data.pct_change().dropna()
        
        # Calculate portfolio metrics
        expected_returns, cov_matrix = calculate_portfolio_metrics(returns)
        
        # Optimize portfolio
        optimized_weights = optimize_portfolio(
            cov_matrix,
            len(all_tickers),
            len(on_list)
        )
        
        # Create results DataFrame
        allocation = pd.DataFrame({
            'Ticker': all_tickers,
            'Weight': optimized_weights,
            'Group': ['On-List'] * len(on_list) + ['Off-List'] * len(off_list)
        })
        
        # Display results
        console.print("\n[bold]Portfolio Analysis Results[/bold]")
        console.print("=" * 50)
        
        # Display allocation table
        console.print(create_portfolio_table(allocation))
        
        # Calculate and display portfolio statistics
        portfolio_return = (expected_returns * optimized_weights).sum()
        portfolio_risk = np.sqrt(optimized_weights.T @ cov_matrix @ optimized_weights)
        sharpe_ratio = portfolio_return / portfolio_risk
        
        display_portfolio_stats(portfolio_return, portfolio_risk, sharpe_ratio)
        
        # Display group allocations
        console.print("\n[bold]Group Allocations[/bold]")
        console.print("=" * 50)
        for group in ['On-List', 'Off-List']:
            group_allocation = allocation[allocation['Group'] == group]['Weight'].sum()
            console.print(f"{group}: [cyan]{group_allocation:.1%}[/cyan]")
        
    except Exception as e:
        console.print(f"[red bold]Portfolio optimization failed: {e}[/red bold]")
        raise

if __name__ == "__main__":
    main()
