"""
Synthetic Stock Price Generator
===============================

Generates realistic stock price series using:
  • Geometric random walk with drift
  • Sinusoidal trend overlay
  • Mean-reversion dampening
  • Stochastic volatility noise

No external datasets required.
"""

import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PRICE_CONFIG


def generate_stock_prices(
    num_steps: int,
    initial_price: Optional[float] = None,
    drift: Optional[float] = None,
    volatility: Optional[float] = None,
    mean_reversion: Optional[float] = None,
    trend_period: Optional[int] = None,
    trend_amplitude: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic stock price series.

    Parameters
    ----------
    num_steps : int
        Length of the price series.
    initial_price : float, optional
        Starting price (defaults to config value).
    drift : float, optional
        Per-step drift (log-return bias).
    volatility : float, optional
        Per-step standard deviation of log-returns.
    mean_reversion : float, optional
        Strength of mean-reversion toward the initial price.
    trend_period : int, optional
        Wavelength of the sinusoidal trend overlay.
    trend_amplitude : float, optional
        Amplitude of the sinusoidal trend.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    prices : np.ndarray, shape (num_steps,)
        Simulated stock prices.
    """
    # ---- Resolve defaults from config ----
    initial_price = initial_price or PRICE_CONFIG["initial_price"]
    drift = drift if drift is not None else PRICE_CONFIG["drift"]
    volatility = volatility if volatility is not None else PRICE_CONFIG["volatility"]
    mean_reversion = mean_reversion if mean_reversion is not None else PRICE_CONFIG["mean_reversion_strength"]
    trend_period = trend_period or PRICE_CONFIG["trend_period"]
    trend_amplitude = trend_amplitude or PRICE_CONFIG["trend_amplitude"]
    seed = seed if seed is not None else PRICE_CONFIG["seed"]

    rng = np.random.default_rng(seed)

    prices = np.zeros(num_steps)
    prices[0] = initial_price

    for t in range(1, num_steps):
        # Random walk component (geometric Brownian motion)
        noise = rng.normal(0, volatility)

        # Mean-reversion pull toward initial price
        reversion = mean_reversion * (np.log(initial_price) - np.log(prices[t - 1]))

        # Sinusoidal trend overlay
        trend = (trend_amplitude / initial_price) * np.sin(2 * np.pi * t / trend_period) * 0.01

        # Log-return
        log_return = drift + reversion + trend + noise

        # Apply log-return
        prices[t] = prices[t - 1] * np.exp(log_return)

        # Floor price to prevent negatives
        prices[t] = max(prices[t], 1.0)

    return prices


def generate_multiple_stocks(
    num_stocks: int,
    num_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate multiple correlated synthetic stock series.

    Parameters
    ----------
    num_stocks : int
        Number of stocks to simulate.
    num_steps : int
        Length of each price series.
    seed : int
        Base random seed.

    Returns
    -------
    all_prices : np.ndarray, shape (num_stocks, num_steps)
    """
    all_prices = np.zeros((num_stocks, num_steps))
    for i in range(num_stocks):
        # Vary volatility and drift slightly per stock
        rng_local = np.random.default_rng(seed + i)
        vol = PRICE_CONFIG["volatility"] * rng_local.uniform(0.8, 1.5)
        d = PRICE_CONFIG["drift"] * rng_local.uniform(-0.5, 2.0)
        init_p = PRICE_CONFIG["initial_price"] * rng_local.uniform(0.5, 2.0)
        all_prices[i] = generate_stock_prices(
            num_steps=num_steps,
            initial_price=init_p,
            drift=d,
            volatility=vol,
            seed=seed + i * 17,
        )
    return all_prices


def get_real_stock_prices(
    ticker: str = "AAPL",
    period: str = "2y",
    interval: str = "1d",
) -> np.ndarray:
    """
    Fetch real historical stock prices using yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'SPY').
    period : str
        Time period to download (e.g., '1y', '2y', '6mo').
    interval : str
        Data interval (e.g., '1d', '1wk', '1mo').

    Returns
    -------
    prices : np.ndarray
        Close prices of the selected stock.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[ERROR] yfinance not installed. Run: pip install yfinance pandas")
        sys.exit(1)

    print(f"Fetching real data for {ticker} (Period: {period}, Interval: {interval})...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if data.empty:
        print(f"[ERROR] Failed to fetch data for {ticker}. Check the ticker symbol.")
        sys.exit(1)

    # Use the 'Close' prices, filling any missing values
    prices = data['Close'].ffill().to_numpy().flatten()
    print(f"Fetched {len(prices)} days of {ticker} history.")
    
    # Floor to 1.0 just in case and return
    return np.maximum(prices, 1.0)


# ------------------------------------------------------------------
# Quick sanity-check when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    prices = generate_stock_prices(500)
    print(f"Generated {len(prices)} price points.")
    print(f"  Start : ${prices[0]:.2f}")
    print(f"  End   : ${prices[-1]:.2f}")
    print(f"  Min   : ${prices.min():.2f}")
    print(f"  Max   : ${prices.max():.2f}")
    print(f"  Mean  : ${prices.mean():.2f}")
    print(f"  Std   : ${prices.std():.2f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(prices, linewidth=0.8)
        plt.title("Synthetic Stock Price")
        plt.xlabel("Time Step")
        plt.ylabel("Price ($)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("sample_prices.png", dpi=150)
        plt.show()
        print("Plot saved to sample_prices.png")
    except ImportError:
        print("matplotlib not installed — skipping plot.")
