"""
Configuration file for the Stock Trading RL Environment.

Centralizes all hyperparameters and environment settings for
reproducibility and easy experimentation.
"""

# ============================================================
# Environment Configuration
# ============================================================

ENV_CONFIG = {
    # Data / episode length
    "num_steps": 500,               # Number of trading steps per episode
    "initial_balance": 10_000.0,    # Starting cash balance ($)
    "min_balance": 1_000.0,         # Bankruptcy threshold ($)
    "transaction_fee_pct": 0.001,   # 0.1 % transaction fee per trade
    "max_shares": 100,              # Position limit (shares)
    "stop_loss_pct": 0.10,          # 10 % stop-loss from peak portfolio

    # State features
    "state_size": 7,                # price, prev_price, balance, shares,
                                    # ma5, ma10, portfolio_value

    # Action space
    "action_size": 3,               # 0=HOLD, 1=BUY, 2=SELL
}

# ============================================================
# Synthetic Price Generator
# ============================================================

PRICE_CONFIG = {
    "initial_price": 100.0,
    "drift": 0.0002,               # Slight upward drift (trend)
    "volatility": 0.015,           # Daily volatility (~1.5 %)
    "mean_reversion_strength": 0.01,
    "trend_period": 50,            # Sinusoidal trend wavelength
    "trend_amplitude": 5.0,        # Amplitude of cyclical trend
    "seed": 42,                    # For reproducibility
}

# ============================================================
# Real Data Configuration (yfinance)
# ============================================================

REAL_DATA_CONFIG = {
    "use_real_data": True,
    "ticker": "AAPL",              # Default stock ticker (Apple)
    "period": "2y",                # 2 years of historical data
    "interval": "1d",              # Daily data
}

# ============================================================
# DQN Agent Hyperparameters
# ============================================================

DQN_CONFIG = {
    # Network
    "hidden_dim_1": 64,
    "hidden_dim_2": 64,

    # Training
    "learning_rate": 1e-3,
    "gamma": 0.99,                 # Discount factor
    "batch_size": 64,
    "replay_buffer_size": 50_000,

    # Exploration
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,

    # Target network
    "target_update_freq": 10,      # Episodes between target syncs

    # Training schedule
    "num_episodes": 500,
    "save_path": "saved_models/dqn_trading.pth",
}

# ============================================================
# Reward Shaping
# ============================================================

REWARD_CONFIG = {
    "profit_scale": 1.0,
    "large_loss_penalty": -2.0,     # Extra penalty if loss > threshold
    "large_loss_threshold": 100.0,  # Dollar loss threshold
    "overtrade_penalty": -0.05,     # Penalty per consecutive trade
    "risk_penalty_scale": 0.001,    # Penalize portfolio concentration
    "consistency_bonus": 0.5,       # Bonus for consecutive positive rewards
    "risk_reduction_bonus": 0.3,    # Bonus when reducing exposure
}

# ============================================================
# Visualization / Logging
# ============================================================

VIS_CONFIG = {
    "plot_dir": "plots",
    "log_interval": 10,             # Print every N episodes
}
