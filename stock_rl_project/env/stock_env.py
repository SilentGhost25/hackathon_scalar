"""
StockTradingEnv — Gymnasium-compatible RL Environment
====================================================

A gymnasium-compatible reinforcement learning environment that simulates stock
trading with realistic dynamics including transaction fees, position
limits, and shaped rewards.

Re-written based on PPO + Gymnasium best practices:
- Uses `gymnasium.Env`
- Continuous Action Space for position sizing
- Normalised State / Reward handled by VecNormalize wrapper in train.py.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class StockTradingEnv(gym.Env):
    """
    Custom RL environment for single-stock trading using Gymnasium.

    State:
        Market data features from dataframe (OHLCV, RSI, MACD, etc.)
        + Portfolio State (balance, shares_held, unrealized PnL)

    Actions:
        Continuous [-1, 1] 
        Negative values = Sell fraction of shares
        Positive values = Buy fraction of max possible shares with current cash
        0 = Hold

    Rewards:
        Shaped reward based on portfolio return, minus a transaction cost
        penalty and a drawdown penalty to discourage reckless trading.
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        transaction_fee_pct: float = 0.001,
        max_drawdown_penalty: float = 0.1,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee_pct = transaction_fee_pct
        self.max_drawdown_penalty = max_drawdown_penalty

        self.num_steps = len(self.df)
        
        # Action space: Continuous from -1.0 to 1.0
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        
        # State space includes market features + balance, shares_held, unrealized_pnl
        self.price_features = self.df.columns.tolist()
        self.state_size = len(self.price_features) + 3 
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        
        # History
        self.history = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        
        self.history = {
            "portfolio_value": [self.balance],
            "reward": [],
            "action": []
        }

        return self._get_state(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"{action} invalid"
        
        current_price = float(self.df.iloc[self.current_step]["Close"])
        act = float(action[0])
        trade_cost = 0.0
        
        # Keep track of previous state for reward computation
        prev_portfolio_value = self.portfolio_value
        
        # Execute Action
        if act > 0:
            # BUY
            amount_to_spend = self.balance * act
            cost_per_share = current_price * (1 + self.transaction_fee_pct)
            shares_to_buy = int(amount_to_spend / cost_per_share)
            
            if shares_to_buy > 0:
                fee = shares_to_buy * current_price * self.transaction_fee_pct
                self.balance -= (shares_to_buy * current_price + fee)
                self.shares_held += shares_to_buy
                trade_cost = fee
        elif act < 0:
            # SELL
            shares_to_sell = int(self.shares_held * abs(act))
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                fee = revenue * self.transaction_fee_pct
                self.balance += (revenue - fee)
                self.shares_held -= shares_to_sell
                trade_cost = fee

        # Update Portfolio
        self.portfolio_value = self.balance + (self.shares_held * current_price)
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)
        
        # Calculate Reward
        # 1. Base profit (dollar amount)
        portfolio_return = self.portfolio_value - prev_portfolio_value
        
        # 2. Drawdown
        drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio
        
        # Shaped Reward Formula
        reward = portfolio_return - trade_cost - (self.max_drawdown_penalty * drawdown * self.initial_balance)
        
        # Advance Step
        self.current_step += 1
        
        # Terminal conditions
        terminated = self.current_step >= self.num_steps - 1
        
        if self.portfolio_value < 0.1 * self.initial_balance: # 90% loss stop-out
            terminated = True
            reward -= self.initial_balance * 0.1  # Heavy bankruptcy penalty

        # Logging
        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["reward"].append(reward)
        self.history["action"].append(act)

        # Return format for Gymnasium environment: (obs, reward, terminated, truncated, info)
        return self._get_state(), float(reward), terminated, False, self._get_info()

    def _get_state(self) -> np.ndarray:
        # Market data
        # Handle mixed types gracefully if present
        market_features = self.df.iloc[self.current_step].values
        # Force numeric types and convert to float32
        market_features = pd.to_numeric(market_features, errors='coerce')
        market_features = np.nan_to_num(market_features).astype(np.float32)
        
        # Portfolio state
        unrealized_pnl = self.portfolio_value - self.initial_balance
        portfolio_state = np.array([
            self.balance,
            float(self.shares_held),
            unrealized_pnl
        ], dtype=np.float32)
        
        state = np.concatenate((market_features, portfolio_state))
        return state

    def _get_info(self) -> Dict[str, Any]:
        return {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "step": self.current_step,
            "final_portfolio": self.portfolio_value
        }

    def render(self):
        print(
            f"Step {self.current_step:>4d} | "
            f"Balance: ${self.balance:>10.2f} | "
            f"Shares: {self.shares_held:>3d} | "
            f"Portfolio: ${self.portfolio_value:>10.2f}"
        )
