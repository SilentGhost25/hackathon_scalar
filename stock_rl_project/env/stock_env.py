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

from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - local fallback for lightweight evaluation
    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def contains(self, x) -> bool:
            try:
                value = int(x)
            except (TypeError, ValueError):
                return False
            return 0 <= value < self.n

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

        def contains(self, x) -> bool:
            arr = np.asarray(x, dtype=self.dtype)
            if arr.shape != self.shape:
                return False
            return np.all(arr >= self.low) and np.all(arr <= self.high)

    class _Env:
        metadata = {"render_modes": ["human"]}

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            return None

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _GymModule:
        Env = _Env
        spaces = _Spaces()

    gym = _GymModule()

try:
    from config import ENV_CONFIG, REWARD_CONFIG
except ImportError:  # pragma: no cover - supports package-style imports
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import ENV_CONFIG, REWARD_CONFIG

class StockTradingEnv(gym.Env):
    """
    Custom RL environment for single-stock trading using Gymnasium.

    State:
        Market data features from dataframe (OHLCV, RSI, MACD, etc.)
        + Portfolio State (balance, shares_held, unrealized PnL)

    Actions:
        Discrete {0, 1, 2}
        0 = Hold
        1 = Buy one or more shares
        2 = Sell one or more shares

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
        slippage_pct: float = 0.0005,
        trade_fraction: float = 0.25,
        cooldown_steps: int = 3,
        max_drawdown_penalty: float = 0.1,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee_pct = transaction_fee_pct
        self.slippage_pct = slippage_pct
        self.trade_fraction = trade_fraction
        self.cooldown_steps = cooldown_steps
        self.max_drawdown_penalty = max_drawdown_penalty
        self.max_shares = int(ENV_CONFIG.get("max_shares", 100))

        self.num_steps = len(self.df)
        
        # Action space: Discrete buy / hold / sell for DQN compatibility.
        self.action_space = gym.spaces.Discrete(3)
        
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
        self.last_action = 0
        self.last_trade_step = -10_000
        
        # History
        self.history = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.last_action = 0
        self.last_trade_step = -10_000
        
        self.history = {
            "price": [],
            "portfolio_value": [self.balance],
            "reward": [],
            "action": [],
            "balance": [self.balance],
            "shares_held": [self.shares_held],
        }

        return self._get_state(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"{action} invalid"
        
        current_price = float(self.df.iloc[self.current_step]["Close"])
        act = int(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else int(action)
        trade_cost = 0.0
        trade_notional = 0.0
        
        # Keep track of previous state for reward computation
        prev_portfolio_value = self.portfolio_value
        did_trade = False
        
        # Execute Action
        if act == 1:
            if self.current_step - self.last_trade_step <= self.cooldown_steps:
                act = 0
                reward = 0.0
                did_trade = False
                trade_cost = 0.0
                trade_notional = 0.0
            else:
                # BUY a capped fraction of available cash to avoid all-in behavior.
                amount_to_spend = self.balance * self.trade_fraction
                cost_per_share = current_price * (1 + self.transaction_fee_pct + self.slippage_pct)
                shares_to_buy = int(amount_to_spend / cost_per_share)

                if shares_to_buy > 0:
                    shares_to_buy = min(shares_to_buy, self.max_shares - self.shares_held)
                if shares_to_buy > 0:
                    effective_price = current_price * (1 + self.slippage_pct)
                    fee = shares_to_buy * current_price * self.transaction_fee_pct
                    trade_notional = shares_to_buy * effective_price
                    self.balance -= (trade_notional + fee)
                    self.shares_held += shares_to_buy
                    trade_cost = fee
                    did_trade = True
        elif act == 2:
            if self.current_step - self.last_trade_step <= self.cooldown_steps:
                act = 0
            else:
                # SELL only a fraction of held shares.
                shares_to_sell = int(max(1, self.shares_held * self.trade_fraction))
                shares_to_sell = min(shares_to_sell, self.shares_held)

                if shares_to_sell > 0:
                    effective_price = current_price * (1 - self.slippage_pct)
                    revenue = shares_to_sell * effective_price
                    fee = revenue * self.transaction_fee_pct
                    self.balance += (revenue - fee)
                    self.shares_held -= shares_to_sell
                    trade_notional = revenue
                    trade_cost = fee
                    did_trade = True

        # Update Portfolio
        self.portfolio_value = self.balance + (self.shares_held * current_price)
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)
        
        # Calculate Reward
        # Use a scaled percentage return so the signal stays numerically stable.
        portfolio_return = self.portfolio_value - prev_portfolio_value
        return_pct = portfolio_return / self.initial_balance
        drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio

        reward = (
            REWARD_CONFIG["profit_scale"] * return_pct
            - REWARD_CONFIG["fee_scale"] * (trade_cost / self.initial_balance)
            - REWARD_CONFIG["slippage_scale"] * ((trade_notional * self.slippage_pct) / self.initial_balance if did_trade else 0.0)
            - REWARD_CONFIG["drawdown_scale"] * drawdown
        )

        if act != 0:
            reward -= REWARD_CONFIG["trade_penalty"]

        if did_trade and self.current_step - self.last_trade_step <= 1:
            reward -= REWARD_CONFIG["cooldown_penalty"]

        if did_trade and act == self.last_action and act != 0:
            reward -= REWARD_CONFIG["repeat_trade_penalty"]

        if portfolio_return > 0:
            reward += REWARD_CONFIG["positive_step_bonus"]
        elif portfolio_return < 0:
            reward -= REWARD_CONFIG["negative_step_penalty"]
        
        # Advance Step
        self.current_step += 1
        
        # Terminal conditions
        terminated = self.current_step >= self.num_steps - 1
        
        if self.portfolio_value < 0.1 * self.initial_balance: # 90% loss stop-out
            terminated = True
            reward -= REWARD_CONFIG["bankruptcy_penalty"]

        # Logging
        self.history["portfolio_value"].append(self.portfolio_value)
        self.history["reward"].append(reward)
        self.history["action"].append(act)
        self.history["price"].append(current_price)
        self.history["balance"].append(self.balance)
        self.history["shares_held"].append(self.shares_held)
        if did_trade:
            self.last_trade_step = self.current_step
        self.last_action = act

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

    def get_metrics(self) -> Dict[str, Any]:
        """Summarize an episode using the tracked history."""
        portfolio = np.asarray(self.history.get("portfolio_value", [self.initial_balance]), dtype=np.float32)
        if portfolio.size == 0:
            portfolio = np.array([self.initial_balance], dtype=np.float32)

        final_value = float(portfolio[-1])
        profit = final_value - self.initial_balance
        return_pct = (profit / self.initial_balance) * 100.0

        running_peak = np.maximum.accumulate(portfolio)
        drawdowns = np.divide(
            running_peak - portfolio,
            running_peak,
            out=np.zeros_like(portfolio),
            where=running_peak > 0,
        )
        max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0

        rewards = np.asarray(self.history.get("reward", []), dtype=np.float32)
        wins = float(np.mean(rewards > 0)) if rewards.size else 0.0
        trades = int(np.sum(np.asarray(self.history.get("action", []), dtype=np.int32) != 0))

        return {
            "total_profit": float(profit),
            "return_pct": float(return_pct),
            "win_rate": wins,
            "max_drawdown": max_drawdown,
            "total_trades": trades,
        }

    def render(self):
        print(
            f"Step {self.current_step:>4d} | "
            f"Balance: ${self.balance:>10.2f} | "
            f"Shares: {self.shares_held:>3d} | "
            f"Portfolio: ${self.portfolio_value:>10.2f}"
        )
