from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import ENV_CONFIG, PRICE_CONFIG
from data.price_generator import generate_stock_prices
from features import STATE_SIZE, build_state_from_row, prepare_market_frame, flatten_columns

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None


@dataclass
class EpisodeMetrics:
    total_reward: float
    final_portfolio_value: float
    total_trades: int
    max_drawdown: float
    win_rate: float
    decision_accuracy: float
    profitable_trade_rate: float


def load_market_history(
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    if yf is not None:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
            if data is not None and not data.empty:
                data = flatten_columns(data)
                if "Volume" not in data.columns:
                    data["Volume"] = 0.0
                return data[["Close", "Volume"]].copy()
        except Exception:
            pass

    prices = generate_stock_prices(900, initial_price=PRICE_CONFIG["initial_price"], seed=PRICE_CONFIG["seed"])
    rng = np.random.default_rng(PRICE_CONFIG["seed"])
    volumes = rng.integers(750_000, 5_000_000, size=len(prices))
    return pd.DataFrame({"Close": prices, "Volume": volumes})


class TradingEpisodeEnv:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        initial_balance: float = ENV_CONFIG["initial_balance"],
        transaction_fee_pct: float = ENV_CONFIG["transaction_fee_pct"],
        max_shares: int = ENV_CONFIG["max_shares"],
        episode_length: int = 252,
        buy_fraction: float = 0.20,
        sell_fraction: float = 0.25,
    ) -> None:
        prepared = prepare_market_frame(frame)
        self.frame = prepared.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.transaction_fee_pct = float(transaction_fee_pct)
        self.max_shares = int(max_shares)
        self.episode_length = int(min(episode_length, len(self.frame)))
        self.buy_fraction = float(buy_fraction)
        self.sell_fraction = float(sell_fraction)

        self.current_step = 0
        self.window_start = 0
        self.window_end = self.episode_length - 1
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.trade_count = 0
        self.trade_wins = 0
        self.decision_matches = 0
        self.decision_count = 0
        self.reward_history: list[float] = []
        self.portfolio_history: list[float] = []

    @property
    def state_size(self) -> int:
        return STATE_SIZE

    def reset(self, start_index: Optional[int] = None, episode_length: Optional[int] = None) -> np.ndarray:
        if episode_length is not None:
            self.episode_length = int(min(episode_length, len(self.frame)))

        max_start = max(0, len(self.frame) - self.episode_length - 1)
        self.window_start = int(np.random.randint(0, max_start + 1)) if start_index is None else int(np.clip(start_index, 0, max_start))
        self.window_end = min(len(self.frame) - 1, self.window_start + self.episode_length - 1)
        self.current_step = self.window_start
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.trade_count = 0
        self.trade_wins = 0
        self.decision_matches = 0
        self.decision_count = 0
        self.reward_history = []
        self.portfolio_history = [self.portfolio_value]
        return self._get_state()

    def _current_row(self) -> pd.Series:
        return self.frame.iloc[self.current_step]

    def _get_state(self) -> np.ndarray:
        row = self._current_row()
        return build_state_from_row(
            row,
            shares_held=float(self.shares_held),
            balance=float(self.balance),
            portfolio_value=float(self.portfolio_value),
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        row = self._current_row()
        current_price = float(row["Close"])
        next_price = float(self.frame.iloc[min(self.current_step + 1, self.window_end)]["Close"])
        prev_portfolio_value = self.portfolio_value
        trade_fee = 0.0
        trade_penalty = 0.0

        if action == 1:
            budget = max(self.balance * self.buy_fraction, current_price)
            qty = min(
                self.max_shares - self.shares_held,
                int(budget / (current_price * (1 + self.transaction_fee_pct))),
            )
            if qty > 0:
                trade_fee = qty * current_price * self.transaction_fee_pct
                self.balance -= qty * current_price + trade_fee
                self.shares_held += qty
                self.trade_count += 1
                if next_price > current_price:
                    self.trade_wins += 1
            else:
                trade_penalty = 0.001
        elif action == 2:
            qty = min(self.shares_held, max(1, int(self.shares_held * self.sell_fraction)))
            if qty > 0:
                revenue = qty * current_price
                trade_fee = revenue * self.transaction_fee_pct
                self.balance += revenue - trade_fee
                self.shares_held -= qty
                self.trade_count += 1
                if next_price < current_price:
                    self.trade_wins += 1
            else:
                trade_penalty = 0.001

        self.portfolio_value = self.balance + self.shares_held * next_price
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)

        portfolio_return = (self.portfolio_value - prev_portfolio_value) / self.initial_balance
        drawdown = max(0.0, (self.peak_portfolio - self.portfolio_value) / self.initial_balance)
        momentum_bonus = 0.0
        if action == 1 and next_price >= current_price:
            momentum_bonus = 0.003
        elif action == 2 and next_price <= current_price:
            momentum_bonus = 0.003
        elif action == 0:
            momentum_bonus = 0.0005 if abs(next_price - current_price) < current_price * 0.002 else 0.0

        reward = portfolio_return - (drawdown * 0.2) - (trade_fee / self.initial_balance) - trade_penalty + momentum_bonus

        self.current_step = min(self.current_step + 1, self.window_end)
        done = self.current_step >= self.window_end

        self.reward_history.append(float(reward))
        self.portfolio_history.append(float(self.portfolio_value))

        oracle_action = 0
        if next_price > current_price * 1.002:
            oracle_action = 1
        elif next_price < current_price * 0.998:
            oracle_action = 2
        self.decision_count += 1
        if action == oracle_action:
            self.decision_matches += 1

        info = {
            "portfolio_value": float(self.portfolio_value),
            "balance": float(self.balance),
            "shares_held": int(self.shares_held),
            "step": int(self.current_step),
            "total_trades": int(self.trade_count),
            "trade_wins": int(self.trade_wins),
            "decision_matches": int(self.decision_matches),
            "decision_count": int(self.decision_count),
            "price": current_price,
            "next_price": next_price,
            "oracle_action": int(oracle_action),
        }
        return self._get_state(), float(reward), done, info

    def metrics(self) -> EpisodeMetrics:
        rewards = np.array(self.reward_history, dtype=np.float32)
        portfolio = np.array(self.portfolio_history, dtype=np.float32)
        running_max = np.maximum.accumulate(portfolio) if len(portfolio) else np.array([self.initial_balance], dtype=np.float32)
        drawdown = ((running_max - portfolio) / np.maximum(running_max, 1.0)).max() if len(portfolio) else 0.0
        win_rate = float((rewards > 0).mean()) if len(rewards) else 0.0
        decision_accuracy = float(self.decision_matches / self.decision_count) if self.decision_count else 0.0
        profitable_trade_rate = float(self.trade_wins / self.trade_count) if self.trade_count else 0.0
        return EpisodeMetrics(
            total_reward=float(rewards.sum()) if len(rewards) else 0.0,
            final_portfolio_value=float(self.portfolio_value),
            total_trades=int(self.trade_count),
            max_drawdown=float(drawdown),
            win_rate=win_rate,
            decision_accuracy=decision_accuracy,
            profitable_trade_rate=profitable_trade_rate,
        )
