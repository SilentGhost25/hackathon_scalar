"""
DQN training script for the stock trading environment.

This script trains the existing `DQNAgent` against the corrected
`StockTradingEnv` using synthetic price series, then saves the
resulting checkpoint to `saved_models/dqn_trading.pth`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ENV_CONFIG, DQN_CONFIG, VIS_CONFIG
from data.price_generator import generate_stock_prices
from env.stock_env import StockTradingEnv
from models.dqn_agent import DQNAgent


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    profit: float = 0.0
    trades: int = 0
    steps: int = 0


def build_market_frame(prices: np.ndarray):
    import pandas as pd

    close = np.asarray(prices, dtype=np.float32)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def train(num_episodes: int = 120, seed_offset: int = 2000) -> str:
    print("=" * 60)
    print("  Stock Trading DQN - Training")
    print("=" * 60)

    agent = DQNAgent(
        state_size=ENV_CONFIG["state_size"],
        action_size=ENV_CONFIG["action_size"],
    )

    best_profit = -float("inf")
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DQN_CONFIG["save_path"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), VIS_CONFIG["plot_dir"]), exist_ok=True)

    reward_history = []
    profit_history = []

    for episode in range(1, num_episodes + 1):
        prices = generate_stock_prices(ENV_CONFIG["num_steps"], seed=seed_offset + episode)
        env = StockTradingEnv(df=build_market_frame(prices))
        state, _ = env.reset(seed=seed_offset + episode)

        stats = EpisodeStats()

        while True:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, _info = env.step(action)

            agent.store_transition(state, action, reward, next_state, terminated or truncated)
            loss = agent.learn()
            if loss is not None and agent.train_step_count % 100 == 0:
                print(f"    step={agent.train_step_count:>6d} loss={loss:>8.4f} eps={agent.epsilon:.3f}")

            state = next_state
            stats.total_reward += reward
            stats.steps += 1

            if terminated or truncated:
                break

        agent.decay_epsilon()
        if episode % DQN_CONFIG["target_update_freq"] == 0:
            agent.update_target_network()

        metrics = env.get_metrics()
        stats.profit = metrics["total_profit"]
        stats.trades = metrics["total_trades"]

        reward_history.append(stats.total_reward)
        profit_history.append(stats.profit)

        if stats.profit > best_profit:
            best_profit = stats.profit
            agent.save(save_path)

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:>3d}/{num_episodes} | "
                f"Reward {stats.total_reward:>8.2f} | "
                f"Profit {stats.profit:>8.2f} | "
                f"Trades {stats.trades:>3d} | "
                f"Eps {agent.epsilon:.3f}"
            )

    agent.save(save_path)
    print(f"\nSaved model to: {save_path}")
    return save_path


if __name__ == "__main__":
    train()
