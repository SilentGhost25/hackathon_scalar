from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DQN_CONFIG, ENV_CONFIG, VIS_CONFIG
from features import STATE_SIZE
from models.dqn_agent import DQNAgent
from pipeline import TradingEpisodeEnv, load_market_history

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


def run_episode(agent: DQNAgent, env: TradingEpisodeEnv, *, start_index: int) -> dict[str, float]:
    state = env.reset(start_index=start_index)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _info = env.step(action)
        total_reward += reward
        state = next_state

    metrics = env.metrics()
    return {
        "total_reward": float(total_reward),
        "final_portfolio_value": metrics.final_portfolio_value,
        "total_trades": metrics.total_trades,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "decision_accuracy": metrics.decision_accuracy,
        "profitable_trade_rate": metrics.profitable_trade_rate,
    }


def run_random_episode(env: TradingEpisodeEnv, *, start_index: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    state = env.reset(start_index=start_index)
    done = False
    total_reward = 0.0

    while not done:
        action = int(rng.integers(0, 3))
        next_state, reward, done, _info = env.step(action)
        total_reward += reward
        state = next_state

    metrics = env.metrics()
    return {
        "total_reward": float(total_reward),
        "final_portfolio_value": metrics.final_portfolio_value,
        "total_trades": metrics.total_trades,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "decision_accuracy": metrics.decision_accuracy,
        "profitable_trade_rate": metrics.profitable_trade_rate,
    }


def plot_episode(env: TradingEpisodeEnv, title: str, output_path: Path) -> None:
    if not HAS_MPL:
        return

    portfolio = np.array(env.portfolio_history, dtype=np.float32)
    rewards = np.array(env.reward_history, dtype=np.float32)
    steps = np.arange(len(portfolio))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, portfolio, color="teal", linewidth=1.4)
    ax.axhline(ENV_CONFIG["initial_balance"], color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Portfolio Value")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value ($)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps[1:], np.cumsum(rewards), color="purple", linewidth=1.4)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    price_window = env.frame.iloc[env.window_start : env.window_end + 1]["Close"].to_numpy()
    ax.plot(np.arange(len(price_window)), price_window, color="steelblue", linewidth=1.2)
    ax.set_title("Price Window")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price ($)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    drawdown = np.maximum.accumulate(portfolio) - portfolio
    ax.plot(steps, drawdown, color="crimson", linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_xlabel("Step")
    ax.set_ylabel("$")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate() -> None:
    print("=" * 72)
    print("  Stock Trading DQN - Evaluation")
    print("=" * 72)

    model_path = DQN_CONFIG["save_path"]
    if not os.path.exists(model_path):
        print(f"[ERROR] No saved model found at: {model_path}")
        print("Run `python train.py` first.")
        sys.exit(1)

    agent = DQNAgent(state_size=STATE_SIZE, action_size=ENV_CONFIG["action_size"])
    agent.load(model_path)
    print(f"Loaded model from: {model_path}")

    history = load_market_history("AAPL")
    split_index = max(60, int(len(history) * 0.8))
    test_frame = history.iloc[split_index:].copy()
    env = TradingEpisodeEnv(test_frame, episode_length=min(252, len(test_frame)))

    window_starts = [0]
    if len(test_frame) > env.episode_length:
        window_starts.append(max(0, (len(test_frame) - env.episode_length) // 2))
        window_starts.append(max(0, len(test_frame) - env.episode_length - 1))

    results = []
    random_results = []
    for idx, start in enumerate(window_starts, start=1):
        model_env = TradingEpisodeEnv(test_frame, episode_length=min(252, len(test_frame)))
        random_env = TradingEpisodeEnv(test_frame, episode_length=min(252, len(test_frame)))
        metrics = run_episode(agent, model_env, start_index=start)
        random_metrics = run_random_episode(random_env, start_index=start, seed=1000 + idx)
        results.append(metrics)
        random_results.append(random_metrics)
        print(
            f"Window {idx:>2d} | "
            f"Portfolio ${metrics['final_portfolio_value']:>10.2f} | "
            f"Reward {metrics['total_reward']:>9.4f} | "
            f"Trades {metrics['total_trades']:>3d} | "
            f"Win {metrics['win_rate']:.2%} | "
            f"Acc {metrics['decision_accuracy']:.2%} | "
            f"TradeWin {metrics['profitable_trade_rate']:.2%} | "
            f"DD {metrics['max_drawdown']:.2%}"
        )

        if idx == len(window_starts):
            plot_episode(model_env, f"DQN Evaluation Window {idx}", ROOT / "plots" / "evaluation_detail.png")

    avg_value = float(np.mean([row["final_portfolio_value"] for row in results]))
    avg_reward = float(np.mean([row["total_reward"] for row in results]))
    avg_dd = float(np.mean([row["max_drawdown"] for row in results]))
    avg_win = float(np.mean([row["win_rate"] for row in results]))
    avg_trades = float(np.mean([row["total_trades"] for row in results]))
    avg_accuracy = float(np.mean([row["decision_accuracy"] for row in results]))
    avg_trade_win = float(np.mean([row["profitable_trade_rate"] for row in results]))
    rand_value = float(np.mean([row["final_portfolio_value"] for row in random_results]))
    rand_reward = float(np.mean([row["total_reward"] for row in random_results]))
    rand_accuracy = float(np.mean([row["decision_accuracy"] for row in random_results]))
    rand_trade_win = float(np.mean([row["profitable_trade_rate"] for row in random_results]))

    buy_and_hold_value = ENV_CONFIG["initial_balance"]
    start_price = float(test_frame["Close"].iloc[0])
    end_price = float(test_frame["Close"].iloc[-1])
    shares = int(buy_and_hold_value / start_price)
    cash = buy_and_hold_value - shares * start_price
    buy_and_hold_value = cash + shares * end_price

    print("\n" + "-" * 72)
    print("  Aggregate")
    print("-" * 72)
    print(f"Average portfolio value : ${avg_value:,.2f}")
    print(f"Average reward           : {avg_reward:.4f}")
    print(f"Average max drawdown     : {avg_dd:.2%}")
    print(f"Average win rate         : {avg_win:.2%}")
    print(f"Average trades           : {avg_trades:.1f}")
    print(f"Decision accuracy        : {avg_accuracy:.2%}")
    print(f"Profitable trade rate    : {avg_trade_win:.2%}")
    print(f"Random policy value      : ${rand_value:,.2f}")
    print(f"Random policy reward     : {rand_reward:.4f}")
    print(f"Random decision accuracy  : {rand_accuracy:.2%}")
    print(f"Random trade win rate    : {rand_trade_win:.2%}")
    print(f"Buy-and-hold value       : ${buy_and_hold_value:,.2f}")


def _build_market_frame(prices: np.ndarray) -> pd.DataFrame:
    """Build a 4-column OHLC frame so the loaded checkpoint sees 7 features."""
    close = np.asarray(prices, dtype=np.float32)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        }
    )


if __name__ == "__main__":
    evaluate()
