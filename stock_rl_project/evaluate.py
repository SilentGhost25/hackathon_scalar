"""
Evaluation Script for the Stock Trading DQN Agent
===================================================

Usage:
    python evaluate.py

Loads a trained DQN model, runs it on fresh synthetic data,
prints performance metrics, and generates detailed visualisation plots.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ENV_CONFIG, DQN_CONFIG, VIS_CONFIG, REAL_DATA_CONFIG
from data.price_generator import generate_stock_prices, get_real_stock_prices
from env.stock_env import StockTradingEnv
from models.dqn_agent import DQNAgent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — plots will be skipped.")


def evaluate(num_eval_episodes: int = 5) -> None:
    """
    Evaluate a trained DQN agent.

    Parameters
    ----------
    num_eval_episodes : int
        Number of evaluation episodes to run.
    """
    print("=" * 60)
    print("  Stock Trading DQN — Evaluation")
    print("=" * 60)

    # ---- Load agent ----
    model_path = DQN_CONFIG["save_path"]
    if not os.path.exists(model_path):
        print(f"[ERROR] No saved model found at: {model_path}")
        print("  Run train.py first to train a model.")
        sys.exit(1)

    agent = DQNAgent(
        state_size=ENV_CONFIG["state_size"],
        action_size=ENV_CONFIG["action_size"],
    )
    agent.load(model_path)
    print(f"Loaded model from: {model_path}\n")

    # ---- Run evaluation episodes ----
    all_metrics = []

    if REAL_DATA_CONFIG.get("use_real_data", False):
        print(f"Loading real data for {REAL_DATA_CONFIG['ticker']}...")
        base_prices = get_real_stock_prices(
            ticker=REAL_DATA_CONFIG["ticker"],
            period=REAL_DATA_CONFIG["period"],
            interval=REAL_DATA_CONFIG["interval"],
        )
        ENV_CONFIG["num_steps"] = len(base_prices)
    else:
        base_prices = None

    for ep in range(1, num_eval_episodes + 1):
        if base_prices is not None:
            prices = base_prices
        else:
            prices = generate_stock_prices(
                ENV_CONFIG["num_steps"],
                seed=1000 + ep,  # Unseen data
            )
        env = StockTradingEnv(prices=prices)
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        metrics = env.get_metrics()
        metrics["total_reward"] = total_reward
        all_metrics.append(metrics)

        profit = metrics["total_profit"]
        win_rate = metrics["win_rate"]
        max_dd = metrics["max_drawdown"]
        trades = metrics["total_trades"]

        print(
            f"  Episode {ep:>2d} | "
            f"Profit: ${profit:>8.2f} | "
            f"Return: {metrics['return_pct']:>6.2f}% | "
            f"Win: {win_rate:.2%} | "
            f"DD: {max_dd:.2%} | "
            f"Trades: {trades:>3d} | "
            f"Reward: {total_reward:>8.2f}"
        )

        # Generate detailed plot for last episode
        if ep == num_eval_episodes and HAS_MPL:
            _generate_eval_plots(env, ep)

    # ---- Aggregate metrics ----
    print("\n" + "-" * 60)
    print("  Aggregate Results")
    print("-" * 60)
    avg_profit = np.mean([m["total_profit"] for m in all_metrics])
    avg_return = np.mean([m["return_pct"] for m in all_metrics])
    avg_win = np.mean([m["win_rate"] for m in all_metrics])
    avg_dd = np.mean([m["max_drawdown"] for m in all_metrics])
    avg_trades = np.mean([m["total_trades"] for m in all_metrics])

    print(f"  Avg Profit     : ${avg_profit:>8.2f}")
    print(f"  Avg Return     : {avg_return:>6.2f}%")
    print(f"  Avg Win Rate   : {avg_win:.2%}")
    print(f"  Avg Max DD     : {avg_dd:.2%}")
    print(f"  Avg Trades     : {avg_trades:.1f}")

    # ---- Buy-and-hold baseline ----
    print("\n" + "-" * 60)
    print("  Buy-and-Hold Baseline")
    print("-" * 60)
    for ep in range(1, num_eval_episodes + 1):
        prices = generate_stock_prices(ENV_CONFIG["num_steps"], seed=1000 + ep)
        initial_shares = int(ENV_CONFIG["initial_balance"] / prices[0])
        remaining_cash = ENV_CONFIG["initial_balance"] - initial_shares * prices[0]
        bh_final = remaining_cash + initial_shares * prices[-1]
        bh_profit = bh_final - ENV_CONFIG["initial_balance"]
        bh_return = (bh_profit / ENV_CONFIG["initial_balance"]) * 100
        print(f"  Episode {ep:>2d} | B&H Profit: ${bh_profit:>8.2f} | Return: {bh_return:>6.2f}%")

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    print("=" * 60)


def _generate_eval_plots(env: StockTradingEnv, episode: int) -> None:
    """Generate detailed evaluation plots for one episode."""
    plot_dir = VIS_CONFIG["plot_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    hist = env.history
    steps = range(len(hist["price"]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"DQN Stock Trading — Evaluation Episode {episode}",
        fontsize=16, fontweight="bold",
    )

    # 1. Stock price with buy/sell markers
    ax = axes[0, 0]
    ax.plot(steps, hist["price"], color="steelblue", linewidth=1, label="Price")
    buys = [i for i, a in enumerate(hist["action"]) if a == 1]
    sells = [i for i, a in enumerate(hist["action"]) if a == 2]
    if buys:
        ax.scatter(buys, [hist["price"][i] for i in buys],
                   marker="^", color="green", s=30, zorder=5, label="BUY")
    if sells:
        ax.scatter(sells, [hist["price"][i] for i in sells],
                   marker="v", color="red", s=30, zorder=5, label="SELL")
    ax.set_title("Stock Price & Trade Actions")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Portfolio value
    ax = axes[0, 1]
    ax.plot(steps, hist["portfolio_value"], color="teal", linewidth=1.2)
    ax.axhline(ENV_CONFIG["initial_balance"], color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, label="Initial Balance")
    ax.fill_between(
        steps, hist["portfolio_value"], ENV_CONFIG["initial_balance"],
        alpha=0.1, color="teal",
    )
    ax.set_title("Portfolio Value")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Cumulative reward
    ax = axes[1, 0]
    cum_reward = np.cumsum(hist["reward"])
    ax.plot(steps, cum_reward, color="purple", linewidth=1.2)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)

    # 4. Profit over time
    ax = axes[1, 1]
    profit_series = np.array(hist["portfolio_value"]) - ENV_CONFIG["initial_balance"]
    ax.plot(steps, profit_series, color="green", linewidth=1.2)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(
        steps, profit_series, 0,
        where=(profit_series >= 0), alpha=0.15, color="green", label="Profit",
    )
    ax.fill_between(
        steps, profit_series, 0,
        where=(profit_series < 0), alpha=0.15, color="red", label="Loss",
    )
    ax.set_title("Profit Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Profit ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "evaluation_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Evaluation detail saved to {path}")


if __name__ == "__main__":
    evaluate()
