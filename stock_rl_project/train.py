from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DQN_CONFIG, ENV_CONFIG
from features import STATE_SIZE
from models.dqn_agent import DQNAgent
from pipeline import TradingEpisodeEnv, load_market_history


def run_episode(agent: DQNAgent, env: TradingEpisodeEnv, *, evaluate: bool = False, start_index: int | None = None):
    state = env.reset(start_index=start_index)
    done = False
    total_reward = 0.0
    last_loss = None

    while not done:
        action = agent.select_action(state, evaluate=evaluate)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if not evaluate:
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

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
        "loss": last_loss,
    }


def make_episode_starts(frame_length: int, episode_length: int, episodes: int) -> list[int]:
    max_start = max(0, frame_length - episode_length - 1)
    if max_start == 0:
        return [0] * episodes
    rng = np.random.default_rng(42)
    return [int(rng.integers(0, max_start + 1)) for _ in range(episodes)]


def main() -> None:
    print("=" * 72)
    print("  Stock Trading DQN - Retraining with normalized multi-factor features")
    print("=" * 72)

    history = load_market_history("AAPL")
    split_index = max(60, int(len(history) * 0.8))
    train_frame = history.iloc[:split_index].copy()
    test_frame = history.iloc[split_index:].copy()

    train_env = TradingEpisodeEnv(train_frame, episode_length=min(252, len(train_frame)))
    test_env = TradingEpisodeEnv(test_frame, episode_length=min(252, len(test_frame)))

    agent = DQNAgent(state_size=STATE_SIZE, action_size=ENV_CONFIG["action_size"])

    episodes = int(DQN_CONFIG["num_episodes"])
    target_update_freq = int(DQN_CONFIG["target_update_freq"])
    best_validation_value = -float("inf")
    episode_starts = make_episode_starts(len(train_frame), train_env.episode_length, episodes)

    history_rows: list[dict[str, float]] = []

    print(f"Train rows: {len(train_frame)} | Test rows: {len(test_frame)} | State size: {STATE_SIZE}")

    for episode in range(1, episodes + 1):
        start_index = episode_starts[episode - 1]
        metrics = run_episode(agent, train_env, evaluate=False, start_index=start_index)
        agent.decay_epsilon()

        validation = None
        if episode % max(5, target_update_freq) == 0 or episode == episodes:
            validation = run_episode(agent, test_env, evaluate=True, start_index=0)
            if validation["final_portfolio_value"] > best_validation_value:
                best_validation_value = validation["final_portfolio_value"]
                agent.save()

        if episode % max(1, DQN_CONFIG.get("log_interval", 10)) == 0 or episode == 1:
            validation_value = validation["final_portfolio_value"] if validation else float("nan")
            print(
                f"Episode {episode:>3d} | "
                f"Reward {metrics['total_reward']:>9.4f} | "
                f"Portfolio ${metrics['final_portfolio_value']:>10.2f} | "
                f"Trades {metrics['total_trades']:>3d} | "
                f"Win {metrics['win_rate']:.2%} | "
                f"Acc {metrics['decision_accuracy']:.2%} | "
                f"TradeWin {metrics['profitable_trade_rate']:.2%} | "
                f"Val ${validation_value:>10.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

        history_rows.append(
            {
                "episode": episode,
                "reward": metrics["total_reward"],
                "portfolio_value": metrics["final_portfolio_value"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "decision_accuracy": metrics["decision_accuracy"],
                "profitable_trade_rate": metrics["profitable_trade_rate"],
                "epsilon": agent.epsilon,
                "validation_value": validation["final_portfolio_value"] if validation else float("nan"),
            }
        )

        if episode % target_update_freq == 0:
            agent.update_target_network()

    save_path = agent.save()
    print(f"\nSaved model to: {save_path}")

    os.makedirs(ROOT / "plots", exist_ok=True)
    summary_path = ROOT / "plots" / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(history_rows, handle, indent=2)
    print(f"Saved training summary to: {summary_path}")

    final_eval = run_episode(agent, test_env, evaluate=True, start_index=0)
    print("\n" + "=" * 72)
    print("  Final validation")
    print("=" * 72)
    print(f"Final portfolio value: ${final_eval['final_portfolio_value']:,.2f}")
    print(f"Total reward         : {final_eval['total_reward']:.4f}")
    print(f"Total trades         : {final_eval['total_trades']}")
    print(f"Max drawdown         : {final_eval['max_drawdown']:.2%}")
    print(f"Win rate             : {final_eval['win_rate']:.2%}")
    print(f"Decision accuracy    : {final_eval['decision_accuracy']:.2%}")
    print(f"Profitable trade rate: {final_eval['profitable_trade_rate']:.2%}")


if __name__ == "__main__":
    main()
