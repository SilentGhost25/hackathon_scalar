from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
from yfinance.cache import set_cache_location, set_tz_cache_location

from config import DQN_CONFIG, ENV_CONFIG
from data.price_generator import generate_stock_prices
from env.stock_env import StockTradingEnv
from models.dqn_agent import DQNAgent


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "frontend")
MODEL_PATH = os.path.join(PROJECT_ROOT, DQN_CONFIG["save_path"])
CACHE_ROOT = os.path.join(PROJECT_ROOT, ".yfinance_cache")

os.makedirs(CACHE_ROOT, exist_ok=True)
set_cache_location(CACHE_ROOT)
set_tz_cache_location(CACHE_ROOT)

app = Flask(
    __name__,
    static_folder=FRONTEND_ROOT,
    static_url_path="",
)

_model_lock = threading.Lock()
_cached_agent: DQNAgent | None = None


def build_market_frame(prices: np.ndarray) -> pd.DataFrame:
    close = np.asarray(prices, dtype=np.float32)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def fetch_market_frame(
    ticker: str = "AAPL",
    period: str = "5d",
    interval: str = "15m",
) -> tuple[pd.DataFrame, str]:
    """Fetch live market bars, falling back to synthetic data if unavailable."""
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is None or data.empty:
            raise ValueError("empty download")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        cols = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]
        if len(cols) < 4:
            raise ValueError("missing OHLC columns")
        frame = data[["Open", "High", "Low", "Close"]].dropna().copy()
        frame.columns = ["Open", "High", "Low", "Close"]
        frame.reset_index(drop=True, inplace=True)
        if len(frame) < 10:
            raise ValueError("not enough live rows")
        return frame, "live"
    except Exception:
        prices = generate_stock_prices(ENV_CONFIG["num_steps"], seed=1001)
        return build_market_frame(prices), "synthetic"


def compute_moving_averages(prices: List[float]) -> Dict[str, List[float]]:
    series = pd.Series(prices, dtype="float64")
    return {
        "ma5": series.rolling(5, min_periods=1).mean().astype(float).tolist(),
        "ma10": series.rolling(10, min_periods=1).mean().astype(float).tolist(),
    }


def load_agent() -> DQNAgent:
    global _cached_agent
    with _model_lock:
        if _cached_agent is None:
            agent = DQNAgent(
                state_size=ENV_CONFIG["state_size"],
                action_size=ENV_CONFIG["action_size"],
            )
            if os.path.exists(MODEL_PATH):
                agent.load(MODEL_PATH)
            _cached_agent = agent
    return _cached_agent


def run_episode(seed: int = 1001) -> Dict[str, Any]:
    agent = load_agent()
    market_df, source = fetch_market_frame()
    prices = market_df["Close"].astype(float).to_numpy()
    env = StockTradingEnv(df=market_df)
    state, _ = env.reset(seed=seed)

    actions: List[int] = []
    portfolio: List[float] = []
    rewards: List[float] = []
    while True:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, _info = env.step(action)
        actions.append(int(action))
        portfolio.append(float(env.portfolio_value))
        rewards.append(float(reward))
        state = next_state
        if terminated or truncated:
            break

    metrics = env.get_metrics()
    ma = compute_moving_averages(prices.tolist())
    history = {
        "price": env.history.get("price", []),
        "ma5": ma["ma5"],
        "ma10": ma["ma10"],
        "portfolioValue": env.history.get("portfolio_value", [ENV_CONFIG["initial_balance"]]),
        "balance": env.history.get("balance", [ENV_CONFIG["initial_balance"]]),
        "sharesHeld": env.history.get("shares_held", [0]),
        "action": env.history.get("action", []),
        "reward": env.history.get("reward", []),
    }
    return {
        "seed": seed,
        "source": source,
        "ticker": "AAPL",
        "period": "5d",
        "interval": "15m",
        "timestamps": list(range(len(prices))),
        "prices": prices.tolist(),
        "ma5": ma["ma5"],
        "ma10": ma["ma10"],
        "actions": actions,
        "portfolio": portfolio,
        "rewards": rewards,
        "metrics": metrics,
        "history": history,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def build_openenv_discovery() -> Dict[str, Any]:
    agent = load_agent()
    return {
        "name": "openenv",
        "path": "/openenv",
        "methods": ["GET", "POST"],
        "description": "Single entrypoint for discovering and running the stock trading OpenEnv service.",
        "operations": {
            "GET /openenv": "Returns endpoint metadata, supported operations, and default configuration.",
            "POST /openenv": "Executes JSON operations such as summary, live-data, or evaluate.",
        },
        "post_examples": [
            {"operation": "summary"},
            {"operation": "live-data", "seed": 1001},
            {"operation": "evaluate", "episodes": 3, "seed": 1001},
        ],
        "defaults": {
            "ticker": "AAPL",
            "period": "5d",
            "interval": "15m",
            "episodes": 3,
            "seed": 1001,
        },
        "model": {
            "state_size": ENV_CONFIG["state_size"],
            "action_size": ENV_CONFIG["action_size"],
            "epsilon": agent.epsilon,
            "model_path": MODEL_PATH,
            "model_loaded": os.path.exists(MODEL_PATH),
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def build_summary_payload() -> Dict[str, Any]:
    agent = load_agent()
    return {
        "state_size": ENV_CONFIG["state_size"],
        "action_size": ENV_CONFIG["action_size"],
        "model_path": MODEL_PATH,
        "epsilon": agent.epsilon,
        "trade_cooldown": int(getattr(agent, "cfg", {}).get("min_trade_advantage", 0) > 0),
        "default_ticker": "AAPL",
        "default_period": "5d",
        "default_interval": "15m",
    }


def build_evaluation_payload(episodes: int, base_seed: int) -> Dict[str, Any]:
    runs = [run_episode(seed=base_seed + i) for i in range(episodes)]

    profits = [run["metrics"]["total_profit"] for run in runs]
    returns = [run["metrics"]["return_pct"] for run in runs]
    dir_accs = []
    for run in runs:
        actions = run["actions"]
        prices = run["prices"]
        correct = 0
        attempts = 0
        for idx, action in enumerate(actions):
            if action not in (1, 2):
                continue
            attempts += 1
            if idx + 1 >= len(prices):
                continue
            if (action == 1 and prices[idx + 1] >= prices[idx]) or (action == 2 and prices[idx + 1] <= prices[idx]):
                correct += 1
        dir_accs.append(correct / attempts if attempts else None)

    latest = runs[-1]
    return {
        "episodes": episodes,
        "aggregate": {
            "avg_profit": float(np.mean(profits)),
            "avg_return_pct": float(np.mean(returns)),
            "avg_directional_accuracy": float(np.mean([x for x in dir_accs if x is not None])) if any(x is not None for x in dir_accs) else None,
            "avg_trades": float(np.mean([run["metrics"]["total_trades"] for run in runs])),
        },
        "latest": latest,
    }


@app.get("/")
def index():
    return send_from_directory(FRONTEND_ROOT, "index.html")


@app.get("/<path:filename>")
def frontend_asset(filename: str):
    if filename.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    if filename in {"index.html", "style.css", "app.js"}:
        return send_from_directory(FRONTEND_ROOT, filename)
    return jsonify({"error": "not found"}), 404


@app.get("/api/health")
def health():
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify(
        {
            "ok": True,
            "model_loaded": model_exists,
            "model_path": MODEL_PATH,
        }
    )


@app.get("/api/summary")
def summary():
    return jsonify(build_summary_payload())


@app.get("/api/live-data")
def live_data():
    seed = int(request.args.get("seed", "1001"))
    return jsonify(run_episode(seed=seed))


@app.post("/api/evaluate")
def evaluate():
    payload = request.get_json(silent=True) or {}
    episodes = int(payload.get("episodes", 3))
    base_seed = int(payload.get("seed", 1001))
    return jsonify(build_evaluation_payload(episodes=episodes, base_seed=base_seed))


@app.route("/openenv", methods=["GET", "POST"])
def openenv_endpoint():
    if request.method == "GET":
        return jsonify(build_openenv_discovery())

    payload = request.get_json(silent=True) or {}
    operation = str(payload.get("operation", "summary")).strip().lower()

    if operation == "summary":
        return jsonify({"operation": "summary", **build_summary_payload()})

    if operation in {"live-data", "live_data", "episode", "run"}:
        seed = int(payload.get("seed", 1001))
        return jsonify({"operation": "live-data", "result": run_episode(seed=seed)})

    if operation == "evaluate":
        episodes = int(payload.get("episodes", 3))
        base_seed = int(payload.get("seed", 1001))
        return jsonify({"operation": "evaluate", **build_evaluation_payload(episodes=episodes, base_seed=base_seed)})

    return (
        jsonify(
            {
                "error": "unsupported operation",
                "supported_operations": ["summary", "live-data", "evaluate"],
            }
        ),
        400,
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5000)
