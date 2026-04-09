from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from .config import ENV_CONFIG
    from .data.price_generator import generate_stock_prices
    from .env.stock_env import StockTradingEnv
except ImportError:
    from config import ENV_CONFIG
    from data.price_generator import generate_stock_prices
    from env.stock_env import StockTradingEnv


class TradingObservation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    step_index: int
    steps_remaining: int
    current_price: float
    previous_price: float
    ma5: float
    ma10: float
    cash_balance: float
    shares_held: int
    portfolio_value: float
    unrealized_pnl: float
    cumulative_reward: float
    last_action: Literal["hold", "buy", "sell"]
    available_actions: List[str]


class TradingAction(BaseModel):
    action: Literal["hold", "buy", "sell"]
    rationale: Optional[str] = None


class TradingReward(BaseModel):
    value: float
    progress: float
    risk_penalty: float
    cost_penalty: float
    task_alignment: float
    message: str


class TradingState(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    step_index: int
    done: bool
    cumulative_reward: float
    total_profit: float
    return_pct: float
    max_drawdown: float
    total_trades: int
    baseline_return_pct: float
    last_reward: TradingReward
    history: Dict[str, List[float | int]]


class TaskScore(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, float]
    passed: bool


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    objective: str
    seed: int
    target_return_pct: float
    max_drawdown_limit: float
    min_trades: int
    max_trades: int
    beat_baseline_by_pct: float = 0.0


TASKS: Dict[str, TaskConfig] = {
    "easy_profit": TaskConfig(
        task_id="easy_profit",
        difficulty="easy",
        title="Preserve capital and end green",
        objective="Finish the episode with a positive return while avoiding obviously wasteful overtrading.",
        seed=1101,
        target_return_pct=1.0,
        max_drawdown_limit=0.12,
        min_trades=1,
        max_trades=24,
    ),
    "medium_outperform": TaskConfig(
        task_id="medium_outperform",
        difficulty="medium",
        title="Beat buy-and-hold with controlled risk",
        objective="Outperform the buy-and-hold baseline and keep maximum drawdown below 9%.",
        seed=2202,
        target_return_pct=3.0,
        max_drawdown_limit=0.09,
        min_trades=2,
        max_trades=22,
        beat_baseline_by_pct=1.0,
    ),
    "hard_risk_managed": TaskConfig(
        task_id="hard_risk_managed",
        difficulty="hard",
        title="Deliver disciplined returns",
        objective="Achieve strong positive returns, stay under 6% drawdown, and trade selectively.",
        seed=3303,
        target_return_pct=5.0,
        max_drawdown_limit=0.06,
        min_trades=3,
        max_trades=14,
        beat_baseline_by_pct=2.0,
    ),
}


ACTION_MAP = {"hold": 0, "buy": 1, "sell": 2}
ACTION_NAMES = {0: "hold", 1: "buy", 2: "sell"}


def build_market_frame(prices: np.ndarray) -> pd.DataFrame:
    close = np.asarray(prices, dtype=np.float32)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def compute_baseline_return(prices: np.ndarray, initial_balance: float) -> float:
    initial_price = float(prices[0])
    shares = int(initial_balance // initial_price)
    cash = initial_balance - (shares * initial_price)
    final_value = cash + (shares * float(prices[-1]))
    return ((final_value - initial_balance) / initial_balance) * 100.0


class StockTradingOpenEnv:
    def __init__(self, task_id: str = "easy_profit"):
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.initial_balance = float(ENV_CONFIG["initial_balance"])
        self.env: Optional[StockTradingEnv] = None
        self.market_df: Optional[pd.DataFrame] = None
        self.prices: Optional[np.ndarray] = None
        self.done = False
        self.cumulative_reward = 0.0
        self.last_reward = TradingReward(
            value=0.0,
            progress=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
            task_alignment=0.0,
            message="Episode not started.",
        )

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "title": task.title,
                "objective": task.objective,
            }
            for task in TASKS.values()
        ]

    def set_task(self, task_id: str) -> None:
        if task_id not in TASKS:
            raise KeyError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.task = TASKS[task_id]

    def reset(self, task_id: Optional[str] = None) -> TradingObservation:
        if task_id is not None:
            self.set_task(task_id)

        prices = generate_stock_prices(ENV_CONFIG["num_steps"], seed=self.task.seed)
        self.prices = np.asarray(prices, dtype=np.float32)
        self.market_df = build_market_frame(self.prices)
        self.env = StockTradingEnv(df=self.market_df, initial_balance=self.initial_balance)
        self.env.reset(seed=self.task.seed)
        self.done = False
        self.cumulative_reward = 0.0
        self.last_reward = TradingReward(
            value=0.0,
            progress=0.0,
            risk_penalty=0.0,
            cost_penalty=0.0,
            task_alignment=0.0,
            message="Environment reset.",
        )
        return self._build_observation()

    def step(
        self,
        action: TradingAction | Dict[str, Any],
    ) -> Tuple[TradingObservation, TradingReward, bool, Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new run.")

        action_model = action if isinstance(action, TradingAction) else TradingAction.model_validate(action)
        discrete_action = ACTION_MAP[action_model.action]
        _, env_reward, terminated, truncated, info = self.env.step(discrete_action)
        self.done = bool(terminated or truncated)

        reward = self._build_reward(float(env_reward), action_model.action)
        self.last_reward = reward
        self.cumulative_reward += reward.value

        response_info: Dict[str, Any] = {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "objective": self.task.objective,
            "metrics": self.env.get_metrics(),
            "env_info": info,
        }
        if self.done:
            response_info["grader"] = self.grade_run().model_dump()

        return self._build_observation(), reward, self.done, response_info

    def state(self) -> TradingState:
        if self.env is None:
            raise RuntimeError("Call reset() before state().")
        metrics = self.env.get_metrics()
        baseline_return = compute_baseline_return(self.prices, self.initial_balance) if self.prices is not None else 0.0
        history = {
            key: [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in values]
            for key, values in self.env.history.items()
        }
        return TradingState(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            step_index=self.env.current_step,
            done=self.done,
            cumulative_reward=self.cumulative_reward,
            total_profit=float(metrics["total_profit"]),
            return_pct=float(metrics["return_pct"]),
            max_drawdown=float(metrics["max_drawdown"]),
            total_trades=int(metrics["total_trades"]),
            baseline_return_pct=float(baseline_return),
            last_reward=self.last_reward,
            history=history,
        )

    def grade_run(self) -> TaskScore:
        state = self.state()
        task = self.task

        return_score = np.clip(state.return_pct / max(task.target_return_pct, 0.01), 0.0, 1.0)
        drawdown_score = np.clip(1.0 - max(0.0, state.max_drawdown - task.max_drawdown_limit) / max(task.max_drawdown_limit, 1e-6), 0.0, 1.0)
        if task.min_trades <= state.total_trades <= task.max_trades:
            trade_score = 1.0
        else:
            distance = min(abs(state.total_trades - task.min_trades), abs(state.total_trades - task.max_trades))
            trade_score = np.clip(1.0 - (distance / max(task.max_trades, 1)), 0.0, 1.0)
        baseline_margin = state.return_pct - state.baseline_return_pct
        baseline_score = np.clip((baseline_margin + max(task.beat_baseline_by_pct, 0.0)) / max(task.beat_baseline_by_pct + 2.0, 2.0), 0.0, 1.0)

        score = float(np.clip((0.45 * return_score) + (0.25 * drawdown_score) + (0.15 * trade_score) + (0.15 * baseline_score), 0.0, 1.0))
        passed = (
            state.return_pct >= task.target_return_pct
            and state.max_drawdown <= task.max_drawdown_limit
            and task.min_trades <= state.total_trades <= task.max_trades
            and baseline_margin >= task.beat_baseline_by_pct
        )
        return TaskScore(
            task_id=task.task_id,
            score=score,
            passed=passed,
            details={
                "return_score": float(return_score),
                "drawdown_score": float(drawdown_score),
                "trade_score": float(trade_score),
                "baseline_score": float(baseline_score),
                "return_pct": float(state.return_pct),
                "max_drawdown": float(state.max_drawdown),
                "total_trades": float(state.total_trades),
                "baseline_return_pct": float(state.baseline_return_pct),
            },
        )

    def _build_observation(self) -> TradingObservation:
        if self.env is None or self.market_df is None:
            raise RuntimeError("Call reset() before building observations.")

        step_index = min(self.env.current_step, len(self.market_df) - 1)
        close_prices = self.market_df["Close"].astype(float).iloc[: step_index + 1]
        ma5 = float(close_prices.tail(5).mean())
        ma10 = float(close_prices.tail(10).mean())
        current_price = float(self.market_df.iloc[step_index]["Close"])
        previous_price = float(self.market_df.iloc[max(step_index - 1, 0)]["Close"])

        return TradingObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            step_index=step_index,
            steps_remaining=max(len(self.market_df) - step_index - 1, 0),
            current_price=current_price,
            previous_price=previous_price,
            ma5=ma5,
            ma10=ma10,
            cash_balance=float(self.env.balance),
            shares_held=int(self.env.shares_held),
            portfolio_value=float(self.env.portfolio_value),
            unrealized_pnl=float(self.env.portfolio_value - self.initial_balance),
            cumulative_reward=float(self.cumulative_reward),
            last_action=ACTION_NAMES.get(int(self.env.last_action), "hold"),
            available_actions=["hold", "buy", "sell"],
        )

    def _build_reward(self, env_reward: float, action_name: str) -> TradingReward:
        metrics = self.env.get_metrics() if self.env is not None else {"return_pct": 0.0, "max_drawdown": 0.0}
        progress = float(np.clip(metrics["return_pct"] / max(self.task.target_return_pct, 1.0), -1.0, 1.0))
        risk_penalty = float(max(0.0, metrics["max_drawdown"] - self.task.max_drawdown_limit))
        cost_penalty = 0.02 if action_name != "hold" else 0.0
        task_alignment = float(np.clip(progress - risk_penalty - cost_penalty, -1.0, 1.0))
        total = float(env_reward + (0.1 * progress) - (0.2 * risk_penalty) - cost_penalty)
        return TradingReward(
            value=total,
            progress=progress,
            risk_penalty=risk_penalty,
            cost_penalty=cost_penalty,
            task_alignment=task_alignment,
            message=f"Action={action_name} progress={progress:.3f} drawdown_penalty={risk_penalty:.3f}",
        )


def summarize_all_tasks(scores: List[TaskScore]) -> Dict[str, Any]:
    aggregate = float(np.mean([score.score for score in scores])) if scores else 0.0
    return {
        "aggregate_score": aggregate,
        "tasks": [score.model_dump() for score in scores],
    }


def format_observation_for_agent(observation: TradingObservation) -> str:
    return json.dumps(observation.model_dump(), indent=2, sort_keys=True)
