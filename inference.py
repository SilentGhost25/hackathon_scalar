from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from stock_rl_project.openenv_env import (
    TASKS,
    StockTradingOpenEnv,
    TaskScore,
    TradingAction,
    summarize_all_tasks,
)


SYSTEM_PROMPT = """You are an agent operating a stock-trading simulation.
You must reply with strict JSON using this schema:
{"action":"hold"|"buy"|"sell","rationale":"short explanation"}
Prioritize the task objective, avoid overtrading, and manage drawdown."""


def scripted_action(observation: Dict[str, Any]) -> TradingAction:
    price = float(observation["current_price"])
    prev_price = float(observation["previous_price"])
    ma5 = float(observation["ma5"])
    ma10 = float(observation["ma10"])
    shares = int(observation["shares_held"])
    pnl = float(observation["unrealized_pnl"])
    step_index = int(observation["step_index"])

    bullish = price > ma5 > ma10 and price >= prev_price
    bearish = price < ma5 or price < prev_price or ma5 < ma10

    if shares == 0 and bullish and step_index % 12 == 0:
        return TradingAction(action="buy", rationale="Momentum is positive and short averages are supportive.")
    if shares > 0 and (pnl > 300 or (bearish and step_index % 12 == 0)):
        return TradingAction(action="sell", rationale="Protect gains or cut weakening exposure.")
    return TradingAction(action="hold", rationale="No clear edge right now.")


class OpenAIActionPolicy:
    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def act(self, observation: Dict[str, Any]) -> TradingAction:
        user_prompt = (
            "Task observation:\n"
            f"{json.dumps(observation, indent=2, sort_keys=True)}\n\n"
            "Choose exactly one next action."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return TradingAction.model_validate(payload)
        except Exception:
            return TradingAction(action="hold", rationale="Fallback hold after malformed or failed model response.")


def run_task(task_id: str, agent: str, model: str) -> Dict[str, Any]:
    env = StockTradingOpenEnv(task_id=task_id)
    observation = env.reset().model_dump()
    policy = OpenAIActionPolicy(model=model) if agent == "openai" else None

    while True:
        action = policy.act(observation) if policy is not None else scripted_action(observation)
        next_observation, reward, done, info = env.step(action)
        observation = next_observation.model_dump()
        if done:
            result = env.grade_run()
            return {
                "task_id": task_id,
                "difficulty": TASKS[task_id].difficulty,
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
                "final_reward": reward.model_dump(),
                "final_info": info,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline inference for the Stock Trading OpenEnv tasks.")
    parser.add_argument("--agent", choices=["openai", "scripted"], default="scripted")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    if args.agent == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when --agent=openai")

    results: List[Dict[str, Any]] = []
    for task_id in TASKS:
        results.append(run_task(task_id=task_id, agent=args.agent, model=args.model))

    summary = summarize_all_tasks(
        [TaskScore(task_id=result["task_id"], score=result["score"], details=result["details"], passed=result["passed"]) for result in results]
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
