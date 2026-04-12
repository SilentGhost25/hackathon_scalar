---
title: Stock Trading OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
  - finance
  - reinforcement-learning
---

# Stock Trading OpenEnv

This repository packages a stock-trading simulator as a real-world OpenEnv task suite. It models a human-relevant workflow: deciding when to buy, hold, or sell under return, drawdown, and overtrading constraints. The environment is deterministic per task seed, exposes typed Pydantic models, includes three graded tasks, and ships with both a Hugging Face Spaces-ready Docker setup and a baseline inference runner.

## Why this is a real-world task

Portfolio management is a real operational task performed by human traders and analysts. The agent is not playing a game; it is making sequential decisions under incomplete information, transaction costs, and risk limits, with explicit tradeoffs between upside and capital preservation.

## OpenEnv compliance

The OpenEnv implementation lives in `stock_rl_project/openenv_env.py`.

- Typed models:
  - `TradingObservation`
  - `TradingAction`
  - `TradingReward`
  - `TradingState`
  - `TaskScore`
- Required methods:
  - `reset() -> TradingObservation`
  - `step(action) -> (TradingObservation, TradingReward, done, info)`
  - `state() -> TradingState`
- Metadata:
  - `openenv.yaml`

## Action space

The agent chooses one of three actions:

- `hold`: do nothing
- `buy`: increase exposure
- `sell`: reduce exposure

These are represented by the typed model:

```python
class TradingAction(BaseModel):
    action: Literal["hold", "buy", "sell"]
    rationale: Optional[str] = None
```

## Observation space

Each step returns a typed observation with market state, portfolio state, task context, and available actions.

- `task_id`
- `difficulty`
- `objective`
- `step_index`
- `steps_remaining`
- `current_price`
- `previous_price`
- `ma5`
- `ma10`
- `cash_balance`
- `shares_held`
- `portfolio_value`
- `unrealized_pnl`
- `cumulative_reward`
- `last_action`
- `available_actions`

## Reward design

The reward is dense and shaped across the full trajectory instead of only at the terminal step.

- Positive progress signal from realized portfolio improvement
- Task alignment bonus when return is moving toward the task target
- Risk penalty when drawdown breaches the task limit
- Cost penalty for non-hold actions to discourage churning
- Existing simulator penalties for fees, slippage, repeated trading, and poor risk behavior

The typed `TradingReward` returns:

- `value`
- `progress`
- `risk_penalty`
- `cost_penalty`
- `task_alignment`
- `message`

## Tasks and graders

The environment includes three deterministic tasks with programmatic graders that score from `0.0` to `1.0`.

1. `easy_profit` (`easy`)
   Objective: finish with a positive return while avoiding wasteful overtrading.

2. `medium_outperform` (`medium`)
   Objective: beat buy-and-hold while keeping max drawdown below 9%.

3. `hard_risk_managed` (`hard`)
   Objective: achieve strong positive returns under a 6% drawdown cap with selective trades.

Each task is graded on weighted components:

- return score
- drawdown score
- trade discipline score
- baseline outperformance score

The final grader is deterministic and implemented in `grade_run()`.

## Baseline inference

The repo-root `inference.py` is the baseline runner.

- `--agent openai`: uses the OpenAI API client and reads `OPENAI_API_KEY`
- `--agent scripted`: deterministic offline baseline for local verification

Example commands:

```bash
python inference.py --agent scripted
```

```bash
set OPENAI_API_KEY=your_key_here
python inference.py --agent openai --model gpt-4.1-mini
```

### Reproducible local baseline scores

These are the current deterministic `--agent scripted` scores from this repo state:

| Task | Score |
|---|---:|
| `easy_profit` | `0.3813` |
| `medium_outperform` | `0.6310` |
| `hard_risk_managed` | `0.5523` |
| Aggregate | `0.5215` |

## Hugging Face Spaces deployment

This repo is configured for Docker Spaces.

- Docker entrypoint: `Dockerfile`
- Space app entrypoint: `app.py`
- Add the `openenv` tag in the Space metadata

Build locally:

```bash
docker build -t stock-trading-openenv .
docker run -p 5000:5000 stock-trading-openenv
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web app locally:

```bash
python app.py
```

Run the OpenEnv baseline:

```bash
python inference.py --agent scripted
```

## Repo structure

- `stock_rl_project/openenv_env.py`: OpenEnv wrapper, tasks, graders, typed models
- `stock_rl_project/env/stock_env.py`: core trading simulator
- `openenv.yaml`: environment metadata
- `inference.py`: baseline inference runner
- `Dockerfile`: containerized runtime
- `app.py`: Space/web server entrypoint

## Validation note

This repo now contains the pieces needed for `openenv validate`: typed models, `reset/step/state`, deterministic tasks, graders, and `openenv.yaml`. I was able to validate the Python runtime locally, but I did not run the external `openenv validate` CLI here because that tool is not installed in this workspace.
