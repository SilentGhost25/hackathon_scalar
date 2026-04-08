# Stock Trading Reinforcement Learning Environment

> **Meta × Scaler OpenEnv Hackathon — Round 1 Submission**

A production-quality reinforcement learning environment built with **PyTorch** and **OpenEnv** that simulates a realistic stock market trading system. A DQN agent learns to **buy, hold, and sell** stocks to maximize profit while minimizing risk.

This model also powers the main app's explainable signal engine and scenario lab, so the hackathon demo can show both live recommendations and what-if trade outcomes.

The latest retrain uses a normalized 12-feature state, Double DQN targets, and a cleaned-up training / evaluation pipeline.

Evaluation now also reports decision accuracy, profitable trade rate, a random-policy baseline, and buy-and-hold comparison.

---

## 🏗️ Project Structure

```
stock_rl_project/
├── config.py                # Centralised hyperparameters & settings
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation & visualisation
├── README.md
├── env/
│   ├── __init__.py
│   └── stock_env.py         # OpenEnv-compatible StockTradingEnv
├── models/
│   ├── __init__.py
│   └── dqn_agent.py         # DQN agent (PyTorch)
├── data/
│   ├── __init__.py
│   └── price_generator.py   # Synthetic stock price generation
├── saved_models/            # (created at runtime)
└── plots/                   # (created at runtime)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy matplotlib
```

> No GPU required — runs entirely on CPU.

### 2. Train the Agent

```bash
cd stock_rl_project
python train.py
```

This will:
- Train a DQN agent for 300 episodes
- Log metrics every 10 episodes
- Save the best model to `saved_models/dqn_trading.pth`
- Generate training plots in `plots/training_summary.png`

### 3. Evaluate

```bash
python evaluate.py
```

This will:
- Load the trained model
- Run on fresh (unseen) synthetic data
- Print performance metrics
- Compare against a buy-and-hold baseline
- Compare against a random-policy baseline
- Report decision accuracy and profitable trade rate
- Generate evaluation plots in `plots/evaluation_detail.png`

---

## 🎯 Environment Design

### State Space (7 features)

| Feature | Description |
|---------|-------------|
| `current_price` | Current stock price |
| `prev_price` | Previous step's price |
| `balance` | Available cash ($) |
| `shares_held` | Number of shares owned |
| `ma_5` | 5-step moving average |
| `ma_10` | 10-step moving average |
| `portfolio_value` | Total portfolio value |

### Action Space

| Action | Code | Description |
|--------|------|-------------|
| HOLD | 0 | Do nothing |
| BUY | 1 | Purchase 1 share |
| SELL | 2 | Sell 1 share |

### Reward Function

The reward is shaped to encourage long-term profitability:

- **Primary signal**: Change in portfolio value
- **Large-loss penalty**: Extra penalty for losses exceeding $100
- **Overtrading penalty**: Penalises consecutive trading
- **Risk concentration penalty**: Penalises large position sizes
- **Consistency bonus**: Rewards consecutive profitable steps
- **Risk-reduction bonus**: Rewards selling when overexposed
- **Transaction cost**: Deducted from reward

### Episode Termination

An episode ends when:
1. All time steps are exhausted (500 steps)
2. The balance drops below $1,000 (bankruptcy)
3. A 10% stop-loss is triggered from peak portfolio value

---

## 🧠 RL Model (DQN)

| Component | Details |
|-----------|---------|
| Network | Feed-forward: Input → 64 → ReLU → 64 → ReLU → Output |
| Loss | Huber (Smooth L1) |
| Optimiser | Adam (lr = 1e-3) |
| Replay Buffer | 50,000 transitions |
| Target Network | Hard sync every 10 episodes |
| Exploration | ε-greedy with decay (1.0 → 0.01) |
| Gradient Clipping | max_norm = 1.0 |

---

## 📊 Synthetic Data

Prices are generated using:
- **Geometric Brownian motion** (random walk with drift)
- **Mean-reversion** toward initial price
- **Sinusoidal trend** overlay
- **Stochastic volatility** noise

No external APIs or datasets are needed.

---

## 📈 Performance Metrics

The system tracks:
- **Total Profit** ($)
- **Return** (%)
- **Win Rate** (fraction of positive return steps)
- **Maximum Drawdown** (peak-to-trough decline)
- **Total Trades**
- Comparison against **buy-and-hold baseline**

---

## ⚡ Bonus Features Implemented

- ✅ Transaction fees (0.1% per trade)
- ✅ Position limits (max 100 shares)
- ✅ Stop-loss logic (10% drawdown trigger)
- ✅ Risk management via reward shaping
- ✅ Buy-and-hold baseline comparison

---

## 📝 Configuration

All hyperparameters are centralised in `config.py`:

```python
ENV_CONFIG     # Environment settings
PRICE_CONFIG   # Price generation parameters
DQN_CONFIG     # Agent hyperparameters
REWARD_CONFIG  # Reward shaping weights
VIS_CONFIG     # Visualisation settings
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch (CPU)
- NumPy
- Matplotlib

---

## 📜 License

This project is submitted as part of the Meta × Scaler OpenEnv Hackathon.
