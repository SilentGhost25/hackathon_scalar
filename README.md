# StockRL Trading Dashboard

This repo now has two pieces:

- `frontend/` - the live trading UI
- `backend/` - FastAPI + SQLite API for quotes, trades, portfolio, and transactions

## Run the backend

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## Open the frontend

Open `frontend/index.html` in a browser, or serve the folder with any static file server.

The UI expects the API at `http://127.0.0.1:8000/api`.

## What it does

- Fetches live quotes with `yfinance`
- Persists wallet balance, portfolio, and transactions in SQLite
- Supports `BUY`, `SELL`, and `HOLD`
- Shows account totals and transaction history
- Displays an explainable AI trade blueprint with edge score, target, stop, and signal factors
- Uses a multi-factor market signal engine so recommendations are less fragile than a single moving-average rule
- Presents a cleaner brokerage-style UI with watchlist chips, fast order entry, and portfolio snapshots
- Includes a scenario lab that lets you test upside/downside price shocks before placing a trade
- Retrains the saved DQN with a normalized 12-feature state, Double DQN updates, and a cleaner evaluation loop
- Reports decision accuracy, profitable trade rate, random-policy baseline, and buy-and-hold comparison
