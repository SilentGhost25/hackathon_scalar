from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "trading.db"
INITIAL_BALANCE = 100_000.0
DEFAULT_USER_ID = 1
DEFAULT_SYMBOL = "AAPL"

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    from stock_rl_project.models.dqn_agent import DQNAgent
except Exception:  # pragma: no cover - optional dependency
    DQNAgent = None

try:
    from stock_rl_project.features import build_live_state
except Exception:  # pragma: no cover - optional dependency
    build_live_state = None


app = FastAPI(title="StockRL Trading API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_lock = threading.Lock()
quote_lock = threading.Lock()
quote_cache: dict[str, dict[str, Any]] = {}
ai_agent: Optional[Any] = None


class TradeRequest(BaseModel):
    user_id: int = DEFAULT_USER_ID
    symbol: str = Field(..., min_length=1, max_length=16)
    quantity: int = Field(1, ge=1)
    price: Optional[float] = Field(default=None, gt=0)
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = Field(default=None, gt=0)


class RecommendationRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    market_data: dict[str, Any] = Field(default_factory=dict)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def dict_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                balance REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                last_updated TEXT NOT NULL,
                UNIQUE(user_id, symbol)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        ensure_column(conn, "transactions", "order_type", "TEXT NOT NULL DEFAULT 'MARKET'")
        ensure_column(conn, "transactions", "limit_price", "REAL")
        ensure_column(conn, "transactions", "execution_price", "REAL")
        ensure_column(conn, "transactions", "filled_at", "TEXT")

        user = conn.execute("SELECT id FROM users WHERE id = ?", (DEFAULT_USER_ID,)).fetchone()
        if user is None:
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, balance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_USER_ID,
                    "demo_trader",
                    "demo@stockrl.local",
                    "",
                    INITIAL_BALANCE,
                    utc_now(),
                ),
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def ensure_user(user_id: int) -> sqlite3.Row:
    with connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None:
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, balance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, f"user{user_id}", f"user{user_id}@stockrl.local", "", INITIAL_BALANCE, utc_now()),
            )
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return user


def normalize_order_type(order_type: str) -> str:
    return order_type.strip().lower()


def get_order_limit(payload: TradeRequest) -> Optional[float]:
    if payload.limit_price is not None:
        return payload.limit_price
    return payload.price


def load_quote(symbol: str, period: str = "3mo", interval: str = "1d") -> dict[str, Any]:
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    cache_key = f"{symbol}:{period}:{interval}"
    now = datetime.now(timezone.utc)

    with quote_lock:
        cached = quote_cache.get(cache_key)
        if cached:
            age = now - cached["fetched_at"]
            if age.total_seconds() < 30:
                return cached["payload"]

    if yf is None:
        raise HTTPException(status_code=503, detail="yfinance is not installed on the server")

    try:
        frame = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to fetch data for {symbol}: {exc}") from exc

    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")

    if hasattr(frame.columns, "levels"):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    closes = frame["Close"].dropna()
    if closes.empty:
        raise HTTPException(status_code=404, detail=f"No close prices available for {symbol}")

    volumes = frame["Volume"].fillna(0) if "Volume" in frame else None
    current_raw = closes.iloc[-1]
    previous_raw = closes.iloc[-2] if len(closes) > 1 else current_raw
    current = float(current_raw.item() if hasattr(current_raw, "item") else current_raw)
    previous = float(previous_raw.item() if hasattr(previous_raw, "item") else previous_raw)
    change = current - previous
    change_pct = (change / previous * 100) if previous else 0.0
    ma5 = float(closes.tail(5).mean())
    ma10 = float(closes.tail(10).mean())
    ma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else float(closes.mean())
    ma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
    trend = "Bullish" if current >= ma5 and ma5 >= ma10 else "Bearish" if current <= ma5 and ma5 <= ma10 else "Neutral"

    returns = closes.pct_change().dropna()
    momentum_3d = float((current / float(closes.iloc[-4])) - 1) * 100 if len(closes) >= 4 else change_pct
    momentum_5d = float((current / float(closes.iloc[-6])) - 1) * 100 if len(closes) >= 6 else change_pct
    momentum_20d = float((current / float(closes.iloc[-21])) - 1) * 100 if len(closes) >= 21 else change_pct
    volatility_20d = float(returns.tail(20).std() * math.sqrt(252) * 100) if len(returns) else 0.0

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if len(rsi_series) and pd.notna(rsi_series.iloc[-1]) else 50.0

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    avg_volume_20d = float(volumes.tail(20).mean()) if volumes is not None and len(volumes) else 0.0
    current_volume = float(volumes.iloc[-1]) if volumes is not None and len(volumes) else 0.0
    volume_ratio = (current_volume / avg_volume_20d) if avg_volume_20d else 0.0

    history = []
    close_values = closes.tail(60)
    for timestamp, close in close_values.items():
        vol = 0
        if volumes is not None and timestamp in volumes.index:
            try:
                vol = int(volumes.loc[timestamp])
            except Exception:
                vol = 0
        history.append(
            {
                "label": timestamp.strftime("%b %d"),
                "timestamp": timestamp.isoformat(),
                "close": float(close),
                "volume": vol,
            }
        )

    payload = {
        "symbol": symbol,
        "current_price": current,
        "previous_close": previous,
        "change": change,
        "change_pct": change_pct,
        "volume": int(volumes.iloc[-1]) if volumes is not None and len(volumes) else 0,
        "trend": trend,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma50": ma50,
        "momentum_3d": momentum_3d,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "volatility_20d": volatility_20d,
        "rsi": rsi,
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "volume_ratio": volume_ratio,
        "history": history,
        "updated_at": utc_now(),
    }

    with quote_lock:
        quote_cache[cache_key] = {"fetched_at": now, "payload": payload}

    return payload


def load_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        result[symbol] = load_quote(symbol)
    return result


def get_portfolio_rows(user_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT symbol, quantity, avg_price, last_updated
            FROM portfolio
            WHERE user_id = ?
            ORDER BY symbol ASC
            """,
            (user_id,),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        quote = load_quote(row["symbol"])
        quantity = int(row["quantity"])
        avg_price = float(row["avg_price"])
        current_price = float(quote["current_price"])
        market_value = quantity * current_price
        invested = quantity * avg_price
        pnl = market_value - invested
        items.append(
            {
                "symbol": row["symbol"],
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "invested": invested,
                "pnl": pnl,
                "last_updated": row["last_updated"],
            }
        )

    return items


def get_user_summary(user_id: int) -> dict[str, Any]:
    process_pending_orders(user_id)
    user = ensure_user(user_id)
    balance = float(user["balance"])
    portfolio = get_portfolio_rows(user_id)
    total_invested = sum(item["invested"] for item in portfolio)
    portfolio_value = balance + sum(item["market_value"] for item in portfolio)
    pnl = portfolio_value - INITIAL_BALANCE
    return {
        "user_id": user_id,
        "balance": balance,
        "total_invested": total_invested,
        "portfolio_value": portfolio_value,
        "pnl": pnl,
    }


def transaction_rows(user_id: int, range_key: str = "all") -> list[dict[str, Any]]:
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]

    if range_key != "all":
        days = 1 if range_key == "today" else 7 if range_key == "7d" else 30
        start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        clauses.append("timestamp >= ?")
        params.append(start)

    query = f"""
        SELECT id, user_id, symbol, type, quantity, price, total, status, timestamp, order_type, limit_price, execution_price, filled_at
        FROM transactions
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp DESC, id DESC
    """

    with connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def _apply_order_execution(
    user_id: int,
    symbol: str,
    side: str,
    quantity: int,
    execution_price: float,
) -> None:
    total = execution_price * quantity
    if side == "BUY":
        user = ensure_user(user_id)
        if float(user["balance"]) < total:
            raise HTTPException(status_code=400, detail="Insufficient wallet balance")
        update_balance(user_id, -total)
        upsert_portfolio(user_id, symbol, quantity, execution_price, "BUY")
    elif side == "SELL":
        upsert_portfolio(user_id, symbol, quantity, execution_price, "SELL")
        update_balance(user_id, total)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported side {side}")


def update_transaction_execution(
    transaction_id: int,
    *,
    status: str,
    execution_price: Optional[float] = None,
    total: Optional[float] = None,
    filled_at: Optional[str] = None,
) -> None:
    assignments = ["status = ?"]
    params: list[Any] = [status]

    if execution_price is not None:
        assignments.append("execution_price = ?")
        params.append(execution_price)
        assignments.append("price = ?")
        params.append(execution_price)

    if total is not None:
        assignments.append("total = ?")
        params.append(total)

    if filled_at is not None:
        assignments.append("filled_at = ?")
        params.append(filled_at)

    params.append(transaction_id)

    with connect() as conn:
        conn.execute(
            f"UPDATE transactions SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        conn.commit()


def process_pending_orders(user_id: int) -> None:
    with connect() as conn:
        pending_rows = conn.execute(
            """
            SELECT id, symbol, type, quantity, price, total, status, order_type, limit_price
            FROM transactions
            WHERE user_id = ? AND status = 'pending' AND order_type = 'LIMIT'
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

    for row in pending_rows:
        symbol = row["symbol"].upper()
        current_price = float(load_quote(symbol)["current_price"])
        limit_price = float(row["limit_price"] or row["price"])
        side = row["type"].upper()

        should_fill = (
            side == "BUY" and current_price <= limit_price
        ) or (
            side == "SELL" and current_price >= limit_price
        )

        if not should_fill:
            continue

        try:
            _apply_order_execution(user_id, symbol, side, int(row["quantity"]), current_price)
        except HTTPException:
            continue

        update_transaction_execution(
            int(row["id"]),
            status="executed",
            execution_price=current_price,
            total=current_price * int(row["quantity"]),
            filled_at=utc_now(),
        )


def upsert_portfolio(user_id: int, symbol: str, quantity_delta: int, price: float, action: str) -> None:
    symbol = symbol.upper()
    with db_lock, connect() as conn:
        row = conn.execute(
            "SELECT id, quantity, avg_price FROM portfolio WHERE user_id = ? AND symbol = ?",
            (user_id, symbol),
        ).fetchone()

        if action == "BUY":
            if row is None:
                conn.execute(
                    """
                    INSERT INTO portfolio (user_id, symbol, quantity, avg_price, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, symbol, quantity_delta, price, utc_now()),
                )
            else:
                quantity = int(row["quantity"]) + quantity_delta
                avg_price = ((float(row["quantity"]) * float(row["avg_price"])) + (quantity_delta * price)) / quantity
                conn.execute(
                    """
                    UPDATE portfolio
                    SET quantity = ?, avg_price = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (quantity, avg_price, utc_now(), row["id"]),
                )
        elif action == "SELL":
            if row is None or int(row["quantity"]) < quantity_delta:
                raise HTTPException(status_code=400, detail=f"Not enough shares of {symbol} to sell")

            quantity = int(row["quantity"]) - quantity_delta
            if quantity == 0:
                conn.execute("DELETE FROM portfolio WHERE id = ?", (row["id"],))
            else:
                conn.execute(
                    """
                    UPDATE portfolio
                    SET quantity = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (quantity, utc_now(), row["id"]),
                )

        conn.commit()


def update_balance(user_id: int, delta: float) -> None:
    with db_lock, connect() as conn:
        conn.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (delta, user_id))
        conn.commit()


def log_transaction(
    user_id: int,
    symbol: str,
    tx_type: str,
    quantity: int,
    price: float,
    total: float,
    status: str,
    *,
    order_type: str = "MARKET",
    limit_price: Optional[float] = None,
    execution_price: Optional[float] = None,
    filled_at: Optional[str] = None,
) -> dict[str, Any]:
    record = {
        "user_id": user_id,
        "symbol": symbol.upper(),
        "type": tx_type,
        "quantity": quantity,
        "price": price,
        "total": total,
        "status": status,
        "timestamp": utc_now(),
        "order_type": order_type.upper(),
        "limit_price": limit_price,
        "execution_price": execution_price,
        "filled_at": filled_at,
    }

    with db_lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO transactions (
                user_id, symbol, type, quantity, price, total, status, timestamp,
                order_type, limit_price, execution_price, filled_at
            )
            VALUES (
                :user_id, :symbol, :type, :quantity, :price, :total, :status, :timestamp,
                :order_type, :limit_price, :execution_price, :filled_at
            )
            """,
            record,
        )
        conn.commit()

    return record


def get_ai_agent() -> Optional[Any]:
    global ai_agent
    if ai_agent is not None:
        return ai_agent

    if DQNAgent is None or torch is None:
        return None

    model_path = ROOT / "stock_rl_project" / "saved_models" / "dqn_trading.pth"
    if not model_path.exists():
        return None

    agent = DQNAgent(state_size=12, action_size=3)
    try:
        agent.load(str(model_path))
    except Exception:
        return None

    ai_agent = agent
    return ai_agent


def build_ai_recommendation(symbol: str, summary: dict[str, Any], quote: dict[str, Any], portfolio: list[dict[str, Any]]) -> dict[str, Any]:
    symbol = symbol.upper()
    position = next((item for item in portfolio if item["symbol"] == symbol), None)
    shares_held = int(position["quantity"]) if position else 0
    current_price = float(quote["current_price"])
    previous_close = float(quote["previous_close"])
    balance = float(summary["balance"])
    portfolio_value = float(summary["portfolio_value"])
    ma5 = float(quote["ma5"])
    ma10 = float(quote["ma10"])
    ma20 = float(quote.get("ma20", ma10))
    ma50 = float(quote.get("ma50", ma20))
    rsi = float(quote.get("rsi", 50.0))
    momentum_3d = float(quote.get("momentum_3d", quote["change_pct"]))
    momentum_5d = float(quote.get("momentum_5d", quote["change_pct"]))
    momentum_20d = float(quote.get("momentum_20d", quote["change_pct"]))
    volatility_20d = float(quote.get("volatility_20d", 0.0))
    macd_hist = float(quote.get("macd_hist", 0.0))
    volume_ratio = float(quote.get("volume_ratio", 0.0))

    if build_live_state is not None:
        state = build_live_state(quote, summary, shares_held=shares_held)
    else:
        state = [
            current_price,
            previous_close,
            balance,
            shares_held,
            ma5,
            ma10,
            ma20,
            ma50,
            rsi / 100.0,
            macd_hist / max(current_price, 1.0),
            volume_ratio,
            balance / max(portfolio_value, 1.0),
        ]
    agent = get_ai_agent()

    dqn_action = None
    dqn_label = None
    dqn_confidence = 0.0
    if agent is not None:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = agent.policy_net(tensor)
            probabilities = torch.softmax(q_values[0], dim=0)
            dqn_action = int(torch.argmax(probabilities).item())
            dqn_confidence = float(probabilities[dqn_action].item())
    else:
        dqn_action = None
        dqn_confidence = 0.0

    short_trend = ((current_price - ma5) / ma5 * 100) if ma5 else 0.0
    long_trend = ((ma20 - ma50) / ma50 * 100) if ma50 else 0.0

    bullish = 50.0
    bearish = 50.0
    factors: list[dict[str, Any]] = []

    def add_factor(name: str, value: float, impact: float, direction: str) -> None:
        factors.append(
            {
                "name": name,
                "value": value,
                "impact": impact,
                "direction": direction,
            }
        )

    trend_score = max(min(short_trend * 3.5, 12.0), -12.0)
    momentum_score = max(min((momentum_3d * 0.8) + (momentum_5d * 0.9) + (momentum_20d * 0.4), 18.0), -18.0)
    macd_score = max(min(macd_hist * 220.0, 10.0), -10.0)
    rsi_score = 0.0
    if rsi < 35:
        rsi_score = (35 - rsi) * 0.6
    elif rsi > 70:
        rsi_score = -(rsi - 70) * 0.6
    volume_score = 0.0
    if volume_ratio >= 1.5:
        volume_score = 6.0 if momentum_5d >= 0 else -6.0
    elif volume_ratio > 0 and volume_ratio < 0.7:
        volume_score = -2.0

    regime_score = long_trend * 2.5
    volatility_penalty = min(volatility_20d / 4.0, 10.0)

    bullish += max(trend_score, 0) + max(momentum_score, 0) + max(macd_score, 0) + max(rsi_score, 0) + max(volume_score, 0) + max(regime_score, 0)
    bearish += max(-trend_score, 0) + max(-momentum_score, 0) + max(-macd_score, 0) + max(-rsi_score, 0) + max(-volume_score, 0) + max(-regime_score, 0) + volatility_penalty

    add_factor("Short trend", short_trend, trend_score, "bullish" if trend_score >= 0 else "bearish")
    add_factor("5D momentum", momentum_5d, momentum_score, "bullish" if momentum_score >= 0 else "bearish")
    add_factor("MACD histogram", macd_hist, macd_score, "bullish" if macd_score >= 0 else "bearish")
    add_factor("RSI", rsi, rsi_score, "bullish" if rsi_score >= 0 else "bearish")
    add_factor("Volume ratio", volume_ratio, volume_score, "bullish" if volume_score >= 0 else "bearish")
    add_factor("Regime", long_trend, regime_score, "bullish" if regime_score >= 0 else "bearish")
    add_factor("Volatility", volatility_20d, -volatility_penalty, "bearish")

    if dqn_action is not None:
        dqn_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        dqn_label = dqn_map[dqn_action]
        if dqn_label == "BUY":
            bullish += 6.0 * dqn_confidence
            add_factor("Model vote", dqn_confidence, 6.0 * dqn_confidence, "bullish")
        elif dqn_label == "SELL":
            bearish += 6.0 * dqn_confidence
            add_factor("Model vote", dqn_confidence, -6.0 * dqn_confidence, "bearish")
        else:
            add_factor("Model vote", dqn_confidence, 0.0, "neutral")

    edge = bullish - bearish
    edge_score = max(0, min(100, int(round(50 + edge))))
    if edge_score >= 60 and current_price >= ma20:
        action = "BUY"
    elif edge_score <= 40 and shares_held > 0:
        action = "SELL"
    else:
        action = "HOLD"

    if action == "BUY":
        reason = "Momentum, trend, and participation are aligned to the upside."
    elif action == "SELL":
        reason = "The signal stack is weakening and risk is higher than reward."
    else:
        reason = "The market is mixed. Waiting for a cleaner setup is the higher-quality trade."

    confidence = max(0.5, min(0.98, 0.58 + abs(edge) / 160))
    confidence = min(0.98, confidence + (0.08 if dqn_action is not None and ((action == "BUY" and dqn_action == 1) or (action == "SELL" and dqn_action == 2) or (action == "HOLD" and dqn_action == 0)) else -0.04 if dqn_action is not None else 0.0))

    if action == "BUY":
        target_price = current_price * (1 + max(0.015, min(0.08, volatility_20d / 120 + max(momentum_5d, 0) / 200)))
        stop_price = current_price * (1 - max(0.01, min(0.05, volatility_20d / 180 + 0.015)))
    elif action == "SELL":
        target_price = current_price * (1 - max(0.015, min(0.08, volatility_20d / 120 + max(-momentum_5d, 0) / 200)))
        stop_price = current_price * (1 + max(0.01, min(0.05, volatility_20d / 180 + 0.015)))
    else:
        target_price = current_price * (1 + 0.01)
        stop_price = current_price * (1 - 0.01)

    if stop_price != current_price:
        if action == "BUY":
            risk_reward = (target_price - current_price) / max(current_price - stop_price, 0.01)
        elif action == "SELL":
            risk_reward = (current_price - target_price) / max(stop_price - current_price, 0.01)
        else:
            risk_reward = 1.0
    else:
        risk_reward = 1.0

    return {
        "action": action,
        "confidence": confidence,
        "edge_score": edge_score,
        "regime": "Bullish breakout" if edge_score >= 60 else "Bearish pressure" if edge_score <= 40 else "Neutral / wait",
        "reason": reason,
        "target_price": target_price,
        "stop_price": stop_price,
        "risk_reward": round(risk_reward, 2),
        "dqn_action": dqn_label,
        "signal_stack": {
            "bullish": round(bullish, 2),
            "bearish": round(bearish, 2),
        },
        "factors": factors,
        "state": {
            "current_price": current_price,
            "previous_close": previous_close,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma50": ma50,
            "rsi": rsi,
            "momentum_5d": momentum_5d,
            "volatility_20d": volatility_20d,
            "shares_held": shares_held,
            "balance": balance,
            "portfolio_value": portfolio_value,
        },
    }


def build_scenario_analysis(
    symbol: str,
    move_pct: float,
    quantity: int,
    summary: dict[str, Any],
    quote: dict[str, Any],
    portfolio: list[dict[str, Any]],
) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    shocked_price = max(1.0, float(quote["current_price"]) * (1 + move_pct / 100.0))
    previous_close = float(quote["previous_close"])
    shocked_change = shocked_price - previous_close
    shocked_change_pct = (shocked_change / previous_close * 100) if previous_close else 0.0

    shocked_quote = dict(quote)
    shocked_quote.update(
        {
            "current_price": shocked_price,
            "change": shocked_change,
            "change_pct": shocked_change_pct,
            "trend": "Bullish" if shocked_price >= float(quote["ma5"]) else "Bearish" if shocked_price <= float(quote["ma5"]) else "Neutral",
            "momentum_3d": float(quote.get("momentum_3d", shocked_change_pct)),
            "momentum_5d": float(quote.get("momentum_5d", shocked_change_pct)),
            "momentum_20d": float(quote.get("momentum_20d", shocked_change_pct)),
        }
    )

    live_recommendation = build_ai_recommendation(symbol, summary, quote, portfolio)
    scenario_recommendation = build_ai_recommendation(symbol, summary, shocked_quote, portfolio)
    current_position = next((item for item in portfolio if item["symbol"] == symbol), None)
    held_qty = int(current_position["quantity"]) if current_position else 0
    avg_price = float(current_position["avg_price"]) if current_position else float(quote["current_price"])
    position_value_now = held_qty * float(quote["current_price"])
    position_value_shocked = held_qty * shocked_price
    position_pnl_now = held_qty * (float(quote["current_price"]) - avg_price)
    position_pnl_shocked = held_qty * (shocked_price - avg_price)
    account_value_now = float(summary["portfolio_value"])
    account_value_shocked = account_value_now + (held_qty * (shocked_price - float(quote["current_price"])))
    hypothetical_trade_value = quantity * shocked_price
    hypothetical_trade_pl = quantity * (shocked_price - float(quote["current_price"]))

    return {
        "symbol": symbol,
        "move_pct": move_pct,
        "quantity": quantity,
        "current_price": float(quote["current_price"]),
        "shocked_price": shocked_price,
        "price_delta": shocked_price - float(quote["current_price"]),
        "position": {
            "shares_held": held_qty,
            "avg_price": avg_price,
            "value_now": position_value_now,
            "value_scenario": position_value_shocked,
            "pnl_now": position_pnl_now,
            "pnl_scenario": position_pnl_shocked,
        },
        "account": {
            "portfolio_value_now": account_value_now,
            "portfolio_value_scenario": account_value_shocked,
            "delta": account_value_shocked - account_value_now,
        },
        "hypothetical_trade": {
            "quantity": quantity,
            "value_at_scenario_price": hypothetical_trade_value,
            "pnl_vs_now": hypothetical_trade_pl,
        },
        "signal_snapshot": scenario_recommendation,
        "live_action": live_recommendation["action"],
        "signal_flip": scenario_recommendation["action"] != live_recommendation["action"],
    }


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "timestamp": utc_now()}


@app.get("/api/quote/{symbol}")
def get_quote(symbol: str) -> dict[str, Any]:
    return load_quote(symbol)


@app.get("/api/balance")
def get_balance(user_id: int = DEFAULT_USER_ID) -> dict[str, Any]:
    summary = get_user_summary(user_id)
    return {"user_id": user_id, "balance": summary["balance"]}


@app.get("/api/portfolio")
def get_portfolio(user_id: int = DEFAULT_USER_ID) -> dict[str, Any]:
    process_pending_orders(user_id)
    summary = get_user_summary(user_id)
    return {"summary": summary, "portfolio": get_portfolio_rows(user_id)}


@app.get("/api/transactions")
def get_transactions(
    range: str = Query("all", pattern="^(today|7d|30d|all)$"),
    user_id: int = DEFAULT_USER_ID,
) -> dict[str, Any]:
    process_pending_orders(user_id)
    return {"transactions": transaction_rows(user_id, range)}


@app.get("/api/scenario")
def scenario_lab(
    symbol: str = Query(DEFAULT_SYMBOL, min_length=1, max_length=16),
    move_pct: float = Query(0.0, ge=-50.0, le=50.0),
    quantity: int = Query(10, ge=1, le=100000),
    user_id: int = DEFAULT_USER_ID,
) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    quote = load_quote(symbol)
    summary = get_user_summary(user_id)
    portfolio = get_portfolio_rows(user_id)
    return build_scenario_analysis(symbol, move_pct, quantity, summary, quote, portfolio)


@app.post("/api/trade/buy")
def buy_trade(payload: TradeRequest) -> dict[str, Any]:
    symbol = payload.symbol.upper().strip()
    quote = load_quote(symbol)
    order_type = normalize_order_type(payload.order_type)
    limit_price = get_order_limit(payload) if order_type == "limit" else None
    market_price = float(quote["current_price"])

    if order_type == "limit" and limit_price is None:
        raise HTTPException(status_code=400, detail="Limit price is required for limit orders")

    executable_price = market_price if order_type == "market" else market_price if market_price <= float(limit_price) else None

    if executable_price is None:
        record = log_transaction(
            payload.user_id,
            symbol,
            "BUY",
            payload.quantity,
            float(limit_price),
            0.0,
            "pending",
            order_type="LIMIT",
            limit_price=float(limit_price),
        )
        return {
            "message": f"BUY limit order queued for {symbol} at ${float(limit_price):.2f}",
            "transaction": record,
            "summary": get_user_summary(payload.user_id),
        }

    _apply_order_execution(payload.user_id, symbol, "BUY", payload.quantity, executable_price)
    record = log_transaction(
        payload.user_id,
        symbol,
        "BUY",
        payload.quantity,
        executable_price,
        executable_price * payload.quantity,
        "executed",
        order_type="MARKET" if order_type == "market" else "LIMIT",
        limit_price=float(limit_price) if limit_price is not None else None,
        execution_price=executable_price,
        filled_at=utc_now(),
    )

    return {
        "message": f"Bought {payload.quantity} share(s) of {symbol} at ${executable_price:.2f}",
        "transaction": record,
        "summary": get_user_summary(payload.user_id),
    }


@app.post("/api/trade/sell")
def sell_trade(payload: TradeRequest) -> dict[str, Any]:
    symbol = payload.symbol.upper().strip()
    quote = load_quote(symbol)
    order_type = normalize_order_type(payload.order_type)
    limit_price = get_order_limit(payload) if order_type == "limit" else None
    market_price = float(quote["current_price"])

    if order_type == "limit" and limit_price is None:
        raise HTTPException(status_code=400, detail="Limit price is required for limit orders")

    executable_price = market_price if order_type == "market" else market_price if market_price >= float(limit_price) else None

    if executable_price is None:
        record = log_transaction(
            payload.user_id,
            symbol,
            "SELL",
            payload.quantity,
            float(limit_price),
            0.0,
            "pending",
            order_type="LIMIT",
            limit_price=float(limit_price),
        )
        return {
            "message": f"SELL limit order queued for {symbol} at ${float(limit_price):.2f}",
            "transaction": record,
            "summary": get_user_summary(payload.user_id),
        }

    _apply_order_execution(payload.user_id, symbol, "SELL", payload.quantity, executable_price)
    record = log_transaction(
        payload.user_id,
        symbol,
        "SELL",
        payload.quantity,
        executable_price,
        executable_price * payload.quantity,
        "executed",
        order_type="MARKET" if order_type == "market" else "LIMIT",
        limit_price=float(limit_price) if limit_price is not None else None,
        execution_price=executable_price,
        filled_at=utc_now(),
    )

    return {
        "message": f"Sold {payload.quantity} share(s) of {symbol} at ${executable_price:.2f}",
        "transaction": record,
        "summary": get_user_summary(payload.user_id),
    }


@app.post("/api/trade/hold")
def hold_trade(payload: TradeRequest) -> dict[str, Any]:
    symbol = payload.symbol.upper().strip()
    quote = load_quote(symbol)
    record = log_transaction(
        payload.user_id,
        symbol,
        "HOLD",
        payload.quantity,
        float(payload.price or quote["current_price"]),
        0.0,
        "logged",
        order_type="MARKET",
    )

    return {
        "message": f"HOLD logged for {symbol}",
        "transaction": record,
        "summary": get_user_summary(payload.user_id),
    }


@app.post("/api/ai/recommendation")
def ai_recommendation(payload: RecommendationRequest) -> dict[str, Any]:
    symbol = payload.symbol.upper().strip()
    quote = load_quote(symbol)
    summary = get_user_summary(DEFAULT_USER_ID)
    portfolio = get_portfolio_rows(DEFAULT_USER_ID)
    return build_ai_recommendation(symbol, summary, quote, portfolio)


@app.get("/api/dashboard")
def dashboard(
    symbol: str = Query(DEFAULT_SYMBOL, min_length=1, max_length=16),
    range: str = Query("today", pattern="^(today|7d|30d|all)$"),
    user_id: int = DEFAULT_USER_ID,
) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    quote = load_quote(symbol)
    summary = get_user_summary(user_id)
    portfolio = get_portfolio_rows(user_id)
    transactions = transaction_rows(user_id, range)
    recommendation = build_ai_recommendation(symbol, summary, quote, portfolio)

    return {
        "user_id": user_id,
        "quote": quote,
        "account": summary,
        "portfolio": portfolio,
        "transactions": transactions,
        "recommendation": recommendation,
    }


@app.websocket("/api/ws/quotes")
async def ws_quotes(websocket: WebSocket) -> None:
    await websocket.accept()
    symbol = (websocket.query_params.get("symbol") or DEFAULT_SYMBOL).upper().strip()
    interval = float(websocket.query_params.get("interval") or 5)
    try:
        while True:
            try:
                payload = await asyncio.to_thread(load_quote, symbol)
                await websocket.send_json(payload)
            except HTTPException as exc:
                await websocket.send_json({"symbol": symbol, "error": exc.detail})
            await asyncio.sleep(max(2.0, interval))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
