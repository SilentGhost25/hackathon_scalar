from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

STATE_SIZE = 12
MAX_SHARES = 100.0


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df.columns, "levels"):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = flatten_columns(df).copy()
    frame.columns = frame.columns.astype(str)

    if "Close" not in frame.columns:
        raise ValueError("Expected a Close column in market data")

    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    close = pd.to_numeric(frame["Close"], errors="coerce").astype(float)
    volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0.0).astype(float)

    frame["Close"] = close
    frame["Volume"] = volume
    frame["PreviousClose"] = close.shift(1).fillna(close.iloc[0])
    frame["Return1D"] = close.pct_change().fillna(0.0)
    frame["Momentum5D"] = close.pct_change(5).fillna(0.0) * 100.0
    frame["Momentum20D"] = close.pct_change(20).fillna(0.0) * 100.0
    frame["Ma5"] = close.rolling(5).mean()
    frame["Ma10"] = close.rolling(10).mean()
    frame["Ma20"] = close.rolling(20).mean()
    frame["Ma50"] = close.rolling(50).mean()
    frame["Volatility20D"] = close.pct_change().rolling(20).std().fillna(0.0) * np.sqrt(252.0) * 100.0
    frame["Rsi14"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["MacdHist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    frame["AvgVolume20D"] = volume.rolling(20).mean()
    frame["VolumeRatio"] = volume / frame["AvgVolume20D"].replace(0, np.nan)

    frame = frame.dropna().reset_index(drop=True)
    return frame


def build_state_from_row(
    row: Mapping[str, Any],
    *,
    shares_held: float,
    balance: float,
    portfolio_value: float,
) -> np.ndarray:
    close = float(row["Close"])
    prev_close = float(row["PreviousClose"])
    ma5 = float(row["Ma5"])
    ma10 = float(row["Ma10"])
    ma20 = float(row["Ma20"])
    ma50 = float(row["Ma50"])
    rsi = float(row["Rsi14"]) if not pd.isna(row["Rsi14"]) else 50.0
    macd_hist = float(row["MacdHist"])
    volume_ratio = float(row["VolumeRatio"]) if not pd.isna(row["VolumeRatio"]) else 0.0
    momentum5 = float(row["Momentum5D"])
    volatility20 = float(row["Volatility20D"])

    state = np.array(
        [
            _clip(_safe_div(close, prev_close, 1.0), 0.5, 1.5),
            _clip(_safe_div(close, ma5, 1.0), 0.5, 1.5),
            _clip(_safe_div(close, ma10, 1.0), 0.5, 1.5),
            _clip(_safe_div(close, ma20, 1.0), 0.5, 1.7),
            _clip(_safe_div(close, ma50, 1.0), 0.5, 1.8),
            _clip(rsi / 100.0, 0.0, 1.0),
            _clip(macd_hist / max(close, 1.0), -0.25, 0.25),
            _clip(np.log1p(max(volume_ratio, 0.0)) / np.log1p(10.0), 0.0, 1.0),
            _clip(momentum5 / 100.0, -0.25, 0.25),
            _clip(volatility20 / 100.0, 0.0, 1.0),
            _clip(shares_held / MAX_SHARES, 0.0, 1.0),
            _clip(_safe_div(balance, max(portfolio_value, 1.0), 1.0), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return state


def build_live_state(
    quote: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    shares_held: float,
    max_shares: float = MAX_SHARES,
) -> np.ndarray:
    current_price = float(quote["current_price"])
    previous_close = float(quote["previous_close"])
    ma5 = float(quote["ma5"])
    ma10 = float(quote["ma10"])
    ma20 = float(quote.get("ma20", ma10))
    ma50 = float(quote.get("ma50", ma20))
    rsi = float(quote.get("rsi", 50.0))
    macd_hist = float(quote.get("macd_hist", 0.0))
    volume_ratio = float(quote.get("volume_ratio", 0.0))
    momentum_5d = float(quote.get("momentum_5d", quote.get("change_pct", 0.0)))
    volatility_20d = float(quote.get("volatility_20d", 0.0))
    balance = float(summary["balance"])
    portfolio_value = float(summary["portfolio_value"])

    return np.array(
        [
            _clip(_safe_div(current_price, previous_close, 1.0), 0.5, 1.5),
            _clip(_safe_div(current_price, ma5, 1.0), 0.5, 1.5),
            _clip(_safe_div(current_price, ma10, 1.0), 0.5, 1.5),
            _clip(_safe_div(current_price, ma20, 1.0), 0.5, 1.7),
            _clip(_safe_div(current_price, ma50, 1.0), 0.5, 1.8),
            _clip(rsi / 100.0, 0.0, 1.0),
            _clip(macd_hist / max(current_price, 1.0), -0.25, 0.25),
            _clip(np.log1p(max(volume_ratio, 0.0)) / np.log1p(10.0), 0.0, 1.0),
            _clip(momentum_5d / 100.0, -0.25, 0.25),
            _clip(volatility_20d / 100.0, 0.0, 1.0),
            _clip(shares_held / max(max_shares, 1.0), 0.0, 1.0),
            _clip(_safe_div(balance, max(portfolio_value, 1.0), 1.0), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
