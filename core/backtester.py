from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    both_flat = avg_gain.eq(0.0) & avg_loss.eq(0.0)
    only_gains = avg_gain.gt(0.0) & avg_loss.eq(0.0)
    only_losses = avg_gain.eq(0.0) & avg_loss.gt(0.0)
    rsi = rsi.mask(both_flat, 50.0)
    rsi = rsi.mask(only_gains, 100.0)
    rsi = rsi.mask(only_losses, 0.0)
    return rsi.clip(lower=0.0, upper=100.0)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Price data must contain a Close column")

    result = df.copy()
    close = result["Close"].astype(float)

    result["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    result["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    result["EMA_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["RSI_14"] = _compute_rsi(close, 14)

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema_fast - ema_slow
    result["MACD_signal"] = result["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()

    return result


class Strategy:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class MACrossover(Strategy):
    short: int = 20
    long: int = 50

    def __post_init__(self) -> None:
        if self.short <= 0 or self.long <= 0:
            raise ValueError("Moving-average windows must be positive")
        if self.short >= self.long:
            raise ValueError("short window must be smaller than long window")

    @property
    def name(self) -> str:
        return f"MACrossover(short={self.short}, long={self.long})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].astype(float)
        short_ma = df["SMA_20"] if self.short == 20 and "SMA_20" in df.columns else close.rolling(self.short, min_periods=self.short).mean()
        long_ma = df["SMA_50"] if self.long == 50 and "SMA_50" in df.columns else close.rolling(self.long, min_periods=self.long).mean()

        buy = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        sell = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        return pd.DataFrame({"buy": buy.fillna(False), "sell": sell.fillna(False)}, index=df.index)


@dataclass(frozen=True)
class RSIStrategy(Strategy):
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("RSI period must be positive")
        if not 0.0 <= self.oversold <= 100.0 or not 0.0 <= self.overbought <= 100.0:
            raise ValueError("RSI thresholds must be between 0 and 100")
        if self.oversold >= self.overbought:
            raise ValueError("RSI oversold must be smaller than overbought")

    @property
    def name(self) -> str:
        return (
            f"RSIStrategy(period={self.period}, "
            f"oversold={self.oversold}, overbought={self.overbought})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].astype(float)
        rsi = df["RSI_14"] if self.period == 14 and "RSI_14" in df.columns else _compute_rsi(close, self.period)

        buy = (rsi <= self.oversold) & (rsi.shift(1) > self.oversold)
        sell = (rsi >= self.overbought) & (rsi.shift(1) < self.overbought)
        return pd.DataFrame({"buy": buy.fillna(False), "sell": sell.fillna(False)}, index=df.index)


@dataclass(frozen=True)
class MACDStrategy(Strategy):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __post_init__(self) -> None:
        if min(self.fast, self.slow, self.signal) <= 0:
            raise ValueError("MACD periods must be positive")
        if self.fast >= self.slow:
            raise ValueError("MACD fast period must be smaller than slow period")

    @property
    def name(self) -> str:
        return f"MACDStrategy(fast={self.fast}, slow={self.slow}, signal={self.signal})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].astype(float)
        if self.fast == 12 and self.slow == 26 and "MACD" in df.columns:
            macd = df["MACD"]
        else:
            ema_fast = close.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
            macd = ema_fast - ema_slow

        if self.fast == 12 and self.slow == 26 and self.signal == 9 and "MACD_signal" in df.columns:
            signal_line = df["MACD_signal"]
        else:
            signal_line = macd.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()

        buy = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        sell = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        return pd.DataFrame({"buy": buy.fillna(False), "sell": sell.fillna(False)}, index=df.index)


class BacktestEngine:
    def _fetch_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")

        history = yf.Ticker(ticker).history(
            start=start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=True,
        )
        if history.empty:
            raise ValueError(f"No price history returned for {ticker}")

        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        result = history.copy()
        result.index = pd.to_datetime(result.index)
        if getattr(result.index, "tz", None) is not None:
            result.index = result.index.tz_localize(None)
        result = result.sort_index()
        result = result[~result.index.duplicated(keep="last")]
        result = result.dropna(subset=["Close"])
        if result.empty:
            raise ValueError(f"No usable closing prices returned for {ticker}")
        return result

    @staticmethod
    def _sample_equity_curve(equity_curve: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
        if len(equity_curve) <= 5:
            return equity_curve
        sampled = equity_curve[::5]
        if sampled[-1]["date"] != equity_curve[-1]["date"]:
            sampled.append(equity_curve[-1])
        return sampled

    def run(
        self,
        ticker: str,
        strategy: Strategy,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        history = self._fetch_history(ticker, start_date, end_date)
        df = calculate_indicators(history)
        signals = strategy.generate_signals(df).reindex(df.index).fillna(False)

        cash = float(initial_capital)
        shares = 0.0
        entry_price = 0.0
        entry_date: pd.Timestamp | None = None
        trades: list[dict[str, Any]] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []

        for date, row in df.iterrows():
            price = float(row["Close"])
            buy_signal = bool(signals.at[date, "buy"])
            sell_signal = bool(signals.at[date, "sell"])

            if shares > 0.0 and sell_signal:
                exit_value = shares * price
                pnl = exit_value - (shares * entry_price)
                pnl_pct = (price / entry_price - 1.0) * 100.0 if entry_price else 0.0
                trades.append(
                    {
                        "entry_date": _format_date(entry_date),
                        "exit_date": _format_date(date),
                        "entry_price": float(entry_price),
                        "exit_price": float(price),
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl_pct),
                        "side": "long",
                    }
                )
                cash = float(exit_value)
                shares = 0.0
                entry_price = 0.0
                entry_date = None

            if shares == 0.0 and buy_signal and price > 0.0:
                shares = cash / price
                cash = 0.0
                entry_price = price
                entry_date = pd.Timestamp(date)

            equity_points.append((pd.Timestamp(date), float(cash + (shares * price))))

        if shares > 0.0 and entry_date is not None:
            final_date = pd.Timestamp(df.index[-1])
            final_price = float(df["Close"].iloc[-1])
            exit_value = shares * final_price
            pnl = exit_value - (shares * entry_price)
            pnl_pct = (final_price / entry_price - 1.0) * 100.0 if entry_price else 0.0
            trades.append(
                {
                    "entry_date": _format_date(entry_date),
                    "exit_date": _format_date(final_date),
                    "entry_price": float(entry_price),
                    "exit_price": float(final_price),
                    "pnl": float(pnl),
                    "pnl_pct": float(pnl_pct),
                    "side": "long",
                }
            )
            cash = float(exit_value)
            shares = 0.0

        equity_series = pd.Series(
            [value for _, value in equity_points],
            index=pd.DatetimeIndex([date for date, _ in equity_points]),
            dtype=float,
        )

        final_value = float(cash if shares == 0.0 else equity_series.iloc[-1])
        total_return_pct = ((final_value / initial_capital) - 1.0) * 100.0

        periods = max(len(equity_series) - 1, 0)
        if periods > 0 and final_value > 0.0:
            annualized_return_pct = ((final_value / initial_capital) ** (252.0 / periods) - 1.0) * 100.0
        else:
            annualized_return_pct = 0.0

        daily_returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if not daily_returns.empty and daily_returns.std(ddof=0) > 0.0:
            sharpe_ratio = float((daily_returns.mean() / daily_returns.std(ddof=0)) * sqrt(252))
        else:
            sharpe_ratio = 0.0

        rolling_peak = equity_series.cummax()
        drawdowns = (equity_series / rolling_peak) - 1.0
        max_drawdown_pct = abs(float(drawdowns.min())) * 100.0 if not drawdowns.empty else 0.0

        gross_profit = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0.0)
        gross_loss = sum(trade["pnl"] for trade in trades if trade["pnl"] < 0.0)
        wins = sum(1 for trade in trades if trade["pnl"] > 0.0)
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100.0 if total_trades else 0.0
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0.0 else 0.0

        equity_curve = [
            {"date": _format_date(date), "value": float(value)}
            for date, value in equity_points
        ]

        return {
            "ticker": ticker.upper(),
            "strategy_name": strategy.name,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital),
            "final_value": final_value,
            "total_return_pct": float(total_return_pct),
            "annualized_return_pct": float(annualized_return_pct),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown_pct),
            "win_rate": float(win_rate),
            "total_trades": total_trades,
            "profit_factor": float(profit_factor),
            "equity_curve": self._sample_equity_curve(equity_curve),
            "trades": trades,
        }
