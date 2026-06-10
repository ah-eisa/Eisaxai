import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_sma(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculates MACD, Signal line, and Histogram."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return {
        "macd": macd,
        "signal": signal_line,
        "histogram": histogram
    }

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculates Maximum Drawdown from peak."""
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculates Value at Risk (VaR) using historical method."""
    if returns.empty:
        return 0.0
    return np.percentile(returns.dropna(), (1 - confidence) * 100)

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculates Beta relative to market benchmark."""
    # Align dates
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.empty or len(df) < 30:
        return 1.0 # Default to 1 if insufficient data
    
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    if var == 0:
        return 1.0
    return cov / var
def generate_technical_summary(ticker: str, data: Any) -> Dict[str, Any]:
    """
    Generates technical summary handling both Series and DataFrames.
    """
    import pandas as pd
    import pandas_ta as ta
    
    # 1. تحويل البيانات لـ DataFrame لو كانت Series
    if isinstance(data, pd.Series):
        df = data.to_frame(name='Close')
        # تخليق أعمدة وهمية للـ High/Low لو مش موجودين (عشان الـ ADX ميفشلش)
        df['High'] = df['Close']
        df['Low'] = df['Close']
    else:
        df = data.copy()

    # 2. تنظيف أسماء الأعمدة (حل مشكلة الـ Multi-Index أو الحروف الكبيرة/الصغيرة)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c).capitalize() for c in df.columns]

    # التأكد من وجود عمود السعر
    if 'Close' not in df.columns:
        # لو ملقاش Close، ياخد أول عمود متاح
        df['Close'] = df.iloc[:, 0]

    close = df['Close']
    current_price = close.iloc[-1]

    # 3. الحسابات
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    
    # RSI using pandas_ta
    rsi_series = ta.rsi(close, length=14)
    rsi = 50.0
    if rsi_series is not None and not rsi_series.empty:
        try: rsi = float(rsi_series.iloc[-1] or 50.0)
        except Exception: pass
    
    # MACD using pandas_ta
    macd_result = ta.macd(close, fast=12, slow=26, signal=9)
    macd_val = 0.0
    signal_val = 0.0
    if macd_result is not None and not macd_result.empty:
        try: macd_val = float(macd_result.iloc[-1, 0] or 0)
        except Exception: pass
        try: signal_val = float(macd_result.iloc[-1, 2] or 0) if len(macd_result.columns) > 2 else 0.0
        except Exception: pass

    # حساب ADX و ATR بأمان
    adx = 0.0
    atr = 0.0
    if 'High' in df.columns and 'Low' in df.columns:
        import pandas as _pd
        try:
            adx_series = ta.adx(df['High'], df['Low'], close, length=14)
            if adx_series is not None and not adx_series.empty:
                # ta.adx() returns a DataFrame (ADX_14, DMP_14, DMN_14) — extract first column
                adx_col = "ADX_14"
                if isinstance(adx_series, _pd.DataFrame):
                    col = adx_col if adx_col in adx_series.columns else adx_series.columns[0]
                    adx_raw = adx_series[col].iloc[-1]
                else:
                    adx_raw = adx_series.iloc[-1]
                adx = float(adx_raw) if not _pd.isna(adx_raw) else 0.0
        except:
            pass
        try:
            atr_series = ta.atr(df['High'], df['Low'], close, length=14)
            if atr_series is not None and not atr_series.empty:
                atr_raw = atr_series.iloc[-1] if isinstance(atr_series, _pd.Series) else atr_series.iloc[-1, 0]
                atr = float(atr_raw) if not _pd.isna(atr_raw) else 0.0
        except:
            pass

    # Ensure all values are clean floats — NaN / inf become 0.0 (not nan)
    import math as _math
    def _safe_float(v, default=0.0):
        try:
            f = float(v)
            return default if (_math.isnan(f) or _math.isinf(f)) else f
        except Exception:
            return default

    rsi      = _safe_float(rsi, 50.0)
    sma_50   = _safe_float(sma_50, 0.0)
    sma_200  = _safe_float(sma_200, 0.0)
    macd_val = _safe_float(macd_val, 0.0)
    signal_val = _safe_float(signal_val, 0.0)

    trend = "Bullish" if current_price > sma_200 else "Bearish"
    momentum = "Bullish" if macd_val > signal_val else "Bearish"
    
    if rsi > 70:
        condition = "Overbought"
    elif rsi >= 60:
        condition = "Near Overbought"
    elif rsi <= 30:
        condition = "Oversold"
    elif rsi <= 40:
        condition = "Near Oversold"
    else:
        condition = "Neutral"

    return {
        "ticker": ticker,
        "price": current_price,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": signal_val,
        "adx": adx,
        "atr": atr,
        "trend": trend,
        "momentum": momentum,
        "condition": condition,
    }

def calculate_monte_carlo(prices: pd.Series, days: int = 252, simulations: int = 1000) -> pd.DataFrame:
    """
    Generates Monte Carlo simulation paths for a given asset.
    Returns DataFrame where each column is a simulation path.
    """
    if prices.empty:
        return pd.DataFrame()

    last_price = prices.iloc[-1]
    returns = np.log(1 + prices.pct_change().dropna())
    
    mu = returns.mean()
    var = returns.var()
    drift = mu - (0.5 * var)
    sigma = returns.std()
    
    # Generate random shocks
    daily_shocks = np.random.normal(0, 1, (days, simulations))
    
    # Calculate daily returns for all paths
    # drift and sigma are scalars, daily_shocks is (days, sims)
    daily_returns = np.exp(drift + sigma * daily_shocks)
    
    # Accumulate price paths
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = last_price
    
    for t in range(1, days + 1):
        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
        
    return pd.DataFrame(price_paths)

def get_simulation_stats(paths: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts P10 (Worst), P50 (Expected), P90 (Best) from simulation paths.
    """
    if paths.empty:
        return {}
        
    final_prices = paths.iloc[-1]
    return {
        "p10": np.percentile(final_prices, 10),
        "p50": np.percentile(final_prices, 50),
        "p90": np.percentile(final_prices, 90),
        "mean": final_prices.mean(),
        "min": final_prices.min(),
        "max": final_prices.max()
    }
def calculate_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put") -> Dict[str, float]:
    """
    Calculates Black-Scholes price and Greeks (Delta, Theta).
    S: Spot, K: Strike, T: Time (years), r: Risk-free rate, sigma: Volatility
    """
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        
    return {
        "price": price,
        "delta": delta,
        "theta": theta
    }

# ── NEW: CVaR ─────────────────────────────────────────────────────────────────
def calculate_cvar(returns, confidence=0.95):
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()

def calculate_sharpe_ratio(returns, rf=0.03):
    if returns.empty or returns.std() == 0: return 0.0
    return (returns.mean() * 252 - rf) / (returns.std() * np.sqrt(252))

def calculate_sortino_ratio(returns, rf=0.03):
    downside = returns[returns < 0].std() * np.sqrt(252)
    if downside == 0: return 0.0
    return (returns.mean() * 252 - rf) / downside

def forecast_garch_volatility(returns, horizon=30):
    try:
        from arch import arch_model
        scaled = returns.dropna() * 100
        model = arch_model(scaled, vol='Garch', p=1, q=1, dist='normal')
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=horizon)
        vol = np.sqrt(forecast.variance.iloc[-1].values) / 100
        return {
            "current_vol_annualized": float(returns.std() * np.sqrt(252)),
            "forecast_vol_30d": float(vol.mean() * np.sqrt(252)),
            "vol_trend": "Rising" if vol[-1] > vol[0] else "Falling",
        }
    except Exception as e:
        return {"error": str(e)}

def forecast_arima(prices, steps=30):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(prices.dropna(), order=(2, 1, 2)).fit()
        forecast = model.forecast(steps=steps)
        current = prices.iloc[-1]
        predicted = forecast.iloc[-1]
        change_pct = ((predicted - current) / current) * 100
        return {
            "current_price": float(current),
            "forecast_30d": float(predicted),
            "change_pct": float(change_pct),
            "direction": "UP ↑" if change_pct > 0 else "DOWN ↓",
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_advanced_indicators(prices):
    try:
        import ta
        import pandas as pd
        df = prices.to_frame(name="close")
        df["high"] = df["close"] * 1.005
        df["low"] = df["close"] * 0.995
        df["volume"] = 1_000_000
        return {
            "adx": ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx().iloc[-1],
            "cci": ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci().iloc[-1],
            "bb_upper": ta.volatility.BollingerBands(df["close"]).bollinger_hband().iloc[-1],
            "bb_lower": ta.volatility.BollingerBands(df["close"]).bollinger_lband().iloc[-1],
            "atr": ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range().iloc[-1],
            "stoch_k": ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch().iloc[-1],
        }
    except Exception as e:
        return {"error": str(e)}

# ── STRESS TEST ENGINE ────────────────────────────────────────────────────────
HISTORICAL_CRISES = {
    "2008_financial_crisis": {
        "name": "2008 Global Financial Crisis",
        "market_drop": -0.565,
        "duration_days": 365,
        "volatility_multiplier": 3.5,
        "description": "Lehman collapse, credit crunch"
    },
    "covid_2020": {
        "name": "COVID-19 Crash (2020)",
        "market_drop": -0.34,
        "duration_days": 33,
        "volatility_multiplier": 4.0,
        "description": "Fastest 30% crash in history"
    },
    "dotcom_2000": {
        "name": "Dot-com Bust (2000-2002)",
        "market_drop": -0.49,
        "duration_days": 750,
        "volatility_multiplier": 2.5,
        "description": "Tech bubble collapse"
    },
    "rate_shock_2022": {
        "name": "Fed Rate Shock (2022)",
        "market_drop": -0.25,
        "duration_days": 280,
        "volatility_multiplier": 2.0,
        "description": "Fastest rate hike cycle in 40 years"
    },
    "black_monday_1987": {
        "name": "Black Monday (1987)",
        "market_drop": -0.22,
        "duration_days": 1,
        "volatility_multiplier": 8.0,
        "description": "22% single day crash"
    }
}

def run_stress_test(prices: pd.Series, beta: float = 1.0) -> dict:
    """
    Run portfolio through historical crisis scenarios
    Returns impact for each crisis based on beta-adjusted market move
    """
    import numpy as np
    returns = prices.pct_change().dropna()
    current_price = prices.iloc[-1]
    vol = returns.std() * np.sqrt(252)

    results = {}
    for crisis_id, crisis in HISTORICAL_CRISES.items():
        # Beta-adjusted portfolio drop (capped at -85%)
        portfolio_drop = crisis["market_drop"] * min(beta, 2.0)
        vol_adjustment = (vol - 0.3) * 0.1 if vol > 0.3 else 0
        total_drop = max(portfolio_drop - vol_adjustment, -0.85)

        stressed_price = current_price * (1 + total_drop)
        dollar_loss_per_share = current_price - stressed_price
        recovery_estimate = crisis["duration_days"] * 1.5  # rough recovery

        results[crisis_id] = {
            "name": crisis["name"],
            "description": crisis["description"],
            "portfolio_drop_pct": round(total_drop * 100, 1),
            "stressed_price": round(stressed_price, 2),
            "loss_per_share": round(dollar_loss_per_share, 2),
            "recovery_days_estimate": int(recovery_estimate),
            "severity": "Critical" if total_drop < -0.4 else "Severe" if total_drop < -0.25 else "Moderate"
        }

    # Worst case
    worst = min(results.values(), key=lambda x: x["portfolio_drop_pct"])
    best  = max(results.values(), key=lambda x: x["portfolio_drop_pct"])

    return {
        "scenarios": results,
        "worst_case": worst,
        "best_case": best,
        "current_price": round(current_price, 2),
        "beta": beta,
        "annual_vol": round(vol * 100, 1)
    }


def format_stress_test(ticker: str, data: dict) -> str:
    """Format stress test results as markdown"""
    lines = [
        f"## 🔥 Stress Test: {ticker}",
        f"**Current Price:** ${data['current_price']} | **Beta:** {data['beta']} | **Annual Vol:** {data['annual_vol']}%\n",
        f"### Crisis Scenarios",
        f"| Scenario | Drop | Stressed Price | Severity |",
        f"|----------|------|----------------|----------|"
    ]
    for s in data["scenarios"].values():
        emoji = "🔴" if s["severity"] == "Critical" else "🟠" if s["severity"] == "Severe" else "🟡"
        lines.append(f"| {s['name']} | {s['portfolio_drop_pct']}% | ${s['stressed_price']} | {emoji} {s['severity']} |")

    worst = data["worst_case"]
    lines += [
        f"\n### ⚠️ Worst Case: {worst['name']}",
        f"- **Drop:** {worst['portfolio_drop_pct']}%",
        f"- **Stressed Price:** ${worst['stressed_price']}",
        f"- **Loss/Share:** ${worst['loss_per_share']}",
        f"- **Est. Recovery:** ~{worst['recovery_days_estimate']} days",
    ]
    return "\n".join(lines)
