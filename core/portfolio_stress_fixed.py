#!/usr/bin/env python3
import yfinance as yf

def stress_test():
    tickers = ['NVDA', 'AAPL', 'TSLA']
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1y", auto_adjust=True)
            # MultiIndex fix
            if isinstance(data.columns, pd.MultiIndex):
                close_col = data['Close'].iloc[-1]
            else:
                close_col = data['Close'].iloc[-1]
            
            current = float(close_col)
            crash = current * 0.6
            print(f"{ticker}: ${current:.2f} → Crash: ${crash:.2f}")
        except Exception as e:
            print(f"{ticker}: ERROR {e}")

stress_test()
