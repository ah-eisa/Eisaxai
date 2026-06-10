#!/usr/bin/env python3
import yfinance as yf
import pandas as pd

def stress_test():
    tickers = ['NVDA', 'AAPL', 'TSLA']
    for ticker in tickers:
        try:
            # الحل الجذري هنا
            data = yf.download(ticker, period="1y", auto_adjust=True)
            current = data['Close'].iloc[-1]
            crash = current * 0.6  # -40%
            print(f"{ticker}: ${current:.2f} → Crash: ${crash:.2f}")
        except Exception as e:
            print(f"{ticker}: ERROR {e}")
            
stress_test()
