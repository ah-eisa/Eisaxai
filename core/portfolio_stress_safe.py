#!/usr/bin/env python3
import yfinance as yf

def stress_test():
    tickers = ['NVDA', 'AAPL', 'TSLA']
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", auto_adjust=True)
        current = float(hist['Close'].iloc[-1])
        crash = current * 0.6
        print(f"{ticker}: ${current:.2f} → Crash: ${crash:.2f}")

stress_test()
