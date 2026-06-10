def dcf_valuation(ticker):
    r = requests.get(f"https://financialmodelingprep.com/stable/cash-flow-statement/{ticker}?limit=5&apikey=$FMP_KEY")
    # DCF calculation
    fcf_growth = 0.25  # 25% expected
    discount_rate = 0.10
    terminal_value = fcf * (1 + fcf_growth) / (discount_rate - fcf_growth)
    intrinsic = terminal_value / shares_outstanding
    return f"DCF Fair Value: \${intrinsic:.0f}"
