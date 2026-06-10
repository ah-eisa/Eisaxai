def stress_portfolio(portfolio):
    # 2008 crash -40%
    stressed = {t: p * 0.6 for t,p in portfolio.items()}
    return stressed
