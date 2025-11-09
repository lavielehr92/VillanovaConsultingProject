import yfinance as yf

stock = yf.Ticker('AAPL')
balance_sheet = stock.balance_sheet
income_stmt = stock.income_stmt
cash_flow = stock.cashflow

print("Balance Sheet columns:")
print(balance_sheet.index.tolist() if not balance_sheet.empty else "Empty")

print("\nIncome Stmt columns:")
print(income_stmt.index.tolist() if not income_stmt.empty else "Empty")

print("\nCash Flow columns:")
print(cash_flow.index.tolist() if not cash_flow.empty else "Empty")

print("\nBalance Sheet shape:", balance_sheet.shape)
print("Income Stmt shape:", income_stmt.shape)
print("Cash Flow shape:", cash_flow.shape)