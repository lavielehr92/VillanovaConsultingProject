import yfinance as yf
import pandas as pd

def calculate_m_score_manual(
    receivables_current, revenue_current, gross_profit_current, current_assets_current,
    total_assets_current, ppe_current, depreciation_current, sga_expenses_current,
    current_liabilities_current, long_term_debt_current, net_income_current,
    nonoperating_income_current, operating_cash_flow_current,
    receivables_previous, revenue_previous, gross_profit_previous, current_assets_previous,
    total_assets_previous, ppe_previous, depreciation_previous, sga_expenses_previous,
    current_liabilities_previous, long_term_debt_previous, net_income_previous,
    nonoperating_income_previous, operating_cash_flow_previous,
    debug=False
):
    # Calculate the financial ratios
    # DSRI: Days Sales in Receivables Index
    dsri = (receivables_current / revenue_current) / (receivables_previous / revenue_previous)

    # GMI: Gross Margin Index
    gmi = (gross_profit_previous / revenue_previous) / (gross_profit_current / revenue_current)

    # AQI: Asset Quality Index
    aqi = (1 - (current_assets_current + ppe_current) / total_assets_current) / \
          (1 - (current_assets_previous + ppe_previous) / total_assets_previous)

    # SGI: Sales Growth Index
    sgi = revenue_current / revenue_previous

    # DEPI: Depreciation Index
    depi = (depreciation_previous / (ppe_previous + depreciation_previous)) / \
           (depreciation_current / (ppe_current + depreciation_current))

    # SGAI: Sales, General, and Administrative Expenses Index
    sgai = (sga_expenses_current / revenue_current) / (sga_expenses_previous / revenue_previous)

    # LVGI: Leverage Index
    lvgi = ((current_liabilities_current + long_term_debt_current) / total_assets_current) / \
           ((current_liabilities_previous + long_term_debt_previous) / total_assets_previous)

    # TATA: Total Accruals to Total Assets
    if operating_cash_flow_current is not None:
        tata = (net_income_current - nonoperating_income_current - operating_cash_flow_current) / total_assets_current
    else:
        tata = None  # Unable to calculate if operating cash flow data is missing

    if debug:
        print("Intermediate Ratios:")
        print(f"DSRI: {dsri}")
        print(f"GMI: {gmi}")
        print(f"AQI: {aqi}")
        print(f"SGI: {sgi}")
        print(f"DEPI: {depi}")
        print(f"SGAI: {sgai}")
        print(f"LVGI: {lvgi}")
        print(f"TATA: {tata}")

    # Calculate the M-Score
    if tata is not None:
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + \
                  0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
        if debug:
            print(f"M-Score components: -4.84 + 0.92*{dsri} + 0.528*{gmi} + 0.404*{aqi} + 0.892*{sgi} + 0.115*{depi} - 0.172*{sgai} + 4.679*{tata} - 0.327*{lvgi}")
    else:
        m_score = None  # M-Score cannot be calculated if TATA is missing

    return m_score

# Test for PLTR
ticker = 'PLTR'
stock = yf.Ticker(ticker)

income_stmt = stock.financials
balance_sheet = stock.balance_sheet
cashflow = stock.cashflow

dates = sorted(income_stmt.columns, reverse=True)[:2]
current_date = dates[0]
previous_date = dates[1]

print(f"Dates: {current_date}, {previous_date}")

receivables_current = balance_sheet.loc['Accounts Receivable', current_date]
revenue_current = income_stmt.loc['Total Revenue', current_date]
gross_profit_current = income_stmt.loc['Gross Profit', current_date]
current_assets_current = balance_sheet.loc['Current Assets', current_date]
total_assets_current = balance_sheet.loc['Total Assets', current_date]
ppe_current = balance_sheet.loc['Machinery Furniture Equipment', current_date]
depreciation_current = cashflow.loc['Depreciation And Amortization', current_date]
sga_expenses_current = income_stmt.loc['Selling General And Administration', current_date]
current_liabilities_current = balance_sheet.loc['Current Liabilities', current_date]
long_term_debt_current = balance_sheet.loc['Long Term Debt And Capital Lease Obligation', current_date] if 'Long Term Debt And Capital Lease Obligation' in balance_sheet.index and not pd.isna(balance_sheet.loc['Long Term Debt And Capital Lease Obligation', current_date]) else 0
net_income_current = income_stmt.loc['Net Income', current_date]
nonoperating_income_current = income_stmt.loc['Other Income Expense', current_date]
operating_cash_flow_current = cashflow.loc['Operating Cash Flow', current_date]

receivables_previous = balance_sheet.loc['Accounts Receivable', previous_date]
revenue_previous = income_stmt.loc['Total Revenue', previous_date]
gross_profit_previous = income_stmt.loc['Gross Profit', previous_date]
current_assets_previous = balance_sheet.loc['Current Assets', previous_date]
total_assets_previous = balance_sheet.loc['Total Assets', previous_date]
ppe_previous = balance_sheet.loc['Machinery Furniture Equipment', previous_date]
depreciation_previous = cashflow.loc['Depreciation And Amortization', previous_date]
sga_expenses_previous = income_stmt.loc['Selling General And Administration', previous_date]
current_liabilities_previous = balance_sheet.loc['Current Liabilities', previous_date]
long_term_debt_previous = balance_sheet.loc['Long Term Debt And Capital Lease Obligation', previous_date] if 'Long Term Debt And Capital Lease Obligation' in balance_sheet.index and not pd.isna(balance_sheet.loc['Long Term Debt And Capital Lease Obligation', previous_date]) else 0
net_income_previous = income_stmt.loc['Net Income', previous_date]
nonoperating_income_previous = income_stmt.loc['Other Income Expense', previous_date]
operating_cash_flow_previous = cashflow.loc['Operating Cash Flow', previous_date]

print("Values:")
print(f"receivables_current: {receivables_current}")
print(f"revenue_current: {revenue_current}")
# etc.

m_score = calculate_m_score_manual(
    receivables_current, revenue_current, gross_profit_current, current_assets_current,
    total_assets_current, ppe_current, depreciation_current, sga_expenses_current,
    current_liabilities_current, long_term_debt_current, net_income_current,
    nonoperating_income_current, operating_cash_flow_current,
    receivables_previous, revenue_previous, gross_profit_previous, current_assets_previous,
    total_assets_previous, ppe_previous, depreciation_previous, sga_expenses_previous,
    current_liabilities_previous, long_term_debt_previous, net_income_previous,
    nonoperating_income_previous, operating_cash_flow_previous,
    debug=True
)

print(f"M-Score: {m_score}")