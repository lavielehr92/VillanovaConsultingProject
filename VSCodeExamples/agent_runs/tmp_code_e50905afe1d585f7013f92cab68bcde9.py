import yfinance as yf
import sys

def fetch_field(data, field_names):
    for name in field_names:
        if name in data:
            return data[name].iloc[0]  # Get most recent value
    return None

def calculate_m_score_manual(
    receivables_current, revenue_current, gross_profit_current, current_assets_current,
    total_assets_current, ppe_current, depreciation_current, sga_expenses_current,
    current_liabilities_current, long_term_debt_current, net_income_current,
    nonoperating_income_current, operating_cash_flow_current,
    receivables_previous, revenue_previous, gross_profit_previous, current_assets_previous,
    total_assets_previous, ppe_previous, depreciation_previous, sga_expenses_previous,
    current_liabilities_previous, long_term_debt_previous, net_income_previous,
    nonoperating_income_previous, operating_cash_flow_previous
):
    dsri = (receivables_current / revenue_current) / (receivables_previous / revenue_previous)
    gmi = (gross_profit_previous / revenue_previous) / (gross_profit_current / revenue_current)
    aqi = (1 - (current_assets_current + ppe_current) / total_assets_current) / \
          (1 - (current_assets_previous + ppe_previous) / total_assets_previous)
    sgi = revenue_current / revenue_previous
    depi = (depreciation_previous / (ppe_previous + depreciation_previous)) / \
           (depreciation_current / (ppe_current + depreciation_current))
    sgai = (sga_expenses_current / revenue_current) / (sga_expenses_previous / revenue_previous)
    lvgi = ((current_liabilities_current + long_term_debt_current) / total_assets_current) / \
           ((current_liabilities_previous + long_term_debt_previous) / total_assets_previous)

    if operating_cash_flow_current is not None:
        tata = (net_income_current - nonoperating_income_current - operating_cash_flow_current) / total_assets_current
    else:
        tata = None

    if tata is not None:
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + \
                  0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    else:
        m_score = None

    return m_score

def main():
    ticker = input("Enter a stock ticker: ").strip().upper()
    stock = yf.Ticker(ticker)

    # Fetching financial data
    income_stmt = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cashflow_stmt = stock.cashflow.T

    # Current year (latest data)
    income_current = income_stmt.iloc[0]
    balance_current = balance_sheet.iloc[0]
    cashflow_current = cashflow_stmt.iloc[0]
    
    # Previous year (second latest data)
    income_previous = income_stmt.iloc[1]
    balance_previous = balance_sheet.iloc[1]
    cashflow_previous = cashflow_stmt.iloc[1]

    # Extracting fields with possible variations
    fields = {
        "receivables": fetch_field(balance_sheet, ['Accounts Receivable', 'Receivables']),
        "revenue": fetch_field(income_stmt, ['Total Revenue', 'TotalRevenue', 'Revenue']),
        "gross_profit": fetch_field(income_stmt, ['Gross Profit']),
        "current_assets": fetch_field(balance_sheet, ['Current Assets']),
        "total_assets": fetch_field(balance_sheet, ['Total Assets']),
        "ppe": fetch_field(balance_sheet, ['Net PPE', 'Property Plant Equipment']),
        "depreciation": fetch_field(cashflow_stmt, ['Depreciation', 'DepreciationAmortization']),
        "sga_expenses": fetch_field(income_stmt, ['Selling General And Administrative', 'SellingGeneralAdministrative']),
        "current_liabilities": fetch_field(balance_sheet, ['Current Liabilities']),
        "long_term_debt": fetch_field(balance_sheet, ['Long Term Debt']),
        "net_income": fetch_field(income_stmt, ['Net Income', 'NetIncome']),
        "nonoperating_income": fetch_field(income_stmt, ['Other Income Expense', 'NonOperatingIncomeExpense']),
        "operating_cash_flow": fetch_field(cashflow_stmt, ['Operating Cash Flow', 'OperatingCashFlow']),
    }

    # Preparing the current and previous value set
    inputs = {
        "current": {key: fields[key] for key in fields},
        "previous": {key: fields[key] for key in fields}
    }

    # Checking for missing fields
    missing_fields = [key for key, value in fields.items() if value is None]
    if missing_fields:
        print(f"Missing fields: {', '.join(missing_fields)}. Exiting.")
        sys.exit()

    # Print the parameters
    print("Using the following parameters:")
    for key, value in inputs['current'].items():
        print(f"{key}: {value})")

    # Calculate M-Score
    m_score = calculate_m_score_manual(
        inputs["current"]["receivables"], inputs["current"]["revenue"],
        inputs["current"]["gross_profit"], inputs["current"]["current_assets"],
        inputs["current"]["total_assets"], inputs["current"]["ppe"],
        inputs["current"]["depreciation"], inputs["current"]["sga_expenses"],
        inputs["current"]["current_liabilities"], inputs["current"]["long_term_debt"],
        inputs["current"]["net_income"], inputs["current"]["nonoperating_income"],
        inputs["current"]["operating_cash_flow"],
        
        inputs["previous"]["receivables"], inputs["previous"]["revenue"],
        inputs["previous"]["gross_profit"], inputs["previous"]["current_assets"],
        inputs["previous"]["total_assets"], inputs["previous"]["ppe"],
        inputs["previous"]["depreciation"], inputs["previous"]["sga_expenses"],
        inputs["previous"]["current_liabilities"], inputs["previous"]["long_term_debt"],
        inputs["previous"]["net_income"], inputs["previous"]["nonoperating_income"],
        inputs["previous"]["operating_cash_flow"]
    )

    print(f"M-Score: {m_score:.3f}")
    if m_score > -2.22:
        print("This suggests a higher likelihood of earnings manipulation.")
    else:
        print("This suggests that earnings manipulation is less likely.")

if __name__ == "__main__":
    main()