import yfinance as yf
import pandas as pd


def _select_first_available(series, candidates, default=None):
    for key in candidates:
        if key in series.index and pd.notna(series[key]):
            return series[key]
    return default


def _prepare_statements(income_stmt, balance_sheet, cash_flow):
    if income_stmt is None or income_stmt.empty:
        raise ValueError("Income statement data is unavailable.")
    if balance_sheet is None or balance_sheet.empty:
        raise ValueError("Balance sheet data is unavailable.")
    if cash_flow is None or cash_flow.empty:
        raise ValueError("Cash flow data is unavailable.")

    income_cols = sorted(income_stmt.columns, reverse=True)
    balance_cols = sorted(balance_sheet.columns, reverse=True)
    cash_cols = sorted(cash_flow.columns, reverse=True)

    if len(income_cols) < 2 or len(balance_cols) < 2 or len(cash_cols) < 2:
        raise ValueError("Not enough historical data (need at least two periods).")

    return (
        income_stmt[income_cols[0]],
        income_stmt[income_cols[1]],
        balance_sheet[balance_cols[0]],
        balance_sheet[balance_cols[1]],
        cash_flow[cash_cols[0]],
        cash_flow[cash_cols[1]],
    )

# === BEGIN PROFESSOR FUNCTION (VERBATIM) ===
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
# === END PROFESSOR FUNCTION (VERBATIM) ===

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow

    return income_stmt, balance_sheet, cash_flow

def extract_data(income_stmt, balance_sheet, cash_flow):
    (
        inc_current,
        inc_previous,
        bal_current,
        bal_previous,
        cash_current,
        cash_previous,
    ) = _prepare_statements(income_stmt, balance_sheet, cash_flow)

    fields = {
        'revenue': None,
        'net_income': None,
        'gross_profit': None,
        'receivables': None,
        'current_assets': None,
        'total_assets': None,
        'ppe': None,
        'depreciation': None,
        'sga_expenses': None,
        'current_liabilities': None,
        'long_term_debt': None,
        'nonoperating_income': None,
        'operating_cash_flow': None,
        'revenue_previous': None,
        'net_income_previous': None,
        'gross_profit_previous': None,
        'receivables_previous': None,
        'current_assets_previous': None,
        'total_assets_previous': None,
        'ppe_previous': None,
        'depreciation_previous': None,
        'sga_expenses_previous': None,
        'current_liabilities_previous': None,
        'long_term_debt_previous': None,
        'nonoperating_income_previous': None,
        'operating_cash_flow_previous': None,
    }

    fields['revenue'] = _select_first_available(inc_current, ['Total Revenue', 'Revenue'])
    fields['net_income'] = _select_first_available(inc_current, ['Net Income', 'NetIncome'])
    fields['gross_profit'] = _select_first_available(inc_current, ['Gross Profit', 'GrossProfit'])
    fields['sga_expenses'] = _select_first_available(
        inc_current,
        ['Selling General Administrative', 'Selling General And Administration', 'SG&A'],
    )
    fields['nonoperating_income'] = _select_first_available(
        inc_current,
        ['Other Income Expense', 'Nonoperating Income', 'Other Non Operating Income Expenses'],
        default=0,
    )

    fields['receivables'] = _select_first_available(bal_current, ['Accounts Receivable', 'Receivables'])
    fields['current_assets'] = _select_first_available(bal_current, ['Total Current Assets', 'Current Assets'])
    fields['total_assets'] = _select_first_available(bal_current, ['Total Assets'])
    fields['ppe'] = _select_first_available(
        bal_current,
        ['Property Plant Equipment', 'Net PPE', 'Plant Property Equipment'],
    )
    fields['current_liabilities'] = _select_first_available(
        bal_current,
        ['Total Current Liabilities', 'Current Liabilities'],
    )
    fields['long_term_debt'] = _select_first_available(
        bal_current,
        ['Long Term Debt And Capital Lease Obligation', 'Long Term Debt', 'Total Debt'],
        default=0,
    )

    fields['depreciation'] = _select_first_available(
        cash_current,
        ['Depreciation And Amortization', 'Depreciation & Amortization', 'Depreciation'],
    )
    fields['operating_cash_flow'] = _select_first_available(
        cash_current,
        ['Operating Cash Flow', 'Cash From Operating Activities', 'Net Cash Provided By Operating Activities'],
    )

    fields['revenue_previous'] = _select_first_available(inc_previous, ['Total Revenue', 'Revenue'])
    fields['net_income_previous'] = _select_first_available(inc_previous, ['Net Income', 'NetIncome'])
    fields['gross_profit_previous'] = _select_first_available(inc_previous, ['Gross Profit', 'GrossProfit'])
    fields['sga_expenses_previous'] = _select_first_available(
        inc_previous,
        ['Selling General Administrative', 'Selling General And Administration', 'SG&A'],
    )
    fields['nonoperating_income_previous'] = _select_first_available(
        inc_previous,
        ['Other Income Expense', 'Nonoperating Income', 'Other Non Operating Income Expenses'],
        default=0,
    )

    fields['receivables_previous'] = _select_first_available(
        bal_previous, ['Accounts Receivable', 'Receivables']
    )
    fields['current_assets_previous'] = _select_first_available(
        bal_previous, ['Total Current Assets', 'Current Assets']
    )
    fields['total_assets_previous'] = _select_first_available(bal_previous, ['Total Assets'])
    fields['ppe_previous'] = _select_first_available(
        bal_previous,
        ['Property Plant Equipment', 'Net PPE', 'Plant Property Equipment'],
    )
    fields['current_liabilities_previous'] = _select_first_available(
        bal_previous,
        ['Total Current Liabilities', 'Current Liabilities'],
    )
    fields['long_term_debt_previous'] = _select_first_available(
        bal_previous,
        ['Long Term Debt And Capital Lease Obligation', 'Long Term Debt', 'Total Debt'],
        default=0,
    )

    fields['depreciation_previous'] = _select_first_available(
        cash_previous,
        ['Depreciation And Amortization', 'Depreciation & Amortization', 'Depreciation'],
    )
    fields['operating_cash_flow_previous'] = _select_first_available(
        cash_previous,
        ['Operating Cash Flow', 'Cash From Operating Activities', 'Net Cash Provided By Operating Activities'],
    )
    
    return fields

def main():
    ticker = input("Enter a stock ticker (e.g., AAPL): ")
    
    income_stmt, balance_sheet, cash_flow = get_financial_data(ticker)
    fields = extract_data(income_stmt, balance_sheet, cash_flow)

    missing_fields = [k for k, v in fields.items() if v is None or pd.isna(v)]
    if missing_fields:
        print(f"Missing fields: {', '.join(missing_fields)}")
        return

    print("\nData used for M-Score calculation:")
    for key, value in fields.items():
        print(f"{key}: {value}")

    m_score = calculate_m_score_manual(
        fields['receivables'], fields['revenue'], fields['gross_profit'], fields['current_assets'],
        fields['total_assets'], fields['ppe'], fields['depreciation'], fields['sga_expenses'],
        fields['current_liabilities'], fields['long_term_debt'], fields['net_income'],
        fields['nonoperating_income'], fields['operating_cash_flow'],
        fields['receivables_previous'], fields['revenue_previous'], fields['gross_profit_previous'], 
        fields['current_assets_previous'], fields['total_assets_previous'], fields['ppe_previous'], 
        fields['depreciation_previous'], fields['sga_expenses_previous'], 
        fields['current_liabilities_previous'], fields['long_term_debt_previous'], 
        fields['net_income_previous'], fields['nonoperating_income_previous'], fields['operating_cash_flow_previous']
    )

    print(f"\nBeneish M-Score: {m_score:.3f}")
    if m_score > -2.22:
        print("Higher risk of earnings manipulation.")
    else:
        print("Lower risk of earnings manipulation.")

if __name__ == "__main__":
    main()

