import yfinance as yf

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

def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)

    # Try to get income statement and balance sheet data
    try:
        financials = stock.financials.transpose()
        balance_sheet = stock.balance_sheet.transpose()

        # Show the fetched data for debugging
        print("Income Statement Data:\n", financials)
        print("Balance Sheet Data:\n", balance_sheet)

        # Ensure we have data for two fiscal years
        if len(financials) < 2 or len(balance_sheet) < 2:
            print("Not enough data available for the specified ticker.")
            return

        current_year = financials.index[0]
        previous_year = financials.index[1]

        # Extract the required fields
        fields = {
            'revenue': 'Total Revenue',
            'net_income': 'Net Income',
            'gross_profit': 'Gross Profit',
            'receivables': 'Net Receivables',
            'current_assets': 'Total Current Assets',
            'total_assets': 'Total Assets',
            'ppe': 'Property Plant and Equipment',
            'depreciation': 'Depreciation',
            'sga_expenses': 'Selling General and Administrative Expenses',
            'current_liabilities': 'Total Current Liabilities',
            'long_term_debt': 'Long Term Debt',
            'nonoperating_income': 'Nonoperating Income',
            'operating_cash_flow': 'Operating Cash Flow'
        }

        data = {}
        missing_fields = []

        for key, field in fields.items():
            try:
                if key in ['receivables', 'current_assets', 'total_assets', 'ppe', 'depreciation', 'sga_expenses', 'current_liabilities', 'long_term_debt']:
                    data[key + '_current'] = balance_sheet.loc[current_year, field]
                    data[key + '_previous'] = balance_sheet.loc[previous_year, field]
                else:
                    data[key + '_current'] = financials.loc[current_year, field]
                    data[key + '_previous'] = financials.loc[previous_year, field]
            except KeyError:
                missing_fields.append(field)

        # Check for missing fields
        if missing_fields:
            print("Missing fields:", missing_fields)
            return

        # Display the inputs
        print("Inputs for Calculation:")
        for key in data.keys():
            print(f"{key}: {data[key]}")

        # Call M-Score calculation function
        m_score = calculate_m_score_manual(
            data['receivables_current'], data['revenue_current'], data['gross_profit_current'], 
            data['current_assets_current'], data['total_assets_current'], data['ppe_current'], 
            data['depreciation_current'], data['sga_expenses_current'], 
            data['current_liabilities_current'], data['long_term_debt_current'], 
            data['net_income_current'], data['nonoperating_income_current'], 
            data['operating_cash_flow_current'], 
            data['receivables_previous'], data['revenue_previous'], 
            data['gross_profit_previous'], data['current_assets_previous'], 
            data['total_assets_previous'], data['ppe_previous'], 
            data['depreciation_previous'], data['sga_expenses_previous'], 
            data['current_liabilities_previous'], data['long_term_debt_previous'], 
            data['net_income_previous'], data['nonoperating_income_previous'], 
            data['operating_cash_flow_previous']
        )

        print("Beneish M-Score:", m_score)

        # Interpretation of M-Score
        if m_score is not None:
            if m_score > -2.22:
                print("The company is likely a 'manipulator'.")
            else:
                print("The company is likely 'not a manipulator'.")

    except Exception as e:
        print("Error while fetching financial data:", e)

if __name__ == "__main__":
    ticker_input = input("Enter stock ticker (e.g., AAPL): ")
    fetch_financial_data(ticker_input)