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
        print(f"DSRI: {dsri:.4f}")
        print(f"GMI: {gmi:.4f}")
        print(f"AQI: {aqi:.4f}")
        print(f"SGI: {sgi:.4f}")
        print(f"DEPI: {depi:.4f}")
        print(f"SGAI: {sgai:.4f}")
        print(f"LVGI: {lvgi:.4f}")
        print(f"TATA: {tata:.4f}" if tata is not None else "TATA: None")

    # Calculate the M-Score
    if tata is not None:
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + \
                  0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
        if debug:
            print(f"M-Score components: -4.84 + 0.92*{dsri:.4f} + 0.528*{gmi:.4f} + 0.404*{aqi:.4f} + 0.892*{sgi:.4f} + 0.115*{depi:.4f} - 0.172*{sgai:.4f} + 4.679*{tata:.4f} - 0.327*{lvgi:.4f}")
    else:
        m_score = None  # M-Score cannot be calculated if TATA is missing

    return m_score

def main():
    while True:
        ticker = input("Enter stock ticker (or 'quit' to exit): ").strip().upper()
        if ticker == 'QUIT':
            break

        try:
            stock = yf.Ticker(ticker)

            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            # Get the most recent two years
            dates = sorted(income_stmt.columns, reverse=True)[:2]
            if len(dates) < 2:
                print("Not enough financial data available for two years.")
                continue

            current_date = dates[0]
            previous_date = dates[1]

            # Extract current year values
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

            # Extract previous year values
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

            # Calculate M-Score
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

            if m_score is not None:
                print(f"M-Score for {ticker}: {m_score:.4f}")
            else:
                print("M-Score could not be calculated due to missing data.")

            # Display financial input parameters
            print("\nFinancial Input Parameters:")
            print(f"Current Year ({current_date.year}):")
            print(f"  Receivables: {receivables_current}")
            print(f"  Revenue: {revenue_current}")
            print(f"  Gross Profit: {gross_profit_current}")
            print(f"  Current Assets: {current_assets_current}")
            print(f"  Total Assets: {total_assets_current}")
            print(f"  PPE: {ppe_current}")
            print(f"  Depreciation: {depreciation_current}")
            print(f"  SGA Expenses: {sga_expenses_current}")
            print(f"  Current Liabilities: {current_liabilities_current}")
            print(f"  Long Term Debt: {long_term_debt_current}")
            print(f"  Net Income: {net_income_current}")
            print(f"  Nonoperating Income: {nonoperating_income_current}")
            print(f"  Operating Cash Flow: {operating_cash_flow_current}")

            print(f"\nPrevious Year ({previous_date.year}):")
            print(f"  Receivables: {receivables_previous}")
            print(f"  Revenue: {revenue_previous}")
            print(f"  Gross Profit: {gross_profit_previous}")
            print(f"  Current Assets: {current_assets_previous}")
            print(f"  Total Assets: {total_assets_previous}")
            print(f"  PPE: {ppe_previous}")
            print(f"  Depreciation: {depreciation_previous}")
            print(f"  SGA Expenses: {sga_expenses_previous}")
            print(f"  Current Liabilities: {current_liabilities_previous}")
            print(f"  Long Term Debt: {long_term_debt_previous}")
            print(f"  Net Income: {net_income_previous}")
            print(f"  Nonoperating Income: {nonoperating_income_previous}")
            print(f"  Operating Cash Flow: {operating_cash_flow_previous}")

        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")

if __name__ == "__main__":
    main()