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

    # Calculate the M-Score
    if tata is not None:
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + \
                  0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    else:
        m_score = None  # M-Score cannot be calculated if TATA is missing

    return m_score

# Example usage
m_score = calculate_m_score_manual(
    receivables_current, revenue_current, gross_profit_current, current_assets_current,
    total_assets_current, ppe_current, depreciation_current, sga_expenses_current,
    current_liabilities_current, long_term_debt_current, net_income_current,
    nonoperating_income_current, operating_cash_flow_current,
    receivables_previous, revenue_previous, gross_profit_previous, current_assets_previous,
    total_assets_previous, ppe_previous, depreciation_previous, sga_expenses_previous,
    current_liabilities_previous, long_term_debt_previous, net_income_previous,
    nonoperating_income_previous, operating_cash_flow_previous
)
if m_score is not None:
    print(f"M-Score: {m_score}")
else:
    print("M-Score could not be calculated due to missing data.")