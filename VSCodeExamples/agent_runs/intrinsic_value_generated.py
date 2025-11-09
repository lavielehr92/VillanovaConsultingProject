import yfinance as yf

def calculate_intrinsic_value(current_free_cash_flow, growth_rate, terminal_growth_rate, reqd_rate_of_return, shares_outstanding):
    years = 10
    expected_fcfs = [current_free_cash_flow * (1 + growth_rate) ** year for year in range(1, years + 1)]
    terminal_value = expected_fcfs[-1] * (1 + terminal_growth_rate) / (reqd_rate_of_return - terminal_growth_rate)
    discount_factors = [(1 + reqd_rate_of_return) ** year for year in range(1, years + 1)]
    present_values = [fcf / df for fcf, df in zip(expected_fcfs, discount_factors)]
    terminal_value_pv = terminal_value / (1 + reqd_rate_of_return) ** years
    current_value_future_cash_flows = sum(present_values) + terminal_value_pv
    intrinsic_value_per_share = current_value_future_cash_flows / shares_outstanding
    return intrinsic_value_per_share

def main():
    tickers_input = input("Enter ticker symbols separated by commas (e.g., AAPL,MSFT,GOOG): ")
    growth_rate_input = input("Enter growth_rate, terminal_growth_rate, reqd_rate_of_return separated by commas (e.g., 0.05,0.02,0.08): ")
    
    try:
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        growth_rate, terminal_growth_rate, reqd_rate_of_return = map(float, [x.strip() for x in growth_rate_input.split(',')])
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            current_free_cash_flow = stock.info.get('freeCashflow')
            shares_outstanding = stock.info.get('sharesOutstanding')
            
            if current_free_cash_flow is None or shares_outstanding is None:
                print(f"Data not available for {ticker}.")
                continue
            
            if shares_outstanding == 0 or reqd_rate_of_return <= terminal_growth_rate:
                print(f"Invalid data for {ticker}: shares_outstanding={shares_outstanding}, reqd_rate_of_return={reqd_rate_of_return}, terminal_growth_rate={terminal_growth_rate}.")
                continue
            
            intrinsic_value = calculate_intrinsic_value(current_free_cash_flow, growth_rate, terminal_growth_rate, reqd_rate_of_return, shares_outstanding)
            print(f"The intrinsic value per share for {ticker} is: ${intrinsic_value:.2f}")
            
    except ValueError:
        print("Invalid input. Please enter numbers in the correct format.")

if __name__ == "__main__":
    main()
