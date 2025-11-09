# test_mscore_agents.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from autogen.agentchat import AssistantAgent, UserProxyAgent

# ---- API key ----
here = Path(__file__).resolve().parent
out_dir = here / "agent_runs"
out_dir.mkdir(parents=True, exist_ok=True)

# Load .env from MyKeys
mykeys_env = here.parent / "MyKeys/.env"
if mykeys_env.exists():
    load_dotenv(mykeys_env)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

config_list = [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]

# === Task for Code Writer Agent ===
# # TASK = r"""
You are the Code Writer Agent. Your task is to generate complete, runnable Python code that computes the Beneish M-Score using the provided sample financial data.

The code should:
- Use the provided sample data for current and previous year.
- Display the financial input parameters used in the calculation.
- Compute the M-Score using the provided function.
- Output the M-Score and interpretation.

Include the exact function and data below verbatim in your code:

# Data for Current Year
receivables_current = 6284
revenue_current = 523964
gross_profit_current = 129359
current_assets_current = 61806
total_assets_current = 236495
ppe_current = 105208
depreciation_current = 10987
sga_expenses_current = 108791
current_liabilities_current = 77790
long_term_debt_current = 64372
net_income_current = 14881
nonoperating_income_current = 1958
operating_cash_flow_current = 25255  # CF from Operations

# Data for Previous Year
receivables_previous = 6283
revenue_previous = 514405
gross_profit_previous = 129104
current_assets_previous = 61890
total_assets_previous = 219295
ppe_previous = 104317
depreciation_previous = 10678
sga_expenses_previous = 107147
current_liabilities_previous = 77477
long_term_debt_previous = 50203
net_income_previous = 6670
nonoperating_income_previous = -8368
operating_cash_flow_previous = None  # Data not needed

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

Provide only the Python code in your response, wrapped in ```python ```.
"""

# === Agents ===
code_writer = AssistantAgent(
    name="CodeWriter",
    system_message="You are a code writer agent that generates Python code for financial calculations.",
    llm_config={"config_list": config_list},
)

code_executor = UserProxyAgent(
    name="CodeExecutor",
    system_message="You are a code executor agent. Execute the code provided by the CodeWriter and report the results.",
    code_execution_config={"work_dir": str(out_dir), "use_docker": False},
    human_input_mode="NEVER",
)

# === Initiate Chat ===
while True:
    tickers_input = input("Enter stock tickers separated by commas (or 'quit' to exit): ")
    if tickers_input.lower() == 'quit':
        break
    tickers = [t.strip() for t in tickers_input.split(',')]
    for ticker in tickers:
        print(f"Processing {ticker}...")
        code_executor.initiate_chat(
            code_writer,
            message=TASK + f"\n\nGenerate the code for ticker '{ticker}'.",
        )