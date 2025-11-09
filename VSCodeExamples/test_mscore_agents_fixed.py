# test_mscore_agents.py
import os
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
TASK = r"""
You are the Code Writer Agent. Your task is to generate complete, runnable Python code that computes the Beneish M-Score for a given stock ticker.

The code should:
- Accept a single stock ticker as user input (e.g., 'AAPL').
- Use yfinance to retrieve financial data for the latest two fiscal years.
- Extract all necessary fields: revenue, net_income, gross_profit, receivables, current_assets, total_assets, ppe, depreciation, sga_expenses, current_liabilities, long_term_debt, nonoperating_income, operating_cash_flow (current and previous year).
- Display the inputs used in the calculation.
- Compute the M-Score using the provided function.
- Handle missing data by listing missing fields and exiting.
- Output the M-Score and interpretation.

Include the exact function below verbatim in your code:

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