# run_mscore_agents.py
import os, re
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from docx import Document
from autogen.agentchat import AssistantAgent, UserProxyAgent

# ---- API key ----
here = Path(__file__).resolve().parent
out_dir = here / "agent_runs"
out_dir.mkdir(parents=True, exist_ok=True)

# Load your key from MyKeys, or fall back to nearest .env
mykeys_env_path = os.getenv("MYKEYS_ENV_PATH")
if mykeys_env_path:
    mykeys_env = Path(mykeys_env_path)
elif (here / "MyKeys/.env").exists():
    mykeys_env = here / "MyKeys/.env"
else:
    mykeys_env = None

if mykeys_env and mykeys_env.exists():
    load_dotenv(mykeys_env)
config_list = [{"model": "gpt-4o", "api_key": OPENAI_API_KEY}]
    load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

config_list = [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]

# === Paste your professor's EXACT function below; no sample numbers ===
TASK = r"""
Create complete, runnable Python code that:

- Accepts a single stock ticker as user input (e.g., AAPL).
- Retrieves all required financial data for the latest two fiscal years using yfinance (preferred).
- Displays a clear table/list of the exact inputs used (current & previous year).
- Computes and prints the Beneish M-Score using the EXACT function below (include it verbatim in your script).
- Handles missing data gracefully: list any missing fields and exit politely.
- Output ONLY one Python code block containing the final script. No explanations.

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

    # SGAI: SG&A Expenses Index
    sgai = (sga_expenses_current / revenue_current) / (sga_expenses_previous / revenue_previous)

    # LVGI: Leverage Index
    lvgi = ((current_liabilities_current + long_term_debt_current) / total_assets_current) / \
           ((current_liabilities_previous + long_term_debt_previous) / total_assets_previous)

    # TATA: Total Accruals to Total Assets
    if operating_cash_flow_current is not None:
        tata = (net_income_current - nonoperating_income_current - operating_cash_flow_current) / total_assets_current
    else:
        tata = None  # Unable to calculate if operating cash flow data is missing

    # M-Score
    if tata is not None:
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + \
                  0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    else:
        m_score = None  # M-Score cannot be calculated if TATA is missing

    return m_score
# === END PROFESSOR FUNCTION (VERBATIM) ===

Implementation notes for the script you generate:
- Use yfinance.Ticker(t).income_stmt, .balance_sheet, and .cashflow for annual data.
- Pull two most recent fiscal years (current & previous).
- For each required field, try common label variants (e.g., 'Total Revenue'/'Revenue', 'Accounts Receivable'/'Receivables',
  'Selling General And Administration'/'SellingGeneralAdministrative', 'Net PPE'/'Property Plant Equipment', etc.).
- Print a clean table/list of all inputs actually used (both years) before computing.
- If any required field (for either year) is missing, print which and exit.
- Print the final M-Score to 3 decimals and a one-line interpretation (e.g., > -2.22 suggests higher risk).
"""

# ---- Agents ----
code_writer = AssistantAgent(
    name="CodeWriter",
    system_message=(
        "You are a Python code writer. Produce robust, runnable code. "
        "Return exactly one ```python code block``` containing the full script and nothing else."
    ),
    llm_config={"config_list": config_list},
)

code_executor = UserProxyAgent(
    name="CodeExecutor",
    system_message=(
        "You execute Python code blocks you receive and report outputs concisely. "
        "If a required package (yfinance) is missing, install it and rerun."
    ),
    code_execution_config={"work_dir": str(out_dir), "use_docker": False},
    human_input_mode="NEVER",
)

# ---- Helpers ----
CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE | re.MULTILINE)

def extract_code_block(text: str) -> str:
    m = CODE_FENCE_RE.search(text or "")
    if not m:
        raise ValueError("No Python code block found in the writer's response.\n\nRaw reply:\n" + (text or ""))
    return m.group(1).strip() + "\n"

def save_word_with_code(code_text: str, doc_path: Path):
    """
    Save the provided code text to a Word document at the specified path.
    """
    doc = Document()
    doc.add_paragraph(code_text)
    doc.save(doc_path)
    writer_msg = code_writer.generate_reply(messages=[{"role": "user", "content": TASK}])
    if writer_msg is None:
        raise ValueError("No response from code_writer.")
    elif isinstance(writer_msg, dict):
        writer_text = writer_msg.get("content", "")
    code_text = extract_code_block(writer_text)

    # Remove any embedded triple backticks from the code to avoid formatting issues
    sanitized_code_text = code_text.replace("```", "")

    py_path = out_dir / "mscore_generated.py"
    py_path.write_text(code_text, encoding="utf-8")
    docx_path = out_dir / "MScore_Code.docx"   # submit this
    save_word_with_code(code_text, docx_path)

    # Optional: quick non-interactive run
    code_executor.initiate_chat(
        code_writer,
        message="Please execute the code block below:\n```python\n" + sanitized_code_text + "\n```",
        max_consecutive_auto_reply=2,
    )

    print(f"\nSaved script: {py_path}")
    print(f"Saved Word doc (submit this): {docx_path}")```python\n" + code_text + "\n```",
        max_consecutive_auto_reply=2,
    )

    print(f"\nSaved script: {py_path}")
    print(f"Saved Word doc (submit this): {docx_path}")