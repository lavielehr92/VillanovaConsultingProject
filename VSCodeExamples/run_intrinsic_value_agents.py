# run_intrinsic_value_agents.py
import os
import re
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from docx import Document  # pip install python-docx
from autogen.agentchat import AssistantAgent, UserProxyAgent  # pip install pyautogen

# ----------------- Config & API key -----------------
here = Path(__file__).resolve().parent
out_dir = here / "agent_runs"
out_dir.mkdir(parents=True, exist_ok=True)

# Load API key from your known location, with a fallback
mykeys_env = Path(r"C:\Users\LAVie\Documents\MBA8583\MyKeys\.env")
if mykeys_env.exists():
    load_dotenv(mykeys_env)
else:
    # fallback: nearest .env if you ever move it
    load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

config_list = [
    {"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
]

# ----------------- Assignment task -----------------
TASK = """
Create complete, runnable Python code that:

- Imports yfinance as yf.
- Asks the user to enter a list of ticker symbols separated by commas (e.g., AAPL,MSFT,GOOG).
- Asks the user to enter growth_rate, terminal_growth_rate, reqd_rate_of_return separated by commas (e.g., 0.05,0.02,0.08).
- For each ticker, fetches the current free cash flow (use stock.info.get('freeCashflow')) and shares outstanding (stock.info.get('sharesOutstanding')) using yfinance as yf.Ticker(ticker).
- Computes the intrinsic value per share using this function verbatim:

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

- Handle errors gracefully (missing data, bad inputs, denominator issues).
- Print the intrinsic value per share for each ticker.
- Output ONLY one Python code block containing the final script. Do not include explanations.
"""

# ----------------- Agents -----------------
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
        "If a required package is missing (e.g., yfinance), install it safely and rerun."
    ),
    code_execution_config={"work_dir": str(out_dir), "use_docker": False},
)

# ----------------- Helper: extract code block & save -----------------
# Accept ```python ... ``` or just ``` ... ```
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_code_block(text: str) -> str:
    m = CODE_FENCE_RE.search(text or "")
    if not m:
        # Helpful failure for debugging submissions
        raise ValueError("No Python code block found in the writer's response.\n\nRaw reply:\n" + (text or ""))
    return m.group(1).rstrip() + "\n"

def save_word_with_code(code_text: str, doc_path: Path):
    doc = Document()
    doc.add_paragraph(code_text)  # only the generated code (no prompts)
    doc.save(doc_path)

# ----------------- Run conversation -----------------
if __name__ == "__main__":
    # Ask Writer to produce the program
    writer_msg = code_writer.generate_reply(messages=[{"role": "user", "content": TASK}])
    writer_text = str(writer_msg)  # FIXED: no .get

    code_text = extract_code_block(writer_text)

    # Save to .py and .docx for submission
    py_path = out_dir / "intrinsic_value_generated.py"
    py_path.write_text(code_text, encoding="utf-8")

    docx_path = out_dir / "Practice_Attempt_4_Code.docx"
    save_word_with_code(code_text, docx_path)

    # Send code to the Executor to run
    code_executor.initiate_chat(
        code_writer,
        message="Please execute the code block below:\n```python\n" + code_text + "\n```",
    )

    print(f"\nSaved script: {py_path}")
    print(f"Saved Word doc (submit this): {docx_path}")
    # Ask Executor to run the code block
    code_executor.initiate_chat(
        code_writer,
        message="Please execute the code block below:\n```python\n" + code_text + "\n```",
    )

    print(f"\nSaved script: {py_path}")
    print(f"Saved Word doc (submit this): {docx_path}")