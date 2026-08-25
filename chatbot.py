"""
chatbot.py
Natural-language Q&A over your traffic video's results, powered by Groq
(free, no credit card required -- see https://console.groq.com/keys).

Unique feature: instead of dumping the whole dataset into the LLM and
hoping it "reads" the numbers correctly (unreliable for counting/math),
this uses a text-to-pandas approach:

    1. The LLM reads the table's schema (not the raw data) and writes a
       single pandas expression that would answer the question.
    2. We validate that expression is safe (no imports, no file access,
       no dunder attributes -- just data operations on the dataframe)
       and execute it ourselves.
    3. The LLM turns the actual computed result into a friendly answer.

This keeps numeric answers accurate (real pandas math, not LLM guessing)
while still letting you ask in plain English/Roman Urdu/etc.

Requires a free Groq API key: https://console.groq.com/keys
Set it as an environment variable: GROQ_API_KEY
"""

import ast
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # picks up GROQ_API_KEY from a local .env file, per README setup instructions

MODEL = "openai/gpt-oss-20b"  # free tier, currently supported on Groq (llama-3.3 models were deprecated)

# ---------------------------------------------------------------------------
# Safety: only allow a narrow, read-only subset of Python expressions.
# No imports, no dunder/attribute tricks, no builtins beyond df/pd.
# ---------------------------------------------------------------------------
_ALLOWED_NODES = (
    ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.keyword,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert,
)
_ALLOWED_NAMES = {"df", "pd", "len"}

# Attribute/method names that read from or write to disk, or execute
# arbitrary code/strings, even though they don't start with "_" and so
# would otherwise slip past the dunder check (e.g. pd.read_csv("/etc/passwd"),
# df.to_pickle(...), pd.read_pickle(...) -- the latter can even trigger
# arbitrary code execution via unpickling).
_BLOCKED_ATTRS = {
    "read_csv", "read_pickle", "read_excel", "read_json", "read_html",
    "read_sql", "read_parquet", "read_feather", "read_table", "read_clipboard",
    "to_csv", "to_pickle", "to_excel", "to_json", "to_html", "to_sql",
    "to_parquet", "to_feather", "to_clipboard",
    "eval", "query", "apply", "applymap", "agg", "aggregate", "transform",
    "pipe", "eval_expr",
}


def _is_safe_expression(code: str) -> bool:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            return False
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False
            if node.attr in _BLOCKED_ATTRS:
                return False

    return True


def _schema_description(df: pd.DataFrame) -> str:
    lines = ["Columns available in `df` (one row per tracked vehicle):"]
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    lines.append("\nSample rows:")
    lines.append(df.head(5).to_string(index=False))
    return "\n".join(lines)


def _get_client(api_key: str = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No Groq API key found. Set the GROQ_API_KEY environment "
            "variable, or pass api_key= explicitly. Get a free key at "
            "https://console.groq.com/keys (no credit card required)."
        )
    return Groq(api_key=key)


def _generate_code(client: Groq, code_prompt: str) -> str:
    code_response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        # reasoning_effort isn't a named param in the pinned groq==0.11.0 SDK
        # (added to the API after that SDK release) -- extra_body forwards it
        # straight into the request body instead.
        extra_body={"reasoning_effort": "low"},
        messages=[{"role": "user", "content": code_prompt}],
    )
    content = code_response.choices[0].message.content or ""
    code = content.strip().strip("`").strip()
    # strip a leading "python" if the model added a language hint out of habit
    if code.lower().startswith("python\n"):
        code = code[7:].strip()
    return code


def _format_structured_result(result) -> str:
    """
    Deterministically renders a multi-value pandas result (DataFrame or Series)
    as a markdown list, with every number/label copied directly from the real
    data -- no LLM involved, so it can't be mis-transcribed or hallucinated.
    """
    if isinstance(result, pd.DataFrame):
        lines = []
        for _, row in result.iterrows():
            parts = [f"**{col}**: {row[col]}" for col in result.columns]
            lines.append("- " + ", ".join(parts))
        return "\n".join(lines)

    if isinstance(result, pd.Series):
        lines = [f"- **{idx}**: {val}" for idx, val in result.items()]
        return "\n".join(lines)

    return str(result)


def ask(question: str, df: pd.DataFrame, api_key: str = None) -> dict:
    """
    Ask a natural-language question about the processed vehicle data.

    Returns a dict:
        {
            "answer": str,          # natural-language answer
            "pandas_code": str,     # the expression that was run
            "raw_result": Any,      # the raw computed value
        }
    Raises ValueError if the question can't be safely translated, so the
    caller can show the user a clear error instead of a wrong answer.
    """
    client = _get_client(api_key)
    schema = _schema_description(df)

    code_prompt = f"""You answer questions about a pandas DataFrame called `df`.

{schema}

Notes:
- `crossed_line` is True if the vehicle crossed the counting line (i.e. was "counted").
- `crossing_direction` says which way it was moving when it crossed (if it crossed).
- `crossing_time_sec` / `first_seen_sec` / `last_seen_sec` are seconds since the video started.
- Speeds are in km/h.
- If the question asks to identify specific vehicle(s) (e.g. "the fastest one",
  "top 3"), make the result include their `track_id` and `label` columns, not
  just the numeric value alone, so it's clear which vehicle each number belongs to.

Write ONE single-line Python expression (not a statement, no assignment, no print)
that uses only `df` and `pandas` (as `pd`) to answer this question:

"{question}"

Respond with ONLY the expression, nothing else. No explanation, no markdown, no backticks."""

    code = _generate_code(client, code_prompt)
    if not code:
        # Empty response is a rare transient hiccup, not a real answer -- retry once
        code = _generate_code(client, code_prompt)

    if not code:
        raise ValueError(
            "The model returned an empty response. This is usually a transient "
            "hiccup -- please try asking again."
        )

    if not _is_safe_expression(code):
        raise ValueError(
            f"Generated code failed the safety check and was not run:\n{code}"
        )

    try:
        result = eval(code, {"__builtins__": {}}, {"df": df, "pd": pd, "len": len})
    except Exception as e:
        raise ValueError(f"Generated code raised an error: {e}\nCode was: {code}")

    # Multi-value results (a DataFrame or Series -- e.g. "top 3 fastest vehicles")
    # are formatted directly from the real data below, with NO LLM step touching
    # the actual numbers/labels -- this is what fixes the mismatched-track-ID bug,
    # since an LLM asked to freely narrate several rows can transpose or invent
    # values, but it can't when it's only wrapping pre-formatted, already-correct text.
    if isinstance(result, (pd.DataFrame, pd.Series)):
        structured = _format_structured_result(result)
        if isinstance(result, pd.DataFrame) and result.empty:
            answer = "No vehicles matched that."
        else:
            answer = structured
        return {"answer": answer, "pandas_code": code, "raw_result": result}

    # Scalar results (a single number/string/bool) -- proven reliable to hand to
    # the LLM for natural-language phrasing, since there's only one value to relay.
    answer_prompt = f"""The question was: "{question}"

Running this pandas code: `{code}`
Produced this result: {result!r}

Give a short, direct, natural-language answer (1-2 sentences) using that exact result.
This is a single already-computed value -- state it plainly, don't round it
differently, and don't attach a vehicle type (car/truck/bus) unless the result
itself is or contains that label -- if it's just a bare number, call it a
"vehicle", not a "car".
Don't mention pandas, code, or dataframes -- just answer like a helpful analyst."""

    answer_response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        temperature=0,  # precision matters here -- this call relays an already-computed
                         # number, so no creativity is needed
        extra_body={"reasoning_effort": "low"},
        messages=[{"role": "user", "content": answer_prompt}],
    )
    answer = (answer_response.choices[0].message.content or "").strip()
    if not answer:
        answer = f"The result is: {result!r}"

    return {"answer": answer, "pandas_code": code, "raw_result": result}