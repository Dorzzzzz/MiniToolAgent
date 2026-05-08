from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedAction:
    tool: str
    args: dict
    raw_thought: str = ""


def parse_llm_response(text: str) -> ParsedAction:
    """Parse an LLM response into a thought and an action.

    Expects format:
        Thought: ...
        Action: {"tool": "...", "args": {...}}
    """
    thought = extract_thought(text)
    action_json = extract_action_json(text)

    if action_json:
        tool = action_json.get("tool", "")
        args = action_json.get("args", {})
        if isinstance(args, str):
            args = {"query": args}
        return ParsedAction(tool=tool, args=args, raw_thought=thought)

    # Fallback: try to find tool name and args from malformed output
    fallback = _fallback_parse(text)
    if fallback:
        return ParsedAction(tool=fallback[0], args=fallback[1], raw_thought=thought)

    return ParsedAction(tool="", args={}, raw_thought=thought)


def extract_thought(text: str) -> str:
    match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_action_json(text: str) -> dict | None:
    """Extract the JSON action from the text, with multiple recovery strategies."""
    # Strategy 1: find JSON after "Action:"
    match = re.search(r"Action:\s*({.+})", text, re.DOTALL)
    if match:
        parsed = _try_parse_json(match.group(1))
        if parsed:
            return parsed

    # Strategy 2: find any JSON block containing "tool" (supports nested braces)
    for match in re.finditer(r"\{[^{}]*\"tool\".*\}", text, re.DOTALL):
        parsed = _try_parse_json(match.group(0))
        if parsed:
            return parsed

    # Strategy 3: find JSON in code blocks
    for match in re.finditer(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL):
        parsed = _try_parse_json(match.group(1))
        if parsed:
            return parsed

    # Strategy 4: extract tool name and args separately from text
    tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    if tool_match:
        tool_name = tool_match.group(1).strip()
        args = {}
        args_match = re.search(r'"args"\s*:\s*(\{[^}]*\})', text)
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass
        return {"tool": tool_name, "args": args}

    return None


def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON with progressive repair."""
    text = text.strip()
    # Remove trailing garbage after the JSON object
    depth = 0
    end_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx >= 0:
        text = text[: end_idx + 1]

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix single quotes
    try:
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: fix unquoted keys
    try:
        fixed = re.sub(r"(\w+)\s*:", r'"\1":', text)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 4: escape raw newlines inside JSON strings
    try:
        fixed = re.sub(r'(?<=")\s*\n\s*', r"\\n", text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    logger.debug("Failed to parse JSON: %s", text[:200])
    return None


def _fallback_parse(text: str) -> tuple[str, dict] | None:
    """Last-resort extraction for severely malformed output."""
    # Try to detect tool name
    tool_match = re.search(r"(?:tool|action)\s*[:=]\s*[\"']?(\w+)[\"']?", text, re.IGNORECASE)
    if not tool_match:
        return None

    tool_name = tool_match.group(1)
    args = {}

    # Try to extract query/code argument
    for key in ("query", "code", "answer"):
        pat = re.search(rf'["\']?{key}["\']?\s*[:=]\s*["\'](.+?)["\']', text, re.DOTALL)
        if pat:
            args[key] = pat.group(1)

    return (tool_name, args) if args else None


# ── Code block extraction ──────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """Extract Python code from markdown code blocks in the response."""
    # Match ```python ... ``` or ``` ... ```
    patterns = [
        r"```python\s*\n(.+?)```",
        r"```\s*\n(.+?)```",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to extract code from "code" field with multiline content
    match = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"\s*}', text, re.DOTALL)
    if match:
        code = match.group(1)
        code = code.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        if code.strip():
            return code.strip()

    return ""


# ── Final answer extraction ────────────────────────────────────────

def extract_final_answer(text: str) -> str | None:
    """Extract the final answer from agent output using multiple patterns."""
    patterns = [
        r"Final\s*Answer\s*:\s*(.+?)(?:\n|$)",
        r"\\boxed\{(.+?)\}",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        r"[Aa]nswer\s*:\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return normalize_answer(match.group(1).strip())
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().strip('"').strip("'").strip(".")
    answer = re.sub(r"\s+", " ", answer)
    # Remove leading $ and trailing $ (LaTeX)
    answer = answer.strip("$")
    # Remove \text{} wrapper
    answer = re.sub(r"\\text\{(.+?)\}", r"\1", answer)
    return answer.strip()


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(predicted).lower()
    gold_norm = normalize_answer(gold).lower()

    if pred_norm == gold_norm:
        return True

    # Numeric comparison (plain numbers and simple fractions)
    pred_num = _try_parse_number(pred_norm)
    gold_num = _try_parse_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6

    # Symbolic / LaTeX comparison (handles \frac, \sqrt, \cdot, etc.)
    pred_sym = _try_parse_symbolic(pred_norm)
    gold_sym = _try_parse_symbolic(gold_norm)
    if pred_sym is not None and gold_sym is not None:
        return abs(pred_sym - gold_sym) < 1e-6

    # Multi-value answers separated by commas (e.g. two roots)
    if "," in gold_norm:
        gold_parts = [p.strip() for p in gold_norm.split(",")]
        pred_parts = [p.strip() for p in pred_norm.split(",")]
        if len(gold_parts) == len(pred_parts):
            if all(answers_match(p, g) for p, g in zip(pred_parts, gold_parts)):
                return True
        # Order-insensitive set comparison
        gold_syms = [_try_parse_symbolic(p) for p in gold_parts]
        pred_syms = [_try_parse_symbolic(p) for p in pred_parts]
        if None not in gold_syms and None not in pred_syms:
            gold_rounded = sorted(round(v, 6) for v in gold_syms)
            pred_rounded = sorted(round(v, 6) for v in pred_syms)
            if gold_rounded == pred_rounded:
                return True

    # Containment fallback (for short gold answers)
    if len(gold_norm) > 2 and gold_norm in pred_norm:
        return True

    return False


def _latex_to_expr(text: str) -> str:
    """Convert simple LaTeX math notation to a sympy-parseable expression."""
    # \frac{a}{b} → (a)/(b)
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', text)
    # \sqrt{n} → sqrt(n)
    text = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', text)
    # \sqrt n (no braces) → sqrt(n)
    text = re.sub(r'\\sqrt\s+(\w+)', r'sqrt(\1)', text)
    # x^{n} → x**(n)
    text = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', text)
    # n! → factorial(n)
    text = re.sub(r'(\d+)!', r'factorial(\1)', text)
    # \cdot → *
    text = text.replace(r'\cdot', '*')
    # \pi → pi
    text = text.replace(r'\pi', 'pi')
    # Remove any remaining LaTeX backslash commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()


def _try_parse_symbolic(text: str) -> float | None:
    """Try to evaluate a (possibly LaTeX) math expression numerically via sympy."""
    try:
        from sympy import sympify, N as sympy_N
        expr_str = _latex_to_expr(text)
        if not expr_str:
            return None
        expr = sympify(expr_str, evaluate=True)
        return float(sympy_N(expr))
    except Exception:
        return None


def _try_parse_number(text: str) -> float | None:
    """Try to parse a string as a number, handling fractions."""
    text = text.strip().replace(",", "")
    # Handle fractions like 1/576
    frac_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", text)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        return num / den if den != 0 else None
    try:
        return float(text)
    except ValueError:
        return None
