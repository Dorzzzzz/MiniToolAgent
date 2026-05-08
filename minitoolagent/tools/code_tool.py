from __future__ import annotations

import logging
import re

from .base import Tool, ToolParameter
from ..models import ChatModel
from ..parsing import extract_code_block
from .._vendor.local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    MAX_EXECUTION_TIME_SECONDS,
    InterpreterError,
    evaluate_python_code,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_LEN = 3000
DEFAULT_TIMEOUT = MAX_EXECUTION_TIME_SECONDS

CODE_GEN_SYSTEM_PROMPT = """\
You are a Python code generator. Given a natural-language description of a \
computation, output exactly ONE fenced Python code block and nothing else.

Rules:
- Use exact arithmetic: sympy for symbolic algebra, fractions.Fraction for \
exact rationals. Avoid floating-point when an exact answer is needed.
- Print all important intermediate values and the final result with print().
- Do NOT include any explanation outside the code block.
- Available libraries: sympy, fractions, math, itertools, collections, \
random, re, statistics, datetime, numpy, scipy.

Output format (strictly):
```python
<your code here>
```
"""


def _extract_code_from_response(response: str) -> str:
    """Try multiple strategies to extract Python code from the LLM response."""
    code = extract_code_block(response)
    if code:
        return code

    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    lines = response.strip().splitlines()
    code_lines = [l for l in lines if not l.startswith("#") or "import" in l]
    if code_lines:
        return "\n".join(code_lines)

    return ""


class CodeTool(Tool):
    """Translate a natural-language computation description into Python code,
    execute it in a sandboxed interpreter, and return both the code and results."""

    name = "codetool"
    description = (
        "Describe a computation in plain English and this tool will generate "
        "Python code, execute it, and return the results. "
        "Use for symbolic math, exact arithmetic, enumeration, and verification. "
        "Available libraries: math, sympy, fractions, numpy, scipy, statistics, "
        "itertools, collections, random, re, datetime. "
        "Variables persist across multiple calls within the same task."
    )
    parameters = [
        ToolParameter(
            name="description",
            type="string",
            description="Natural language description of the computation to perform.",
        ),
    ]

    def __init__(
        self,
        model: ChatModel,
        timeout: int = DEFAULT_TIMEOUT,
        additional_authorized_imports: list[str] | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.authorized_imports = list(
            set(BASE_BUILTIN_MODULES) | set(additional_authorized_imports or [])
        )
        self._state: dict = {}

    def reset(self) -> None:
        """Clear persistent state between tasks."""
        self._state = {}

    def execute(self, description: str = "", **kwargs) -> str:
        if not description.strip():
            return "Error: empty description."

        # Step 1: Generate code via LLM
        messages = [
            {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        try:
            response = self.model.chat(messages, temperature=0.0)
        except Exception as e:
            return f"Error calling LLM for code generation: {e}"

        code = _extract_code_from_response(response)
        if not code:
            return (
                f"Error: could not extract Python code from LLM response.\n"
                f"[raw_response]\n{response[:1000]}"
            )

        # Step 2: Execute generated code in sandbox
        try:
            output, _is_final = evaluate_python_code(
                code,
                static_tools=dict(BASE_PYTHON_TOOLS),
                state=self._state,
                authorized_imports=self.authorized_imports,
                timeout_seconds=self.timeout,
            )
        except InterpreterError as e:
            return (
                f"[generated_code]\n{code}\n\n"
                f"[execution_error]\n{e}"
            )
        except Exception as e:
            return (
                f"[generated_code]\n{code}\n\n"
                f"[execution_error]\n{type(e).__name__}: {e}"
            )

        # Step 3: Format output
        prints = str(self._state.get("_print_outputs", "")).strip()
        parts: list[str] = [f"[generated_code]\n{code}"]
        if prints:
            parts.append(f"[stdout]\n{prints}")
        if output is not None:
            parts.append(f"[result]\n{output!r}")
        if not prints and output is None:
            parts.append("[No output produced. The description may need to "
                         "explicitly request printing results.]")

        result = "\n\n".join(parts)
        if len(result) > MAX_OUTPUT_LEN:
            result = result[:MAX_OUTPUT_LEN] + "\n... [truncated]"
        return result
