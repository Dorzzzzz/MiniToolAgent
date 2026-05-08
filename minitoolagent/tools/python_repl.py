from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import queue

from .base import Tool, ToolParameter
from .._vendor.local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    MAX_EXECUTION_TIME_SECONDS,
    InterpreterError,
    evaluate_python_code,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_LEN = 2000
DEFAULT_TIMEOUT = MAX_EXECUTION_TIME_SECONDS


def _pickleable_state(state: dict) -> dict:
    safe_state = {}
    for key, value in state.items():
        try:
            pickle.dumps(value)
        except Exception:
            continue
        safe_state[key] = value
    return safe_state


def _execute_python_code_worker(
    result_queue,
    code: str,
    state: dict,
    authorized_imports: list[str],
) -> None:
    try:
        output, is_final = evaluate_python_code(
            code,
            static_tools=dict(BASE_PYTHON_TOOLS),
            state=state,
            authorized_imports=authorized_imports,
            timeout_seconds=None,
        )
        result_queue.put(("ok", output, is_final, _pickleable_state(state)))
    except InterpreterError as e:
        result_queue.put(("interpreter_error", str(e), None, _pickleable_state(state)))
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        result_queue.put(("execution_error", message, None, _pickleable_state(state)))


class PythonREPLTool(Tool):
    """Execute Python code in a killable subprocess around the AST sandbox."""

    name = "python_repl"
    description = (
        "Execute a Python code snippet in a sandboxed in-process interpreter. "
        "Use for symbolic math, exact arithmetic, enumeration, and verification. "
        "Both print() output and the final expression value are returned. "
        "Available imports: math, sympy, fractions, numpy, scipy, statistics, "
        "itertools, collections, random, re, datetime. "
        "Variables persist across multiple calls within the same task. "
        "os/subprocess/sys are forbidden."
    )
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description="The Python code to execute.",
        ),
    ]

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        additional_authorized_imports: list[str] | None = None,
    ):
        self.timeout = timeout
        self.authorized_imports = list(
            set(BASE_BUILTIN_MODULES) | set(additional_authorized_imports or [])
        )
        self._state: dict = {}

    def reset(self) -> None:
        """Clear persistent state between tasks."""
        self._state = {}

    def execute(self, code: str = "", **kwargs) -> str:
        if not code.strip():
            return "Error: empty code."

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_execute_python_code_worker,
            args=(result_queue, code, dict(self._state), self.authorized_imports),
        )
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            result_queue.close()
            result_queue.join_thread()
            return f"Error: code execution exceeded the maximum execution time of {self.timeout} seconds."

        try:
            status, payload, _is_final, new_state = result_queue.get_nowait()
        except queue.Empty:
            return f"Error executing code: subprocess exited with code {process.exitcode} without returning a result."
        finally:
            result_queue.close()
            result_queue.join_thread()

        self._state = new_state
        if status == "interpreter_error":
            return f"Error: {payload}"
        if status == "execution_error":
            return f"Error executing code: {payload}"

        output = payload

        prints = str(self._state.get("_print_outputs", "")).strip()
        parts: list[str] = []
        if prints:
            parts.append(f"[stdout]\n{prints}")
        if output is not None:
            parts.append(f"[result]\n{output!r}")
        if not parts:
            parts.append("[No output produced. Did you forget to print()?]")

        result = "\n".join(parts)
        if len(result) > MAX_OUTPUT_LEN:
            result = result[:MAX_OUTPUT_LEN] + "\n... [truncated]"
        return result
