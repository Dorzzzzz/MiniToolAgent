"""Simulate calling the MiniToolAgent python_repl tool with Python code.

Examples:

    python scripts/simulate_python_repl.py
    python scripts/simulate_python_repl.py --code "print(1 + 2)"
    python scripts/simulate_python_repl.py --file path/to/example.py
    python scripts/simulate_python_repl.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from minitoolagent.tools.python_repl import PythonREPLTool


EXAMPLE_SNIPPETS = [
    (
        "basic calculation",
        """
print("hello from python_repl")
print("2 + 3 =", 2 + 3)
""",
    ),
    (
        "loop and function",
        """
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

for n in range(1, 6):
    print(n, factorial(n))
""",
    ),
    (
        "debug an error",
        """
numbers = [10, 20, 30]
print("numbers =", numbers)
print(numbers[5])
""",
    ),
]


def run_tool(tool: PythonREPLTool, code: str, title: str = "custom code") -> None:
    action = {
        "tool": "python_repl",
        "args": {
            "code": code.strip(),
        },
    }

    print("=" * 72)
    print(f"Case: {title}")
    print("-" * 72)
    print("Action:")
    print(json.dumps(action, indent=2, ensure_ascii=False))
    print("\nObservation:")
    print(tool.execute(code=code))
    print()


def run_examples(tool: PythonREPLTool) -> None:
    for title, code in EXAMPLE_SNIPPETS:
        run_tool(tool, code, title)


def run_interactive(tool: PythonREPLTool) -> None:
    print("Interactive python_repl simulation")
    print("Paste Python code. Type :run to execute, :clear to reset, or :quit to exit.")
    print()

    buffer: list[str] = []
    while True:
        try:
            line = input("py> " if not buffer else "... ")
        except EOFError:
            print()
            break

        command = line.strip()
        if command == ":quit":
            break
        if command == ":clear":
            buffer.clear()
            print("Buffer cleared.")
            continue
        if command == ":run":
            run_tool(tool, "\n".join(buffer), "interactive input")
            buffer.clear()
            continue

        buffer.append(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate how an agent calls the python_repl tool."
    )
    parser.add_argument("--timeout", type=int, default=5, help="Execution timeout in seconds.")
    parser.add_argument("--code", help="Run one Python code snippet.")
    parser.add_argument("--file", type=Path, help="Run Python code loaded from a file.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter a tiny prompt where :run sends your code to python_repl.",
    )
    args = parser.parse_args()

    tool = PythonREPLTool(timeout=args.timeout)

    if args.code is not None:
        run_tool(tool, args.code)
        return 0

    if args.file is not None:
        code = args.file.read_text(encoding="utf-8")
        run_tool(tool, code, str(args.file))
        return 0

    if args.interactive:
        run_interactive(tool)
        return 0

    run_examples(tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
