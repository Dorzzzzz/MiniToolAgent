"""Smoke tests for the MiniToolAgent python_repl tool.

Run from the project root:

    python scripts/test_python_repl.py

The script uses only the standard library so it can run in the same minimal
environment as the project itself.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from minitoolagent.tools.python_repl import MAX_OUTPUT_LEN, PythonREPLTool


@dataclass
class TestCase:
    name: str
    run: Callable[[], None]


def assert_contains(text: str, expected: str) -> None:
    if expected not in text:
        raise AssertionError(f"Expected output to contain {expected!r}, got:\n{text}")


def test_metadata_and_schema() -> None:
    tool = PythonREPLTool()
    schema = tool.get_schema()

    assert tool.name == "python_repl"
    assert schema["name"] == "python_repl"
    assert schema["parameters"]["required"] == ["code"]
    assert schema["parameters"]["properties"]["code"]["type"] == "string"


def test_stdout_capture() -> None:
    tool = PythonREPLTool(timeout=5)
    output = tool.execute(
        code="""
total = sum(i * i for i in range(6))
print("sum_squares =", total)
"""
    )

    assert_contains(output, "[stdout]")
    assert_contains(output, "sum_squares = 55")


def test_multiline_stdout_capture() -> None:
    tool = PythonREPLTool(timeout=5)
    output = tool.execute(
        code="""
for i in range(3):
    print(f"line-{i}")
"""
    )

    assert_contains(output, "line-0")
    assert_contains(output, "line-1")
    assert_contains(output, "line-2")


def test_stderr_capture_for_exceptions() -> None:
    tool = PythonREPLTool(timeout=5)
    output = tool.execute(
        code="""
print("before error")
raise ValueError("intentional failure")
"""
    )

    assert_contains(output, "[stdout]")
    assert_contains(output, "before error")
    assert_contains(output, "[stderr]")
    assert_contains(output, "ValueError: intentional failure")


def test_no_output_hint() -> None:
    tool = PythonREPLTool(timeout=5)
    output = tool.execute(code="x = 1 + 1")

    assert output == "[No output produced. Did you forget to print()?]"


def test_empty_code_rejected() -> None:
    tool = PythonREPLTool(timeout=5)
    assert tool.execute(code="   \n\t") == "Error: empty code."


def test_output_truncation() -> None:
    tool = PythonREPLTool(timeout=5)
    output = tool.execute(code="print('x' * 2500)")

    assert len(output) > MAX_OUTPUT_LEN
    assert len(output) < 2200
    assert_contains(output, "... [truncated]")


def test_timeout() -> None:
    tool = PythonREPLTool(timeout=1)
    output = tool.execute(
        code="""
import time
time.sleep(5)
print("should not reach here")
"""
    )

    assert output == "Error: code execution timed out after 1s."


def build_cases(skip_timeout: bool) -> list[TestCase]:
    cases = [
        TestCase("metadata/schema", test_metadata_and_schema),
        TestCase("stdout capture", test_stdout_capture),
        TestCase("multiline stdout capture", test_multiline_stdout_capture),
        TestCase("stderr capture for exceptions", test_stderr_capture_for_exceptions),
        TestCase("no-output hint", test_no_output_hint),
        TestCase("empty-code rejection", test_empty_code_rejected),
        TestCase("output truncation", test_output_truncation),
    ]
    if not skip_timeout:
        cases.append(TestCase("timeout", test_timeout))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the MiniToolAgent python_repl tool.")
    parser.add_argument(
        "--skip-timeout",
        action="store_true",
        help="Skip the timeout case if you need the quickest possible smoke test.",
    )
    args = parser.parse_args()

    failures: list[tuple[str, Exception]] = []
    cases = build_cases(skip_timeout=args.skip_timeout)

    print(f"Running {len(cases)} python_repl tests...\n")
    for case in cases:
        try:
            case.run()
        except Exception as exc:
            failures.append((case.name, exc))
            print(f"[FAIL] {case.name}: {exc}")
        else:
            print(f"[PASS] {case.name}")

    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        return 1

    print("\nAll python_repl tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
