"""Evaluate a smolagents math agent on AIME25 and HMMT.

This script mirrors the MiniToolAgent math evaluation setup as closely as is
reasonable while letting smolagents own the agent loop and code execution.
It reuses:
- the same LLM config file format as MiniToolAgent;
- the same processed datasets;
- the same answer-matching logic.

Example:
    conda activate cbev35
    python others/run_smolagents_math.py --dataset all --log-root logs/smolagents_math
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOLAGENTS_SRC = PROJECT_ROOT / "smolagents" / "src"

sys.path.insert(0, str(PROJECT_ROOT))

from minitoolagent.config import Config
from minitoolagent.evaluate import load_dataset, strip_asymptote
from minitoolagent.models import ChatModel as MiniToolChatModel
from minitoolagent.parsing import answers_match

logger = logging.getLogger(__name__)

DATASETS = {
    "aime25": PROJECT_ROOT / "data" / "processed" / "aime25.jsonl",
    "hmmt": PROJECT_ROOT / "data" / "processed" / "hmmt_feb_2025.jsonl",
}

SMOL_MATH_TASK_PROMPT = """\
Solve the following math competition problem step by step.

## Strategy
1. Analyze the problem type: number theory, algebra, geometry, combinatorics,
   probability, or another competition-math area.
2. Plan the mathematical approach before writing code.
3. Use Python for exact computation, enumeration, symbolic algebra, or
   verification. Avoid floating-point arithmetic when an exact answer is needed.
4. Use sympy for symbolic algebra and fractions.Fraction for exact rationals.
5. Verify the result with a sanity check or alternate method before submitting.
6. Finish by calling final_answer() with only the requested value.

## Available Libraries
- sympy: symbols, solve, simplify, expand, factor, factorint, isprime,
  Rational, sqrt, pi, cos, sin, tan, acos, asin, atan2, binomial, factorial,
  gcd, lcm, geometry helpers
- fractions: Fraction
- math: factorial, comb, gcd, isqrt, log, sqrt, pi
- itertools, collections, statistics, random, re, numpy, scipy

## Answer Format Rules
- Give the exact answer; never use a decimal approximation unless the problem
  asks for one.
- Integers: output as a plain integer, e.g. 588.
- Fractions: output as p/q in lowest terms, e.g. 7/18.
- Radicals: output as a*sqrt(b) or a + b*sqrt(c), e.g. 10*sqrt(3).
- AIME answers are integers from 0 to 999.
- HMMT answers may be fractions, radicals, or expressions involving pi.
- Use sympy.simplify() or sympy.nsimplify() to obtain a clean exact form.

## Examples

Example 1 - Number Theory
Problem: Find the number of positive divisors of 9!.

Thought: I will compute the prime factorization of 9! exactly, then multiply
the exponent-plus-one factors.
```python
import math
from sympy import factorint

n = math.factorial(9)
factors = factorint(n)
num_div = 1
for exp in factors.values():
    num_div *= exp + 1
print(n, factors, num_div)
final_answer("160")
```

Example 2 - Algebra
Problem: Find the sum of the real roots of 2x^3 - 11x^2 + 17x - 6 = 0.

Thought: I will solve the polynomial exactly and sum the real roots.
```python
from sympy import symbols, solve, simplify

x = symbols("x")
roots = solve(2*x**3 - 11*x**2 + 17*x - 6, x)
real_roots = [r for r in roots if r.is_real]
print(roots, real_roots, simplify(sum(real_roots)))
final_answer("11/2")
```

Example 3 - Combinatorics
Problem: How many 4-digit numbers whose digits are each from 1 to 9 have digit
sum exactly 10?

Thought: I will enumerate all digit tuples exactly and count the valid ones.
```python
count = 0
for a in range(1, 10):
    for b in range(1, 10):
        for c in range(1, 10):
            d = 10 - a - b - c
            if 1 <= d <= 9:
                count += 1
print(count)
final_answer("84")
```

Now solve the target problem. Use one or more Python code blocks as needed.
Call final_answer() only when you are confident.

Problem: {question}
"""


@dataclass
class EvalRecord:
    idx: int
    question: str
    gold: str
    predicted: str = ""
    correct: bool = False
    steps: int = 0
    error: str = ""
    attempts: int = 0


def import_smolagents_classes() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import smolagents, preferring the active environment installation."""
    try:
        from smolagents import ChatMessage, CodeAgent, LogLevel, MessageRole, Model, TokenUsage

        return CodeAgent, LogLevel, Model, ChatMessage, MessageRole, TokenUsage
    except (ImportError, ModuleNotFoundError) as first_error:
        sys.modules.pop("smolagents", None)
        if SMOLAGENTS_SRC.exists():
            sys.path.insert(0, str(SMOLAGENTS_SRC))
            try:
                from smolagents import ChatMessage, CodeAgent, LogLevel, MessageRole, Model, TokenUsage

                return CodeAgent, LogLevel, Model, ChatMessage, MessageRole, TokenUsage
            except (ImportError, ModuleNotFoundError) as second_error:
                raise RuntimeError(
                    "Could not import smolagents. Activate the cbev35 environment "
                    "or install smolagents dependencies, then rerun this script. "
                    f"Initial import error: {first_error}. Local source import error: {second_error}."
                ) from second_error
        raise RuntimeError(
            "Could not import smolagents. Activate the cbev35 environment "
            f"or install smolagents. Import error: {first_error}."
        ) from first_error


def build_model(
    config: Config,
    smol_model_cls: Any,
    smol_chat_message_cls: Any,
    smol_message_role_cls: Any,
    smol_token_usage_cls: Any,
) -> Any:
    """Build a smolagents model adapter backed by MiniToolAgent's ChatModel.

    This intentionally keeps the LLM call path identical to scripts/run_math.py:
    Config.from_yaml(...) -> minitoolagent.models.ChatModel.chat(...).
    """

    class MiniToolAgentChatAdapter(smol_model_cls):
        def __init__(self, cfg: Config):
            super().__init__(model_id=cfg.model_name, flatten_messages_as_text=True)
            self.config = cfg
            self.chat_model = MiniToolChatModel(cfg)
            self._last_prompt_tokens = 0
            self._last_completion_tokens = 0

        def generate(
            self,
            messages: list[Any],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, Any] | None = None,
            tools_to_call_from: list[Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del response_format, tools_to_call_from, kwargs
            before_prompt = self.chat_model.total_prompt_tokens
            before_completion = self.chat_model.total_completion_tokens

            content = self.chat_model.chat(
                messages=_smol_messages_to_openai_dicts(messages, smol_message_role_cls),
                stop=stop_sequences,
            )

            self._last_prompt_tokens = self.chat_model.total_prompt_tokens - before_prompt
            self._last_completion_tokens = self.chat_model.total_completion_tokens - before_completion
            return smol_chat_message_cls(
                role=smol_message_role_cls.ASSISTANT,
                content=content,
                token_usage=smol_token_usage_cls(
                    input_tokens=self._last_prompt_tokens,
                    output_tokens=self._last_completion_tokens,
                ),
            )

        @property
        def token_usage(self) -> dict[str, int]:
            usage = self.chat_model.token_usage
            return {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }

    return MiniToolAgentChatAdapter(config)


def build_agent(code_agent_cls: Any, model: Any, max_steps: int, verbosity_level: Any) -> Any:
    """Create the smolagents CodeAgent used for math tasks."""
    return code_agent_cls(
        tools=[],
        model=model,
        max_steps=max_steps,
        additional_authorized_imports=[
            "sympy",
            "fractions",
            "numpy",
            "scipy",
            "math",
            "itertools",
            "collections",
            "statistics",
            "random",
            "re",
        ],
        return_full_result=True,
        verbosity_level=verbosity_level,
        code_block_tags="markdown",
    )


def _smol_messages_to_openai_dicts(messages: list[Any], message_role_cls: Any) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", "")
        role_value = getattr(role, "value", role)

        if role_value == message_role_cls.SYSTEM.value:
            openai_role = "system"
        elif role_value == message_role_cls.ASSISTANT.value or role_value == message_role_cls.TOOL_CALL.value:
            openai_role = "assistant"
        else:
            openai_role = "user"

        converted.append({"role": openai_role, "content": _flatten_smol_content(content)})
    return converted


def _flatten_smol_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def run_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    agent: Any,
    limit: int | None,
    resume: bool,
    item_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]

    results_file = output_dir / "results.jsonl"
    existing = _load_existing_records(results_file) if resume else {}

    records: list[EvalRecord] = []
    skipped_existing = 0
    rerun_failed_existing = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    start_time = time.time()

    with results_file.open("w", encoding="utf-8") as f_out:
        for i, item in enumerate(dataset, start=1):
            idx = int(item.get("idx", i))
            question = str(item["question"])
            gold = str(item["answer"])

            logger.info("[%s %d/%d] idx=%s", dataset_name, i, len(dataset), idx)

            old_record = existing.get(idx)
            if old_record and _can_resume_record(old_record, question, gold):
                record = EvalRecord(
                    idx=idx,
                    question=old_record.get("question", question),
                    gold=old_record.get("gold", gold),
                    predicted=old_record.get("predicted", ""),
                    correct=bool(old_record.get("correct", False)),
                    steps=int(old_record.get("steps", 0) or 0),
                    error=old_record.get("error", ""),
                    attempts=int(old_record.get("attempts", 1) or 1),
                )
                skipped_existing += 1
                records.append(record)
                f_out.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n")
                f_out.flush()
                continue
            if old_record and old_record.get("error"):
                rerun_failed_existing += 1

            task = SMOL_MATH_TASK_PROMPT.format(question=strip_asymptote(question))
            record = EvalRecord(idx=idx, question=question, gold=gold)

            for attempt in range(1, max(item_retries, 0) + 2):
                record.attempts = attempt
                try:
                    result = agent.run(task, reset=True, return_full_result=True)
                    predicted = "" if result.output is None else str(result.output).strip()
                    record.predicted = predicted
                    record.steps = _count_action_steps(result.steps)
                    record.correct = answers_match(predicted, gold)
                    record.error = ""

                    _add_token_usage(token_usage, result.token_usage)
                    _save_trajectory(output_dir / f"trajectory_{idx}.json", result)
                    break
                except Exception as exc:
                    record.predicted = ""
                    record.correct = False
                    record.error = f"{type(exc).__name__}: {exc}"
                    if attempt <= item_retries:
                        sleep_seconds = retry_sleep * attempt
                        logger.warning(
                            "idx=%s attempt %d/%d failed: %s. Retrying in %.1fs",
                            idx,
                            attempt,
                            item_retries + 1,
                            exc,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    else:
                        logger.exception("idx=%s failed after %d attempt(s)", idx, attempt)

            records.append(record)
            f_out.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n")
            f_out.flush()

            correct_so_far = sum(1 for r in records if r.correct)
            logger.info(
                "  -> predicted=%r gold=%r correct=%s running=%d/%d",
                record.predicted[:80],
                gold[:80],
                record.correct,
                correct_so_far,
                len(records),
            )

    total_time = time.time() - start_time
    summary = {
        "framework": "smolagents",
        "agent_type": "CodeAgent",
        "dataset": dataset_name,
        "total": len(records),
        "correct": sum(1 for r in records if r.correct),
        "failed": sum(1 for r in records if r.error),
        "accuracy": round(sum(1 for r in records if r.correct) / max(len(records), 1), 4),
        "avg_steps": round(sum(r.steps for r in records) / max(len(records), 1), 2),
        "total_time_seconds": round(total_time, 1),
        "token_usage": token_usage,
        "resume_enabled": resume,
        "skipped_existing": skipped_existing,
        "rerun_failed_existing": rerun_failed_existing,
        "item_retries": item_retries,
        "retry_sleep_seconds": retry_sleep,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_summary(summary)
    return summary


def _load_existing_records(results_file: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not results_file.exists():
        return records
    with results_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[int(record["idx"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Ignoring malformed line %d in %s: %s", line_no, results_file, exc)
    return records


def _can_resume_record(record: dict[str, Any], question: str, gold: str) -> bool:
    if record.get("question") != question or str(record.get("gold")) != str(gold):
        return False
    if record.get("error"):
        return False
    predicted = record.get("predicted")
    return predicted is not None and str(predicted).strip() != ""


def _record_to_dict(record: EvalRecord) -> dict[str, Any]:
    return {
        "idx": record.idx,
        "question": record.question,
        "gold": record.gold,
        "predicted": record.predicted,
        "correct": record.correct,
        "steps": record.steps,
        "error": record.error,
        "attempts": record.attempts,
    }


def _count_action_steps(steps: list[dict[str, Any]]) -> int:
    return sum(1 for step in steps if "step_number" in step)


def _add_token_usage(total: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    usage_dict = usage.dict() if hasattr(usage, "dict") else dict(usage)
    total["input_tokens"] += int(usage_dict.get("input_tokens", 0) or 0)
    total["output_tokens"] += int(usage_dict.get("output_tokens", 0) or 0)
    total["total_tokens"] += int(usage_dict.get("total_tokens", 0) or 0)


def _save_trajectory(path: Path, result: Any) -> None:
    payload = {
        "output": result.output,
        "state": result.state,
        "token_usage": result.token_usage.dict() if result.token_usage else None,
        "timing": result.timing.dict() if result.timing else None,
        "steps": result.steps,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print(f"SMOLAGENTS SUMMARY: {summary['dataset']}")
    print("=" * 50)
    print(f"Total:      {summary['total']}")
    print(f"Correct:    {summary['correct']}")
    print(f"Failed:     {summary['failed']}")
    print(f"Accuracy:   {summary['accuracy']:.2%}")
    print(f"Avg steps:  {summary['avg_steps']:.1f}")
    print(f"Total time: {summary['total_time_seconds']:.0f}s")
    print(f"Tokens:     {summary['token_usage']}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate smolagents CodeAgent on math datasets")
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--item-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs" / "smolagents_math"),
        help="Parent directory for dataset logs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment folder under --log-root, e.g. qwen2.5_72b.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["off", "error", "info", "debug"],
        default="error",
        help="smolagents console verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    CodeAgent, LogLevel, Model, ChatMessage, MessageRole, TokenUsage = import_smolagents_classes()
    config = Config.from_yaml(args.config)
    model = build_model(
        config=config,
        smol_model_cls=Model,
        smol_chat_message_cls=ChatMessage,
        smol_message_role_cls=MessageRole,
        smol_token_usage_cls=TokenUsage,
    )
    verbosity = {
        "off": LogLevel.OFF,
        "error": LogLevel.ERROR,
        "info": LogLevel.INFO,
        "debug": LogLevel.DEBUG,
    }[args.verbosity]
    agent = build_agent(
        code_agent_cls=CodeAgent,
        model=model,
        max_steps=args.max_steps,
        verbosity_level=verbosity,
    )

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    log_root = Path(args.log_root)
    if args.run_name:
        log_root = log_root / args.run_name

    summaries = []
    for dataset_name in selected:
        summaries.append(
            run_dataset(
                dataset_name=dataset_name,
                dataset_path=DATASETS[dataset_name],
                output_dir=log_root / dataset_name,
                agent=agent,
                limit=args.limit,
                resume=not args.no_resume,
                item_retries=args.item_retries,
                retry_sleep=args.retry_sleep,
            )
        )

    if len(summaries) > 1:
        with (log_root / "summary_all.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
