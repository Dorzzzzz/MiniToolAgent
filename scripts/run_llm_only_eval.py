"""Run direct LLM-only evaluations on HotpotQA, AIME25, and HMMT.

This script is intentionally tool-free: each example is answered by one
OpenAI-compatible chat completion call, then scored against the gold answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minitoolagent.config import Config
from minitoolagent.evaluate import EvalSummary, load_dataset
from minitoolagent.models import ChatModel
from minitoolagent.parsing import answers_match, extract_final_answer, normalize_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    task_type: str


DATASETS = {
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        path=PROJECT_ROOT / "data" / "raw" / "hotpotqa200.jsonl",
        task_type="qa",
    ),
    "aime25": DatasetSpec(
        name="aime25",
        path=PROJECT_ROOT / "data" / "processed" / "aime25.jsonl",
        task_type="math",
    ),
    "hmmt": DatasetSpec(
        name="hmmt",
        path=PROJECT_ROOT / "data" / "processed" / "hmmt_feb_2025.jsonl",
        task_type="math",
    ),
}

SYSTEM_PROMPT = """\
You are an evaluation assistant. Answer using only the LLM call itself:
do not use external tools, browsing, code execution, or retrieval.
Keep any reasoning concise, and always end with exactly one line:
Final Answer: <your answer>
"""

QA_PROMPT = """\
Answer the following HotpotQA question using only your internal knowledge and reasoning.
Give a short factual answer, not a full sentence, whenever possible.

Question: {question}
"""

MATH_PROMPT = """\
Solve the following math contest problem using only your own reasoning.
The final answer should be the exact requested value, usually an integer.

Problem: {question}
"""


def build_messages(question: str, task_type: str) -> list[dict[str, str]]:
    prompt_template = MATH_PROMPT if task_type == "math" else QA_PROMPT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_template.format(question=question)},
    ]


def extract_prediction(raw_response: str, task_type: str) -> str:
    """Extract the answer to score while preserving a useful fallback."""
    answer = extract_final_answer(raw_response)
    if answer:
        return answer

    patterns = [
        r"(?:^|\n)\s*(?:\*\*)?Final\s*Answer(?:\*\*)?\s*:\s*(.+?)(?=\n|$)",
        r"(?:^|\n)\s*(?:\*\*)?Answer(?:\*\*)?\s*:\s*(.+?)(?=\n|$)",
        r"\\boxed\{([^{}]+)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_answer(_strip_markup(match.group(1)))

    nonempty_lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
    if not nonempty_lines:
        return ""

    last_line = _strip_markup(nonempty_lines[-1])
    if task_type == "math":
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", last_line.replace(",", ""))
        if numbers:
            return normalize_answer(numbers[-1])

    return normalize_answer(last_line)


def _strip_markup(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^`+|`+$", "", text)
    text = re.sub(r"^\*+|\*+$", "", text)
    return text.strip()


def evaluate_dataset(
    model: ChatModel,
    spec: DatasetSpec,
    output_dir: Path,
    limit: int | None,
    temperature: float | None,
    max_tokens: int | None,
    sleep_seconds: float,
    resume: bool,
    item_retries: int,
    retry_sleep: float,
) -> EvalSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(spec.path)
    if limit is not None:
        dataset = dataset[:limit]

    results_file = output_dir / "results.jsonl"
    existing_records = _load_existing_records(results_file) if resume else {}
    start_time = time.time()
    correct = 0
    failed = 0
    skipped_existing = 0
    rerun_failed_existing = 0
    processed = 0

    logging.info("Evaluating %s: %d examples", spec.name, len(dataset))
    token_usage_before = model.token_usage

    with open(results_file, "w", encoding="utf-8") as f_out:
        for i, item in enumerate(dataset, start=1):
            idx = item.get("idx", i)
            question = item["question"]
            gold = item["answer"]
            raw_response = ""
            predicted = ""
            error = ""
            is_correct = False

            logging.info("[%s %d/%d] %s", spec.name, i, len(dataset), question[:100])

            existing = existing_records.get(int(idx))
            if existing and _can_resume_record(existing, question, gold):
                f_out.write(json.dumps(existing, ensure_ascii=False) + "\n")
                f_out.flush()
                predicted = existing.get("predicted", "")
                is_correct = bool(existing.get("correct", False))
                error = existing.get("error", "")
                correct += int(is_correct)
                failed += int(bool(error))
                skipped_existing += 1
                processed += 1
                logging.info(
                    "  -> Skipped existing result for idx=%s | Predicted: %s | Gold: %s",
                    idx,
                    str(predicted)[:80],
                    str(gold)[:80],
                )
                continue
            if existing and existing.get("error"):
                rerun_failed_existing += 1
                logging.info("  -> Rerunning failed checkpoint for idx=%s: %s", idx, existing.get("error"))

            attempts = 0
            for attempt in range(1, max(item_retries, 0) + 2):
                attempts = attempt
                try:
                    raw_response = model.chat(
                        build_messages(question, spec.task_type),
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    predicted = extract_prediction(raw_response, spec.task_type)
                    is_correct = answers_match(predicted, gold)
                    error = ""
                    break
                except Exception as exc:
                    error = str(exc)
                    if attempt <= item_retries:
                        retry_seconds = retry_sleep * attempt
                        logging.warning(
                            "Error on %s item %s attempt %d/%d: %s. Retrying in %.1fs",
                            spec.name,
                            idx,
                            attempt,
                            item_retries + 1,
                            exc,
                            retry_seconds,
                        )
                        time.sleep(retry_seconds)
                    else:
                        logging.exception("Error on %s item %s after %d attempt(s)", spec.name, idx, attempt)

            correct += int(is_correct)
            failed += int(bool(error))

            record = {
                "idx": idx,
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "raw_response": raw_response,
                "steps": 1 if not error else 0,
                "error": error,
                "attempts": attempts,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            processed += 1

            logging.info(
                "  -> Predicted: %s | Gold: %s | %s | Running acc: %d/%d (%.1f%%)",
                predicted[:80],
                str(gold)[:80],
                "OK" if is_correct else "WRONG",
                correct,
                processed,
                100 * correct / processed,
            )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    total_time = time.time() - start_time
    summary = EvalSummary(
        total=processed,
        correct=correct,
        failed=failed,
        accuracy=correct / max(processed, 1),
        avg_steps=1.0 if processed else 0.0,
        total_time=total_time,
        token_usage=_token_usage_delta(token_usage_before, model.token_usage),
    )

    summary_dict = {
        "dataset": spec.name,
        "total": summary.total,
        "correct": summary.correct,
        "failed": summary.failed,
        "accuracy": round(summary.accuracy, 4),
        "avg_steps": round(summary.avg_steps, 2),
        "total_time_seconds": round(summary.total_time, 1),
        "token_usage": summary.token_usage,
        "resume_enabled": resume,
        "skipped_existing": skipped_existing,
        "rerun_failed_existing": rerun_failed_existing,
        "item_retries": item_retries,
        "retry_sleep_seconds": retry_sleep,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    print_summary(summary_dict)
    return summary


def _load_existing_records(results_file: Path) -> dict[int, dict]:
    if not results_file.exists():
        return {}

    records: dict[int, dict] = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                idx = int(record["idx"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logging.warning("Ignoring malformed checkpoint line %d in %s: %s", line_no, results_file, exc)
                continue
            records[idx] = record
    return records


def _can_resume_record(record: dict, question: str, gold: str) -> bool:
    if record.get("question") != question or str(record.get("gold")) != str(gold):
        return False
    if record.get("error"):
        return False
    predicted = record.get("predicted")
    return predicted is not None and str(predicted).strip() != ""


def _token_usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 50)
    print(f"LLM-ONLY SUMMARY: {summary['dataset']}")
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
    parser = argparse.ArgumentParser(
        description="Evaluate HotpotQA/AIME25/HMMT with direct LLM API calls only."
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to run. Use 'all' to run hotpotqa, aime25, and hmmt.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Max examples per dataset")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. For --dataset all, this is treated as the parent directory.",
    )
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs"),
        help="Parent directory for logs when --output-dir is not set.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment folder under --log-root, e.g. qwen2.5_72b.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls, useful for rate limits.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and rerun the dataset from scratch.",
    )
    parser.add_argument(
        "--item-retries",
        type=int,
        default=2,
        help="Retry a failed example this many extra times before writing it as failed.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=30.0,
        help="Base seconds to sleep between item retries; later retries use a linear backoff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config.from_yaml(args.config)
    model = ChatModel(config)

    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    output_root = resolve_output_root(args)

    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        output_dir = (
            output_root / f"llm_only_{dataset_name}"
            if args.dataset == "all" or args.output_dir is None
            else output_root
        )
        evaluate_dataset(
            model=model,
            spec=spec,
            output_dir=output_dir,
            limit=args.limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sleep_seconds=args.sleep,
            resume=not args.no_resume,
            item_retries=args.item_retries,
            retry_sleep=args.retry_sleep,
        )


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_name:
        return Path(args.log_root) / args.run_name
    return Path(args.log_root)


if __name__ == "__main__":
    main()
