from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import ReActAgent, AgentResult
from .parsing import answers_match, normalize_answer


def strip_asymptote(text: str) -> str:
    """Remove [asy]...[/asy] drawing blocks from problem text."""
    text = re.sub(r'\[asy\].*?\[/asy\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    idx: int
    question: str
    gold_answer: str
    predicted_answer: str = ""
    correct: bool = False
    num_steps: int = 0
    error: str = ""
    attempts: int = 0


@dataclass
class EvalSummary:
    total: int = 0
    correct: int = 0
    failed: int = 0
    accuracy: float = 0.0
    avg_steps: float = 0.0
    total_time: float = 0.0
    token_usage: dict = field(default_factory=dict)


def load_dataset(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_evaluation(
    agent: ReActAgent,
    dataset_path: str | Path,
    task_prompt_template: str,
    output_dir: str | Path,
    limit: int | None = None,
    resume: bool = True,
    item_retries: int = 0,
    retry_sleep: float = 30.0,
) -> EvalSummary:
    """Run the agent on a dataset and compute metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    results_file = output_dir / "results.jsonl"
    existing_records = _load_existing_records(results_file) if resume else {}
    skipped_existing = 0
    rerun_failed_existing = 0

    results: list[EvalItem] = []
    start_time = time.time()
    interrupted = False

    with open(results_file, "w", encoding="utf-8") as f_out:
        for i, item in enumerate(dataset):
            idx = item.get("idx", i + 1)
            question = item["question"]
            gold = item["answer"]

            logger.info("[%d/%d] Question: %s", i + 1, len(dataset), question[:80])

            existing = existing_records.get(int(idx))
            if existing and _can_resume_record(existing, question, gold):
                eval_item = _eval_item_from_record(existing)
                results.append(eval_item)
                skipped_existing += 1
                f_out.write(json.dumps(existing, ensure_ascii=False) + "\n")
                f_out.flush()
                logger.info(
                    "  -> Skipped existing result for idx=%s | Predicted: %s | Gold: %s",
                    idx,
                    eval_item.predicted_answer[:50],
                    str(gold)[:50],
                )
                continue
            if existing and existing.get("error"):
                rerun_failed_existing += 1
                logger.info("  -> Rerunning failed checkpoint for idx=%s: %s", idx, existing.get("error"))

            task_prompt = task_prompt_template.format(question=strip_asymptote(question))
            eval_item = EvalItem(idx=idx, question=question, gold_answer=gold)

            for attempt in range(1, max(item_retries, 0) + 2):
                eval_item.attempts = attempt
                try:
                    result: AgentResult = agent.run(task_prompt)
                    eval_item.predicted_answer = result.answer
                    eval_item.num_steps = result.trajectory.total_steps
                    eval_item.correct = answers_match(result.answer, gold)
                    eval_item.error = ""

                    trajectory_file = output_dir / f"trajectory_{idx}.json"
                    result.trajectory.save(trajectory_file)
                    break

                except KeyboardInterrupt:
                    interrupted = True
                    eval_item.error = "interrupted by user"
                    eval_item.predicted_answer = ""
                    eval_item.correct = False
                    logger.warning(
                        "Interrupted on item %d attempt %d/%d; saving partial results.",
                        idx,
                        attempt,
                        item_retries + 1,
                    )
                    break
                except Exception as e:
                    eval_item.error = str(e)
                    eval_item.predicted_answer = ""
                    eval_item.correct = False
                    if attempt <= item_retries:
                        sleep_seconds = retry_sleep * attempt
                        logger.warning(
                            "Error on item %d attempt %d/%d: %s. Retrying in %.1fs",
                            idx,
                            attempt,
                            item_retries + 1,
                            e,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    else:
                        logger.error("Error on item %d after %d attempt(s): %s", idx, attempt, e)

            results.append(eval_item)

            # Write result line
            f_out.write(json.dumps(_record_from_eval_item(eval_item), ensure_ascii=False) + "\n")
            f_out.flush()

            # Running accuracy
            correct_so_far = sum(1 for r in results if r.correct)
            logger.info(
                "  -> Predicted: %s | Gold: %s | %s | Running acc: %d/%d (%.1f%%)",
                eval_item.predicted_answer[:50],
                gold[:50],
                "✓" if eval_item.correct else "✗",
                correct_so_far, len(results),
                100 * correct_so_far / len(results),
            )

            if interrupted:
                break

    # Compute summary
    summary = EvalSummary(
        total=len(results),
        correct=sum(1 for r in results if r.correct),
        failed=sum(1 for r in results if r.error),
        avg_steps=sum(r.num_steps for r in results) / max(len(results), 1),
        total_time=time.time() - start_time,
        token_usage=agent.model.token_usage,
    )
    summary.accuracy = summary.correct / max(summary.total, 1)

    # Save summary
    summary_dict = {
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
        "interrupted": interrupted,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    _print_summary(summary)
    if interrupted:
        print("Interrupted: partial results were saved.")
        raise KeyboardInterrupt
    return summary


def _record_from_eval_item(eval_item: EvalItem) -> dict:
    return {
        "idx": eval_item.idx,
        "question": eval_item.question,
        "gold": eval_item.gold_answer,
        "predicted": eval_item.predicted_answer,
        "correct": eval_item.correct,
        "steps": eval_item.num_steps,
        "error": eval_item.error,
        "attempts": eval_item.attempts,
    }


def _eval_item_from_record(record: dict) -> EvalItem:
    return EvalItem(
        idx=int(record["idx"]),
        question=record.get("question", ""),
        gold_answer=record.get("gold", ""),
        predicted_answer=record.get("predicted", ""),
        correct=bool(record.get("correct", False)),
        num_steps=int(record.get("steps", 0) or 0),
        error=record.get("error", ""),
        attempts=int(record.get("attempts", 1) or 1),
    )


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
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning("Ignoring malformed checkpoint line %d in %s: %s", line_no, results_file, e)
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


def _print_summary(s: EvalSummary) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total:      {s.total}")
    print(f"Correct:    {s.correct}")
    print(f"Failed:     {s.failed}")
    print(f"Accuracy:   {s.accuracy:.2%}")
    print(f"Avg steps:  {s.avg_steps:.1f}")
    print(f"Total time: {s.total_time:.0f}s")
    if s.token_usage:
        print(f"Tokens:     {s.token_usage}")
    print("=" * 50)
