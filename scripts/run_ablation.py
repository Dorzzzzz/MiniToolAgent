"""Run ReflectionAgent evaluations.

This script runs only ReflectionAgent. It keeps the historical filename because
the project already uses scripts/run_ablation.py for the reflection experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minitoolagent.pipeline import MiniToolAgentPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_ablation(
    dataset_name: str,
    limit: int | None = None,
    config_path: str | Path = PROJECT_ROOT / "llm.yaml",
    log_root: str | Path = PROJECT_ROOT / "logs",
    run_name: str | None = None,
    resume: bool = True,
    item_retries: int = 2,
    retry_sleep: float = 30.0,
) -> dict:
    """Run ReflectionAgent on one dataset and return metrics."""
    run_name = run_name or f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 72)
    print(f"Running {dataset_name}: ReflectionAgent")
    print("=" * 72)

    pipeline = MiniToolAgentPipeline.reflection(
        dataset_name=dataset_name,
        config_path=config_path,
        log_root=log_root,
        run_name=run_name,
    )
    output_dir = pipeline.spec.output_dir
    dataset_dir = output_dir.parent

    summary = pipeline.run_evaluation(
        limit=limit,
        resume=resume,
        item_retries=item_retries,
        retry_sleep=retry_sleep,
    )

    metrics = {
        "dataset": dataset_name,
        "agent": "ReflectionAgent",
        **_build_run_metrics(summary, output_dir),
    }
    metrics_path = dataset_dir / "reflection_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _print_metrics(metrics)
    print(f"\nSaved ReflectionAgent metrics to {metrics_path}")
    return metrics


def _build_run_metrics(summary, output_dir: Path) -> dict:
    diagnostics = _collect_trajectory_diagnostics(output_dir)
    tokens = summary.token_usage or {}
    total_tokens = int(tokens.get("total_tokens", 0) or 0)
    correct = int(summary.correct)
    return {
        "total": int(summary.total),
        "correct": correct,
        "failed": int(summary.failed),
        "accuracy": round(summary.accuracy, 4),
        "avg_steps": round(summary.avg_steps, 2),
        "prompt_tokens": int(tokens.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(tokens.get("completion_tokens", 0) or 0),
        "total_tokens": total_tokens,
        "cost_per_correct": round(total_tokens / correct, 2) if correct else None,
        "reflection_count": diagnostics["reflection_count"],
        "error_steps": diagnostics["error_steps"],
        "trajectory_files": diagnostics["trajectory_files"],
    }


def _collect_trajectory_diagnostics(output_dir: Path) -> dict:
    reflection_count = 0
    error_steps = 0
    trajectory_files = 0

    for path in sorted(output_dir.glob("trajectory_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        trajectory_files += 1
        reflection_count += int(data.get("reflection_count", 0) or 0)
        for step in data.get("steps", []):
            if step.get("error"):
                error_steps += 1
                if not data.get("reflection_count") and step.get("error") in {
                    "parse_error",
                    "tool_error",
                    "search_error",
                    "empty_search",
                    "no_output",
                    "repeated_action",
                }:
                    reflection_count += 1

    return {
        "reflection_count": reflection_count,
        "error_steps": error_steps,
        "trajectory_files": trajectory_files,
    }


def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 72)
    print(f"REFLECTIONAGENT RESULT: {metrics['dataset']}")
    print("=" * 72)
    print(f"{'Accuracy':<22} {metrics['accuracy']:>17.2%}")
    print(f"{'Avg Steps':<22} {metrics['avg_steps']:>18.2f}")
    print(f"{'Total Tokens':<22} {metrics['total_tokens']:>18,}")
    print(f"{'Prompt Tokens':<22} {metrics['prompt_tokens']:>18,}")
    print(f"{'Completion Tokens':<22} {metrics['completion_tokens']:>18,}")
    print(f"{'Reflection Count':<22} {metrics['reflection_count']:>18}")
    print(f"{'Error Steps':<22} {metrics['error_steps']:>18}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReflectionAgent evaluations")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "aime25", "hmmt", "math", "all"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples per dataset. Defaults to the full dataset.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--log-root", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and rerun each dataset from scratch",
    )
    parser.add_argument(
        "--item-retries",
        type=int,
        default=2,
        help="Retry a failed example this many extra times before writing it as failed",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=30.0,
        help="Base seconds to sleep between item retries; later retries use a linear backoff",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_name = args.run_name or f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.dataset == "all":
        datasets = ["hotpotqa", "aime25", "hmmt"]
    elif args.dataset == "math":
        datasets = ["aime25", "hmmt"]
    else:
        datasets = [args.dataset]
    all_results: dict[str, dict] = {}

    print(f"Datasets will run serially: {', '.join(datasets)}")
    try:
        for dataset_name in datasets:
            all_results[dataset_name] = run_ablation(
                dataset_name=dataset_name,
                limit=args.limit,
                config_path=args.config,
                log_root=args.log_root,
                run_name=run_name,
                resume=not args.no_resume,
                item_retries=args.item_retries,
                retry_sleep=args.retry_sleep,
            )
    except KeyboardInterrupt:
        output_dir = Path(args.log_root) / "ablation" / run_name
        print(f"\nInterrupted by user. Partial results were saved under {output_dir}")
        return

    if args.dataset in {"all", "math"}:
        aggregate_name = "reflection_all.json" if args.dataset == "all" else "reflection_math.json"
        output_path = Path(args.log_root) / "ablation" / run_name / aggregate_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregate ReflectionAgent results to {output_path}")


if __name__ == "__main__":
    main()
