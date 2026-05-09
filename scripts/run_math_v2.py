"""Run math evaluation with a search-then-code v2 pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minitoolagent.agent import ReActAgent
from minitoolagent.config import Config
from minitoolagent.evaluate import run_evaluation
from minitoolagent.math_v2 import MathSearchThenCodeAgent
from minitoolagent.models import ChatModel
from minitoolagent.prompts import MATH_TASK_PROMPT
from minitoolagent.tools.python_repl import PythonREPLTool
from minitoolagent.tools.search import WikipediaSearchTool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "aime25": PROJECT_ROOT / "data" / "processed" / "aime25.jsonl",
    "hmmt": PROJECT_ROOT / "data" / "processed" / "hmmt_feb_2025.jsonl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math agent with term search before code execution"
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="aime25")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-search-terms", type=int, default=3)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and rerun the dataset from scratch",
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
    parser.add_argument("--output-dir", default=None, help="Exact output directory for this run")
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs"),
        help="Parent directory for logs when --output-dir is not set",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment folder under --log-root, e.g. qwen2.5_72b",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config.from_yaml(args.config)
    model = ChatModel(config)

    search_tool = WikipediaSearchTool(brave_api_key=config.brave_api_key)
    solver_agent = ReActAgent(
        model=model,
        tools=[
            search_tool,
            PythonREPLTool(
                timeout=60,
                additional_authorized_imports=["sympy", "fractions", "numpy", "scipy"],
            ),
        ],
        max_steps=args.max_steps,
        enable_reflection=not args.no_reflection,
    )
    agent = MathSearchThenCodeAgent(
        model=model,
        solver_agent=solver_agent,
        search_tool=search_tool,
        max_terms=args.max_search_terms,
    )

    run_evaluation(
        agent=agent,
        dataset_path=DATASETS[args.dataset],
        task_prompt_template=MATH_TASK_PROMPT,
        output_dir=str(resolve_output_dir(args)),
        limit=args.limit,
        resume=not args.no_resume,
        item_retries=args.item_retries,
        retry_sleep=args.retry_sleep,
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    dataset_name = f"{args.dataset}_math_v2"
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_name:
        return Path(args.log_root) / args.run_name / dataset_name
    return Path(args.log_root) / dataset_name


if __name__ == "__main__":
    main()
