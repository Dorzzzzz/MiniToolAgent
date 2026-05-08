"""Run the agent on HotpotQA (search-augmented multi-hop QA)."""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minitoolagent.config import Config
from minitoolagent.models import ChatModel
from minitoolagent.agent import ReActAgent
from minitoolagent.tools.search import WikipediaSearchTool
from minitoolagent.prompts import SEARCH_TASK_PROMPT
from minitoolagent.evaluate import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotpotqa200.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on HotpotQA")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to evaluate")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--no-reflection", action="store_true", help="Disable self-correction")
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

    tools = [
        WikipediaSearchTool(brave_api_key=config.brave_api_key),
    ]

    agent = ReActAgent(
        model=model,
        tools=tools,
        max_steps=args.max_steps,
        enable_reflection=not args.no_reflection,
    )

    run_evaluation(
        agent=agent,
        dataset_path=DATA_PATH,
        task_prompt_template=SEARCH_TASK_PROMPT,
        output_dir=str(resolve_output_dir(args, "hotpotqa")),
        limit=args.limit,
        resume=not args.no_resume,
        item_retries=args.item_retries,
        retry_sleep=args.retry_sleep,
    )


def resolve_output_dir(args: argparse.Namespace, dataset_name: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_name:
        return Path(args.log_root) / args.run_name / dataset_name
    return Path(args.log_root) / dataset_name


if __name__ == "__main__":
    main()
