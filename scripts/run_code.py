"""Run the agent on math datasets (AIME25 / HMMT) with the code tool.

In this mode the agent focuses on mathematical analysis and describes
computations in natural language.  The CodeTool internally calls the LLM
to generate Python code and executes it in a sandbox.
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minitoolagent.config import Config
from minitoolagent.models import ChatModel
from minitoolagent.agent import ReActAgent
from minitoolagent.tools.code_tool import CodeTool
from minitoolagent.prompts_code import CODE_TASK_PROMPT
from minitoolagent.evaluate import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "aime25": PROJECT_ROOT / "data" / "processed" / "aime25.jsonl",
    "hmmt": PROJECT_ROOT / "data" / "processed" / "hmmt_feb_2025.jsonl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent on math datasets using CodeTool (LLM-generated code)"
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="aime25")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "llm.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Exact output directory for this run")
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs"),
        help="Parent directory for logs when --output-dir is not set",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment folder under --log-root, e.g. qwen2.5_72b_code",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.run_name:
        output_dir = Path(args.log_root) / args.run_name / args.dataset
    else:
        output_dir = Path(args.log_root) / f"code_{args.dataset}"

    config = Config.from_yaml(args.config)
    model = ChatModel(config)

    tools = [
        CodeTool(
            model=model,
            timeout=60,
            additional_authorized_imports=["sympy", "fractions", "numpy", "scipy"],
        ),
    ]

    agent = ReActAgent(
        model=model,
        tools=tools,
        max_steps=args.max_steps,
        enable_reflection=not args.no_reflection,
    )

    data_path = DATASETS[args.dataset]
    run_evaluation(
        agent=agent,
        dataset_path=data_path,
        task_prompt_template=CODE_TASK_PROMPT,
        output_dir=str(output_dir),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
