from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .agent import AgentResult, ReActAgent, ReflectionAgent
from .config import Config
from .evaluate import EvalSummary, run_evaluation as evaluate_dataset
from .memory import ActionStep, AgentTrajectory
from .models import ChatModel
from .prompts import MATH_TASK_PROMPT, SEARCH_TASK_PROMPT
from .prompts_code import CODE_TASK_PROMPT
from .tools.base import Tool
from .tools.code_tool import CodeTool
from .tools.python_repl import PythonREPLTool
from .tools.search import WikipediaSearchTool

logger = logging.getLogger(__name__)

TaskType = Literal["hotpotqa", "math", "code_math", "math_v2", "reflection"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "hotpotqa": PROJECT_ROOT / "data" / "raw" / "hotpotqa200.jsonl",
    "aime25": PROJECT_ROOT / "data" / "processed" / "aime25.jsonl",
    "hmmt": PROJECT_ROOT / "data" / "processed" / "hmmt_feb_2025.jsonl",
}

AUTHORIZED_MATH_IMPORTS = ["sympy", "fractions", "numpy", "scipy"]


@dataclass(frozen=True)
class PipelineSpec:
    task_type: TaskType
    dataset_name: str
    dataset_path: Path
    prompt_template: str
    output_dir: Path
    max_steps: int


@dataclass(frozen=True)
class PipelineRunConfig:
    limit: int | None = None
    resume: bool = True
    item_retries: int = 0
    retry_sleep: float = 30.0


class MiniToolAgentPipeline:
    """Compose prompts, tools, agents, datasets, and evaluation runtime."""

    def __init__(self, spec: PipelineSpec, config: Config, model: ChatModel, agent):
        self.spec = spec
        self.config = config
        self.model = model
        self.agent = agent

    @classmethod
    def hotpotqa(
        cls,
        config_path: str | Path = PROJECT_ROOT / "llm.yaml",
        output_dir: str | Path | None = None,
        log_root: str | Path = PROJECT_ROOT / "logs",
        run_name: str | None = None,
        max_steps: int = 8,
        enable_reflection: bool = True,
    ) -> "MiniToolAgentPipeline":
        config = Config.from_yaml(config_path)
        model = ChatModel(config)
        agent = ReActAgent(
            model=model,
            tools=[WikipediaSearchTool(brave_api_key=config.brave_api_key)],
            max_steps=max_steps,
            enable_reflection=enable_reflection,
        )
        return cls(
            spec=PipelineSpec(
                task_type="hotpotqa",
                dataset_name="hotpotqa",
                dataset_path=DATASETS["hotpotqa"],
                prompt_template=SEARCH_TASK_PROMPT,
                output_dir=_resolve_output_dir(output_dir, log_root, run_name, "hotpotqa"),
                max_steps=max_steps,
            ),
            config=config,
            model=model,
            agent=agent,
        )

    @classmethod
    def math(
        cls,
        dataset_name: str = "aime25",
        config_path: str | Path = PROJECT_ROOT / "llm.yaml",
        output_dir: str | Path | None = None,
        log_root: str | Path = PROJECT_ROOT / "logs",
        run_name: str | None = None,
        max_steps: int = 12,
        enable_reflection: bool = True,
    ) -> "MiniToolAgentPipeline":
        _require_math_dataset(dataset_name)
        config = Config.from_yaml(config_path)
        model = ChatModel(config)
        agent = ReActAgent(
            model=model,
            tools=[_python_repl_tool()],
            max_steps=max_steps,
            enable_reflection=enable_reflection,
        )
        return cls(
            spec=PipelineSpec(
                task_type="math",
                dataset_name=dataset_name,
                dataset_path=DATASETS[dataset_name],
                prompt_template=MATH_TASK_PROMPT,
                output_dir=_resolve_output_dir(output_dir, log_root, run_name, dataset_name),
                max_steps=max_steps,
            ),
            config=config,
            model=model,
            agent=agent,
        )

    @classmethod
    def code_math(
        cls,
        dataset_name: str = "aime25",
        config_path: str | Path = PROJECT_ROOT / "llm.yaml",
        output_dir: str | Path | None = None,
        log_root: str | Path = PROJECT_ROOT / "logs",
        run_name: str | None = None,
        max_steps: int = 12,
        enable_reflection: bool = True,
    ) -> "MiniToolAgentPipeline":
        _require_math_dataset(dataset_name)
        config = Config.from_yaml(config_path)
        model = ChatModel(config)
        agent = ReActAgent(
            model=model,
            tools=[
                CodeTool(
                    model=model,
                    timeout=60,
                    additional_authorized_imports=AUTHORIZED_MATH_IMPORTS,
                )
            ],
            max_steps=max_steps,
            enable_reflection=enable_reflection,
        )
        return cls(
            spec=PipelineSpec(
                task_type="code_math",
                dataset_name=dataset_name,
                dataset_path=DATASETS[dataset_name],
                prompt_template=CODE_TASK_PROMPT,
                output_dir=_resolve_code_output_dir(output_dir, log_root, run_name, dataset_name),
                max_steps=max_steps,
            ),
            config=config,
            model=model,
            agent=agent,
        )

    @classmethod
    def math_v2(
        cls,
        dataset_name: str = "aime25",
        config_path: str | Path = PROJECT_ROOT / "llm.yaml",
        output_dir: str | Path | None = None,
        log_root: str | Path = PROJECT_ROOT / "logs",
        run_name: str | None = None,
        max_steps: int = 12,
        max_search_terms: int = 3,
        enable_reflection: bool = True,
    ) -> "MiniToolAgentPipeline":
        _require_math_dataset(dataset_name)
        config = Config.from_yaml(config_path)
        model = ChatModel(config)
        search_tool = WikipediaSearchTool(brave_api_key=config.brave_api_key)
        solver_agent = ReActAgent(
            model=model,
            tools=[search_tool, _python_repl_tool()],
            max_steps=max_steps,
            enable_reflection=enable_reflection,
        )
        agent = MathSearchThenCodeAgent(
            model=model,
            solver_agent=solver_agent,
            search_tool=search_tool,
            max_terms=max_search_terms,
        )
        return cls(
            spec=PipelineSpec(
                task_type="math_v2",
                dataset_name=dataset_name,
                dataset_path=DATASETS[dataset_name],
                prompt_template=MATH_TASK_PROMPT,
                output_dir=_resolve_output_dir(output_dir, log_root, run_name, f"{dataset_name}_math_v2"),
                max_steps=max_steps,
            ),
            config=config,
            model=model,
            agent=agent,
        )

    @classmethod
    def reflection(
        cls,
        dataset_name: str,
        config_path: str | Path = PROJECT_ROOT / "llm.yaml",
        output_dir: str | Path | None = None,
        log_root: str | Path = PROJECT_ROOT / "logs",
        run_name: str | None = None,
    ) -> "MiniToolAgentPipeline":
        _require_dataset(dataset_name)
        config = Config.from_yaml(config_path)
        model = ChatModel(config)
        prompt_template, tools, max_steps = _reflection_runtime(dataset_name, config)
        default_leaf = Path("ablation") / (run_name or "reflection") / dataset_name / "with_reflection"
        agent = ReflectionAgent(model=model, tools=tools, max_steps=max_steps)
        return cls(
            spec=PipelineSpec(
                task_type="reflection",
                dataset_name=dataset_name,
                dataset_path=DATASETS[dataset_name],
                prompt_template=prompt_template,
                output_dir=_resolve_output_dir(output_dir, log_root, None, default_leaf),
                max_steps=max_steps,
            ),
            config=config,
            model=model,
            agent=agent,
        )

    def run_evaluation(
        self,
        limit: int | None = None,
        resume: bool = True,
        item_retries: int = 0,
        retry_sleep: float = 30.0,
    ) -> EvalSummary:
        return evaluate_dataset(
            agent=self.agent,
            dataset_path=self.spec.dataset_path,
            task_prompt_template=self.spec.prompt_template,
            output_dir=self.spec.output_dir,
            limit=limit,
            resume=resume,
            item_retries=item_retries,
            retry_sleep=retry_sleep,
        )

    def run(self, run_config: PipelineRunConfig | None = None) -> EvalSummary:
        run_config = run_config or PipelineRunConfig()
        return self.run_evaluation(
            limit=run_config.limit,
            resume=run_config.resume,
            item_retries=run_config.item_retries,
            retry_sleep=run_config.retry_sleep,
        )


TERM_DISCOVERY_PROMPT = """\
You are preparing to solve a math competition problem.

Identify up to {max_terms} mathematical terms, named theorems, definitions, or
specialized concepts in the problem that a solver may need to look up before
coding. Prefer searchable terms such as theorem names, geometry objects,
sequence names, counting principles, or specialized contest vocabulary.

Return JSON only, with this schema:
{{"terms": [{{"term": "...", "query": "...", "reason": "..."}}]}}

If there are no useful terms to search, return {{"terms": []}}.

Problem:
{question}
"""


MATH_V2_SOLVER_SUFFIX = """\

## Pre-search Context
The v2 pipeline searched the following mathematical terms before solving.
Use this context to clarify definitions or named results, then proceed with
exact computation. If an important term remains unclear, use wikipedia_search
before python_repl.

{search_context}
"""


class MathSearchThenCodeAgent:
    """Pipeline agent that searches math terms before running a code-capable solver."""

    def __init__(
        self,
        model: ChatModel,
        solver_agent: ReActAgent,
        search_tool: WikipediaSearchTool,
        max_terms: int = 3,
    ):
        self.model = model
        self.solver_agent = solver_agent
        self.search_tool = search_tool
        self.max_terms = max(0, max_terms)

    def run(self, task_prompt: str) -> AgentResult:
        question = _extract_problem(task_prompt)
        pre_steps, search_context = self._search_math_terms(question)
        enriched_prompt = task_prompt + MATH_V2_SOLVER_SUFFIX.format(search_context=search_context)

        result = self.solver_agent.run(enriched_prompt)
        result.trajectory = _merge_trajectories(
            task=enriched_prompt,
            pre_steps=pre_steps,
            solver_trajectory=result.trajectory,
            answer=result.answer,
        )
        return result

    def build_task_prompt(self, question: str) -> str:
        return MATH_TASK_PROMPT.format(question=question)

    def _search_math_terms(self, question: str) -> tuple[list[ActionStep], str]:
        if self.max_terms == 0:
            return [], "Term pre-search disabled by max_terms=0."

        terms = self._discover_terms(question)
        steps: list[ActionStep] = [
            ActionStep(
                step=1,
                thought="Identify math terms that should be searched before coding.",
                tool_name="term_discovery",
                tool_args={"max_terms": self.max_terms},
                observation=json.dumps(terms, ensure_ascii=False),
            )
        ]

        if not terms:
            return steps, "No specialized math terms were identified for pre-search."

        context_parts: list[str] = []
        for term_info in terms[: self.max_terms]:
            term = str(term_info.get("term", "")).strip()
            query = str(term_info.get("query", "") or term).strip()
            if not query:
                continue
            observation = self.search_tool.execute(query=query)
            steps.append(
                ActionStep(
                    step=len(steps) + 1,
                    thought=f"Search background for math term: {term or query}",
                    tool_name=self.search_tool.name,
                    tool_args={"query": query},
                    observation=observation,
                )
            )
            label = term or query
            context_parts.append(f"### {label}\nQuery: {query}\n{observation}")

        if not context_parts:
            return steps, "No valid search queries were produced by term discovery."
        return steps, "\n\n".join(context_parts)

    def _discover_terms(self, question: str) -> list[dict]:
        prompt = TERM_DISCOVERY_PROMPT.format(max_terms=self.max_terms, question=question)
        try:
            response = self.model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning("Math term discovery failed: %s", exc)
            return []

        payload = _extract_json_object(response)
        if not isinstance(payload, dict):
            return []

        terms = payload.get("terms", [])
        if not isinstance(terms, list):
            return []

        normalized: list[dict] = []
        seen: set[str] = set()
        for item in terms:
            if isinstance(item, str):
                item = {"term": item, "query": item, "reason": ""}
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            query = str(item.get("query", "") or term).strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "term": term or query,
                    "query": query,
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        return normalized[: self.max_terms]


def _reflection_runtime(dataset_name: str, config: Config) -> tuple[str, list[Tool], int]:
    if dataset_name == "hotpotqa":
        return (
            SEARCH_TASK_PROMPT,
            [WikipediaSearchTool(brave_api_key=config.brave_api_key)],
            8,
        )
    return (MATH_TASK_PROMPT, [_python_repl_tool()], 12)


def _python_repl_tool() -> PythonREPLTool:
    return PythonREPLTool(
        timeout=60,
        additional_authorized_imports=AUTHORIZED_MATH_IMPORTS,
    )


def _resolve_output_dir(
    output_dir: str | Path | None,
    log_root: str | Path,
    run_name: str | None,
    default_leaf: str | Path,
) -> Path:
    if output_dir:
        return Path(output_dir)
    if run_name:
        return Path(log_root) / run_name / default_leaf
    return Path(log_root) / default_leaf


def _resolve_code_output_dir(
    output_dir: str | Path | None,
    log_root: str | Path,
    run_name: str | None,
    dataset_name: str,
) -> Path:
    if output_dir:
        return Path(output_dir)
    if run_name:
        return Path(log_root) / run_name / dataset_name
    return Path(log_root) / f"code_{dataset_name}"


def _require_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _require_math_dataset(dataset_name: str) -> None:
    if dataset_name not in {"aime25", "hmmt"}:
        raise ValueError(f"Unknown math dataset: {dataset_name}")


def _extract_problem(task_prompt: str) -> str:
    match = re.search(r"Problem:\s*(.+)\s*\Z", task_prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else task_prompt.strip()


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _merge_trajectories(
    task: str,
    pre_steps: list[ActionStep],
    solver_trajectory: AgentTrajectory,
    answer: str,
) -> AgentTrajectory:
    merged = AgentTrajectory(task=task)
    for idx, step in enumerate(pre_steps, start=1):
        merged.add_step(replace(step, step=idx))

    offset = len(merged.steps)
    for step in solver_trajectory.steps:
        merged.add_step(replace(step, step=step.step + offset))

    merged.reflection_count = solver_trajectory.reflection_count
    merged.success = solver_trajectory.success
    merged.start_time = min(merged.start_time, solver_trajectory.start_time)
    merged.finish(answer)
    return merged
