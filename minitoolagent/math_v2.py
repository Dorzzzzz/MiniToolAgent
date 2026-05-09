from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from .agent import AgentResult, ReActAgent
from .memory import ActionStep, AgentTrajectory
from .models import ChatModel
from .prompts import MATH_TASK_PROMPT
from .tools.search import WikipediaSearchTool

logger = logging.getLogger(__name__)


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
    """Pipeline wrapper that searches math terms before running a code agent."""

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
