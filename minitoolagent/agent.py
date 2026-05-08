from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import Config
from .models import ChatModel
from .memory import ActionStep, AgentTrajectory
from .parsing import parse_llm_response, ParsedAction, extract_code_block
from .prompts import SYSTEM_PROMPT_TEMPLATE, REFLECTION_PROMPT, STRUCTURED_REFLECTION_PROMPT
from .tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)

MAX_OBSERVATION_LEN = 3000
MAX_REPEATED_ACTIONS = 3


@dataclass
class AgentResult:
    answer: str
    trajectory: AgentTrajectory
    token_usage: dict = field(default_factory=dict)


class ReActAgent:
    """A ReAct agent that loops through Thought → Action → Observation."""

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool],
        max_steps: int = 10,
        enable_reflection: bool = True,
    ):
        self.model = model
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection

        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)

    def run(self, task_prompt: str) -> AgentResult:
        """Execute the full agent loop on a task."""
        for tool in self.registry._tools.values():
            if hasattr(tool, "reset"):
                tool.reset()
        trajectory = AgentTrajectory(task=task_prompt)
        messages = self._build_initial_messages(task_prompt)
        recent_actions: list[str] = []

        for step_idx in range(1, self.max_steps + 1):
            logger.info("Step %d/%d", step_idx, self.max_steps)

            # 1. Call LLM
            response = self.model.chat(messages)

            # 2. Parse response into thought + action
            parsed = parse_llm_response(response)

            # 2.5 Fuzzy tool name matching
            if parsed.tool and parsed.tool not in self.registry and parsed.tool != "final_answer":
                matched = self._fuzzy_match_tool(parsed.tool)
                if matched:
                    parsed = ParsedAction(tool=matched, args=parsed.args, raw_thought=parsed.raw_thought)

            # 2.6 Extract code from markdown blocks if code arg is empty
            if parsed.tool in ("python_repl",) and not parsed.args.get("code"):
                code = extract_code_block(response)
                if code:
                    parsed = ParsedAction(tool=parsed.tool, args={"code": code}, raw_thought=parsed.raw_thought)
            action_step = ActionStep(
                step=step_idx,
                thought=parsed.raw_thought,
                tool_name=parsed.tool,
                tool_args=parsed.args,
            )

            # 3. Check for final answer
            if parsed.tool == "final_answer":
                answer = parsed.args.get("answer", "")
                action_step.observation = f"Final answer: {answer}"
                trajectory.add_step(action_step)
                trajectory.finish(answer)
                return AgentResult(
                    answer=answer,
                    trajectory=trajectory,
                    token_usage=self.model.token_usage,
                )

            # 4. Check for parse failure
            if not parsed.tool:
                observation = self._handle_parse_failure(parsed, response)
                action_step.error = "Failed to parse action from LLM output."
                action_step.observation = observation
                trajectory.add_step(action_step)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            # 5. Detect repeated actions
            action_key = json.dumps({"tool": parsed.tool, "args": parsed.args}, sort_keys=True)
            recent_actions.append(action_key)
            if self._is_stuck(recent_actions):
                nudge = (
                    "You have repeated the same action multiple times. "
                    "Please try a DIFFERENT approach or search query."
                )
                action_step.error = "Repeated action detected."
                action_step.observation = nudge
                trajectory.add_step(action_step)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {nudge}"})
                recent_actions.clear()
                continue

            # 6. Execute tool
            observation = self._execute_tool(parsed)
            action_step.observation = observation
            trajectory.add_step(action_step)

            # 7. If tool error and reflection enabled, add reflection prompt
            is_error = observation.startswith("Error")
            if is_error and self.enable_reflection:
                reflection = REFLECTION_PROMPT.format(error_info=observation)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}\n\n{reflection}"})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            logger.info(
                "Tool: %s | Observation: %s",
                parsed.tool,
                observation[:100] + "..." if len(observation) > 100 else observation,
            )

        # Max steps reached — force a final answer
        answer = self._force_final_answer(messages)
        trajectory.finish(answer)
        return AgentResult(
            answer=answer,
            trajectory=trajectory,
            token_usage=self.model.token_usage,
        )

    # ── Internal helpers ────────────────────────────────────────────

    def _build_initial_messages(self, task_prompt: str) -> list[dict]:
        tool_names = ", ".join(self.registry.list_names() + ["final_answer"])
        system = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self.registry.describe_all(),
            tool_names=tool_names,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt},
        ]

    def _fuzzy_match_tool(self, name: str) -> str | None:
        """Match a potentially truncated/mangled tool name to the closest registered tool."""
        name_clean = name.strip().lower().replace(" ", "_").replace("-", "_")
        for registered in self.registry.list_names():
            reg_lower = registered.lower()
            if reg_lower == name_clean:
                return registered
            # Handles: "python_reD" -> "python_repl", "w_wikipedia_search" -> "wikipedia_search"
            if reg_lower in name_clean or name_clean in reg_lower:
                return registered
            # Handles prefix matches: "python_re" -> "python_repl"
            if reg_lower.startswith(name_clean) or name_clean.startswith(reg_lower[:6]):
                return registered
        return None

    def _execute_tool(self, parsed: ParsedAction) -> str:
        tool = self.registry.get(parsed.tool)
        if tool is None:
            available = ", ".join(self.registry.list_names())
            return f"Error: unknown tool '{parsed.tool}'. Available tools: {available}"
        try:
            result = tool.execute(**parsed.args)
            return self._truncate(result)
        except Exception as e:
            return f"Error executing {parsed.tool}: {e}"

    def _handle_parse_failure(self, parsed: ParsedAction, raw: str) -> str:
        if self.enable_reflection:
            return (
                "I could not parse a valid action from your response. "
                "Please respond exactly in this format:\n"
                'Thought: <reasoning>\n'
                'Action: {"tool": "<tool_name>", "args": {<arguments>}}'
            )
        return "Error: could not parse action."

    def _is_stuck(self, recent_actions: list[str]) -> bool:
        if len(recent_actions) < MAX_REPEATED_ACTIONS:
            return False
        last_n = recent_actions[-MAX_REPEATED_ACTIONS:]
        return len(set(last_n)) == 1

    def _force_final_answer(self, messages: list[dict]) -> str:
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of steps. "
                "Based on all information gathered so far, provide your best final answer now.\n"
                'Action: {"tool": "final_answer", "args": {"answer": "<your answer>"}}'
            ),
        })
        response = self.model.chat(messages)
        parsed = parse_llm_response(response)
        if parsed.tool == "final_answer":
            return parsed.args.get("answer", "")
        # Last resort: try to extract any answer-like text
        from .parsing import extract_final_answer
        extracted = extract_final_answer(response)
        return extracted or response.strip()[:200]

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) > MAX_OBSERVATION_LEN:
            return text[:MAX_OBSERVATION_LEN] + "\n... [truncated]"
        return text


class ReflectionAgent(ReActAgent):
    """ReAct agent with structured runtime self-correction.

    This agent does not inspect gold answers. It only reflects when the live
    trajectory enters an abnormal state such as parse failure, repeated action,
    empty search results, no code output, or tool execution errors.
    """

    ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
        ("[execution_error]", "tool_error"),
        ("Error executing", "tool_error"),
        ("Error:", "tool_error"),
        ("Search error", "search_error"),
        ("No Wikipedia results found", "empty_search"),
        ("No Wikipedia content found", "empty_search"),
        ("[No output produced", "no_output"),
    )

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool],
        max_steps: int = 10,
    ):
        super().__init__(
            model=model,
            tools=tools,
            max_steps=max_steps,
            enable_reflection=True,
        )
        self.total_reflection_count = 0

    def run(self, task_prompt: str) -> AgentResult:
        """Execute the agent loop with structured reflection on abnormal states."""
        for tool in self.registry._tools.values():
            if hasattr(tool, "reset"):
                tool.reset()

        trajectory = AgentTrajectory(task=task_prompt)
        messages = self._build_initial_messages(task_prompt)
        recent_actions: list[str] = []

        for step_idx in range(1, self.max_steps + 1):
            logger.info("Reflection step %d/%d", step_idx, self.max_steps)

            response = self.model.chat(messages)
            parsed = parse_llm_response(response)

            if parsed.tool and parsed.tool not in self.registry and parsed.tool != "final_answer":
                matched = self._fuzzy_match_tool(parsed.tool)
                if matched:
                    parsed = ParsedAction(tool=matched, args=parsed.args, raw_thought=parsed.raw_thought)

            if parsed.tool in ("python_repl",) and not parsed.args.get("code"):
                code = extract_code_block(response)
                if code:
                    parsed = ParsedAction(tool=parsed.tool, args={"code": code}, raw_thought=parsed.raw_thought)

            action_step = ActionStep(
                step=step_idx,
                thought=parsed.raw_thought,
                tool_name=parsed.tool,
                tool_args=parsed.args,
            )

            if parsed.tool == "final_answer":
                answer = parsed.args.get("answer", "")
                action_step.observation = f"Final answer: {answer}"
                trajectory.add_step(action_step)
                trajectory.finish(answer)
                return AgentResult(
                    answer=answer,
                    trajectory=trajectory,
                    token_usage=self.model.token_usage,
                )

            if not parsed.tool:
                observation = self._handle_parse_failure(parsed, response)
                self._record_reflection(
                    trajectory=trajectory,
                    action_step=action_step,
                    messages=messages,
                    response=response,
                    task_prompt=task_prompt,
                    error_type="parse_error",
                    observation=observation,
                    recent_actions=recent_actions,
                    last_action="<parse failed>",
                )
                continue

            action_key = json.dumps({"tool": parsed.tool, "args": parsed.args}, sort_keys=True)
            recent_actions.append(action_key)
            if self._is_stuck(recent_actions):
                observation = (
                    "Repeated action detected: the same Action JSON was produced "
                    f"{MAX_REPEATED_ACTIONS} times. Try a different query/code path "
                    "or answer from gathered evidence."
                )
                self._record_reflection(
                    trajectory=trajectory,
                    action_step=action_step,
                    messages=messages,
                    response=response,
                    task_prompt=task_prompt,
                    error_type="repeated_action",
                    observation=observation,
                    recent_actions=recent_actions,
                    last_action=action_key,
                )
                recent_actions.clear()
                continue

            observation = self._execute_tool(parsed)
            error_type = self._classify_observation(observation)
            if error_type:
                self._record_reflection(
                    trajectory=trajectory,
                    action_step=action_step,
                    messages=messages,
                    response=response,
                    task_prompt=task_prompt,
                    error_type=error_type,
                    observation=observation,
                    recent_actions=recent_actions,
                    last_action=action_key,
                )
            else:
                action_step.observation = observation
                trajectory.add_step(action_step)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            logger.info(
                "Tool: %s | Observation: %s",
                parsed.tool,
                observation[:100] + "..." if len(observation) > 100 else observation,
            )

        answer = self._force_final_answer(messages)
        trajectory.finish(answer)
        return AgentResult(
            answer=answer,
            trajectory=trajectory,
            token_usage=self.model.token_usage,
        )

    def _classify_observation(self, observation: str) -> str:
        for pattern, error_type in self.ERROR_PATTERNS:
            if pattern in observation:
                return error_type
        return ""

    def _record_reflection(
        self,
        trajectory: AgentTrajectory,
        action_step: ActionStep,
        messages: list[dict],
        response: str,
        task_prompt: str,
        error_type: str,
        observation: str,
        recent_actions: list[str],
        last_action: str,
    ) -> None:
        action_step.error = error_type
        action_step.observation = (
            f"{observation}\n\n"
            f"[reflection_triggered] {error_type}: structured recovery prompt added."
        )
        trajectory.reflection_count += 1
        self.total_reflection_count += 1
        trajectory.add_step(action_step)

        reflection_prompt = self._build_structured_reflection_prompt(
            task_prompt=task_prompt,
            error_type=error_type,
            last_thought=action_step.thought,
            last_action=last_action,
            observation=observation,
            recent_actions=recent_actions,
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation}\n\n{reflection_prompt}"})

    def _build_structured_reflection_prompt(
        self,
        task_prompt: str,
        error_type: str,
        last_thought: str,
        last_action: str,
        observation: str,
        recent_actions: list[str],
    ) -> str:
        recent = "\n".join(f"- {a}" for a in recent_actions[-5:]) or "- <none>"
        tool_names = ", ".join(self.registry.list_names() + ["final_answer"])
        return STRUCTURED_REFLECTION_PROMPT.format(
            task=task_prompt,
            error_type=error_type,
            last_thought=last_thought or "<empty>",
            last_action=last_action,
            observation=observation,
            recent_actions=recent,
            tool_names=tool_names,
        )
