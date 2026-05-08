"""Smoke tests for ReflectionAgent self-correction triggers."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from minitoolagent.agent import ReflectionAgent
from minitoolagent.tools.base import Tool, ToolParameter


class ScriptedModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, stop=None, temperature=None, max_tokens=None):
        if not self.responses:
            raise AssertionError("No scripted response left")
        self.total_prompt_tokens += sum(len(m.get("content", "")) for m in messages)
        response = self.responses.pop(0)
        self.total_completion_tokens += len(response)
        return response

    @property
    def token_usage(self):
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


class StaticTool(Tool):
    name = "dummy_tool"
    description = "Return a fixed observation."
    parameters = [ToolParameter(name="query", type="string", description="Query text.")]

    def __init__(self, observation: str):
        self.observation = observation
        self.calls = 0

    def execute(self, **kwargs) -> str:
        self.calls += 1
        return self.observation


def action(query: str) -> str:
    return f'Thought: try it\nAction: {{"tool": "dummy_tool", "args": {{"query": "{query}"}}}}'


def final(answer: str = "ok") -> str:
    return f'Thought: done\nAction: {{"tool": "final_answer", "args": {{"answer": "{answer}"}}}}'


def run_case(responses: list[str], observation: str, expected_error: str) -> None:
    model = ScriptedModel(responses)
    agent = ReflectionAgent(model=model, tools=[StaticTool(observation)], max_steps=5)
    result = agent.run("test task")
    assert result.answer == "ok"
    assert result.trajectory.reflection_count == 1
    assert any(step.error == expected_error for step in result.trajectory.steps)


def test_parse_failure_triggers_reflection() -> None:
    run_case(["this is not an action", final()], "unused", "parse_error")


def test_empty_search_triggers_reflection() -> None:
    run_case([action("missing page"), final()], "No Wikipedia results found for: missing page", "empty_search")


def test_execution_error_triggers_reflection() -> None:
    run_case([action("bad code"), final()], "[execution_error]\nSyntaxError: invalid syntax", "tool_error")


def test_repeated_action_triggers_reflection() -> None:
    model = ScriptedModel([action("same"), action("same"), action("same"), final()])
    agent = ReflectionAgent(model=model, tools=[StaticTool("normal observation")], max_steps=5)
    result = agent.run("test task")
    assert result.answer == "ok"
    assert result.trajectory.reflection_count == 1
    assert any(step.error == "repeated_action" for step in result.trajectory.steps)


if __name__ == "__main__":
    test_parse_failure_triggers_reflection()
    test_empty_search_triggers_reflection()
    test_execution_error_triggers_reflection()
    test_repeated_action_triggers_reflection()
    print("ReflectionAgent smoke tests passed.")
