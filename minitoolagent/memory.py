from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ActionStep:
    """One step in the agent trajectory."""
    step: int
    thought: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    observation: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrajectory:
    """Full trajectory of an agent run."""
    task: str = ""
    steps: list[ActionStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    reflection_count: int = 0
    total_steps: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_step(self, step: ActionStep) -> None:
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def finish(self, answer: str) -> None:
        self.final_answer = answer
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "final_answer": self.final_answer,
            "success": self.success,
            "reflection_count": self.reflection_count,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(self.elapsed, 2),
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def format_log(self) -> str:
        lines = [f"=== Task: {self.task} ==="]
        for s in self.steps:
            lines.append(f"\n--- Step {s.step} ---")
            if s.thought:
                lines.append(f"Thought: {s.thought}")
            if s.tool_name:
                lines.append(f"Action: {s.tool_name}({json.dumps(s.tool_args, ensure_ascii=False)})")
            if s.observation:
                lines.append(f"Observation: {s.observation}")
            if s.error:
                lines.append(f"Error: {s.error}")
        lines.append(f"\n=== Final Answer: {self.final_answer} ===")
        lines.append(f"=== Steps: {self.total_steps} | Time: {self.elapsed:.1f}s ===")
        return "\n".join(lines)
