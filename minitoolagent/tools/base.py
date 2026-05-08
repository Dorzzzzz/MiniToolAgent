from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a string observation."""

    def get_schema(self) -> dict:
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def describe_for_prompt(self) -> str:
        params_desc = ", ".join(
            f'{p.name} ({p.type}, {"required" if p.required else "optional"}): {p.description}'
            for p in self.parameters
        )
        return f"- **{self.name}**: {self.description}\n  Parameters: {params_desc}"


class ToolRegistry:
    """Registry that holds available tools for the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def describe_all(self) -> str:
        return "\n\n".join(t.describe_for_prompt() for t in self._tools.values())

    def get_all_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
