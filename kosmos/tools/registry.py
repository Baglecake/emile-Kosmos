"""Tool registry: structured actions the agent can take."""

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class Tool:
    """A tool the agent can invoke."""
    name: str
    description: str
    parameters: dict        # param_name -> description
    fn: Callable = None     # bound at runtime by the agent
    category: str = "action"  # action, perception, social, meta

    def schema(self) -> dict:
        """Return a JSON-like schema for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                k: {"type": "string", "description": v}
                for k, v in self.parameters.items()
            },
        }


class ToolRegistry:
    """Registry of available tools, filterable by category."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def schemas(self, categories: list[str] | None = None) -> list[dict]:
        """Return tool schemas, optionally filtered by category."""
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.schema() for t in tools]

    def invoke(self, name: str, **kwargs) -> dict:
        """Invoke a tool by name. Returns result dict."""
        tool = self._tools.get(name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool: {name}"}
        if tool.fn is None:
            return {"success": False, "error": f"Tool '{name}' has no bound function"}
        try:
            result = tool.fn(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
