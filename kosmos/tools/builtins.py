"""Built-in tools for the Kosmos agent."""

from .registry import Tool


def get_builtin_tools() -> list[Tool]:
    """Return all built-in tools (functions are bound by the agent at runtime)."""
    return [
        Tool(
            name="move",
            description="Move one cell in a direction.",
            parameters={"direction": "One of: north, south, east, west"},
            category="action",
        ),
        Tool(
            name="examine",
            description="Look at an object at current position or nearby. Returns details about it.",
            parameters={"target": "Name of object to examine, or 'surroundings' for overview"},
            category="perception",
        ),
        Tool(
            name="pickup",
            description="Pick up a craft item at current position and add to inventory.",
            parameters={"item": "Name of item to pick up"},
            category="action",
        ),
        Tool(
            name="consume",
            description="Eat food or drink water at current position to restore energy/hydration.",
            parameters={"item": "Name of food or water source to consume"},
            category="action",
        ),
        Tool(
            name="craft",
            description="Combine two inventory items to create something new.",
            parameters={
                "item1": "First inventory item name or craft tag",
                "item2": "Second inventory item name or craft tag",
            },
            category="action",
        ),
        Tool(
            name="rest",
            description="Rest in place to recover a small amount of energy. Takes a full turn.",
            parameters={},
            category="action",
        ),
        Tool(
            name="remember",
            description="Search your memory for past experiences relevant to a query.",
            parameters={"query": "What to search for in memory"},
            category="meta",
        ),
        Tool(
            name="wait",
            description="Do nothing this turn. Observe the world.",
            parameters={},
            category="action",
        ),
    ]
