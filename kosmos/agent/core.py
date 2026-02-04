"""KosmosAgent: QSE cognition + tool use + LLM reasoning in a living world."""

import threading
import time
import numpy as np
from typing import Optional

from emile_mini import EmileAgent, QSEConfig
from emile_mini.goal_v2 import GoalModuleV2

from ..world.grid import KosmosWorld
from ..world.objects import (
    Food, Water, Hazard, CraftItem, CRAFT_RECIPES, Biome,
)
from ..tools.registry import ToolRegistry
from ..tools.builtins import get_builtin_tools
from ..llm.ollama import OllamaReasoner


DIRECTIONS = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


class KosmosAgent:
    """
    An agent with:
    - QSE wavefunction (emile-mini) for cognitive dynamics
    - TD(lambda) goal selection
    - Tool registry for structured actions
    - LLM reasoning for tool selection + inner monologue
    - Inventory and survival mechanics
    """

    def __init__(self, world: KosmosWorld, model: str = "llama3.1:8b"):
        self.world = world

        # Position and facing
        self.pos = (world.size // 2, world.size // 2)
        self.facing = "east"

        # Survival stats
        self.energy = 1.0
        self.hydration = 1.0
        self.alive = True
        self.deaths = 0
        self.total_ticks = 0

        # Inventory
        self.inventory: list[CraftItem] = []
        self.crafted: list[str] = []  # names of crafted items

        # QSE cognitive engine
        self.config = QSEConfig()
        self.config.GRID_SIZE = 64
        self.emile = EmileAgent(self.config)
        self.goal_module = GoalModuleV2(self.config)
        self.goal_module.epsilon = 0.2
        self.goal_module.alpha = 0.3
        self.goal_module.gamma = 0.99

        # LLM reasoner
        self.llm = OllamaReasoner(model=model)
        self.use_llm = False  # set True after checking availability

        # Tool registry
        self.tools = ToolRegistry()
        self._register_tools()

        # Background QSE thread
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Cognitive state (read by renderer)
        self.strategy = "explore"
        self.context = 0
        self.entropy = 0.5
        self.surplus_mean = 0.0

        # Memory: simple list of significant events
        self.memories: list[str] = []

        # Stats
        self.food_eaten = 0
        self.water_drunk = 0
        self.damage_taken = 0
        self.cells_visited: set[tuple] = {self.pos}
        self.steps_taken = 0

        # Last action result (for renderer)
        self.last_action: dict = {}
        self.last_thought: str = ""
        self.last_narration: str = ""

    # ------------------------------------------------------------------ #
    #  Tool binding                                                        #
    # ------------------------------------------------------------------ #
    def _register_tools(self):
        for tool in get_builtin_tools():
            # Bind tool functions
            fn_map = {
                "move": self._tool_move,
                "examine": self._tool_examine,
                "pickup": self._tool_pickup,
                "consume": self._tool_consume,
                "craft": self._tool_craft,
                "rest": self._tool_rest,
                "remember": self._tool_remember,
                "wait": self._tool_wait,
            }
            tool.fn = fn_map.get(tool.name)
            self.tools.register(tool)

    # ------------------------------------------------------------------ #
    #  Tool implementations                                                #
    # ------------------------------------------------------------------ #
    def _tool_move(self, direction: str = "north") -> str:
        d = direction.lower()
        if d not in DIRECTIONS:
            return f"Unknown direction: {d}"
        dr, dc = DIRECTIONS[d]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= nr < self.world.size and 0 <= nc < self.world.size):
            return "Blocked by world edge."
        # Check for solid objects
        for obj in self.world.objects_at((nr, nc)):
            if obj.solid:
                return f"Blocked by {obj.name}."
        # Apply move cost (biome-dependent)
        cost = self.world.move_cost((nr, nc))
        # Crafted axe reduces forest cost
        if "axe" in self.crafted:
            biome = self.world.biomes[nr, nc]
            if biome == Biome.FOREST:
                cost *= 0.5
        # Crafted rope reduces water cost
        if "rope" in self.crafted:
            biome = self.world.biomes[nr, nc]
            if biome == Biome.WATER:
                cost *= 0.4
        self.energy -= cost
        self.pos = (nr, nc)
        self.facing = d
        self.cells_visited.add(self.pos)
        self.steps_taken += 1
        # Check for hazards at new position
        hazard_msg = ""
        for obj in self.world.objects_at(self.pos):
            if isinstance(obj, Hazard):
                dmg = obj.damage
                if "sling" in self.crafted:
                    dmg *= 0.3
                self.energy -= dmg
                self.damage_taken += 1
                hazard_msg = f" Ouch! Hit {obj.name} (-{dmg:.0%} energy)."
                self._remember(f"Encountered {obj.name} at {self.pos}, took damage.")
        biome = self.world.biomes[self.pos].value
        return f"Moved {d} to {self.pos} ({biome}).{hazard_msg}"

    def _tool_examine(self, target: str = "surroundings") -> str:
        if target == "surroundings":
            nearby = self.world.objects_near(self.pos, radius=4)
            biome = self.world.biomes[self.pos].value
            tod = self.world.time_of_day
            here = self.world.objects_at(self.pos)
            here_str = ", ".join(o.name for o in here) if here else "nothing"
            near_summary = {}
            for dist, pos, obj in nearby:
                if dist > 0:
                    near_summary[obj.name] = near_summary.get(obj.name, 0) + 1
            near_str = ", ".join(f"{v}x {k}" for k, v in near_summary.items()) if near_summary else "nothing"
            return (
                f"Standing in {biome} at {self.pos}. Time: {tod}. "
                f"Here: {here_str}. Nearby: {near_str}."
            )
        else:
            # Examine specific object here
            for obj in self.world.objects_at(self.pos):
                if target.lower() in obj.name.lower():
                    info = f"{obj.name} ({obj.symbol})"
                    if isinstance(obj, Food):
                        info += f" - edible, energy +{obj.energy_value:.0%}"
                    elif isinstance(obj, Water):
                        info += f" - drinkable, hydration +{obj.hydration_value:.0%}"
                    elif isinstance(obj, Hazard):
                        info += f" - dangerous! damage {obj.damage:.0%}"
                    elif isinstance(obj, CraftItem):
                        info += f" - craft material ({obj.craft_tag})"
                    return info
            return f"No '{target}' here."

    def _tool_pickup(self, item: str = "") -> str:
        for obj in self.world.objects_at(self.pos):
            if isinstance(obj, CraftItem) and item.lower() in obj.name.lower():
                self.world.objects[self.pos].remove(obj)
                if not self.world.objects[self.pos]:
                    del self.world.objects[self.pos]
                self.inventory.append(obj)
                self._remember(f"Picked up {obj.name} at {self.pos}.")
                return f"Picked up {obj.name} ({obj.craft_tag})."
        return f"No '{item}' to pick up here."

    def _tool_consume(self, item: str = "") -> str:
        for obj in self.world.objects_at(self.pos):
            if isinstance(obj, Food) and item.lower() in obj.name.lower():
                self.world.objects[self.pos].remove(obj)
                if not self.world.objects[self.pos]:
                    del self.world.objects[self.pos]
                self.energy = min(1.0, self.energy + obj.energy_value)
                self.food_eaten += 1
                self._remember(f"Ate {obj.name} at {self.pos}, energy now {self.energy:.0%}.")
                return f"Ate {obj.name}. Energy +{obj.energy_value:.0%} -> {self.energy:.0%}."
            if isinstance(obj, Water) and item.lower() in obj.name.lower():
                self.world.objects[self.pos].remove(obj)
                if not self.world.objects[self.pos]:
                    del self.world.objects[self.pos]
                self.hydration = min(1.0, self.hydration + obj.hydration_value)
                self.water_drunk += 1
                return f"Drank {obj.name}. Hydration +{obj.hydration_value:.0%}."
        # If no specific item named, consume first available food/water
        if not item:
            for obj in list(self.world.objects_at(self.pos)):
                if isinstance(obj, Food):
                    return self._tool_consume(item=obj.name)
                if isinstance(obj, Water):
                    return self._tool_consume(item=obj.name)
        return f"Nothing to consume here."

    def _tool_craft(self, item1: str = "", item2: str = "") -> str:
        inv_tags = {i: obj for obj in self.inventory for i in [obj.craft_tag, obj.name]}
        tag1 = None
        tag2 = None
        obj1 = obj2 = None
        for obj in self.inventory:
            if item1.lower() in (obj.name.lower(), obj.craft_tag.lower()) and tag1 is None:
                tag1 = obj.craft_tag
                obj1 = obj
            elif item2.lower() in (obj.name.lower(), obj.craft_tag.lower()) and tag2 is None:
                tag2 = obj.craft_tag
                obj2 = obj
        if tag1 is None or tag2 is None:
            return "Don't have those items."
        # Check recipes (order-independent)
        key = tuple(sorted([tag1, tag2]))
        recipe = CRAFT_RECIPES.get(key)
        if recipe is None:
            return f"Can't combine {tag1} and {tag2}."
        result_name, desc = recipe
        self.inventory.remove(obj1)
        self.inventory.remove(obj2)
        self.crafted.append(result_name)
        self._remember(f"Crafted {result_name}: {desc}")
        return f"Crafted {result_name}! {desc}"

    def _tool_rest(self) -> str:
        recovery = 0.03
        biome = self.world.biomes[self.pos]
        if biome == Biome.FOREST:
            recovery = 0.05  # sheltered
        elif biome == Biome.DESERT:
            recovery = 0.01  # harsh
        self.energy = min(1.0, self.energy + recovery)
        return f"Rested. Energy +{recovery:.0%} -> {self.energy:.0%}."

    def _tool_remember(self, query: str = "") -> str:
        if not self.memories:
            return "No memories yet."
        # Simple keyword search
        words = query.lower().split()
        scored = []
        for mem in self.memories:
            score = sum(1 for w in words if w in mem.lower())
            if score > 0:
                scored.append((score, mem))
        scored.sort(reverse=True)
        if not scored:
            return f"No memories matching '{query}'."
        hits = [m for _, m in scored[:3]]
        return "Memories: " + " | ".join(hits)

    def _tool_wait(self) -> str:
        return "Waited and observed."

    def _remember(self, event: str):
        """Store a memory, keeping last 200."""
        self.memories.append(f"[t={self.total_ticks}] {event}")
        if len(self.memories) > 200:
            self.memories = self.memories[-200:]

    # ------------------------------------------------------------------ #
    #  Background QSE evolution                                            #
    # ------------------------------------------------------------------ #
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._thread.start()
        # Check LLM availability
        self.use_llm = self.llm.check_available()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _bg_loop(self):
        while self._running:
            with self._lock:
                self.emile.step(dt=0.01)
            time.sleep(0.05)

    # ------------------------------------------------------------------ #
    #  Main tick: perceive -> think -> act                                 #
    # ------------------------------------------------------------------ #
    def tick(self) -> dict:
        """One agent tick. Returns action result dict."""
        if not self.alive:
            self._respawn()

        self.total_ticks += 1

        # Basal metabolism
        self.energy -= 0.003
        self.hydration -= 0.001

        # Death check
        if self.energy <= 0:
            self.alive = False
            self.deaths += 1
            self._remember(f"Died of exhaustion at {self.pos}. Death #{self.deaths}.")
            self.last_action = {"tool": "death", "result": "Ran out of energy."}
            self.last_thought = "Everything fades..."
            return self.last_action

        # 1. Update QSE state
        with self._lock:
            result = self.emile.step(dt=0.01, external_input={
                "reward": 0.0,
            })
        self.context = result["context"]
        self.surplus_mean = result["surplus_mean"]
        self.entropy = float(np.clip(1.0 - self.surplus_mean, 0.05, 0.95))
        energy_for_goal = self.emile.body.state.energy if hasattr(self.emile, "body") else 0.5
        self.strategy = self.goal_module.select_goal(self.context, energy_for_goal, self.entropy)

        # 2. Build situation description
        situation = self._build_situation()

        # 3. Decide action (LLM or heuristic)
        if self.use_llm:
            # QSE strategy influences which tool categories the LLM sees
            categories = self._strategy_tool_categories()
            schemas = self.tools.schemas(categories)
            decision = self.llm.reason(
                situation=situation,
                tools=schemas,
                strategy=self.strategy,
                entropy=self.entropy,
                energy=self.energy,
                inventory=[f"{o.name} ({o.craft_tag})" for o in self.inventory],
                memory_hits=self._recent_relevant_memories(),
            )
        else:
            decision = self._heuristic_decide()

        self.last_thought = decision.get("thought", "")

        # 4. Execute tool
        tool_name = decision.get("tool", "wait")
        args = decision.get("args", {})
        result = self.tools.invoke(tool_name, **args)

        # 5. Compute reward for TD(lambda)
        reward = self._compute_reward(tool_name, result)
        self.goal_module.update(reward, self.context, energy_for_goal)

        # 6. Feed reward back to QSE
        with self._lock:
            self.emile.step(dt=0.005, external_input={"reward": reward})

        self.last_action = {
            "tool": tool_name,
            "args": args,
            "result": result.get("result", result.get("error", "")),
            "reward": reward,
        }
        return self.last_action

    def _build_situation(self) -> str:
        """Describe current situation for LLM."""
        biome = self.world.biomes[self.pos].value
        tod = self.world.time_of_day
        here = self.world.objects_at(self.pos)
        here_str = ", ".join(o.name for o in here) if here else "nothing"
        nearby = self.world.objects_near(self.pos, radius=3)
        near_food = sum(1 for d, _, o in nearby if isinstance(o, Food) and d > 0)
        near_water = sum(1 for d, _, o in nearby if isinstance(o, Water) and d > 0)
        near_hazard = sum(1 for d, _, o in nearby if isinstance(o, Hazard) and d > 0)

        parts = [
            f"Position: {self.pos} in {biome}. Time: {tod}.",
            f"Energy: {self.energy:.0%}. Hydration: {self.hydration:.0%}.",
            f"Here: {here_str}.",
        ]
        if near_food:
            parts.append(f"{near_food} food source(s) nearby.")
        if near_water:
            parts.append(f"{near_water} water source(s) nearby.")
        if near_hazard:
            parts.append(f"WARNING: {near_hazard} hazard(s) nearby!")
        if self.energy < 0.25:
            parts.append("CRITICAL: Energy dangerously low!")
        return " ".join(parts)

    def _strategy_tool_categories(self) -> list[str]:
        """QSE strategy determines which tools the LLM considers."""
        base = ["action"]
        if self.strategy in ("explore", "learn"):
            base.append("perception")
        if self.strategy == "learn":
            base.append("meta")
        if self.strategy == "social":
            base.append("social")
        return base

    def _recent_relevant_memories(self) -> list[str]:
        """Return last few memories."""
        return self.memories[-5:] if self.memories else []

    def _compute_reward(self, tool_name: str, result: dict) -> float:
        """Reward signal for TD(lambda) learning."""
        r = -0.01  # small step cost (existence tax)

        res = str(result.get("result", ""))

        if "Ate" in res or "Drank" in res:
            r += 0.5
        if "Crafted" in res:
            r += 0.8
        if "Picked up" in res:
            r += 0.2
        if "Ouch" in res or "damage" in res.lower():
            r -= 0.4
        if tool_name == "move" and self.pos not in self.cells_visited:
            r += 0.1  # exploration bonus
        if self.energy < 0.15:
            r -= 0.2  # danger penalty

        return float(np.clip(r, -1.0, 1.0))

    def _heuristic_decide(self) -> dict:
        """Fallback decision-making when LLM is unavailable."""
        # Emergency: eat if food here and low energy
        if self.energy < 0.3:
            for obj in self.world.objects_at(self.pos):
                if isinstance(obj, Food):
                    return {"tool": "consume", "args": {"item": obj.name},
                            "thought": "Need food urgently."}
            # Move toward nearest food
            nearby = self.world.objects_near(self.pos, radius=5)
            for dist, pos, obj in nearby:
                if isinstance(obj, Food):
                    direction = self._direction_toward(pos)
                    if direction:
                        return {"tool": "move", "args": {"direction": direction},
                                "thought": f"Food nearby, heading {direction}."}

        # Consume if standing on food/water
        for obj in self.world.objects_at(self.pos):
            if isinstance(obj, Food):
                return {"tool": "consume", "args": {"item": obj.name},
                        "thought": "Food right here."}
            if isinstance(obj, Water) and self.hydration < 0.6:
                return {"tool": "consume", "args": {"item": obj.name},
                        "thought": "Should drink."}
            if isinstance(obj, CraftItem) and len(self.inventory) < 6:
                return {"tool": "pickup", "args": {"item": obj.name},
                        "thought": "Useful item."}

        # Try crafting if we have 2+ items
        if len(self.inventory) >= 2:
            for i, a in enumerate(self.inventory):
                for b in self.inventory[i + 1:]:
                    key = tuple(sorted([a.craft_tag, b.craft_tag]))
                    if key in CRAFT_RECIPES and CRAFT_RECIPES[key][0] not in self.crafted:
                        return {"tool": "craft",
                                "args": {"item1": a.name, "item2": b.name},
                                "thought": "Can craft something!"}

        # Rest if energy low
        if self.energy < 0.2 and self.strategy == "rest":
            return {"tool": "rest", "args": {}, "thought": "Must rest."}

        # Default: move in a semi-random direction biased by strategy
        dirs = list(DIRECTIONS.keys())
        if self.strategy == "exploit":
            # Move toward nearest food
            nearby = self.world.objects_near(self.pos, radius=6)
            for _, pos, obj in nearby:
                if isinstance(obj, Food):
                    d = self._direction_toward(pos)
                    if d:
                        return {"tool": "move", "args": {"direction": d},
                                "thought": "Heading toward food."}
        # Random exploration
        d = dirs[np.random.randint(len(dirs))]
        return {"tool": "move", "args": {"direction": d},
                "thought": "Wandering..."}

    def _direction_toward(self, target: tuple) -> str | None:
        """Return cardinal direction from self.pos toward target."""
        dr = target[0] - self.pos[0]
        dc = target[1] - self.pos[1]
        if abs(dr) >= abs(dc):
            return "south" if dr > 0 else "north"
        else:
            return "east" if dc > 0 else "west"

    def _respawn(self):
        """Respawn after death. Keep memories, lose inventory."""
        self.pos = (self.world.size // 2, self.world.size // 2)
        self.energy = 0.8
        self.hydration = 0.8
        self.alive = True
        self.inventory.clear()
        self.crafted.clear()
        self._remember(f"Respawned at {self.pos}. Memories intact. Inventory lost.")

    # ------------------------------------------------------------------ #
    #  State for renderer                                                  #
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        return {
            "pos": self.pos,
            "facing": self.facing,
            "energy": self.energy,
            "hydration": self.hydration,
            "alive": self.alive,
            "strategy": self.strategy,
            "context": self.context,
            "entropy": self.entropy,
            "inventory": [o.name for o in self.inventory],
            "crafted": self.crafted,
            "food_eaten": self.food_eaten,
            "water_drunk": self.water_drunk,
            "deaths": self.deaths,
            "cells_visited": len(self.cells_visited),
            "steps": self.steps_taken,
            "total_ticks": self.total_ticks,
            "thought": self.last_thought,
            "last_action": self.last_action,
            "time_of_day": self.world.time_of_day,
            "season": self.world.season,
            "use_llm": self.use_llm,
        }
