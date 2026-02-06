"""KosmosAgent: QSE cognition + tool use + LLM reasoning in a living world."""

import threading
import time
import numpy as np
from typing import Optional

from emile_mini import EmileAgent, QSEConfig
from emile_mini.goal_v2 import GoalModuleV2
from emile_mini.goal_mapper import GoalMapper

from ..world.grid import KosmosWorld
from ..world.objects import (
    Food, Water, Hazard, CraftItem, CRAFT_RECIPES, Biome,
    Herb, Seed, PlantedCrop,
)
from ..world.weather import WeatherType
from ..tools.registry import ToolRegistry
from ..tools.builtins import get_builtin_tools
from ..llm.ollama import OllamaReasoner, AgentState
from .action_policy import (
    KosmosActionPolicy,
    action_to_tool_call,
    decision_to_action_name,
    KOSMOS_ACTIONS,
)
from .demo_buffer import DemonstrationBuffer, behavior_cloning_update
from .surplus_tension import SurplusTensionModule


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
        self._last_move_was_new_cell = False  # For exploration reward bug fix
        self.steps_taken = 0

        # Last action result (for renderer)
        self.last_action: dict = {}
        self.last_thought: str = ""
        self.last_narration: str = ""

        # Layer 2: GoalMapper (strategy -> embodied goal)
        if self.config.GOAL_MAPPER_LEARNED:
            self.goal_mapper = GoalMapper(
                alpha=0.15, gamma=0.8, lambda_trace=0.7,
                epsilon_base=0.3, warm_start=True,
            )
        else:
            self.goal_mapper = None
        self.embodied_goal = "explore_space"

        # Layer 3: KosmosActionPolicy (goal -> tool action)
        if self.config.LEARNED_POLICY_ENABLED:
            self.action_policy = KosmosActionPolicy(
                hidden_dim=self.config.POLICY_HIDDEN_DIM,
                lr=self.config.POLICY_LR,
                gamma=self.config.POLICY_GAMMA,
            )
        else:
            self.action_policy = None

        # Teacher-student decay state
        self._teacher_prob = 1.0
        self._learned_reward_ema = 0.0
        self._heuristic_reward_ema = 0.0
        self._learned_samples = 0
        self._used_learned = False
        self._decision_source = "teacher"
        self._policy_step_count = 0
        self._consciousness_zone = "healthy"  # crisis/struggling/healthy/transcendent

        # Async LLM reasoning
        self._llm_pending: Optional[dict] = None
        self._llm_busy = False
        self._llm_pending_tick = 0

        # Event detection for 5b: event-triggered LLM calls
        self._prev_biome: Optional[str] = None
        self._prev_strategy: Optional[str] = None
        self._prev_zone: Optional[str] = None
        self._prev_weather: Optional[str] = None
        self._seen_objects: set[str] = set()  # for first-discovery events
        self._ticks_since_llm = 0  # force periodic refresh
        self._last_significant_event: str = ""  # what triggered the LLM call

        # Phase 6b: Strategy dwell time (reduce oscillation for τ′-scheduling)
        self._strategy_dwell_ticks = 0  # How long current strategy has been held
        self._min_strategy_dwell = 10   # Minimum ticks before strategy can change
        self._pending_strategy: Optional[str] = None  # Strategy waiting to be adopted

        # 5c: Surplus-faucet goal pressure
        self._goal_satisfaction = 0.5  # EMA of recent goal achievement
        self._recent_rewards: list[float] = []  # last N rewards for satisfaction calc

        # 5d: Information metabolism (anti-camping)
        self._recent_positions: list[tuple] = []  # last N positions for novelty calc
        self._novelty = 0.5  # current novelty level (0=camping, 1=exploring)

        # 5e: Multi-step LLM planning
        self._current_plan: list[dict] = []  # queue of planned actions
        self._plan_goal: str = ""  # what the plan is trying to achieve
        self._plan_replan_if: list[str] = []  # conditions that trigger replan
        self._plan_strategy: str = ""  # strategy when plan was created
        self._use_planning = True  # enable/disable planning mode

        # 5f: Anti-oscillation (from complete_navigation_system_e.py)
        self._recent_actions: list[str] = []  # last N action names
        self._action_repeat_window = 10  # how many actions to track

        # 5g: Stuckness detection (from maze_environment.py)
        self._stuckness_threshold = 3  # <= this many unique positions = stuck
        self._stuckness_window = 20  # positions to consider
        self._is_stuck = False
        self._stuck_ticks = 0  # how long we've been stuck

        # Demonstration buffer for behavior cloning (consolidation)
        # This is the "repetition + consolidation" that was missing
        self.demo_buffer = DemonstrationBuffer(max_size=20000)
        self._bc_train_interval = 200  # Train from demos every N ticks
        self._bc_batch_size = 64
        self._bc_learning_rate = 0.005

        # Competence-based teacher decay tracking
        self._death_rate_window: list[int] = []  # Recent death tick deltas
        self._last_death_tick = 0
        self._ticks_since_death = 0
        self._death_rate_ema = 30.0  # Deaths per 1000 ticks (start pessimistic)
        self._baseline_death_rate = 30.0  # Teacher's best death rate (frozen after warmup)
        self._baseline_frozen = False  # Freeze baseline at end of warmup

        # Food memory for improved survival
        self._last_known_food_pos: tuple | None = None  # Last position where food was seen

        # Phase 6: Surplus/Tension Module (principled QSE metrics)
        self.surplus_tension = SurplusTensionModule()
        self._st_metrics: dict = {}  # Latest surplus/tension metrics

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
                "plant": self._tool_plant,
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
        # Apply move cost (biome-dependent, weather-aware)
        cost = self.world.move_cost((nr, nc), direction=d)
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
        # Shelter frame reduces night penalty
        if "shelter_frame" in self.crafted and self.world.is_night:
            cost *= 0.7
        self.energy -= cost
        self.pos = (nr, nc)
        self.facing = d
        # Track if this is a new cell BEFORE adding (for exploration reward)
        self._last_move_was_new_cell = self.pos not in self.cells_visited
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
            nearby = self.world.objects_near(self.pos, radius=self.world.examine_radius)
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

    @property
    def inventory_capacity(self) -> int:
        base = 6
        if "basket" in self.crafted:
            base = 10
        return base

    def _tool_pickup(self, item: str = "") -> str:
        if len(self.inventory) >= self.inventory_capacity:
            return "Inventory full."
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
                energy_gain = obj.energy_value
                # Flint enables cooking for better food value
                if "flint" in self.crafted:
                    energy_gain *= 1.3
                self.energy = min(1.0, self.energy + energy_gain)
                self.food_eaten += 1
                # Clear food memory - we ate the food at this location
                if self._last_known_food_pos == self.pos:
                    self._last_known_food_pos = None
                extra = ""
                if isinstance(obj, Herb) and hasattr(obj, 'heal_value'):
                    extra = " Feeling better."
                self._remember(f"Ate {obj.name} at {self.pos}, energy now {self.energy:.0%}.")
                return f"Ate {obj.name}. Energy +{energy_gain:.0%} -> {self.energy:.0%}.{extra}"
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
        # Shelter frame bonus
        if "shelter_frame" in self.crafted:
            recovery *= 1.3
        # Storm penalty if exposed
        w = self.world.weather.current
        if w and w.weather_type == WeatherType.STORM:
            if biome in (Biome.PLAINS, Biome.DESERT):
                recovery *= 0.3
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

    def _tool_plant(self, item: str = "seed") -> str:
        for i, obj in enumerate(self.inventory):
            if isinstance(obj, Seed) or (isinstance(obj, CraftItem) and obj.craft_tag == "seed"):
                self.inventory.remove(obj)
                crop = PlantedCrop(position=self.pos)
                self.world._add_object(crop, self.pos)
                self._remember(f"Planted seed at {self.pos}.")
                return f"Planted a seed at {self.pos}. It will grow into food."
        return "No seeds to plant."

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
    #  Layer 2/3 helpers                                                   #
    # ------------------------------------------------------------------ #
    def _is_food_nearby(self) -> bool:
        nearby = self.world.objects_near(self.pos, radius=3)
        return any(isinstance(o, Food) for _, _, o in nearby)

    def _is_shelter_nearby(self) -> bool:
        if self.world.biomes[self.pos] == Biome.FOREST:
            return True
        r, c = self.pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.world.size and 0 <= nc < self.world.size:
                if self.world.biomes[nr, nc] == Biome.FOREST:
                    return True
        return False

    def _build_visual_field(self, radius: int = 3) -> str:
        """
        Build ASCII mini-map for LLM spatial perception.

        Legend:
          @ = agent (you)
          F = food, ~ = water, ! = hazard, + = craft item
          T = forest, : = desert, ^ = rock, . = plains, # = wall/edge
        """
        BIOME_CHARS = {
            Biome.PLAINS: ".",
            Biome.FOREST: "T",
            Biome.DESERT: ":",
            Biome.WATER: "~",
            Biome.ROCK: "^",
        }

        lines = []
        # Add north indicator
        lines.append("         N")

        for dr in range(-radius, radius + 1):
            row = []
            # West indicator on middle row
            if dr == 0:
                row.append("W ")
            else:
                row.append("  ")

            for dc in range(-radius, radius + 1):
                r, c = self.pos[0] + dr, self.pos[1] + dc

                if (r, c) == self.pos:
                    row.append("@")
                elif not (0 <= r < self.world.size and 0 <= c < self.world.size):
                    row.append("#")  # wall/edge
                else:
                    # Check objects at this cell (priority: hazard > food > water > item)
                    objs = self.world.objects_at((r, c))
                    if any(isinstance(o, Hazard) for o in objs):
                        row.append("!")
                    elif any(isinstance(o, Food) for o in objs):
                        row.append("F")
                    elif any(isinstance(o, Water) for o in objs):
                        row.append("~")
                    elif any(isinstance(o, CraftItem) for o in objs):
                        row.append("+")
                    else:
                        # Show biome
                        biome = self.world.biomes[r, c]
                        row.append(BIOME_CHARS.get(biome, "?"))

            # East indicator on middle row
            if dr == 0:
                row.append(" E")

            lines.append(" ".join(row))

        # Add south indicator
        lines.append("         S")

        return "\n".join(lines)

    def _build_policy_state_dict(self) -> dict:
        nearby = self.world.objects_near(self.pos, radius=3)
        near_food = sum(1 for d, _, o in nearby if isinstance(o, Food) and d > 0)
        near_water = sum(1 for d, _, o in nearby if isinstance(o, Water) and d > 0)
        near_hazard = sum(1 for d, _, o in nearby if isinstance(o, Hazard) and d > 0)
        near_craft = sum(1 for d, _, o in nearby if isinstance(o, CraftItem) and d > 0)

        here = self.world.objects_at(self.pos)
        can_craft = False
        for i, a in enumerate(self.inventory):
            for b in self.inventory[i + 1:]:
                key = tuple(sorted([a.craft_tag, b.craft_tag]))
                if key in CRAFT_RECIPES:
                    can_craft = True
                    break
            if can_craft:
                break

        # Compute directional cues to nearest food and hazard (critical for learning)
        # Larger radius for food (survival priority)
        food_dx, food_dy = 0.0, 0.0
        hazard_dx, hazard_dy = 0.0, 0.0
        search_radius = 8  # Match heuristic's search radius

        nearby_large = self.world.objects_near(self.pos, radius=search_radius)
        nearest_food_dist = float('inf')
        nearest_hazard_dist = float('inf')

        for dist, pos, obj in nearby_large:
            if dist == 0:
                continue  # Skip objects at current position

            if isinstance(obj, Food) and dist < nearest_food_dist:
                nearest_food_dist = dist
                # Normalized relative offset (-1 to 1)
                food_dx = (pos[1] - self.pos[1]) / search_radius
                food_dy = (pos[0] - self.pos[0]) / search_radius

            if isinstance(obj, Hazard) and dist < nearest_hazard_dist:
                nearest_hazard_dist = dist
                hazard_dx = (pos[1] - self.pos[1]) / search_radius
                hazard_dy = (pos[0] - self.pos[0]) / search_radius

        return dict(
            energy=self.energy,
            hydration=self.hydration,
            biome=self.world.biomes[self.pos].value,
            time_of_day=self.world.time_of_day,
            nearby_food=near_food,
            nearby_water=near_water,
            nearby_hazard=near_hazard,
            nearby_craft=near_craft,
            has_food_here=any(isinstance(o, Food) for o in here),
            has_water_here=any(isinstance(o, Water) for o in here),
            has_craft_here=any(isinstance(o, CraftItem) for o in here),
            has_hazard_here=any(isinstance(o, Hazard) for o in here),
            inventory_count=len(self.inventory),
            can_craft=can_craft,
            strategy=self.strategy,
            goal=self.embodied_goal,
            entropy=self.entropy,
            surplus_mean=self.surplus_mean,
            food_dx=food_dx,
            food_dy=food_dy,
            hazard_dx=hazard_dx,
            hazard_dy=hazard_dy,
            sigma_ema=self._st_metrics.get("sigma_ema", 0.0),
        )

    # ------------------------------------------------------------------ #
    #  Main tick: perceive -> think -> act                                 #
    # ------------------------------------------------------------------ #
    def tick(self) -> dict:
        """One agent tick. Returns action result dict."""
        if not self.alive:
            self._respawn()

        self.total_ticks += 1

        # Basal metabolism (tuned for complex behavior emergence)
        # Lower rate gives agent time for multi-step plans, crafting, cultivation
        self.energy -= 0.002  # was 0.003
        self.hydration -= 0.0008  # was 0.001

        # Weather effects on metabolism
        w = self.world.weather.current
        if w:
            if w.weather_type == WeatherType.HEAT_WAVE:
                self.energy -= 0.002 * w.intensity
                self.hydration -= 0.002 * w.intensity
            elif w.weather_type == WeatherType.STORM:
                if self.world.biomes[self.pos] in (Biome.PLAINS, Biome.DESERT):
                    self.energy -= 0.02 * w.intensity

        # Death check
        if self.energy <= 0:
            self.alive = False
            self.deaths += 1
            biome_name = self.world.biomes[self.pos].name
            weather_name = self.world.weather.current.weather_type.name if self.world.weather.current else "clear"
            print(f"[DEATH t={self.total_ticks}] pos={self.pos} biome={biome_name} weather={weather_name} "
                  f"zone={self._consciousness_zone} hydration={self.hydration:.2f} deaths={self.deaths}")
            self._remember(f"Died of exhaustion at {self.pos}. Death #{self.deaths}.")
            self.last_action = {"tool": "death", "result": "Ran out of energy."}

            # Update death rate tracking for competence-based decay
            ticks_since_death = self.total_ticks - self._last_death_tick
            self._death_rate_window.append(ticks_since_death)
            if len(self._death_rate_window) > 20:
                self._death_rate_window = self._death_rate_window[-20:]
            self._last_death_tick = self.total_ticks
            self._ticks_since_death = 0

            # Update death rate EMA (deaths per 1000 ticks)
            if ticks_since_death > 0:
                instant_rate = 1000.0 / ticks_since_death
                self._death_rate_ema = 0.9 * self._death_rate_ema + 0.1 * instant_rate

            # Mark recent demos as death-leading
            self.demo_buffer.on_death()
            # Record death for surplus/tension curvature calculation
            self.surplus_tension.record_death(self.total_ticks)
            self.last_thought = "Everything fades..."
            if self.goal_mapper is not None:
                self.goal_mapper.reset_episode()
            return self.last_action

        # 1. Update QSE state
        with self._lock:
            result = self.emile.step(dt=0.01, external_input={
                "reward": 0.0,
            })
        self.context = result["context"]
        self.surplus_mean = result["surplus_mean"]
        self.entropy = float(np.clip(result.get("normalized_entropy", 0.5), 0.05, 0.95))
        energy_for_goal = self.emile.body.state.energy if hasattr(self.emile, "body") else 0.5

        # 2. L1: Strategy selection with dwell time (reduces oscillation)
        proposed_strategy = self.goal_module.select_goal(self.context, energy_for_goal, self.entropy)

        # Enforce minimum dwell time to reduce strategy oscillation
        # This allows τ′-scheduling to work without constant strategy-change triggers
        self._strategy_dwell_ticks += 1
        if proposed_strategy != self.strategy:
            # Strategy wants to change
            if self._strategy_dwell_ticks >= self._min_strategy_dwell:
                # Dwell time met - allow change
                self.strategy = proposed_strategy
                self._strategy_dwell_ticks = 0
            else:
                # Store pending strategy (for monitoring)
                self._pending_strategy = proposed_strategy
        else:
            self._pending_strategy = None

        # 3. L2: GoalMapper -> embodied goal
        if self.goal_mapper is not None:
            food_nearby = self._is_food_nearby()
            shelter_nearby = self._is_shelter_nearby()
            self.embodied_goal = self.goal_mapper.select_goal(
                self.strategy, self.energy, self.energy,
                food_nearby, shelter_nearby, self.entropy,
            )

        # 4. Build situation description
        situation = self._build_situation()

        # 4.5 Phase 6: Surplus/Tension computation
        self._st_metrics = self.surplus_tension.step(self)

        # Check for rupture (Phase 6c - escape death traps)
        if self._st_metrics.get("should_rupture", False):
            k_eff = self._st_metrics.get('k_effective', 2.0)
            print(f"[RUPTURE t={self.total_ticks}] Σ={self._st_metrics['sigma_ema']:.2f} "
                  f"k={k_eff:.1f} pos={self.pos} ruptures={self.surplus_tension.ruptures_triggered}")
            # Clear current plan and force replan
            self._current_plan.clear()
            self._plan_goal = ""
            # Mark stuckness to trigger escape behavior in heuristic
            self._is_stuck = True
            self._stuck_ticks = 10  # Force escape
            # Reset strategy dwell to allow immediate strategy pivot
            self._strategy_dwell_ticks = 0
            self._pending_strategy = None
            # Reset internal model for fresh start
            self.surplus_tension.on_rupture()
            # Force LLM call for new plan
            self._ticks_since_llm = 999

        # 4.6 Consciousness zone classification (for survival override)
        # Crisis: force heuristic survival actions, no learned policy
        # Struggling: bias toward teacher, slow decay
        # Healthy: normal teacher-student balance
        # Transcendent: allow exploration, faster decay
        # NOTE: Crisis is energy-only — entropy affects creativity, not survival
        # Thresholds tuned to give agent time for complex behaviors
        if self.energy < 0.25:  # was 0.20 — earlier intervention
            self._consciousness_zone = "crisis"
        elif self.energy < 0.40:  # was 0.35 — more buffer
            self._consciousness_zone = "struggling"
        elif self.energy > 0.7 and self.entropy < 0.4:
            self._consciousness_zone = "transcendent"
        else:
            self._consciousness_zone = "healthy"

        # 5. Teacher-student decision (gated by consciousness zone)
        # In crisis zone, ALWAYS use heuristic — survival reflex override
        # EXCEPTION: If rupture triggered this tick, bypass crisis override
        # The heuristic itself may be causing the death loop - let LLM replan
        rupture_override = self._st_metrics.get("should_rupture", False)
        if self._consciousness_zone == "crisis" and not rupture_override:
            # Abort any active plan in crisis
            if self._current_plan:
                self._current_plan.clear()
            decision = self._heuristic_decide()
            self._decision_source = "survival_reflex"
            self._used_learned = False
        elif self.action_policy is not None and np.random.random() > self._teacher_prob:
            # Student: learned policy decides
            self._used_learned = True
            state_dict = self._build_policy_state_dict()
            action_name, _, _ = self.action_policy.select_action(
                state_dict, entropy=self.entropy,
            )
            decision = action_to_tool_call(action_name, self)
            self._decision_source = "learned"
        else:
            # Teacher path: use planning (5e) if enabled
            self._used_learned = False

            # Check for plan interrupts
            should_interrupt, interrupt_reason = self._check_plan_interrupt()
            if should_interrupt and self._current_plan:
                self._current_plan.clear()
                self._plan_goal = ""

            # Try to execute from current plan
            if self._use_planning and self._current_plan:
                decision = self._current_plan.pop(0)
                self._decision_source = "teacher_plan"
            elif self._llm_pending is not None:
                # Check if pending LLM result is a plan or single action
                staleness = self.total_ticks - self._llm_pending_tick
                if staleness <= 3:
                    pending = self._llm_pending
                    if "plan" in pending and isinstance(pending["plan"], list):
                        # It's a plan response
                        self._current_plan = pending["plan"]
                        self._plan_goal = pending.get("goal", "")
                        self._plan_replan_if = pending.get("replan_if", [])
                        self._plan_strategy = self.strategy
                        if self._current_plan:
                            decision = self._current_plan.pop(0)
                            self._decision_source = "teacher_plan"
                        else:
                            decision = self._heuristic_decide()
                            self._decision_source = "teacher_heuristic"
                    else:
                        # Single action response (backwards compatible)
                        decision = pending
                        self._decision_source = "teacher_llm"
                else:
                    decision = self._heuristic_decide()
                    self._decision_source = "teacher_heuristic"
                self._llm_pending = None
            else:
                decision = self._heuristic_decide()
                self._decision_source = "teacher_heuristic"

        # Fire off next LLM reasoning in background (event-triggered, 5b)
        # Only fire if we don't have an active plan (or plan is almost done)
        should_fire, fire_reason = self._should_fire_llm()
        need_new_plan = len(self._current_plan) <= 1  # Request plan when almost empty
        if self.use_llm and not self._llm_busy and should_fire and need_new_plan:
            self._llm_busy = True
            categories = self._strategy_tool_categories()
            schemas = self.tools.schemas(categories)
            tick_snapshot = self.total_ticks
            last_res = str(self.last_action.get("result", "")) if self.last_action else ""
            # Include the trigger reason in the situation for context
            situation_with_trigger = situation
            if fire_reason and fire_reason != "periodic refresh":
                situation_with_trigger = f"[Event: {fire_reason}]\n\n{situation}"
            # Build embodied agent state for LLM (Phase 7)
            nearby = self.world.objects_near(self.pos, radius=3)
            hazard_nearby = any(isinstance(o, Hazard) for _, _, o in nearby)
            food_nearby = any(isinstance(o, Food) for _, _, o in nearby)
            agent_state = AgentState(
                energy=self.energy,
                hydration=self.hydration,
                sigma_ema=self._st_metrics.get("sigma_ema", 0.0),
                hazard_nearby=hazard_nearby,
                food_nearby=food_nearby,
                in_crisis=(self._consciousness_zone == "crisis"),
            )

            llm_args = dict(
                situation=situation_with_trigger,
                tools=schemas,
                strategy=self.strategy,
                entropy=self.entropy,
                energy=self.energy,
                inventory=[f"{o.name} ({o.craft_tag})" for o in self.inventory],
                memory_hits=self._recent_relevant_memories(),
                last_result=last_res,
                agent_state=agent_state,
            )
            use_planning = self._use_planning

            def _llm_reason():
                try:
                    if use_planning:
                        result = self.llm.reason_plan(**llm_args)
                    else:
                        result = self.llm.reason(**llm_args)
                    self._llm_pending_tick = tick_snapshot
                    self._llm_pending = result
                finally:
                    self._llm_busy = False
            threading.Thread(target=_llm_reason, daemon=True).start()

        self.last_thought = decision.get("thought", "")

        # 6. Execute tool
        tool_name = decision.get("tool", "wait")
        args = decision.get("args", {})
        result = self.tools.invoke(tool_name, **args)

        # 7. Compute reward
        reward = self._compute_reward(tool_name, result)

        # 7b. Apply anti-oscillation penalty (5f)
        # Use granular action name (move_north vs move) for directional specificity
        granular_action_for_penalty = decision_to_action_name(decision)
        action_penalty = self._get_action_penalty(granular_action_for_penalty)
        reward = reward - action_penalty

        # 7c. Record teacher demonstrations for behavior cloning
        # Only record when teacher (not learned policy) makes the decision
        if not self._used_learned and granular_action_for_penalty in KOSMOS_ACTIONS:
            state_dict = self._build_policy_state_dict()
            self.demo_buffer.add(
                state_dict=state_dict,
                action_name=granular_action_for_penalty,
                reward=reward,
                source=self._decision_source,
            )

        # Track survival for competence evaluation
        self._ticks_since_death += 1
        # Update demo buffer with survival info periodically
        if self.total_ticks % 50 == 0:
            self.demo_buffer.update_survival(self._ticks_since_death)

        # 8. Update L1: GoalModuleV2
        self.goal_module.update(reward, self.context, energy_for_goal)

        # 9. Update L2: GoalMapper
        if self.goal_mapper is not None:
            food_now = self._is_food_nearby()
            shelter_now = self._is_shelter_nearby()
            self.goal_mapper.update(
                reward, self.strategy, self.energy, self.energy,
                food_now, shelter_now,
            )

        # 10. Update L3: ActionPolicy + teacher-student decay
        if self.action_policy is not None:
            if self._used_learned:
                self.action_policy.record_reward(reward)

            self._policy_step_count += 1
            if self._policy_step_count % self.config.POLICY_UPDATE_INTERVAL == 0:
                self.action_policy.update()

            # EMA reward tracking
            ema_alpha = 0.02
            if self._used_learned:
                self._learned_reward_ema = (
                    (1 - ema_alpha) * self._learned_reward_ema + ema_alpha * reward
                )
                self._learned_samples += 1
            else:
                self._heuristic_reward_ema = (
                    (1 - ema_alpha) * self._heuristic_reward_ema + ema_alpha * reward
                )

            # Competence-based teacher decay (replaces time-based decay)
            # Only reduce teacher probability when student is demonstrably competent
            decay = self.config.POLICY_TEACHER_DECAY
            floor = self.config.POLICY_TEACHER_MIN
            warmup = getattr(self.config, 'POLICY_TEACHER_WARMUP', 2000)

            if self._learned_samples < warmup:
                # During warmup: establish baseline, slow decay
                # Track the BEST (lowest) death rate as baseline - not EMA
                # This captures teacher's best performance, not degrading average
                if self._death_rate_ema < self._baseline_death_rate:
                    self._baseline_death_rate = self._death_rate_ema
                # Very slow decay during warmup
                self._teacher_prob = max(floor, self._teacher_prob * 0.9999)
            else:
                # Freeze baseline at end of warmup (only once)
                if not self._baseline_frozen:
                    self._baseline_frozen = True
                    print(f"[Baseline frozen] death_rate={self._baseline_death_rate:.1f} "
                          f"at samples={self._learned_samples}")
                # After warmup: competence-gated decay
                # Student is competent if:
                # 1. Reward EMA is at least 90% of heuristic EMA
                # 2. Death rate is not much worse than baseline
                reward_competent = self._learned_reward_ema >= 0.9 * self._heuristic_reward_ema
                survival_competent = self._death_rate_ema <= 1.2 * self._baseline_death_rate
                min_samples_for_eval = warmup + 500  # Need enough samples to evaluate

                if self._learned_samples >= min_samples_for_eval and reward_competent and survival_competent:
                    # Student is competent - decay teacher
                    self._teacher_prob = max(floor, self._teacher_prob * decay)
                elif self._learned_samples >= min_samples_for_eval and not survival_competent:
                    # Student failing survival - slow recovery of teacher
                    # But cap at 0.85 so student always gets ~15% of decisions to keep learning
                    self._teacher_prob = min(0.85, self._teacher_prob * 1.0005)
                # else: stay at current level, student still learning

            # Periodic behavior cloning from demo buffer
            if self.total_ticks % self._bc_train_interval == 0 and len(self.demo_buffer) >= self._bc_batch_size:
                demos = self.demo_buffer.sample(self._bc_batch_size, weighted=True)
                bc_stats = behavior_cloning_update(
                    self.action_policy, demos, learning_rate=self._bc_learning_rate
                )
                if self.total_ticks % 1000 == 0:
                    print(f"[BC t={self.total_ticks}] loss={bc_stats['loss']:.3f} "
                          f"acc={bc_stats['accuracy']:.2f} demos={len(self.demo_buffer)}")

        # Console logging every 100 ticks
        if self.total_ticks % 100 == 0:
            teacher_count = self.total_ticks - self._learned_samples
            st = self._st_metrics
            thresh = st.get('dynamic_threshold', 0.65)
            k_eff = st.get('k_effective', 2.0)
            print(
                f"[t={self.total_ticks}] zone={self._consciousness_zone} "
                f"teacher_prob={self._teacher_prob:.3f} "
                f"teacher={teacher_count} learned={self._learned_samples} "
                f"t_ema={self._heuristic_reward_ema:.3f} l_ema={self._learned_reward_ema:.3f} "
                f"death_rate={self._death_rate_ema:.1f} "
                f"S={st.get('surplus_ema', 0):.2f} Σ={st.get('sigma_ema', 0):.2f}/{thresh:.2f} k={k_eff:.1f} τ′={st.get('tau_prime', 1):.2f}"
            )

        # 11. Surplus-faucet goal pressure (5c)
        # Track recent rewards to compute goal satisfaction
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 20:
            self._recent_rewards = self._recent_rewards[-20:]

        # Compute goal satisfaction: how well are recent actions achieving goals?
        # Positive rewards = good, scaled 0-1
        if self._recent_rewards:
            avg_reward = np.mean(self._recent_rewards)
            # Map reward range [-1, 1] to satisfaction [0, 1]
            self._goal_satisfaction = 0.9 * self._goal_satisfaction + 0.1 * ((avg_reward + 1) / 2)
        self._goal_satisfaction = float(np.clip(self._goal_satisfaction, 0.1, 0.9))

        # Apply metabolic pressure: low satisfaction = higher energy cost
        # gamma_effective = gamma * (0.3 + 0.7 * goal_satisfaction)
        # Translate to: extra_cost = base_pressure * (1 - goal_satisfaction)
        # Reduced to give agent more breathing room for complex behaviors
        goal_pressure = 0.001 * (1 - self._goal_satisfaction)  # was 0.002
        self.energy -= goal_pressure

        # 11b. Information metabolism (5d: anti-camping)
        # Track recent positions to compute novelty
        self._recent_positions.append(self.pos)
        window_size = 30
        if len(self._recent_positions) > window_size:
            self._recent_positions = self._recent_positions[-window_size:]

        # Compute novelty: unique positions in recent window / window size
        unique_positions = len(set(self._recent_positions))
        raw_novelty = unique_positions / max(1, len(self._recent_positions))
        # EMA smoothing
        self._novelty = 0.9 * self._novelty + 0.1 * raw_novelty
        self._novelty = float(np.clip(self._novelty, 0.1, 0.9))

        # 11c. Check for stuckness (5g)
        self._check_stuckness()

        # Apply novelty-based energy modulation:
        # High novelty (exploring) = small energy bonus
        # Low novelty (camping) = small energy drain
        # Neutral point at novelty = 0.3 (reduced to encourage staying for cultivation)
        novelty_effect = (self._novelty - 0.3) * 0.002  # was 0.003, neutral at 0.4
        self.energy += novelty_effect

        # Modulate reward fed to QSE: amplify when achieving goals
        qse_reward = reward * (0.3 + 0.7 * self._goal_satisfaction)

        # 12. Feed modulated reward back to QSE
        with self._lock:
            self.emile.step(dt=0.005, external_input={"reward": qse_reward})

        # 13. Track action for anti-oscillation (5f)
        # Use granular action names (move_north, move_south, etc.) so directional
        # oscillation is penalized but exploring new directions is not.
        granular_action = decision_to_action_name(decision)
        self._recent_actions.append(granular_action)
        if len(self._recent_actions) > self._action_repeat_window:
            self._recent_actions = self._recent_actions[-self._action_repeat_window:]

        self.last_action = {
            "tool": tool_name,
            "args": args,
            "result": result.get("result", result.get("error", "")),
            "reward": reward,
        }
        return self.last_action

    def _build_situation(self) -> str:
        """Describe current situation for LLM with visual field."""
        biome = self.world.biomes[self.pos].value
        tod = self.world.time_of_day
        here = self.world.objects_at(self.pos)
        here_str = ", ".join(o.name for o in here) if here else "nothing"

        # Build visual field first
        # Radius=5 so LLM can see what the heuristic's radius=8 search might find
        visual_field = self._build_visual_field(radius=5)

        parts = [
            f"Your field of vision (@ = you):",
            visual_field,
            f"Legend: F=food, ~=water, !=hazard, +=item, T=forest, :=desert, ^=rock, .=plains",
            "",
            f"Position: {self.pos} in {biome}. Time: {tod}.",
            f"Energy: {self.energy:.0%}. Hydration: {self.hydration:.0%}.",
            f"Standing on: {here_str}.",
        ]

        weather_name = self.world.weather_name
        if weather_name != "clear":
            parts.append(f"Weather: {weather_name}.")
            if weather_name == "storm":
                parts.append("DANGER: Storm! Seek forest (T) for shelter.")
            elif weather_name == "fog":
                parts.append("Visibility reduced by fog.")

        if self.energy < 0.25:
            parts.append("CRITICAL: Energy dangerously low! Find food (F) immediately!")
        if self.hydration < 0.25:
            parts.append("CRITICAL: Dehydrated! Find water (~) immediately!")

        # Internal state (5a-alt: QSE-derived metrics)
        # Arousal from entropy (high entropy = high arousal/uncertainty)
        if self.entropy > 0.7:
            arousal = "agitated"
        elif self.entropy > 0.4:
            arousal = "alert"
        else:
            arousal = "calm"

        # Valence from recent reward history (approximate from EMAs)
        avg_ema = (self._learned_reward_ema + self._heuristic_reward_ema) / 2
        if avg_ema > 0.05:
            valence = "positive"
        elif avg_ema < -0.05:
            valence = "negative"
        else:
            valence = "neutral"

        # Consciousness zone already computed in tick()
        zone = self._consciousness_zone

        parts.append("")
        parts.append(f"Internal state: {arousal}, feeling {valence}, zone: {zone}")

        # Plan execution feedback (so LLM knows what happened to previous plans)
        if self._plan_goal:
            parts.append("")
            parts.append(f"Current plan goal: {self._plan_goal}")
            parts.append(f"Steps remaining: {len(self._current_plan)}")
        elif self.last_action and self.last_action.get("tool") != "death":
            # No active plan - tell LLM what just happened
            last_tool = self.last_action.get("tool", "unknown")
            last_result = self.last_action.get("result", "")
            last_reward = self.last_action.get("reward", 0)
            outcome = "successful" if last_reward > 0 else "unsuccessful" if last_reward < 0 else "neutral"
            parts.append("")
            parts.append(f"Last action: {last_tool} was {outcome}. Result: {last_result[:80]}")

        return "\n".join(parts)

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
        """Reward signal for TD(lambda) learning.

        Shaped to encourage survival:
        - Eating/drinking scaled by urgency (low energy = bigger reward)
        - Penalty for ignoring food when hungry
        - Penalty for failed consume attempts (teaches when to eat)
        """
        r = -0.01  # small step cost (existence tax)

        res = str(result.get("result", ""))

        # Eating reward scaled by urgency (0.5 base, up to 1.0 when starving)
        if "Ate" in res or "Drank" in res:
            urgency_bonus = 0.5 * (1.0 - self.energy)  # 0 when full, 0.5 when empty
            r += 0.5 + urgency_bonus
        if "Crafted" in res:
            r += 0.8
        if "Picked up" in res:
            r += 0.2
        if "Ouch" in res or "damage" in res.lower():
            r -= 0.4

        # Exploration bonus (reduced to not overshadow survival)
        # Use _last_move_was_new_cell flag (set before adding to cells_visited)
        if tool_name == "move" and getattr(self, '_last_move_was_new_cell', False):
            r += 0.05  # was 0.1 - reduced to prioritize survival
            self._last_move_was_new_cell = False  # Reset flag

        if self.energy < 0.15:
            r -= 0.2  # danger penalty

        # Penalty for ignoring food when hungry (teaches to eat when appropriate)
        has_food_here = any(isinstance(o, Food) for o in self.world.objects_at(self.pos))
        if has_food_here and self.energy < 0.5 and tool_name != "consume":
            r -= 0.15  # missed opportunity to eat

        # Penalty for failed consume (teaches to only consume when food present)
        if tool_name == "consume" and "Nothing to consume" in res:
            r -= 0.1  # wasted action

        # Phase 6d: Intrinsic reward shaping from surplus/tension
        # - High surplus (surprise) = small exploration bonus
        # - High curvature (structured failure) = penalty
        # Scaled down so survival rewards still dominate
        intrinsic = self.surplus_tension.get_intrinsic_reward()
        r += intrinsic

        return float(np.clip(r, -1.0, 1.0))

    def _heuristic_decide(self) -> dict:
        """Fallback decision-making when LLM is unavailable."""
        # PRIORITY 1: Hazard avoidance — flee if hazard within 2 cells
        nearby = self.world.objects_near(self.pos, radius=2)
        for dist, haz_pos, obj in nearby:
            if isinstance(obj, Hazard) and dist > 0:
                # Move away from hazard
                dr = self.pos[0] - haz_pos[0]
                dc = self.pos[1] - haz_pos[1]
                if abs(dr) >= abs(dc):
                    flee_dir = "south" if dr > 0 else "north"
                else:
                    flee_dir = "east" if dc > 0 else "west"
                # Validate the flee direction is in bounds
                nr = self.pos[0] + DIRECTIONS[flee_dir][0]
                nc = self.pos[1] + DIRECTIONS[flee_dir][1]
                if 0 <= nr < self.world.size and 0 <= nc < self.world.size:
                    return {"tool": "move", "args": {"direction": flee_dir},
                            "thought": "Danger! Fleeing from hazard."}
                # If can't flee that way, try perpendicular
                perp_dirs = ["east", "west"] if abs(dr) >= abs(dc) else ["north", "south"]
                for d in perp_dirs:
                    nr = self.pos[0] + DIRECTIONS[d][0]
                    nc = self.pos[1] + DIRECTIONS[d][1]
                    if 0 <= nr < self.world.size and 0 <= nc < self.world.size:
                        return {"tool": "move", "args": {"direction": d},
                                "thought": "Evading hazard."}

        # PRIORITY 2: Emergency food/water if low energy/hydration
        if self.energy < 0.45:  # was 0.35 — seek food earlier
            for obj in self.world.objects_at(self.pos):
                if isinstance(obj, Food):
                    return {"tool": "consume", "args": {"item": obj.name},
                            "thought": "Need food urgently."}
            # Adaptive search radius: larger when more desperate
            search_radius = 12 if self.energy < 0.30 else 8
            nearby = self.world.objects_near(self.pos, radius=search_radius)
            for dist, pos, obj in nearby:
                if isinstance(obj, Food):
                    # Remember this food location for future reference
                    self._last_known_food_pos = pos
                    direction = self._direction_toward(pos)
                    if direction:
                        return {"tool": "move", "args": {"direction": direction},
                                "thought": f"Food at {pos}, heading {direction}."}
            # No food in search radius - try last known food location
            if self._last_known_food_pos:
                # If we're at the remembered location but no food here, clear memory
                if self._last_known_food_pos == self.pos:
                    self._last_known_food_pos = None
                else:
                    direction = self._direction_toward(self._last_known_food_pos)
                    if direction:
                        return {"tool": "move", "args": {"direction": direction},
                                "thought": f"Returning to last known food at {self._last_known_food_pos}."}

        # PRIORITY 3: Emergency water if low hydration
        if self.hydration < 0.4:  # was 0.3 — seek water earlier
            for obj in self.world.objects_at(self.pos):
                if isinstance(obj, Water):
                    return {"tool": "consume", "args": {"item": obj.name},
                            "thought": "Need water urgently."}
            # Move toward nearest water (larger search radius)
            nearby = self.world.objects_near(self.pos, radius=8)  # was 6
            for dist, pos, obj in nearby:
                if isinstance(obj, Water):
                    direction = self._direction_toward(pos)
                    if direction:
                        return {"tool": "move", "args": {"direction": direction},
                                "thought": f"Water nearby, heading {direction}."}

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

        # If stuck, force random exploration to break out (5g)
        if self._is_stuck and self._stuck_ticks >= 10:
            dirs = ["north", "south", "east", "west"]
            # Avoid current facing direction to encourage new paths
            other_dirs = [d for d in dirs if d != self.facing]
            direction = other_dirs[int(np.random.randint(len(other_dirs)))]
            return {"tool": "move", "args": {"direction": direction},
                    "thought": "Breaking out of stuck area."}

        # Try crafting if we have 2+ items
        if len(self.inventory) >= 2:
            for i, a in enumerate(self.inventory):
                for b in self.inventory[i + 1:]:
                    key = tuple(sorted([a.craft_tag, b.craft_tag]))
                    if key in CRAFT_RECIPES and CRAFT_RECIPES[key][0] not in self.crafted:
                        return {"tool": "craft",
                                "args": {"item1": a.name, "item2": b.name},
                                "thought": "Can craft something!"}

        # Storm: seek shelter (forest)
        w = self.world.weather.current
        if w and w.weather_type == WeatherType.STORM and w.intensity > 0.5:
            if self.world.biomes[self.pos] not in (Biome.FOREST, Biome.ROCK):
                # Move toward forest
                for _, pos, _ in self.world.objects_near(self.pos, radius=5):
                    if self.world.biomes[pos] == Biome.FOREST:
                        d = self._direction_toward(pos)
                        if d:
                            return {"tool": "move", "args": {"direction": d},
                                    "thought": "Storm! Need shelter."}
                # No forest visible, try any non-exposed direction
                for d in DIRECTIONS:
                    nr = self.pos[0] + DIRECTIONS[d][0]
                    nc = self.pos[1] + DIRECTIONS[d][1]
                    if 0 <= nr < self.world.size and 0 <= nc < self.world.size:
                        if self.world.biomes[nr, nc] in (Biome.FOREST, Biome.ROCK):
                            return {"tool": "move", "args": {"direction": d},
                                    "thought": "Seeking shelter from storm."}

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
        d = dirs[int(np.random.randint(len(dirs)))]
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

    def _get_action_penalty(self, action_name: str) -> float:
        """
        Calculate penalty for repeating an action (5f: anti-oscillation).

        From complete_navigation_system_e.py: decay penalties for repeated actions
        to encourage behavioral diversity and prevent oscillation.
        """
        if not self._recent_actions:
            return 0.0
        # Count how many times this action appears in recent history
        recent_count = self._recent_actions.count(action_name)
        # Exponential decay penalty: 0.05 * count^1.5
        penalty = 0.05 * (recent_count ** 1.5)
        return min(penalty, 0.5)  # Cap at 0.5 to avoid over-penalizing

    def _check_stuckness(self) -> bool:
        """
        Check if agent is stuck (5g: stuckness detection).

        From maze_environment.py: stuckness detection triggers context-switch
        when agent revisits the same few positions repeatedly.
        """
        if len(self._recent_positions) < self._stuckness_window:
            return False

        recent = self._recent_positions[-self._stuckness_window:]
        unique_positions = len(set(recent))

        was_stuck = self._is_stuck
        self._is_stuck = unique_positions <= self._stuckness_threshold

        if self._is_stuck:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0

        # Log transition
        if self._is_stuck and not was_stuck:
            print(f"[STUCK t={self.total_ticks}] Agent stuck at {unique_positions} unique positions")
        elif not self._is_stuck and was_stuck:
            print(f"[UNSTUCK t={self.total_ticks}] Agent escaped stuckness")

        return self._is_stuck

    def _check_plan_interrupt(self) -> tuple[bool, str]:
        """
        Check if current plan should be interrupted.

        Returns (should_interrupt, reason).
        """
        if not self._current_plan:
            return False, ""

        # Always interrupt in crisis zone
        if self._consciousness_zone == "crisis":
            return True, "crisis_zone"

        # Check for hazard within 2 cells
        nearby = self.world.objects_near(self.pos, radius=2)
        for dist, _, obj in nearby:
            if isinstance(obj, Hazard) and dist <= 2:
                return True, "hazard_nearby"

        # Check replan conditions
        if "energy_critical" in self._plan_replan_if and self.energy < 0.2:
            return True, "energy_critical"

        if "goal_changed" in self._plan_replan_if and self.strategy != self._plan_strategy:
            return True, "strategy_changed"

        if "weather_change" in self._plan_replan_if:
            if self._prev_weather and self.world.weather_name != self._prev_weather:
                return True, "weather_change"

        if "inventory_full" in self._plan_replan_if:
            if len(self.inventory) >= self.inventory_capacity:
                return True, "inventory_full"

        return False, ""

    def _should_fire_llm(self) -> tuple[bool, str]:
        """
        Check if LLM should be invoked based on τ′-scaled scheduling (Phase 6b).

        Returns (should_fire, reason).

        Phase 6b replaces fixed periodic refresh with τ′-scaled timing:
        - base_interval = 25 ticks
        - effective_interval = base_interval * τ′
        - High Σ (tension) → low τ′ → more frequent LLM calls
        - Low Σ (calm) → high τ′ → less frequent LLM calls

        Critical events still trigger immediately:
        - Zone transitions, near-death, high curvature, first discoveries
        """
        self._ticks_since_llm += 1

        # Current state
        current_biome = self.world.biomes[self.pos].value
        current_weather = self.world.weather_name
        current_zone = self._consciousness_zone
        current_strategy = self.strategy

        reasons = []

        # === Critical events (always trigger immediately) ===

        # 1. Zone transition (crisis/struggling/healthy/transcendent)
        if self._prev_zone is not None and current_zone != self._prev_zone:
            reasons.append(f"zone transition to {current_zone}")

        # 2. Near-death experience
        if self.energy < 0.15:
            reasons.append("near-death")

        # 3. High curvature Σ (Phase 6: structured tension warrants deliberation)
        sigma_ema = self._st_metrics.get("sigma_ema", 0)
        if sigma_ema > 0.5:  # Elevated tension but below rupture threshold
            reasons.append(f"high tension (Σ={sigma_ema:.2f})")

        # 4. First discovery — novel object types
        for obj in self.world.objects_at(self.pos):
            obj_type = type(obj).__name__
            if obj_type not in self._seen_objects:
                self._seen_objects.add(obj_type)
                reasons.append(f"discovered {obj.name}")

        # === Secondary events (trigger if no critical events) ===

        if not reasons:
            # 5. Biome change
            if self._prev_biome is not None and current_biome != self._prev_biome:
                reasons.append(f"entered {current_biome}")

            # 6. Strategy change
            if self._prev_strategy is not None and current_strategy != self._prev_strategy:
                reasons.append(f"strategy shift to {current_strategy}")

            # 7. Weather change
            if self._prev_weather is not None and current_weather != self._prev_weather:
                if current_weather == "clear":
                    reasons.append("weather cleared")
                else:
                    reasons.append(f"{current_weather} started")

            # 8. High entropy removed - τ′-scheduling via Σ handles cognitive tension
            # The QSE naturally runs high entropy (mean ~0.9), making this trigger
            # too sensitive. Curvature Σ is a better measure of cognitive tension.
            # if self.entropy > 0.92:
            #     reasons.append("high entropy")

            # 9. Stuckness triggers replanning
            if self._is_stuck and self._stuck_ticks >= 5:
                reasons.append("stuck in area")

        # Update state for next tick
        self._prev_biome = current_biome
        self._prev_strategy = current_strategy
        self._prev_zone = current_zone
        self._prev_weather = current_weather

        # === Phase 6b: τ′-scaled periodic refresh ===
        # Instead of fixed 25 ticks, scale by emergent time τ′
        # τ′ ∈ [0.5, 2.0]: high tension = shorter interval, low tension = longer
        if not reasons:
            tau_prime = self._st_metrics.get("tau_prime", 1.0)
            base_interval = 25
            effective_interval = base_interval * tau_prime
            if self._ticks_since_llm >= effective_interval:
                reasons.append(f"τ′-refresh (τ′={tau_prime:.2f})")

        # Minimum cooldown: prevent excessive LLM calls from noisy triggers
        # Critical events (zone transition, near-death) bypass this
        MIN_LLM_INTERVAL = 5
        is_critical = any(r for r in reasons if "zone transition" in r or "near-death" in r)
        if reasons and (is_critical or self._ticks_since_llm >= MIN_LLM_INTERVAL):
            self._ticks_since_llm = 0
            reason_str = ", ".join(reasons)
            self._last_significant_event = reason_str
            return True, reason_str

        return False, ""

    def _respawn(self):
        """Respawn after death. Keep memories, lose inventory."""
        self.pos = (self.world.size // 2, self.world.size // 2)
        self.energy = 1.0  # was 0.8 — full energy for fresh start
        self.hydration = 1.0  # was 0.8
        self.alive = True
        self.inventory.clear()
        self.crafted.clear()
        if self.goal_mapper is not None:
            self.goal_mapper.reset_episode()
        self.llm.history.clear()
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
            "embodied_goal": self.embodied_goal,
            "teacher_prob": self._teacher_prob,
            "decision_source": self._decision_source,
            "learned_samples": self._learned_samples,
            "learned_ema": self._learned_reward_ema,
            "heuristic_ema": self._heuristic_reward_ema,
            "weather": self.world.weather_name,
            "consciousness_zone": self._consciousness_zone,
            "llm_trigger": self._last_significant_event,
            "ticks_since_llm": self._ticks_since_llm,
            "goal_satisfaction": self._goal_satisfaction,
            "novelty": self._novelty,
            "plan_goal": self._plan_goal,
            "plan_steps_remaining": len(self._current_plan),
            # 5f: Anti-oscillation
            "action_repeat_penalty": self._get_action_penalty(
                self.last_action.get("tool", "") if self.last_action else ""
            ),
            # 5g: Stuckness detection
            "is_stuck": self._is_stuck,
            "stuck_ticks": self._stuck_ticks,
            # Phase 6: Surplus/Tension
            "surplus": self._st_metrics.get("surplus", 0),
            "surplus_ema": self._st_metrics.get("surplus_ema", 0),
            "curvature": self._st_metrics.get("curvature", 0),
            "sigma_ema": self._st_metrics.get("sigma_ema", 0),
            "tau_prime": self._st_metrics.get("tau_prime", 1.0),
            "dynamic_threshold": self._st_metrics.get("dynamic_threshold", 0.65),
            "k_effective": self._st_metrics.get("k_effective", 2.0),
            "ruptures": self.surplus_tension.ruptures_triggered,
        }
