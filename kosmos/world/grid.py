"""KosmosWorld: the living grid with biomes, objects, and survival pressure."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .objects import (
    Biome, BIOME_MOVE_COST, Food, Water, Hazard, CraftItem, WorldObject,
    Herb, Seed, PlantedCrop,
)
from .weather import WeatherManager, WeatherType

_OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass
class ResourceNode:
    """Fixed location where a resource spawns/respawns.

    Resources exist at fixed nodes, not randomly everywhere.
    When consumed, the node is depleted and respawns after cooldown.
    """
    position: tuple
    resource_type: str  # 'food', 'water', 'craft', 'hazard', 'herb', 'seed'
    cooldown: int = 0           # ticks until respawn (0 = has resource)
    respawn_time: int = 150     # ticks to respawn after depletion


def _opposite_dir(d: str) -> str:
    return _OPPOSITES.get(d, "")


class KosmosWorld:
    """
    A grid world with biomes, decaying resources, hazards, and real stakes.

    Key difference from emile-mini's grid: energy depletion = death.
    Food migrates. The world changes whether the agent acts or not.
    """

    def __init__(self, size: int = 30, seed: int | None = None):
        self.size = size
        self.rng = np.random.RandomState(seed)

        # Generate biome map (Perlin-like noise via smoothed random)
        self.biomes = self._generate_biomes()

        # Objects on the grid: position -> list[WorldObject]
        self.objects: dict[tuple, list[WorldObject]] = {}

        # Resource nodes: fixed spawn points that respawn when depleted
        # position -> ResourceNode
        self.resource_nodes: dict[tuple, ResourceNode] = {}

        # Spawn initial objects at fixed node locations
        self._spawn_initial_objects()

        # World clock
        self.tick_count = 0
        self.day_length = 200        # ticks per day
        self.season_length = 800     # ticks per season

        # Weather system
        self.weather = WeatherManager(self.rng)

        # Event log (renderer reads this)
        self.events: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Biome generation                                                    #
    # ------------------------------------------------------------------ #
    def _generate_biomes(self) -> np.ndarray:
        """Generate biome map using smoothed noise."""
        grid = np.zeros((self.size, self.size), dtype=object)

        # Base noise (low-res upscaled for coherent regions)
        lo = 6
        noise = self.rng.rand(lo, lo)
        # Simple bilinear upscale
        from numpy import interp
        xs = np.linspace(0, lo - 1, self.size)
        ys = np.linspace(0, lo - 1, self.size)
        smooth = np.zeros((self.size, self.size))
        for i, x in enumerate(xs):
            x0, x1 = int(x), min(int(x) + 1, lo - 1)
            xf = x - x0
            for j, y in enumerate(ys):
                y0, y1 = int(y), min(int(y) + 1, lo - 1)
                yf = y - y0
                val = (noise[x0, y0] * (1 - xf) * (1 - yf) +
                       noise[x1, y0] * xf * (1 - yf) +
                       noise[x0, y1] * (1 - xf) * yf +
                       noise[x1, y1] * xf * yf)
                smooth[i, j] = val

        # Map noise values to biomes
        for i in range(self.size):
            for j in range(self.size):
                v = smooth[i, j]
                if v < 0.2:
                    grid[i, j] = Biome.WATER
                elif v < 0.4:
                    grid[i, j] = Biome.FOREST
                elif v < 0.7:
                    grid[i, j] = Biome.PLAINS
                elif v < 0.85:
                    grid[i, j] = Biome.DESERT
                else:
                    grid[i, j] = Biome.ROCK
        return grid

    # ------------------------------------------------------------------ #
    #  Object spawning (fixed resource nodes)                              #
    # ------------------------------------------------------------------ #
    def _spawn_initial_objects(self):
        """Create fixed resource nodes and spawn initial objects.

        Resources spawn at fixed locations. When consumed, they respawn
        at the SAME location after a cooldown - no random new locations.
        """
        n = self.size * self.size

        # Food nodes: ~6% coverage, biased toward plains/forest
        # Respawn time varies by food type
        for _ in range(int(n * 0.06)):
            pos = self._random_pos(prefer=[Biome.PLAINS, Biome.FOREST])
            if pos not in self.resource_nodes:
                node = ResourceNode(pos, 'food', cooldown=0, respawn_time=120)
                self.resource_nodes[pos] = node
                self._add_object(Food(position=pos), pos)

        # Water nodes: ~2% coverage, near water biomes
        # Water respawns quickly
        for _ in range(int(n * 0.02)):
            pos = self._random_pos(prefer=[Biome.WATER, Biome.FOREST])
            if pos not in self.resource_nodes:
                node = ResourceNode(pos, 'water', cooldown=0, respawn_time=80)
                self.resource_nodes[pos] = node
                self._add_object(Water(position=pos), pos)

        # Hazard nodes: ~3% coverage, in harsh biomes
        # Hazards are PERMANENT - don't respawn (no node tracking)
        for _ in range(int(n * 0.03)):
            pos = self._random_pos(prefer=[Biome.DESERT, Biome.ROCK])
            if pos not in self.resource_nodes:
                self._add_object(Hazard(position=pos), pos)

        # Craft item nodes: ~2% coverage
        # Slow respawn (resources are limited)
        for _ in range(int(n * 0.02)):
            pos = self._random_pos()
            if pos not in self.resource_nodes:
                node = ResourceNode(pos, 'craft', cooldown=0, respawn_time=300)
                self.resource_nodes[pos] = node
                self._add_object(CraftItem(position=pos), pos)

        # Herb nodes: ~1%, forest only
        for _ in range(int(n * 0.01)):
            pos = self._random_pos(prefer=[Biome.FOREST])
            if pos not in self.resource_nodes:
                node = ResourceNode(pos, 'herb', cooldown=0, respawn_time=200)
                self.resource_nodes[pos] = node
                self._add_object(Herb(position=pos), pos)

        # Seed nodes: ~0.5%, rare
        for _ in range(int(n * 0.005)):
            pos = self._random_pos(prefer=[Biome.PLAINS, Biome.FOREST])
            if pos not in self.resource_nodes:
                node = ResourceNode(pos, 'seed', cooldown=0, respawn_time=400)
                self.resource_nodes[pos] = node
                self._add_object(Seed(position=pos), pos)

    def _random_pos(self, prefer: list[Biome] | None = None) -> tuple:
        """Pick a random position, optionally biased toward certain biomes."""
        for _ in range(20):
            pos = (self.rng.randint(self.size), self.rng.randint(self.size))
            if prefer is None or self.biomes[pos] in prefer:
                return pos
        # Fallback: any position
        return (self.rng.randint(self.size), self.rng.randint(self.size))

    def _add_object(self, obj: WorldObject, pos: tuple):
        obj.position = pos
        if pos not in self.objects:
            self.objects[pos] = []
        self.objects[pos].append(obj)

    # ------------------------------------------------------------------ #
    #  World tick                                                          #
    # ------------------------------------------------------------------ #
    def tick(self):
        """Advance world by one step. Depleted nodes respawn, objects decay."""
        self.tick_count += 1
        self.events.clear()

        # Weather update
        self.weather.tick()

        # Seasonal modifiers affect respawn speed
        _season_mods = {
            "spring": 1.0,   # normal respawn
            "summer": 0.8,   # faster respawn (growth season)
            "autumn": 1.2,   # slower respawn
            "winter": 2.0,   # much slower respawn
        }
        season_mod = _season_mods.get(self.season, 1.0)

        # Rain speeds up water respawn
        w = self.weather.current
        water_mod = 0.5 if (w and w.weather_type == WeatherType.RAIN) else 1.0

        # Tick depleted resource nodes - respawn when cooldown expires
        for pos, node in self.resource_nodes.items():
            if node.cooldown > 0:
                # Apply season modifier to respawn
                effective_mod = season_mod
                if node.resource_type == 'water':
                    effective_mod *= water_mod

                node.cooldown -= 1
                if node.cooldown <= 0:
                    # Respawn the resource at this node
                    self._respawn_at_node(node)

        # Mature crops become food (player-planted, not node-based)
        for pos in list(self.objects.keys()):
            objs = self.objects.get(pos, [])
            for obj in list(objs):
                if isinstance(obj, PlantedCrop) and obj.is_mature:
                    objs.remove(obj)
                    self._add_object(Food(position=pos), pos)
                    self.events.append({
                        "type": "harvest", "object": "crop", "position": pos
                    })

    def _respawn_at_node(self, node: ResourceNode):
        """Respawn a resource at a depleted node."""
        pos = node.position
        if node.resource_type == 'food':
            self._add_object(Food(position=pos), pos)
        elif node.resource_type == 'water':
            self._add_object(Water(position=pos), pos)
        elif node.resource_type == 'craft':
            self._add_object(CraftItem(position=pos), pos)
        elif node.resource_type == 'herb':
            self._add_object(Herb(position=pos), pos)
        elif node.resource_type == 'seed':
            self._add_object(Seed(position=pos), pos)

        self.events.append({
            "type": "respawn", "object": node.resource_type, "position": pos
        })

    def deplete_node(self, pos: tuple, resource_type: str):
        """Mark a resource node as depleted (called when resource consumed)."""
        if pos in self.resource_nodes:
            node = self.resource_nodes[pos]
            if node.resource_type == resource_type:
                node.cooldown = node.respawn_time

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #
    def objects_at(self, pos: tuple) -> list[WorldObject]:
        return self.objects.get(pos, [])

    def objects_near(self, pos: tuple, radius: int = 3) -> list[tuple]:
        """Return (distance, position, object) tuples within radius."""
        results = []
        r, c = pos
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    dist = abs(dr) + abs(dc)
                    if dist <= radius and (nr, nc) in self.objects:
                        for obj in self.objects[(nr, nc)]:
                            results.append((dist, (nr, nc), obj))
        results.sort(key=lambda x: x[0])
        return results

    def move_cost(self, pos: tuple, direction: str = "") -> float:
        """Energy cost to enter this cell."""
        biome = self.biomes[pos[0] % self.size, pos[1] % self.size]
        base = BIOME_MOVE_COST.get(biome, 1.0)
        # Night penalty
        if self.is_night:
            base *= 1.4
        # Weather effects
        w = self.weather.current
        if w:
            if w.weather_type == WeatherType.RAIN:
                base *= 1.0 + 0.3 * w.intensity
            elif w.weather_type == WeatherType.WIND and direction:
                if direction == w.wind_direction:
                    base *= 0.7  # wind at your back
                elif direction == _opposite_dir(w.wind_direction):
                    base *= 1.0 + 0.5 * w.intensity  # headwind
        return base * 0.008

    @property
    def weather_name(self) -> str:
        return self.weather.weather_name

    @property
    def examine_radius(self) -> int:
        w = self.weather.current
        if w and w.weather_type == WeatherType.FOG:
            return 2
        return 4

    @property
    def is_night(self) -> bool:
        return (self.tick_count % self.day_length) > (self.day_length * 0.6)

    @property
    def time_of_day(self) -> str:
        phase = (self.tick_count % self.day_length) / self.day_length
        if phase < 0.25:
            return "dawn"
        elif phase < 0.55:
            return "day"
        elif phase < 0.65:
            return "dusk"
        else:
            return "night"

    @property
    def season(self) -> str:
        phase = (self.tick_count % (self.season_length * 4)) / (self.season_length * 4)
        if phase < 0.25:
            return "spring"
        elif phase < 0.5:
            return "summer"
        elif phase < 0.75:
            return "autumn"
        else:
            return "winter"
