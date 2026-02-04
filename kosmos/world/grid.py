"""KosmosWorld: the living grid with biomes, objects, and survival pressure."""

import numpy as np
from typing import Optional
from .objects import (
    Biome, BIOME_MOVE_COST, Food, Water, Hazard, CraftItem, WorldObject,
)


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

        # Spawn initial objects
        self._spawn_initial_objects()

        # World clock
        self.tick_count = 0
        self.day_length = 200        # ticks per day
        self.season_length = 800     # ticks per season

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
    #  Object spawning                                                     #
    # ------------------------------------------------------------------ #
    def _spawn_initial_objects(self):
        """Populate world with initial resources."""
        n = self.size * self.size

        # Food: ~8% coverage, biased toward plains/forest
        for _ in range(int(n * 0.08)):
            pos = self._random_pos(prefer=[Biome.PLAINS, Biome.FOREST])
            self._add_object(Food(position=pos), pos)

        # Water: ~3% coverage, biased toward water/forest biomes
        for _ in range(int(n * 0.03)):
            pos = self._random_pos(prefer=[Biome.WATER, Biome.FOREST])
            self._add_object(Water(position=pos), pos)

        # Hazards: ~4% coverage, biased toward desert/rock
        for _ in range(int(n * 0.04)):
            pos = self._random_pos(prefer=[Biome.DESERT, Biome.ROCK])
            self._add_object(Hazard(position=pos), pos)

        # Craft items: ~3%
        for _ in range(int(n * 0.03)):
            pos = self._random_pos()
            self._add_object(CraftItem(position=pos), pos)

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
        """Advance world by one step. Objects decay, new ones may spawn."""
        self.tick_count += 1
        self.events.clear()

        # Seasonal modulation of spawn rates
        _season_mods = {
            "spring": (1.0, 1.0, 1.0),    # (food_target, food_prob, hazard_prob)
            "summer": (0.85, 0.85, 1.8),
            "autumn": (1.4, 1.3, 0.7),
            "winter": (0.5, 0.5, 1.5),
        }
        ft_mod, fp_mod, hp_mod = _season_mods.get(self.season, (1.0, 1.0, 1.0))

        # Decay existing objects
        to_remove = []
        for pos, objs in self.objects.items():
            surviving = []
            for obj in objs:
                if obj.tick():
                    surviving.append(obj)
                else:
                    self.events.append({
                        "type": "decay", "object": obj.name, "position": pos
                    })
            if surviving:
                self.objects[pos] = surviving
            else:
                to_remove.append(pos)
        for pos in to_remove:
            del self.objects[pos]

        # Spawn new food (migration — never in same spot)
        food_count = sum(
            1 for objs in self.objects.values()
            for obj in objs if isinstance(obj, Food)
        )
        target_food = int(self.size * self.size * 0.06 * ft_mod)
        if food_count < target_food and self.rng.random() < 0.15 * fp_mod:
            pos = self._random_pos(prefer=[Biome.PLAINS, Biome.FOREST])
            if pos not in self.objects or not any(
                isinstance(o, Food) for o in self.objects.get(pos, [])
            ):
                self._add_object(Food(position=pos), pos)
                self.events.append({
                    "type": "spawn", "object": "food", "position": pos
                })

        # Occasional hazard spawns
        if self.rng.random() < 0.02 * hp_mod:
            pos = self._random_pos(prefer=[Biome.DESERT, Biome.ROCK])
            self._add_object(Hazard(position=pos), pos)

        # Occasional craft item spawns
        if self.rng.random() < 0.03:
            pos = self._random_pos()
            self._add_object(CraftItem(position=pos), pos)

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

    def move_cost(self, pos: tuple) -> float:
        """Energy cost to enter this cell."""
        biome = self.biomes[pos[0] % self.size, pos[1] % self.size]
        base = BIOME_MOVE_COST.get(biome, 1.0)
        # Night penalty
        if self.is_night:
            base *= 1.4
        return base * 0.008

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
