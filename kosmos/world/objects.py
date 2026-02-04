"""World objects: things that exist on the grid."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Biome(Enum):
    PLAINS = "plains"
    FOREST = "forest"
    DESERT = "desert"
    WATER = "water"
    ROCK = "rock"


# Colors for rendering (R, G, B)
BIOME_COLORS = {
    Biome.PLAINS: (34, 50, 34),
    Biome.FOREST: (20, 42, 20),
    Biome.DESERT: (60, 52, 30),
    Biome.WATER: (20, 30, 55),
    Biome.ROCK: (40, 40, 42),
}

# Energy cost multiplier per biome
BIOME_MOVE_COST = {
    Biome.PLAINS: 1.0,
    Biome.FOREST: 1.3,
    Biome.DESERT: 1.8,
    Biome.WATER: 2.5,
    Biome.ROCK: 3.0,
}


@dataclass
class WorldObject:
    """Base class for objects in the world."""
    name: str
    symbol: str
    color: tuple
    position: tuple = (0, 0)
    solid: bool = False       # blocks movement
    decay_rate: float = 0.0   # per-tick chance of disappearing
    age: int = 0

    def tick(self) -> bool:
        """Advance one step. Returns False if object should be removed."""
        self.age += 1
        if self.decay_rate > 0 and np.random.random() < self.decay_rate:
            return False
        return True


@dataclass
class Food(WorldObject):
    """Food source. Consumed for energy."""
    name: str = "berry"
    symbol: str = "o"
    color: tuple = (180, 50, 50)
    energy_value: float = 0.25
    decay_rate: float = 0.003   # food rots over time

    def __post_init__(self):
        # Vary food types
        variants = [
            ("berry", (180, 50, 50), 0.20, 0.004),
            ("mushroom", (160, 120, 60), 0.15, 0.006),
            ("fruit", (200, 100, 40), 0.35, 0.002),
            ("root", (140, 100, 70), 0.12, 0.001),
        ]
        if self.name == "berry":  # only randomize if default
            v = variants[np.random.randint(len(variants))]
            self.name, self.color, self.energy_value, self.decay_rate = v


@dataclass
class Water(WorldObject):
    """Water source. Consumed for hydration."""
    name: str = "puddle"
    symbol: str = "~"
    color: tuple = (60, 100, 200)
    hydration_value: float = 0.30
    decay_rate: float = 0.001


@dataclass
class Hazard(WorldObject):
    """Dangerous object. Costs energy on contact."""
    name: str = "thorns"
    symbol: str = "x"
    color: tuple = (200, 40, 40)
    damage: float = 0.15
    solid: bool = False
    decay_rate: float = 0.0

    def __post_init__(self):
        variants = [
            ("thorns", (200, 40, 40), 0.12),
            ("snake", (180, 160, 30), 0.20),
            ("pitfall", (80, 60, 40), 0.25),
        ]
        if self.name == "thorns":
            v = variants[np.random.randint(len(variants))]
            self.name, self.color, self.damage = v


@dataclass
class CraftItem(WorldObject):
    """Item that can be picked up and used for crafting."""
    name: str = "stick"
    symbol: str = "+"
    color: tuple = (140, 110, 60)
    craft_tag: str = "wood"
    decay_rate: float = 0.0

    def __post_init__(self):
        variants = [
            ("stick", (140, 110, 60), "wood"),
            ("stone", (150, 150, 155), "stone"),
            ("fiber", (100, 160, 80), "fiber"),
            ("shell", (200, 190, 170), "shell"),
        ]
        if self.name == "stick":
            v = variants[np.random.randint(len(variants))]
            self.name, self.color, self.craft_tag = v


# Craft recipes: (tag1, tag2) -> result_name, result_description
CRAFT_RECIPES = {
    ("wood", "stone"): ("axe", "A crude axe. Reduces forest movement cost."),
    ("wood", "fiber"): ("rope", "A length of rope. Can cross water more easily."),
    ("stone", "fiber"): ("sling", "A sling. Can scare away hazards."),
    ("wood", "shell"): ("bowl", "A bowl. Can carry water."),
}
