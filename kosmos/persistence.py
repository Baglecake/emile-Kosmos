"""Save/load persistence for Kosmos world and agent state."""

import json
import numpy as np
from pathlib import Path

from .world.grid import KosmosWorld
from .world.objects import (
    Biome, Food, Water, Hazard, CraftItem, Herb, Seed, PlantedCrop, WorldObject,
)
from .world.weather import WeatherManager, WeatherEvent
from .agent.core import KosmosAgent


# ------------------------------------------------------------------ #
#  Object serialization                                               #
# ------------------------------------------------------------------ #
_OBJ_TYPE_MAP = {
    "Food": Food,
    "Water": Water,
    "Hazard": Hazard,
    "CraftItem": CraftItem,
    "Herb": Herb,
    "Seed": Seed,
    "PlantedCrop": PlantedCrop,
}


def _serialize_object(obj: WorldObject) -> dict:
    """Serialize a WorldObject to a JSON-safe dict."""
    type_name = type(obj).__name__
    d = {
        "_type": type_name,
        "name": obj.name,
        "symbol": obj.symbol,
        "color": list(obj.color),
        "position": list(obj.position),
        "solid": obj.solid,
        "decay_rate": obj.decay_rate,
        "age": obj.age,
    }
    if isinstance(obj, Food):
        d["energy_value"] = obj.energy_value
        if isinstance(obj, Herb):
            d["heal_value"] = obj.heal_value
    elif isinstance(obj, Water):
        d["hydration_value"] = obj.hydration_value
    elif isinstance(obj, Hazard):
        d["damage"] = obj.damage
    elif isinstance(obj, PlantedCrop):
        d["growth_ticks"] = obj.growth_ticks
        d["mature_at"] = obj.mature_at
    elif isinstance(obj, CraftItem):
        d["craft_tag"] = obj.craft_tag
    return d


def _deserialize_object(d: dict) -> WorldObject:
    """Deserialize a WorldObject from a dict, bypassing __post_init__."""
    type_name = d["_type"]
    cls = _OBJ_TYPE_MAP.get(type_name)
    if cls is None:
        cls = WorldObject

    pos = tuple(d["position"])

    if cls == Food:
        obj = object.__new__(Food)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.energy_value = d.get("energy_value", 0.2)
    elif cls == Herb:
        obj = object.__new__(Herb)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.energy_value = d.get("energy_value", 0.05)
        obj.heal_value = d.get("heal_value", 0.1)
    elif cls == Water:
        obj = object.__new__(Water)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.hydration_value = d.get("hydration_value", 0.3)
    elif cls == Hazard:
        obj = object.__new__(Hazard)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.damage = d.get("damage", 0.15)
    elif cls == Seed:
        obj = object.__new__(Seed)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.craft_tag = d.get("craft_tag", "seed")
    elif cls == PlantedCrop:
        obj = object.__new__(PlantedCrop)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d.get("decay_rate", 0.0)
        obj.age = d.get("age", 0)
        obj.growth_ticks = d.get("growth_ticks", 0)
        obj.mature_at = d.get("mature_at", 100)
    elif cls == CraftItem:
        obj = object.__new__(CraftItem)
        obj.name = d["name"]
        obj.symbol = d["symbol"]
        obj.color = tuple(d["color"])
        obj.position = pos
        obj.solid = d.get("solid", False)
        obj.decay_rate = d["decay_rate"]
        obj.age = d.get("age", 0)
        obj.craft_tag = d.get("craft_tag", "wood")
    else:
        obj = WorldObject(
            name=d["name"], symbol=d["symbol"], color=tuple(d["color"]),
            position=pos, solid=d.get("solid", False),
            decay_rate=d["decay_rate"], age=d.get("age", 0),
        )
    return obj


# ------------------------------------------------------------------ #
#  Save / Load                                                        #
# ------------------------------------------------------------------ #
def save_state(world: KosmosWorld, agent: KosmosAgent, filepath: str):
    """Save full world + agent state to JSON."""
    # Biomes as string grid
    biome_grid = []
    for r in range(world.size):
        row = []
        for c in range(world.size):
            row.append(world.biomes[r, c].value)
        biome_grid.append(row)

    # Objects
    objects_list = []
    for pos, objs in world.objects.items():
        for obj in objs:
            objects_list.append(_serialize_object(obj))

    # Inventory (CraftItems)
    inv_list = [_serialize_object(obj) for obj in agent.inventory]

    state = {
        "version": 1,
        "world": {
            "size": world.size,
            "biomes": biome_grid,
            "objects": objects_list,
            "tick_count": world.tick_count,
            "day_length": world.day_length,
            "season_length": world.season_length,
            "weather": world.weather.to_dict(),
        },
        "agent": {
            "pos": list(agent.pos),
            "facing": agent.facing,
            "energy": agent.energy,
            "hydration": agent.hydration,
            "alive": agent.alive,
            "deaths": agent.deaths,
            "total_ticks": agent.total_ticks,
            "inventory": inv_list,
            "crafted": agent.crafted,
            "memories": agent.memories[-100:],
            "food_eaten": agent.food_eaten,
            "water_drunk": agent.water_drunk,
            "damage_taken": agent.damage_taken,
            "cells_visited": [list(c) for c in agent.cells_visited],
            "steps_taken": agent.steps_taken,
        },
        "learning": {
            "teacher_prob": agent._teacher_prob,
            "learned_reward_ema": agent._learned_reward_ema,
            "heuristic_reward_ema": agent._heuristic_reward_ema,
            "learned_samples": agent._learned_samples,
            "policy_step_count": agent._policy_step_count,
            "embodied_goal": agent.embodied_goal,
            "strategy": agent.strategy,
        },
    }

    Path(filepath).write_text(json.dumps(state, indent=2))


def load_state(filepath: str, world: KosmosWorld, agent: KosmosAgent):
    """Load saved state into existing world + agent instances."""
    data = json.loads(Path(filepath).read_text())

    # World
    wd = data["world"]
    world.tick_count = wd["tick_count"]
    world.day_length = wd.get("day_length", 200)
    world.season_length = wd.get("season_length", 800)

    # Restore biomes
    biome_map = {b.value: b for b in Biome}
    for r in range(world.size):
        for c in range(world.size):
            world.biomes[r, c] = biome_map.get(wd["biomes"][r][c], Biome.PLAINS)

    # Restore objects
    world.objects.clear()
    for od in wd["objects"]:
        obj = _deserialize_object(od)
        pos = obj.position
        if pos not in world.objects:
            world.objects[pos] = []
        world.objects[pos].append(obj)

    # Restore weather
    if wd.get("weather"):
        world.weather = WeatherManager.from_dict(wd["weather"], world.rng)

    # Agent
    ad = data["agent"]
    agent.pos = tuple(ad["pos"])
    agent.facing = ad.get("facing", "east")
    agent.energy = ad["energy"]
    agent.hydration = ad["hydration"]
    agent.alive = ad["alive"]
    agent.deaths = ad["deaths"]
    agent.total_ticks = ad["total_ticks"]
    agent.crafted = ad.get("crafted", [])
    agent.memories = ad.get("memories", [])
    agent.food_eaten = ad.get("food_eaten", 0)
    agent.water_drunk = ad.get("water_drunk", 0)
    agent.damage_taken = ad.get("damage_taken", 0)
    agent.cells_visited = {tuple(c) for c in ad.get("cells_visited", [])}
    agent.steps_taken = ad.get("steps_taken", 0)

    # Restore inventory
    agent.inventory.clear()
    for od in ad.get("inventory", []):
        obj = _deserialize_object(od)
        if isinstance(obj, CraftItem):
            agent.inventory.append(obj)

    # Learning state
    ld = data.get("learning", {})
    agent._teacher_prob = ld.get("teacher_prob", 1.0)
    agent._learned_reward_ema = ld.get("learned_reward_ema", 0.0)
    agent._heuristic_reward_ema = ld.get("heuristic_reward_ema", 0.0)
    agent._learned_samples = ld.get("learned_samples", 0)
    agent._policy_step_count = ld.get("policy_step_count", 0)
    agent.embodied_goal = ld.get("embodied_goal", "explore_space")
    agent.strategy = ld.get("strategy", "explore")
