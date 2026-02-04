"""Weather system for Kosmos world."""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class WeatherType(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"
    HEAT_WAVE = "heat_wave"
    FOG = "fog"
    WIND = "wind"


@dataclass
class WeatherEvent:
    """A single weather event with lifecycle."""
    weather_type: WeatherType
    duration: int
    elapsed: int = 0
    intensity: float = 1.0
    wind_direction: str = "north"

    @property
    def active(self) -> bool:
        return self.elapsed < self.duration

    @property
    def progress(self) -> float:
        return self.elapsed / max(self.duration, 1)

    def tick(self) -> bool:
        """Advance one tick. Returns False when expired."""
        self.elapsed += 1
        # Bell-curve intensity: ramp up, plateau, ramp down
        p = self.progress
        if p < 0.2:
            self.intensity = p / 0.2
        elif p > 0.8:
            self.intensity = (1.0 - p) / 0.2
        else:
            self.intensity = 1.0
        return self.active

    def to_dict(self) -> dict:
        return {
            "type": self.weather_type.value,
            "duration": self.duration,
            "elapsed": self.elapsed,
            "intensity": self.intensity,
            "wind_direction": self.wind_direction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WeatherEvent":
        return cls(
            weather_type=WeatherType(data["type"]),
            duration=data["duration"],
            elapsed=data["elapsed"],
            intensity=data["intensity"],
            wind_direction=data.get("wind_direction", "north"),
        )


class WeatherManager:
    """Manages weather event lifecycle."""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.current: WeatherEvent | None = None
        self.cooldown: int = 0
        self.history: list[str] = []

    def tick(self) -> WeatherEvent | None:
        """Called once per world tick. Returns current event or None."""
        if self.current is not None:
            if not self.current.tick():
                self.history.append(self.current.weather_type.value)
                if len(self.history) > 20:
                    self.history = self.history[-20:]
                self.current = None
                self.cooldown = self.rng.randint(40, 120)
            return self.current

        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if self.rng.random() < 0.005:
            self.current = self._spawn_event()
            return self.current

        return None

    def _spawn_event(self) -> WeatherEvent:
        types = [WeatherType.RAIN, WeatherType.STORM, WeatherType.HEAT_WAVE,
                 WeatherType.FOG, WeatherType.WIND]
        chosen = types[self.rng.randint(len(types))]
        duration = self.rng.randint(30, 81)
        wind_dir = ["north", "south", "east", "west"][self.rng.randint(4)]
        return WeatherEvent(
            weather_type=chosen,
            duration=duration,
            wind_direction=wind_dir,
        )

    @property
    def weather_name(self) -> str:
        if self.current and self.current.active:
            return self.current.weather_type.value
        return "clear"

    def to_dict(self) -> dict:
        return {
            "current": self.current.to_dict() if self.current else None,
            "cooldown": self.cooldown,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict, rng: np.random.RandomState) -> "WeatherManager":
        mgr = cls(rng)
        if data.get("current"):
            mgr.current = WeatherEvent.from_dict(data["current"])
        mgr.cooldown = data.get("cooldown", 0)
        mgr.history = data.get("history", [])
        return mgr
