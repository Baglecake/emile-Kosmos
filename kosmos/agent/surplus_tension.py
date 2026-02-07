"""
SurplusTensionModule: Computes S (surplus) and Σ (curvature) for cognitive scheduling.

Implements the theoretical framework from Antifinity/QSE:
- Φ (phi): observed world state vector (what we see now)
- Ψ (psi): internal model / expectations (what we predict)
- S = |Ψ - Φ|: surplus magnitude (surprise)
- Σ (sigma): curvature / structural tension (how contorted the surprise is)
- τ′ (tau_prime): emergent time (LLM scheduling rate scaled by cognitive intensity)

This replaces heuristic proxies (novelty, stuckness) with principled QSE metrics.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import KosmosAgent


class InternalModel:
    """
    Maintains expectations about the world (Ψ).

    Uses EMA to track what the agent expects based on recent experience.
    """

    def __init__(self, dim: int = 9):
        self.dim = dim
        # Initialize expectations to neutral values
        self.psi = np.array([
            0.7,    # expected energy
            0.7,    # expected hydration
            0.3,    # expected food density (scaled)
            0.2,    # expected water density
            0.1,    # expected hazard density
            0.3,    # expected food at position (probability)
            0.1,    # expected hazard at position (probability)
            0.2,    # expected biome danger level
            0.1,    # expected weather severity
        ], dtype=np.float32)

    def build_psi(self) -> np.ndarray:
        """Return current internal model vector (what we expect)."""
        return self.psi.copy()

    def update(self, phi: np.ndarray, alpha: float = 0.1):
        """EMA update of expectations toward observed reality."""
        if phi.shape != self.psi.shape:
            raise ValueError(f"Phi shape {phi.shape} != Psi shape {self.psi.shape}")
        self.psi = (1 - alpha) * self.psi + alpha * phi

    def reset(self):
        """Reset expectations to neutral (used after rupture)."""
        self.psi = np.array([
            0.7, 0.7, 0.3, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {"psi": self.psi.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "InternalModel":
        """Deserialize from persistence."""
        model = cls()
        model.psi = np.array(data["psi"], dtype=np.float32)
        return model


def build_phi(agent: "KosmosAgent") -> np.ndarray:
    """
    Build external reality vector Φ (what we observe now).

    Dimensions (9):
    0: energy
    1: hydration
    2: nearby_food_count / 5.0 (scaled density)
    3: nearby_water_count / 5.0
    4: nearby_hazard_count / 5.0
    5: 1.0 if food at position, else 0.0
    6: 1.0 if hazard at position, else 0.0
    7: biome danger level (0=safe, 1=dangerous)
    8: weather severity (0=clear, 1=severe)
    """
    from ..world.objects import Food, Water, Hazard, Biome
    from ..world.weather import WeatherType

    # Count nearby objects
    nearby = agent.world.objects_near(agent.pos, radius=4)
    nearby_food = sum(1 for _, _, o in nearby if isinstance(o, Food))
    nearby_water = sum(1 for _, _, o in nearby if isinstance(o, Water))
    nearby_hazard = sum(1 for _, _, o in nearby if isinstance(o, Hazard))

    # Check objects at current position
    here = agent.world.objects_at(agent.pos)
    food_here = 1.0 if any(isinstance(o, Food) for o in here) else 0.0
    hazard_here = 1.0 if any(isinstance(o, Hazard) for o in here) else 0.0

    # Biome danger level
    biome = agent.world.biomes[agent.pos]
    biome_danger = {
        Biome.PLAINS: 0.1,
        Biome.FOREST: 0.2,
        Biome.DESERT: 0.6,
        Biome.WATER: 0.4,
        Biome.ROCK: 0.3,
    }.get(biome, 0.2)

    # Weather severity
    weather_severity = 0.0
    w = agent.world.weather.current
    if w:
        base_severity = {
            WeatherType.CLEAR: 0.0,
            WeatherType.RAIN: 0.2,
            WeatherType.STORM: 0.8,
            WeatherType.HEAT_WAVE: 0.6,
            WeatherType.FOG: 0.1,
            WeatherType.WIND: 0.3,
        }.get(w.weather_type, 0.0)
        weather_severity = base_severity * w.intensity

    return np.array([
        agent.energy,
        agent.hydration,
        min(nearby_food / 5.0, 1.0),
        min(nearby_water / 5.0, 1.0),
        min(nearby_hazard / 5.0, 1.0),
        food_here,
        hazard_here,
        biome_danger,
        weather_severity,
    ], dtype=np.float32)


def compute_surplus(phi: np.ndarray, psi: np.ndarray) -> float:
    """
    Compute scalar surplus S: L2 norm of discrepancy.

    S = |Ψ - Φ| = sqrt(sum((psi - phi)^2))

    High S = high surprise / novelty
    Low S = world matches expectations
    """
    diff = psi - phi
    return float(np.sqrt(np.sum(diff ** 2)))


def compute_curvature(
    surplus_history: list[float],
    position_history: list[tuple],
    death_history: list[int],
    current_tick: int,
) -> float:
    """
    Compute curvature Σ: measures structural tension in recent experience.

    High Σ = localized, repeated failure pattern (death trap)
    Low Σ = uniform surprise (general exploration)

    Components:
    1. Variance in surplus (inconsistency)
    2. Gradient of surplus (rate of change)
    3. Spatial concentration (stuck in same area)
    4. Death clustering (deaths in recent ticks)
    """
    if len(surplus_history) < 10:
        return 0.0

    recent_surplus = np.array(surplus_history[-20:])

    # 1. Variance in surplus (inconsistency in surprise levels)
    variance = float(np.var(recent_surplus))

    # 2. Gradient: how fast surplus is changing
    gradient = float(np.mean(np.abs(np.diff(recent_surplus))))

    # 3. Spatial concentration: are we stuck in same area?
    if len(position_history) >= 20:
        recent_positions = position_history[-20:]
        unique_positions = len(set(recent_positions))
        spatial_concentration = 1.0 - (unique_positions / 20.0)
    else:
        spatial_concentration = 0.0

    # 4. Death clustering: deaths within last 200 ticks
    recent_deaths = sum(1 for t in death_history if current_tick - t < 200)
    death_factor = min(recent_deaths / 5.0, 1.0)  # Cap at 5 deaths

    # Combine into curvature score
    # Weights emphasize death clustering (failure pattern) and spatial concentration
    sigma = (
        0.2 * variance +
        0.2 * gradient +
        0.3 * spatial_concentration +
        0.3 * death_factor
    )

    return float(np.clip(sigma, 0.0, 1.0))


class SurplusTensionModule:
    """
    Computes S (surplus) and Σ (curvature) for cognitive scheduling.

    Provides:
    - τ′ (tau_prime): emergent time scaling for LLM calls
    - Rupture detection: when Σ exceeds critical threshold
    - Metrics for reward shaping and cognitive integrity
    """

    def __init__(self):
        self.internal_model = InternalModel()

        # History tracking
        self.surplus_history: list[float] = []
        self.curvature_history: list[float] = []
        self.death_ticks: list[int] = []  # Ticks when deaths occurred

        # EMA smoothed values for stable decisions
        self.surplus_ema = 0.0
        self.sigma_ema = 0.0

        # Rupture state
        self.rupture_cooldown = 0
        self.ruptures_triggered = 0
        self._last_dynamic_threshold = 0.65  # Track for monitoring
        self._current_tick = 0  # Updated each step for death-sensitive k
        self._last_k_effective = 2.0  # Track for monitoring

        # Configuration
        self.SIGMA_CRIT = 0.65        # Initial/fallback threshold (used during warmup)
        self.TAU_MIN = 0.5            # Min LLM interval scaling (faster)
        self.TAU_MAX = 2.0            # Max LLM interval scaling (slower)
        self.RUPTURE_COOLDOWN = 50    # Ticks between ruptures
        self.HISTORY_SIZE = 100       # Max history to keep

    def step(self, agent: "KosmosAgent") -> dict:
        """
        Compute S, Σ, and τ′ from current agent state.

        Returns dict with:
        - surplus: current surprise magnitude
        - curvature: current structural tension
        - surplus_ema: smoothed surplus
        - sigma_ema: smoothed curvature
        - tau_prime: emergent time scaling for LLM
        - should_rupture: whether to trigger rupture behavior
        """
        # Track current tick for death-sensitive k calculation
        self._current_tick = agent.total_ticks

        # Build observation vectors
        phi = build_phi(agent)
        psi = self.internal_model.build_psi()

        # Compute surplus
        S = compute_surplus(phi, psi)
        self.surplus_history.append(S)
        if len(self.surplus_history) > self.HISTORY_SIZE:
            self.surplus_history = self.surplus_history[-self.HISTORY_SIZE:]

        # EMA smooth surplus
        self.surplus_ema = 0.9 * self.surplus_ema + 0.1 * S

        # Update internal model toward reality (learning rate depends on surprise)
        # Higher surprise = faster adaptation
        alpha = 0.1 + 0.1 * min(S, 1.0)
        self.internal_model.update(phi, alpha=alpha)

        # Compute curvature
        sigma = compute_curvature(
            self.surplus_history,
            agent._recent_positions,
            self.death_ticks,
            agent.total_ticks,
        )
        self.curvature_history.append(sigma)
        if len(self.curvature_history) > self.HISTORY_SIZE:
            self.curvature_history = self.curvature_history[-self.HISTORY_SIZE:]

        # EMA smooth curvature
        self.sigma_ema = 0.9 * self.sigma_ema + 0.1 * sigma

        # Compute emergent time scaling
        tau_prime = self._compute_tau_prime()

        # Check for rupture
        should_rupture = self._check_rupture()

        return {
            "surplus": S,
            "curvature": sigma,
            "surplus_ema": self.surplus_ema,
            "sigma_ema": self.sigma_ema,
            "tau_prime": tau_prime,
            "should_rupture": should_rupture,
            "dynamic_threshold": self._last_dynamic_threshold,
            "k_effective": self._last_k_effective,
            "phi": phi,
            "psi": psi,
        }

    def _compute_tau_prime(self) -> float:
        """
        Compute emergent time τ′: LLM call frequency scaling.

        High tension (sigma_ema) = more frequent LLM calls (lower tau)
        Low tension = less frequent calls (higher tau)

        τ′ = τ_max - (τ_max - τ_min) * sigma_ema
        """
        # Invert: high sigma -> low tau (more calls)
        tau_prime = self.TAU_MAX - (self.TAU_MAX - self.TAU_MIN) * self.sigma_ema
        return float(np.clip(tau_prime, self.TAU_MIN, self.TAU_MAX))

    def _compute_dynamic_threshold(self) -> float:
        """
        Compute adaptive rupture threshold from agent's own curvature statistics.

        Instead of a fixed SIGMA_CRIT, the threshold adapts to the distribution
        of curvature the agent has experienced. Rupture triggers when curvature
        exceeds the agent's baseline by a significant margin.

        threshold = mean(recent_sigma) + k_effective * std(recent_sigma)

        Death-Sensitive k (prevents "Normalization of Deviance"):
        When deaths cluster, the curvature history rises (due to death_factor
        in compute_curvature). This would normally raise the threshold along
        with the tension, preventing ruptures precisely when needed most.

        To counter this, k decreases when recent deaths are detected:
        - k_base = 2.0 (conservative, ~95th percentile)
        - k drops by 0.5 per recent death (within 200 ticks)
        - k_min = 0.5 (still requires some deviation)

        This ensures ruptures become MORE likely when survival is deteriorating,
        not LESS likely as the "death trap" becomes the new normal.
        """
        min_history = 50  # Need enough samples for meaningful statistics
        if len(self.curvature_history) < min_history:
            return self.SIGMA_CRIT  # Fall back to initial value during warmup

        recent = np.array(self.curvature_history[-min_history:])
        sigma_mean = float(np.mean(recent))
        sigma_std = float(np.std(recent))

        # Death-sensitive k: lower threshold when deaths are clustering
        k_base = 2.0
        k_min = 0.5
        death_window = 200  # Same window used in compute_curvature

        # Count deaths within recent window
        recent_deaths = sum(
            1 for t in self.death_ticks
            if self._current_tick - t < death_window
        )

        # Each recent death drops k by 0.75 (more aggressive than 0.5)
        # 0 deaths: k=2.0, 1 death: k=1.25, 2 deaths: k=0.5
        # This triggers ruptures earlier in persistent death traps
        death_penalty = 0.75 * recent_deaths
        k_effective = max(k_min, k_base - death_penalty)
        self._last_k_effective = k_effective  # Track for monitoring

        dynamic_threshold = sigma_mean + k_effective * sigma_std

        # Floor: don't let threshold drop below a minimum
        # (prevents triggering on tiny fluctuations when everything is calm)
        min_threshold = 0.3
        return max(dynamic_threshold, min_threshold)

    def _check_rupture(self) -> bool:
        """
        Check if rupture threshold exceeded using adaptive threshold.

        Rupture occurs when:
        1. Curvature EMA exceeds dynamic threshold (mean + k*std)
        2. Not in cooldown from previous rupture

        The threshold adapts to the agent's own curvature distribution,
        making ruptures context-sensitive rather than fixed.
        """
        if self.rupture_cooldown > 0:
            self.rupture_cooldown -= 1
            return False

        # Compute dynamic threshold from curvature statistics
        dynamic_threshold = self._compute_dynamic_threshold()
        self._last_dynamic_threshold = dynamic_threshold  # Store for monitoring

        if self.sigma_ema > dynamic_threshold:
            self.rupture_cooldown = self.RUPTURE_COOLDOWN
            self.ruptures_triggered += 1
            return True

        return False

    def record_death(self, tick: int):
        """Record a death event for curvature calculation."""
        self.death_ticks.append(tick)
        # Keep only recent deaths
        cutoff = tick - 500
        self.death_ticks = [t for t in self.death_ticks if t > cutoff]

    def on_rupture(self):
        """Called after rupture is executed - reset internal model."""
        self.internal_model.reset()
        # Partially reset history to give fresh start
        self.surplus_history = self.surplus_history[-10:]
        self.curvature_history = self.curvature_history[-10:]

    def get_intrinsic_reward(self) -> float:
        """
        Compute intrinsic reward from surplus (exploration bonus).

        High surplus (surprise) = positive intrinsic reward
        But scaled down if curvature is also high (structured failure)
        """
        # Base reward from surprise
        surprise_reward = 0.1 * self.surplus_ema

        # Penalty if high curvature (we're surprised but in a bad way)
        curvature_penalty = 0.05 * self.sigma_ema

        return surprise_reward - curvature_penalty

    def compute_cognitive_integrity(self, agent: "KosmosAgent") -> dict:
        """
        Measure cognitive integrity: collaboration vs compromise.

        From Antifinity framework:
        - Collaboration: Multiple decision sources contributing additively,
          expanding capability while preserving each source's "elemental truth"
        - Compromise: System collapsing to single mode, reducing potential,
          "flattening" the internal diversity

        Metrics:
        - diversity: entropy of decision source distribution (high = healthy)
        - plan_stability: how often plans complete vs abort (high = coherent)
        - collaboration: composite score of additive integration
        - compromise: indicators of reductive collapse
        - integrity: net cognitive health (collaboration - compromise)

        Returns dict with all metrics for dashboard display.
        """
        # Get decision history from agent
        # Track last 100 decisions for meaningful statistics
        decision_history = getattr(agent, '_decision_history', [])
        if not decision_history:
            # Initialize tracking if not present
            return {
                'collaboration': 0.5,
                'compromise': 0.5,
                'integrity': 0.0,
                'diversity': 0.5,
                'plan_stability': 0.5,
            }

        recent = decision_history[-100:] if len(decision_history) > 100 else decision_history

        # 1. Decision source diversity (entropy of distribution)
        from collections import Counter
        source_counts = Counter(recent)
        total = sum(source_counts.values())

        if total == 0:
            diversity = 0.5
        else:
            probs = [c / total for c in source_counts.values()]
            # Shannon entropy, normalized by max possible (log of num sources)
            import math
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(max(len(source_counts), 1) + 1e-10)
            diversity = entropy / max_entropy if max_entropy > 0 else 0

        # 2. Plan stability: completed plans / started plans
        plans_started = getattr(agent, '_plans_started', 0)
        plans_completed = getattr(agent, '_plans_completed', 0)
        plan_stability = plans_completed / max(1, plans_started)

        # 3. Collaboration score: weighted sum of positive indicators
        # - High diversity = multiple sources contributing
        # - Good plan stability = coherent execution
        # - Moderate teacher prob = balance between learning and teaching
        teacher_prob = getattr(agent, '_teacher_prob', 0.5)
        teacher_balance = 1.0 - abs(teacher_prob - 0.5) * 2  # Peak at 0.5

        collaboration = 0.4 * diversity + 0.4 * plan_stability + 0.2 * teacher_balance

        # 4. Compromise indicators: signs of reductive collapse
        # - Survival reflex dominance = crisis mode taking over
        # - Single source dominance = one mode suppressing others
        # - Low sigma_ema with deaths = normalized deviance (accepting bad patterns)

        survival_dominance = source_counts.get('survival_reflex', 0) / max(1, total)
        single_dominance = max(source_counts.values()) / max(1, total) if source_counts else 0

        # Normalized deviance: low curvature despite recent deaths
        recent_deaths = len([t for t in self.death_ticks if agent.total_ticks - t < 200])
        normalized_deviance = 0.0
        if recent_deaths > 0 and self.sigma_ema < 0.3:
            normalized_deviance = 0.3 * recent_deaths  # System accepting failures

        compromise = 0.4 * survival_dominance + 0.4 * (single_dominance - 0.5) + 0.2 * normalized_deviance
        compromise = float(np.clip(compromise, 0.0, 1.0))

        # 5. Net integrity: collaboration minus compromise
        integrity = collaboration - compromise

        return {
            'collaboration': float(np.clip(collaboration, 0.0, 1.0)),
            'compromise': float(np.clip(compromise, 0.0, 1.0)),
            'integrity': float(np.clip(integrity, -1.0, 1.0)),
            'diversity': float(np.clip(diversity, 0.0, 1.0)),
            'plan_stability': float(np.clip(plan_stability, 0.0, 1.0)),
        }

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "internal_model": self.internal_model.to_dict(),
            "surplus_history": self.surplus_history,
            "curvature_history": self.curvature_history,
            "death_ticks": self.death_ticks,
            "surplus_ema": self.surplus_ema,
            "sigma_ema": self.sigma_ema,
            "rupture_cooldown": self.rupture_cooldown,
            "ruptures_triggered": self.ruptures_triggered,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SurplusTensionModule":
        """Deserialize from persistence."""
        module = cls()
        module.internal_model = InternalModel.from_dict(data["internal_model"])
        module.surplus_history = data.get("surplus_history", [])
        module.curvature_history = data.get("curvature_history", [])
        module.death_ticks = data.get("death_ticks", [])
        module.surplus_ema = data.get("surplus_ema", 0.0)
        module.sigma_ema = data.get("sigma_ema", 0.0)
        module.rupture_cooldown = data.get("rupture_cooldown", 0)
        module.ruptures_triggered = data.get("ruptures_triggered", 0)
        return module
