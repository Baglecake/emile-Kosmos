# Roadmap: Surplus/Tension Module for emile-Kosmos

**Created:** 2026-02-05
**Status:** Design Phase
**Priority:** High - addresses death-trap patterns and principled cognitive scheduling

---

## 1. Motivation

Current Kosmos uses heuristic proxies for cognitive dynamics:
- **Novelty** = unique positions in last 30 ticks (proxy for exploration)
- **Stuckness** = <=3 unique positions (proxy for local trap)
- **Event-triggered LLM** = biome/zone/weather changes (proxy for "meaningful moment")

The theoretical framework (Antifinity/QSE) provides *formal* definitions:
- **Surplus S** = magnitude of discrepancy between internal model (Ψ) and external reality (Φ)
- **Curvature Σ** = *structural tension* of that discrepancy (not just how much, but how contorted)
- **Emergent Time τ′** = processing rate scaled by cognitive intensity

This roadmap formalizes these concepts for Kosmos.

---

## 2. Theoretical Foundation (from emile_mini)

### 2.1 Existing QSE Core (emile_mini/qse_core.py)

```python
# Symbolic fields from surplus
def calculate_symbolic_fields(S: np.ndarray, cfg=CONFIG):
    psi = 1.0 / (1.0 + np.exp(-cfg.K_PSI * (S - cfg.THETA_PSI)))
    phi = np.maximum(0.0, cfg.K_PHI * (S - cfg.THETA_PHI))
    sigma = psi - phi
    return psi, phi, sigma

# Emergent time from sigma change
def calculate_emergent_time(sigma: np.ndarray, sigma_prev: np.ndarray, cfg=CONFIG) -> float:
    delta = np.mean(np.abs(sigma - sigma_prev))
    raw = cfg.TAU_MIN + (cfg.TAU_MAX - cfg.TAU_MIN) / (1.0 + np.exp(cfg.TAU_K * (delta - cfg.TAU_THETA)))
    return float(np.clip(raw, cfg.TAU_MIN, cfg.TAU_MAX))

# Surplus dynamics with rupture
def update_surplus(S: np.ndarray, sigma: np.ndarray, dt: float, cfg=CONFIG) -> np.ndarray:
    # rupture expulsion when |sigma| > threshold
    expel = np.where(np.abs(sigma) > cfg.S_THETA_RUPTURE, cfg.S_EPSILON * dt * S, 0.0)
    S_new = (1.0 + cfg.S_GAMMA * dt) * S + cfg.S_BETA * dt * sigma - expel
    # Laplacian coupling, damping, noise...
    return np.clip(S_new, 0.0, 1.0)
```

### 2.2 Key Insight

The existing QSE runs on a 1D *semantic* grid (GRID_SIZE=64). For Kosmos, we need to map:
- **Φ (external reality)** = what the agent *observes* (percepts, vitals, world state)
- **Ψ (internal model)** = what the agent *expects* (predictions, running averages)
- **S = |Ψ - Φ|** = surprise magnitude
- **Σ = gradient/structure of S** = how "contorted" the surprise is

---

## 3. Kosmos-Specific Design

### 3.1 Define Φ (Observed World State)

A vector of current percepts:

```python
def build_phi(agent) -> np.ndarray:
    """External reality vector (what we observe now)."""
    return np.array([
        agent.energy,                           # Current energy
        agent.hydration,                        # Current hydration
        nearby_food_count / 5.0,                # Food density (scaled)
        nearby_water_count / 5.0,               # Water density
        nearby_hazard_count / 5.0,              # Hazard density
        1.0 if food_at_position else 0.0,       # Food here now
        1.0 if hazard_at_position else 0.0,     # Hazard here now
        biome_danger_level,                     # 0=safe, 1=dangerous
        weather_severity,                       # 0=clear, 1=storm
        # ... can extend
    ])
```

### 3.2 Define Ψ (Internal Model / Predictions)

Running averages of what the agent *expects*:

```python
class InternalModel:
    """Maintains expectations about the world."""

    def __init__(self):
        self.expected_food_density = 0.3      # EMA of food availability
        self.expected_hazard_density = 0.1    # EMA of hazard frequency
        self.expected_energy_delta = -0.002   # Expected energy change per tick
        self.expected_survival_time = 200     # Ticks until death at current rate
        # ...

    def build_psi(self) -> np.ndarray:
        """Internal model vector (what we expect)."""
        return np.array([
            self.expected_energy,
            self.expected_hydration,
            self.expected_food_density,
            self.expected_water_density,
            self.expected_hazard_density,
            self.expected_food_here,
            self.expected_hazard_here,
            self.expected_biome_danger,
            self.expected_weather,
        ])

    def update(self, phi: np.ndarray, alpha: float = 0.1):
        """EMA update of expectations toward reality."""
        self.psi = (1 - alpha) * self.psi + alpha * phi
```

### 3.3 Compute Surplus S

```python
def compute_surplus(phi: np.ndarray, psi: np.ndarray) -> float:
    """Scalar surplus: L2 norm of discrepancy."""
    diff = phi - psi
    return float(np.sqrt(np.sum(diff ** 2)))
```

### 3.4 Compute Curvature Σ

Curvature measures *structural* tension - not just "how much" but "how inconsistent":

```python
def compute_curvature(surplus_history: list[float], position_history: list[tuple]) -> float:
    """
    Curvature Σ: measures structural tension in recent experience.

    High Σ = localized, repeated failure pattern (death trap)
    Low Σ = uniform surprise (general exploration)
    """
    if len(surplus_history) < 10:
        return 0.0

    recent = np.array(surplus_history[-20:])

    # Variance in surplus (inconsistency)
    variance = np.var(recent)

    # Gradient: how fast surplus is changing
    gradient = np.mean(np.abs(np.diff(recent)))

    # Spatial concentration: are we stuck in same area?
    unique_positions = len(set(position_history[-20:]))
    spatial_concentration = 1.0 - (unique_positions / 20.0)

    # Combine into curvature score
    sigma = 0.4 * variance + 0.3 * gradient + 0.3 * spatial_concentration
    return float(np.clip(sigma, 0.0, 1.0))
```

---

## 4. Integration Points in KosmosAgent

### 4.1 New Module: SurplusTensionModule

```python
# kosmos/agent/surplus_tension.py

class SurplusTensionModule:
    """Computes S (surplus) and Σ (curvature) for cognitive scheduling."""

    def __init__(self):
        self.internal_model = InternalModel()
        self.surplus_history: list[float] = []
        self.curvature_history: list[float] = []
        self.sigma_ema = 0.0
        self.rupture_cooldown = 0

        # Thresholds
        self.SIGMA_CRIT = 0.7          # Rupture threshold
        self.TAU_MIN = 0.5             # Min LLM interval scaling
        self.TAU_MAX = 2.0             # Max LLM interval scaling

    def step(self, agent) -> dict:
        """Compute S, Σ, and τ′ from current agent state."""
        # Build observation vectors
        phi = self._build_phi(agent)
        psi = self.internal_model.build_psi()

        # Compute surplus
        S = compute_surplus(phi, psi)
        self.surplus_history.append(S)
        if len(self.surplus_history) > 100:
            self.surplus_history = self.surplus_history[-100:]

        # Update internal model toward reality
        self.internal_model.update(phi)

        # Compute curvature
        sigma = compute_curvature(
            self.surplus_history,
            agent._recent_positions
        )
        self.curvature_history.append(sigma)

        # EMA smooth curvature
        self.sigma_ema = 0.9 * self.sigma_ema + 0.1 * sigma

        # Compute emergent time scaling
        tau_prime = self._compute_tau_prime(sigma)

        # Check for rupture
        should_rupture = self._check_rupture(sigma)

        return {
            'surplus': S,
            'curvature': sigma,
            'sigma_ema': self.sigma_ema,
            'tau_prime': tau_prime,
            'should_rupture': should_rupture,
        }

    def _compute_tau_prime(self, sigma: float) -> float:
        """Emergent time: high tension = more LLM calls."""
        # τ′ = τ_min + (τ_max - τ_min) * sigma
        return self.TAU_MIN + (self.TAU_MAX - self.TAU_MIN) * sigma

    def _check_rupture(self, sigma: float) -> bool:
        """Check if rupture threshold exceeded."""
        if self.rupture_cooldown > 0:
            self.rupture_cooldown -= 1
            return False
        if sigma > self.SIGMA_CRIT:
            self.rupture_cooldown = 50  # Cooldown after rupture
            return True
        return False
```

### 4.2 Modify tick() in KosmosAgent

```python
def tick(self) -> dict:
    # ... existing code ...

    # NEW: Compute surplus/tension
    st_metrics = self.surplus_tension.step(self)

    # Use τ′ to modulate LLM scheduling (replace heuristic event triggers)
    base_llm_interval = 25  # ticks
    effective_interval = base_llm_interval / st_metrics['tau_prime']
    should_fire_llm = self._ticks_since_llm >= effective_interval

    # Use rupture to trigger escape behavior
    if st_metrics['should_rupture']:
        print(f"[RUPTURE t={self.total_ticks}] σ={st_metrics['curvature']:.2f}")
        self._execute_rupture()

    # Use S for intrinsic reward shaping
    intrinsic_reward = 0.1 * st_metrics['surplus']  # Reward for encountering novelty

    # ... rest of tick ...
```

### 4.3 Rupture Behavior

```python
def _execute_rupture(self):
    """Execute cognitive rupture: forced relocation and reset."""
    # Clear current plan
    self._current_plan.clear()
    self._plan_goal = ""

    # Force relocation: move away from current area
    # Pick direction AWAY from recent death locations
    escape_direction = self._compute_escape_direction()

    # Execute several moves in escape direction
    for _ in range(5):
        self._tool_move(escape_direction)

    # Temporarily elevate exploration in policy
    self._teacher_prob = min(0.9, self._teacher_prob + 0.1)

    # Clear internal model (fresh start)
    self.surplus_tension.internal_model.reset()

    # Request LLM plan for new area
    self._ticks_since_llm = 999  # Force LLM call
```

---

## 5. Cognitive Integrity Metrics (Collaboration vs Compromise)

### 5.1 Definition

- **Collaboration** = multiple decision sources contributing additively, expanding capability
- **Compromise** = system collapsing to single mode, reducing potential

### 5.2 Implementation

```python
def compute_cognitive_integrity(agent) -> dict:
    """Measure collaboration vs compromise."""

    # Count active contributors in last 100 decisions
    recent_sources = agent._decision_history[-100:]
    source_counts = Counter(recent_sources)

    # Diversity = entropy of decision source distribution
    total = sum(source_counts.values())
    probs = [c/total for c in source_counts.values()]
    diversity = -sum(p * np.log(p + 1e-10) for p in probs)
    max_diversity = np.log(len(source_counts))
    normalized_diversity = diversity / max_diversity if max_diversity > 0 else 0

    # Plan stability = how often plans complete vs abort
    plan_completion_rate = agent._plans_completed / max(1, agent._plans_started)

    # Collaboration score
    collaboration = 0.5 * normalized_diversity + 0.5 * plan_completion_rate

    # Compromise indicators
    survival_reflex_dominance = source_counts.get('survival_reflex', 0) / total
    single_action_dominance = max(source_counts.values()) / total

    compromise = 0.5 * survival_reflex_dominance + 0.5 * single_action_dominance

    return {
        'collaboration': collaboration,
        'compromise': compromise,
        'integrity': collaboration - compromise,  # Net cognitive health
        'diversity': normalized_diversity,
    }
```

---

## 6. Implementation Order

### Phase 6a: SurplusTensionModule (Core)
1. Create `kosmos/agent/surplus_tension.py`
2. Implement `InternalModel` with EMA expectations
3. Implement `compute_surplus()` and `compute_curvature()`
4. Wire into `KosmosAgent.__init__` and `tick()`

### Phase 6b: τ′-Driven LLM Scheduling
1. Replace heuristic `_should_fire_llm()` with τ′ scaling
2. Remove individual event triggers (biome change, etc.)
3. LLM fires when `ticks_since_llm >= base_interval / tau_prime`

### Phase 6c: Rupture Mechanism
1. Implement `_check_rupture()` with threshold and cooldown
2. Implement `_execute_rupture()` with escape behavior
3. Track death locations for escape direction computation

### Phase 6d: Intrinsic Reward Shaping
1. Add surplus-based intrinsic reward to `_compute_reward()`
2. Curvature penalty for repeated-failure patterns
3. Bonus for reducing curvature (escaping traps)

### Phase 6e: Cognitive Integrity Dashboard
1. Implement `compute_cognitive_integrity()`
2. Add to `get_state()` for renderer
3. Display in pygame panel

---

## 7. Expected Outcomes

| Current Behavior | With Surplus/Tension |
|-----------------|---------------------|
| Heuristic event triggers | τ′-scaled continuous scheduling |
| Stuckness = 3 positions | Σ measures structural tension pattern |
| Escape = random direction | Rupture = principled relocation |
| Fixed novelty bonus | S-scaled intrinsic reward |
| No cognitive health metric | Collaboration/Compromise dashboard |

**Death trap problem**: High Σ (repeated deaths in same area) triggers rupture, forcing relocation before pattern cements.

---

## 8. Files to Create/Modify

### New Files
- `kosmos/agent/surplus_tension.py` (~200 lines)

### Modified Files
- `kosmos/agent/core.py` - wire SurplusTensionModule into tick()
- `kosmos/render/pygame_render.py` - display S, Σ, τ′, integrity
- `kosmos/__init__.py` - export new module

---

## 9. Verification

1. Run extended test (10 min)
2. Check that τ′ varies with cognitive intensity
3. Check that ruptures occur when trapped in death patterns
4. Check that death rate decreases after rupture implementation
5. Check cognitive integrity stays positive (collaboration > compromise)

---

## 10. References

- Theoretical notes: `archive/docs/Compiled Antifinity + Being-in-addition-to-itself notes and equations.txt`
- GPT analysis: `archive/docs/GPT_theory_feedback.md`
- QSE core: `emile_mini/qse_core.py`
- Symbolic reasoner: `emile_mini/symbolic.py`
