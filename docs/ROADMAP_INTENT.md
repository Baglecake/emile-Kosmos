# Intent System Roadmap: LLM as Conscious Cortex

## Vision

Shift the LLM from **micro-planner** (issuing tool sequences) to **strategic cortex** (setting goals, quotas, and constraints). The embodied agent becomes the "body" that executes intents using survival reflexes and learned skills.

```
Current: LLM → ["move", "move", "consume"] → stale, fragile, micromanaging

Proposed: LLM → Intent { quotas, constraints, triggers } → body satisfies intent
```

This mirrors human cognition: you don't consciously plan each muscle twitch, you set goals and let your nervous system handle execution.

---

## Phase 0: Diagnostic Validation (COMPLETE ✓)

Verified with 50K tick run (2026-02-07):

### Checklist

- [x] **LLM adoption happening**: 248 ADOPT, 0 STALE events
- [x] **Plans completing**: 65% completion rate (162/248)
- [x] **Crops maturing**: Mechanism wired, harvest events emitted
- [x] **Teacher distribution**: 0.9% teacher_plan, 27.8% heuristic, 67.7% learned

### Results

| Metric | Value |
|--------|-------|
| Total ticks | 50,700 |
| Deaths | 2 |
| Death rate | 0.04/1000 ticks |
| Healthy zone | 80.6% |

---

## Phase 1: State-Validity for LLM Plans (COMPLETE ✓)

**Goal**: Accept/reject LLM plans based on **context relevance**, not tick age.

### Implementation (2026-02-07)

Added `SituationSignature` dataclass with:
```python
@dataclass
class SituationSignature:
    zone: str           # crisis/struggling/healthy/transcendent
    energy_band: int    # 0=critical, 1=low, 2=ok, 3=high
    hydration_band: int
    strategy: str
    has_food_nearby: bool
    has_hazard_nearby: bool
    weather: str
    biome: str

    def matches(self, other, strict=False) -> bool:
        # Critical: zone and strategy must match
        # Energy/hydration: 1-band difference OK
        # Hazard status must match (safety critical)
```

### Implementation Details

1. ✅ `_compute_situation_signature()` method in `core.py`
2. ✅ `_llm_request_signature` stored when FIRE event
3. ✅ `matches()` method on signature class with tolerance
4. ✅ Plans rejected with `REJECT` log event showing reason
5. ✅ 500-tick hard timeout retained as backup

### Match Rules

- **Critical (must match)**: zone, strategy
- **Tolerant (1-band diff OK)**: energy_band, hydration_band
- **Safety (must match)**: has_hazard_nearby
- **Ignored (non-strict)**: weather, biome

---

## Phase 2: Minimal Quotas

**Goal**: LLM outputs inventory targets, body uses deficits to shape behavior.

### LLM Output Format (Minimal)

```json
{
  "quotas": {"seed": 2, "fiber": 3, "wood": 1},
  "horizon_ticks": 400
}
```

### Body Behavior

Each tick:
1. Compute **deficits**: `deficit[item] = quota[item] - inventory.count(item)`
2. Use deficits in reward shaping: pickup actions get bonus proportional to deficit
3. Use deficits in heuristic: prioritize collecting items with highest deficit
4. GoalMapper considers deficits when selecting embodied goals

### Implementation

1. Add `Intent` dataclass with `quotas: dict[str, int]` and `horizon_ticks: int`
2. Add `self._current_intent: Optional[Intent]` to agent
3. New LLM prompt mode: `reason_intent()` that returns quotas (not tool sequences)
4. Add `_compute_deficits()` method
5. Modify `_compute_reward()` to include deficit-satisfaction bonus
6. Modify `_heuristic_decide()` to prioritize deficit items

### Trigger Conditions

Intent is refreshed when:
- Quotas are satisfied (all deficits ≤ 0)
- Horizon ticks elapsed
- Rupture triggered
- Major zone transition (healthy → crisis)

---

## Phase 3: Priority Stack

**Goal**: LLM sets ordered priorities, body works through them.

### LLM Output Format

```json
{
  "priority_stack": [
    "stabilize hydration",
    "collect 2 seeds",
    "plant near water"
  ],
  "quotas": {"seed": 2},
  "horizon_ticks": 500
}
```

### Body Behavior

1. Work on top priority until satisfied or blocked
2. Pop satisfied priorities from stack
3. Lower priorities provide tie-breaking guidance

### Integration with GoalMapper

Priority strings map to embodied goals:
- "stabilize hydration" → `seek_water`
- "collect seeds" → `explore_space` + pickup bias
- "plant near water" → `seek_water` + plant tool preference

---

## Phase 4: Constraints

**Goal**: LLM sets behavioral constraints the body respects.

### LLM Output Format

```json
{
  "priority_stack": ["start crop cycle"],
  "quotas": {"seed": 2},
  "constraints": [
    "avoid storm exposure",
    "don't plant without nearby water",
    "conserve energy above 0.4"
  ],
  "horizon_ticks": 600
}
```

### Body Behavior

Constraints act as **filters** on action selection:
- "avoid storm exposure" → penalty for plains/desert moves during storm
- "don't plant without nearby water" → disable plant tool if no water in radius
- "conserve energy above 0.4" → earlier rest/food-seeking threshold

### Implementation

Constraints are parsed into action modifiers:
```python
def _apply_constraints(self, action_scores: dict) -> dict:
    for constraint in self._current_intent.constraints:
        if constraint == "avoid storm exposure" and self._in_storm():
            # Penalize exposed biome moves
            ...
```

---

## Phase 5: Replan Triggers

**Goal**: LLM specifies when its intent should be reconsidered.

### LLM Output Format

```json
{
  "priority_stack": ["accumulate crafting materials"],
  "quotas": {"fiber": 3, "wood": 2, "stone": 2},
  "constraints": ["don't craft yet"],
  "replan_triggers": [
    "quotas_satisfied",
    "energy < 0.3",
    "rupture",
    "entered_new_biome"
  ],
  "horizon_ticks": 800
}
```

### Body Behavior

Each tick, check if any trigger condition is true:
```python
def _check_replan_triggers(self) -> bool:
    for trigger in self._current_intent.replan_triggers:
        if trigger == "quotas_satisfied" and all deficits <= 0:
            return True
        if trigger == "energy < 0.3" and self.energy < 0.3:
            return True
        # ...
    return False
```

If triggered, request new intent from LLM.

---

## Implementation Order

| Phase | Effort | Impact | Dependencies |
|-------|--------|--------|--------------|
| 0: Diagnostics | Low | High | Logging (done) |
| 1: State-validity | Medium | High | None |
| 2: Minimal quotas | Medium | Very High | None |
| 3: Priority stack | Medium | Medium | Phase 2 |
| 4: Constraints | Medium | Medium | Phase 2 |
| 5: Replan triggers | Low | Medium | Phase 2 |

**Recommended sequence**: 0 → 1 → 2 → (3,4,5 in any order)

---

## Key Design Principles

1. **Additive, not replacement**: Intent layer adds to existing system, doesn't replace heuristics or learned policy

2. **Graceful degradation**: If LLM unavailable or intent stale, body continues with reflexes

3. **Observable**: All intent activity logged for analysis

4. **Deficit-driven**: Quotas translate to deficits which shape rewards without language

5. **Teacher for intents too**: Demonstrations include (state, intent, outcome) for learning "good intents"

---

## Success Metrics

- **Plan stability**: Intents last longer than micro-plans (fewer refreshes per 1000 ticks)
- **Quota satisfaction**: % of intents that reach quota targets before expiration
- **Constraint compliance**: % of actions that respect active constraints
- **Survival integration**: LLM intents don't increase death rate vs heuristic-only

---

## Connection to Theory

From the Antifinity framework:
- **Intent = compressed potential**: LLM reduces infinite action space to finite goal space
- **Body = actualization**: Embodied agent actualizes potential through action
- **Deficit = gradient**: The "should" vs "is" tension that drives behavior

From QSE:
- **τ′-driven intent refresh**: High tension → more frequent LLM consultation
- **Rupture clears intent**: Cognitive reset includes intent reset
- **Surplus informs intent**: High surprise → intent may need revision
