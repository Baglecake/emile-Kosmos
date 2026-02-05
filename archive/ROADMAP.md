# emile-Kosmos Development Roadmap

## Current State (as of 2026-02-04)

**Completed:**
- Phase 1: Layer 2 (GoalMapper) + Layer 3 (ActionPolicy) wired with teacher-student decay
- Phase 2: Weather system, new objects (Herb/Seed/PlantedCrop), save/load persistence
- Phase 4: Multi-turn LLM reasoning, handoff metrics display
- Fixes 1-3: Hazard avoidance, consciousness zones, slower decay (0.9995, warmup 500)
- Phase 5a: Visual field perception (ASCII mini-map for LLM spatial awareness)
- Phase 5a-alt: Richer internal state (arousal, valence, zone in LLM prompt)
- Phase 5b: Event-triggered LLM calls (biome/strategy/zone/weather changes, discovery, entropy)
- Phase 5c: Surplus-faucet goal pressure (energy cost modulated by goal satisfaction)
- Phase 5d: Information metabolism (novelty tracking, anti-camping energy modulation)
- Phase 5e: Multi-step LLM planning (2-4 step plans, interrupt conditions, replan triggers)

**Resolved: Agent Death Spiral**
Fixed via consciousness zones (survival reflex override in crisis), hazard avoidance in heuristic,
and slower teacher decay. Agent now survives reliably and learns over time.

---

## Immediate Fixes (Pre-Phase)

These are targeted fixes to stabilize the agent before larger enhancements.

### Fix 1: Hazard Avoidance in Heuristic
**File:** `kosmos/agent/core.py` (`_heuristic_decide()`)
**What:** Before any other heuristic logic, check for hazards within 1-2 cells and move away.
**Why:** The heuristic currently ignores hazards. While the LLM is thinking, the agent can walk into danger.

### Fix 2: Survival Reflex Override (Consciousness Zones)
**File:** `kosmos/agent/core.py` (`tick()`)
**What:** Classify agent state into zones based on energy + entropy + surplus:
- **Crisis** (energy < 0.15 OR entropy > 0.8): Force heuristic survival actions only (eat/drink/rest/flee)
- **Struggling** (energy < 0.30): Bias toward teacher decisions, slow decay
- **Healthy** (normal): Normal teacher-student balance
- **Transcendent** (energy > 0.8 AND entropy < 0.3): Allow full exploration, faster decay
**Why:** Prevents the untrained MLP from killing the agent. Borrowed from emile-cogito's consciousness zones.

### Fix 3: Slower Teacher Decay
**File:** `kosmos/agent/core.py` (`tick()`)
**What:**
- Change decay rate: 0.998 -> 0.9995
- Increase warmup: 200 -> 500 samples
- Floor remains 0.1
**Why:** Gives the Student more time to learn before being trusted with survival decisions.

---

## Phase 5: Cogito-Inspired Enhancements

Ideas borrowed from emile-cogito that fit Kosmos's scope.

### 5a. Visual Field Perception (LLM as Perceptual Organ)
**Files:** `kosmos/agent/core.py`, `kosmos/llm/ollama.py`
**What:** Give the LLM spatial awareness via an ASCII mini-map instead of just text descriptions.

**Current (text-only):**
```
You are at (15, 12) in plains biome.
Nearby: berry (north, 2 cells), stone (here), snake (east, 3 cells)
```

**New (visual field):**
```
Your field of vision (5x5):
  .  T  F  T  .
  .  T  .  .  !
  .  .  @  .  .
  ~  .  .  +  .
  .  .  :  :  .

Legend: @ = you, F = food, ~ = water, ! = hazard, + = item
        T = forest, : = desert, ^ = rock, . = plains
```

**Implementation:**
- `_build_visual_field(radius=5)` generates ASCII grid
- Objects override biome display (hazard > food > water > item > biome)
- Include legend and cardinal direction markers
- Pass to LLM alongside situation text

**Perception hierarchy:**
| Layer | Speed | What it sees | Role |
|-------|-------|--------------|------|
| Reflex (heuristic) | Instant | Adjacent cells | Flee/eat urgently |
| Learned (ActionPolicy) | Fast | 30-dim state vector | Habitual actions |
| Conscious (LLM) | Slow | Full visual field | Strategic reasoning |

**Why:**
- Enables spatial reasoning ("hazard blocks path to food")
- Enables path planning ("go north then east")
- Enables pattern recognition ("resource cluster northwest")
- Natural foundation for multi-step planning (5e)
- Maps to cogito's sensorium concept

### 5a-alt. Richer Internal State in LLM Prompts
**Files:** `kosmos/agent/core.py`, `kosmos/llm/ollama.py`
**What:** Pass additional QSE-derived metrics to the LLM prompt:
- `consciousness_level`: derived from surplus mean and stability
- `valence`: positive/negative affect from recent rewards
- `arousal`: derived from entropy
- `regime`: stable_coherence / symbolic_turbulence / flat_rupture
**Why:** Makes narration dramatically richer and more varied. Can be combined with 5a.

### 5b. Event-Triggered Narration
**Files:** `kosmos/agent/core.py`, `kosmos/llm/ollama.py`
**What:** Don't fire narration every tick. Narrate only when meaningful events occur:
- Goal achieved
- Near-death experience
- New discovery (first time seeing an object type)
- Weather change
- Consciousness zone transition
**Why:** Fixes the "narration threads pile up" issue. Reduces LLM load. More natural storytelling.

### 5c. Surplus-Faucet Goal Pressure
**File:** `kosmos/agent/core.py`
**What:** When the agent isn't achieving its current goal, modulate QSE surplus growth rate downward:
```python
gamma_effective = gamma * (0.3 + 0.7 * goal_satisfaction)
```
**Why:** Creates natural metabolic pressure toward goal-directed behavior. More principled than fixed energy decay.

### 5d. Information Metabolism (Anti-Camping)
**File:** `kosmos/agent/core.py`
**What:** Track "novelty" — how many new cells/objects the agent has seen recently. Add a novelty hunger:
- High novelty: bonus energy recovery
- Low novelty (camping): slight energy drain
**Why:** Prevents camping on food piles. The agent must explore to stay healthy.

### 5e. Multi-Step LLM Planning (Strategic Planner)
**Files:** `kosmos/agent/core.py`, `kosmos/llm/ollama.py`
**What:** Shift the LLM from reactive single-step decisions to strategic multi-step planning:
- LLM returns a plan (3-5 tool calls) instead of a single action
- Agent maintains a plan queue, executing steps via ActionPolicy/heuristic
- Re-plan triggers: context change, goal change, plan exhausted, emergency interrupt
- Plan includes goal description and replan conditions

**Response format:**
```json
{
    "plan": [
        {"tool": "move", "args": {"direction": "north"}, "thought": "Heading toward forest"},
        {"tool": "pickup", "args": {"target": "wood"}, "thought": "Need materials"},
        {"tool": "craft", "args": {"item1": "wood", "item2": "stone"}, "thought": "Making an axe"}
    ],
    "goal": "Craft an axe for forest efficiency",
    "replan_if": ["inventory_full", "energy_critical", "hazard_nearby", "goal_changed"]
}
```

**Interrupt conditions:**
- Crisis zone (energy < 0.15) — abort plan, force survival reflex
- Hazard within 2 cells — abort plan, flee
- Context change (strategy shift) — request new plan
- Plan step fails — skip or request new plan

**Why:**
- Reduces LLM latency impact (1 call per 5-10 actions instead of 1 per action)
- Enables strategic reasoning ("gather materials, then craft, then use tool")
- Natural fit for teacher-student: LLM produces plans, ActionPolicy learns to predict/execute
- Richer emergent behavior from goal-directed sequences

**Caveats:**
- World changes during plan execution — validate each step before executing
- Need to balance plan length vs. staleness
- ActionPolicy must learn to handle mid-plan interrupts gracefully

---

## Phase 3: Multi-Agent (Future)

From emile-cogito's `agents.py`:
- Rupture events spawn new cognitive sub-agents
- Shared workspace for collective memory
- Spatial field overlays

Not planned for immediate implementation.

---

## Implementation Order

1. **Fix 1: Hazard Avoidance** — smallest change, biggest survival impact ✓
2. **Fix 2: Consciousness Zones** — cleanest version of survival override ✓
3. **Fix 3: Slower Decay** — simple config change ✓
4. **5a. Visual Field Perception** — LLM sees spatial mini-map (foundation for 5e) ✓
5. **5a-alt. Richer Internal State** — QSE metrics in prompt (can combine with 5a) ✓
6. **5b. Event-Triggered Narration** — fixes pile-up issue ✓
7. **5c. Surplus-Faucet** — more principled goal pressure ✓
8. **5d. Information Metabolism** — anti-camping ✓
9. **5e. Multi-Step Planning** — LLM as strategic planner (requires 5a) ✓

---

## Files to Modify

| Fix/Phase | Files |
|-----------|-------|
| Fix 1-3 | `kosmos/agent/core.py` |
| 5a | `kosmos/agent/core.py`, `kosmos/llm/ollama.py` |
| 5a-alt | `kosmos/agent/core.py`, `kosmos/llm/ollama.py` |
| 5b | `kosmos/agent/core.py`, `kosmos/llm/ollama.py` |
| 5c | `kosmos/agent/core.py` |
| 5d | `kosmos/agent/core.py` |
| 5e | `kosmos/agent/core.py`, `kosmos/llm/ollama.py` |

---

## Verification Criteria

After fixes 1-3:
- Run `python3 -m kosmos --speed 16` for 1000+ ticks
- Agent should survive consistently (< 3 deaths per 1000 ticks in normal weather)
- `teacher_prob` should decay slowly (still > 0.5 at tick 500)
- In crisis zone, agent should always use heuristic (no `learned` decisions)

After phase 5a:
- LLM prompt includes ASCII mini-map
- LLM responses reference spatial relationships ("going north to avoid hazard")
- Path planning emerges from spatial reasoning

After phase 5a-b:
- Narration should include internal state descriptions ("feeling anxious", "confident and energized")
- Narration should only fire on meaningful events (< 20 narrations per 100 ticks)

After phase 5c-d:
- Agent should not camp on single food source
- Agent should explore even when food is nearby

After phase 5e:
- LLM calls should be ~5-10x less frequent than ticks
- Agent should execute multi-step sequences (gather→craft→use)
- Plans should abort correctly on crisis/hazard
- ActionPolicy should learn plan step patterns over time

---

## Phase 6: Deeper Cogito Integration (Speculative)

Ideas from emile-cogito that require more validation before adopting.

### 6a. Self-Sustaining Ecology
World richness depends on agent's expressive success. New biomes/resources appear only if the agent crafts novel objects or produces insightful narrations.

### 6b. Autobiographical Memory
Enhanced memory that constructs narratives from past experiences, influences strategy selection, and produces richer LLM context.

### 6c. Qualia and Sensorium
Perceptual grounding — embedding visual/spatial context into QSE state. Ties environment representation to entropy/temperature modulation.

### 6d. Antifinity Metrics
Collaboration vs. compromise metrics for multi-agent scenarios. Relevant when Phase 3 (multi-agent) is implemented.

---

## Guiding Principles (from GPT Assessment)

1. **Incremental integration**: Cogito is complex and not fully validated. Integrate gradually, starting with clear utility (goal selection, memory) before speculative modules (qualia, antifinity).

2. **Maintain tool-based action space**: Keep the structured tool registry. Avoid free-form action outputs. This enables RL algorithms and behavior monitoring.

3. **Test harnesses**: Write unit tests and benchmarks before merging new components. Use gymnasium integration patterns for comparison.

4. **Performance profiling**: Some Cogito modules (QuTiP-based QSE) are computationally intensive. Profile when adding, simplify if real-time performance suffers.

5. **Documentation**: As modules are incorporated, document purpose, parameters, and expected outputs.

---

## Reference Documents

- `archive/ARCHITECTURE.md` — Original architecture document
- `archive/gemini_feedback.md` — Gemini's assessment
- `archive/comprehensive_review.md` — Full system review
- `GPT_feedback.md` — GPT's comprehensive assessment and recommendations
