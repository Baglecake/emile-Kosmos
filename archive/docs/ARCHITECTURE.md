# emile-Kosmos: Comprehensive Architecture & Development Handoff

**Last Updated:** 2026-02-05
**Latest Commit:** dab4be2 - Improve heuristic food-seeking with adaptive radius and memory

## 1. Project Overview

**emile-Kosmos** is a living world simulation where an autonomous agent survives in a procedurally generated environment. The agent's cognition is driven by a **QSE (Quantum Surplus Emergence) wavefunction** that continuously evolves, producing entropy and context dynamics that modulate an **LLM's** creativity and personality in real time.

**Key innovation**: The QSE wavefunction runs at 20Hz between decisions. Shannon entropy from |psi|^2 determines how "creative" the LLM is (high entropy = high temperature = divergent thinking). TD(lambda) learns which cognitive strategies work in which contexts. The agent doesn't just react — it has genuine cognitive dynamics.

**Repository**: https://github.com/Baglecake/emile-Kosmos
**Parent project**: https://github.com/Baglecake/emile-mini (QSE cognitive engine)
**Environment**: macOS, Python 3.14 via pyenv, Ollama for local LLM inference, pygame-ce for rendering

---

## 2. Directory Structure

```
emile-Kosmos-1/
  emile_mini/                    # Vendored QSE cognitive engine (v0.6.0)
    __init__.py                  # Exports EmileAgent, QSEConfig
    config.py                    # QSEConfig dataclass (52 parameters) + YAML loader
    agent.py                     # EmileAgent: main cognitive orchestration loop
    qse_core.py                  # QSEEngine: Schrodinger equation + surplus dynamics
    symbolic.py                  # SymbolicReasoner: computes Psi, Phi, Sigma fields
    memory.py                    # MemoryModule: working + episodic + semantic memory
    context.py                   # ContextModule: context switching with hysteresis
    goal.py                      # GoalModule: simple goal selection (legacy)
    goal_v2.py                   # GoalModuleV2: TD(lambda) strategy selection (L1)
    goal_mapper.py               # GoalMapper: Layer 2 (strategy -> goal) - WIRED
    action_policy.py             # ActionPolicy: Layer 3 reference (goal -> action)
    embodied_qse_emile.py        # Reference implementation of full 3-layer pipeline
    multimodal.py                # Text/image/audio adapters + fusion layer
    utils/
      tau_prime.py               # Emergent time calculation helpers
      json_logger.py             # JSONL logging utility

  kosmos/                        # The Kosmos application
    __init__.py
    __main__.py                  # Entry point, CLI args (--save, --load, --speed)
    persistence.py               # save_state()/load_state() for world+agent

    agent/
      __init__.py
      core.py                    # KosmosAgent: tick loop, consciousness zones, planning
      action_policy.py           # KosmosActionPolicy: MLP for learned actions (L3)
      demo_buffer.py             # DemonstrationBuffer for behavior cloning

    world/
      __init__.py
      grid.py                    # KosmosWorld: biomes, day/night, seasons, weather
      objects.py                 # Food/Water/Hazard/Herb/Seed/PlantedCrop + recipes
      weather.py                 # WeatherManager: rain/storm/heat_wave/fog/wind

    tools/
      __init__.py
      registry.py                # ToolRegistry: extensible tool system with JSON schemas
      builtins.py                # 9 tools: move, examine, pickup, consume, craft, rest, remember, wait, plant

    llm/
      __init__.py
      ollama.py                  # OllamaReasoner: reason() + reason_plan() for multi-step

    render/
      __init__.py
      pygame_render.py           # Full visualization with zone/plan/novelty bars

  tests/
    __init__.py
    test_extended.py             # Extended integration test with metrics

  archive/
    docs/                        # Documentation and design notes
      ARCHITECTURE.md            # This file
      ROADMAP.md                 # Development roadmap
      NEXT_STEPS.md              # Implementation details
      GPT_*.md                   # AI-assisted analysis
    test_results/                # Historical test result JSON files

  dev_files/
    Emile_full_package_v1.ipynb  # Historical development notebook

  social_qse_agent_v2.py         # Multi-agent reference (for Phase 3)
  pyproject.toml
  .python-version                # pyenv: Python 3.14.0
  .gitignore
```

---

## 3. How to Run

```bash
cd ~/emile-Kosmos-1
eval "$(pyenv init -)"
python3 -m kosmos --model llama3.1:8b --speed 8
```

**Options:**
- `--model MODEL` — Ollama model name (default: llama3.1:8b)
- `--size SIZE` — World grid dimension (default: 30)
- `--speed SPEED` — Ticks per second (default: 8)
- `--seed SEED` — World generation seed
- `--save PATH` — Save state on exit
- `--load PATH` — Load state on start

**Controls:** SPACE=pause, UP/DOWN=speed, Q/ESC=quit

**Requirements:** Python 3.10+, numpy, scipy, matplotlib, pygame-ce, requests, Ollama running locally

**Extended Testing:**
```bash
python3 tests/test_extended.py --minutes 10 --output archive/test_results/my_test.json
```

---

## 4. Architecture Deep Dive

### 4.1 Three-Layer Learning Architecture (ALL WIRED)

```
Layer 1: GoalModuleV2 (TD-lambda)
         Context + Energy + Entropy -> Strategy (explore/exploit/rest/learn/social)
                      |
                      v
Layer 2: GoalMapper (TD-lambda with warm-start)
         Strategy -> Embodied Goal (7 goals: explore_space, seek_food, find_shelter, etc.)
                      |
                      v
Layer 3: KosmosActionPolicy (MLP via REINFORCE)
         Goal + State -> Action (11 actions: move_N/S/E/W, consume, pickup, etc.)

         Teacher-Student: LLM/heuristic teaches policy, competence-gated handoff
```

### 4.2 The QSE Cognitive Engine (emile_mini/)

The mathematical core. A 1D quantum field theory simulation that produces emergent cognitive dynamics.

**The cognitive loop (EmileAgent.step):**
1. **Symbolic fields** (symbolic.py): Surplus field S -> compute Psi (presence), Phi (tension), Sigma (curvature)
2. **QSE dynamics** (qse_core.py): Update surplus S with growth/decay/rupture. Evolve quantum wavefunction psi via split-step FFT Schrodinger equation
3. **Entropy** (qse_core.py): Shannon entropy of |psi|^2 probability density -> normalized to [0,1]
4. **Context** (context.py): Track Sigma EMA. When |delta_sigma| exceeds threshold, shift context
5. **Strategy** (goal_v2.py): TD(lambda) selects from {explore, exploit, rest, learn, social}
6. **Memory** (memory.py): Working memory (last 10 items), episodic memory (last 1000 events)

### 4.3 The Kosmos Agent (kosmos/agent/core.py)

**Current tick cycle (1500 lines):**

1. **Basal metabolism** (tuned): energy -0.002, hydration -0.0008 per tick
2. **Weather effects**: heat_wave extra drain, storm damage in exposed biomes
3. **Death check**: energy <= 0 -> respawn with goal_mapper reset, log death rate
4. **QSE step**: update wavefunction, get context/entropy/surplus
5. **L1**: GoalModuleV2.select_goal() -> strategy
6. **L2**: GoalMapper.select_goal() -> embodied_goal (7 options)
7. **Consciousness zone**: crisis (<0.25) / struggling (<0.40) / healthy / transcendent
8. **Decision**:
   - Crisis zone: survival_reflex (heuristic with adaptive food search)
   - Else: teacher-student decision (roll against teacher_prob)
   - Teacher path: consume LLM plan or heuristic
   - Student path: KosmosActionPolicy.select_action()
9. **Event-triggered LLM** (fires on meaningful events only):
   - Biome/strategy/zone/weather change
   - First discovery, high entropy, stuckness
   - Periodic refresh (every 25 ticks)
10. **Execute tool** via ToolRegistry
11. **Compute reward** with shaping (urgency bonus, missed-food penalty)
12. **Anti-oscillation**: penalty for repeated *directional* actions
13. **Update all layers**: L1, L2, L3 (REINFORCE + behavior cloning)
14. **Teacher decay** (competence-based):
    - Warmup: track baseline death rate
    - After warmup: decay only when student proves competent
    - Recovery cap at 0.85 (student always gets ~15% of decisions)
15. **Surplus-faucet**: goal satisfaction modulates energy cost
16. **Info metabolism**: novelty modulates energy (+explore, -camping)
17. **Stuckness detection**: <=3 unique positions triggers escape
18. **Feed modulated reward to QSE**

### 4.4 Teacher-Student Architecture

```python
# Competence-based decay (lines 866-902 in core.py)
warmup = 2000  # samples before evaluation
floor = 0.1   # minimum teacher probability

if learned_samples < warmup:
    # Track BEST death rate as baseline (frozen at warmup end)
    if death_rate_ema < baseline_death_rate:
        baseline_death_rate = death_rate_ema
    teacher_prob *= 0.9999  # Very slow decay
else:
    # Freeze baseline at warmup end
    reward_competent = learned_ema >= 0.9 * heuristic_ema
    survival_competent = death_rate_ema <= 1.2 * baseline_death_rate

    if reward_competent and survival_competent:
        teacher_prob = max(floor, teacher_prob * 0.9995)  # Decay
    elif not survival_competent:
        teacher_prob = min(0.85, teacher_prob * 1.0005)   # Recover (capped)
```

### 4.5 Behavior Cloning (Consolidation)

```python
# DemonstrationBuffer stores teacher decisions
# Every 200 ticks: cross-entropy training from demonstrations
demos = demo_buffer.sample(batch_size=64, weighted=True)
bc_stats = behavior_cloning_update(action_policy, demos, lr=0.005)
```

Key insight from GPT: "consciousness interprets -> you act -> nervous system consolidates offline"

### 4.6 State Encoding (34-dim)

```python
# Indices 0-28: vitals, biome, time, nearby objects, inventory, strategy, QSE
# Indices 29-32: Directional cues (food_dx/dy, hazard_dx/dy) - CRITICAL
# Index 33: should_eat_urgency (combines food_here + low_energy)
```

---

## 5. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| QSE wavefunction | WORKING | 20Hz background thread |
| Strategy selection (L1) | LEARNING | TD(lambda) |
| Goal mapping (L2) | LEARNING | TD(lambda) with warm-start |
| Action policy (L3) | LEARNING | MLP + REINFORCE + behavior cloning |
| Teacher-student decay | WORKING | Competence-based, baseline frozen |
| LLM tool selection | WORKING | Event-triggered, multi-step plans |
| Multi-step planning | WORKING | LLM returns 2-4 step plans |
| Visual field | WORKING | ASCII mini-map for LLM |
| Consciousness zones | WORKING | Crisis/struggling/healthy/transcendent |
| Surplus-faucet | WORKING | Goal satisfaction modulates energy |
| Info metabolism | WORKING | Novelty modulates energy |
| Anti-oscillation | WORKING | Directional action penalties |
| Stuckness detection | WORKING | Triggers escape behavior |
| Weather system | WORKING | Rain/storm/heat/fog/wind |
| Save/load | WORKING | JSON serialization |
| Pygame rendering | WORKING | Full visualization |
| Multi-agent | NOT STARTED | Reference code available |

**Performance (from tests/test_extended.py):**
- Death rate: ~16 per 1000 ticks (was ~25 before fixes)
- Food per life: ~0.42 (180% improvement)
- Learned policy: reaches 60%+ of decisions
- Survival: 8000+ ticks without death streaks

---

## 6. Known Issues (Resolved)

1. **Anti-oscillation tracked "move" not "move_north"** - FIXED: Now uses directional action names
2. **State encoding lacked directional cues** - FIXED: 34-dim with food_dx/dy, hazard_dx/dy
3. **Exploration reward never firing** - FIXED: Cell added to visited AFTER reward check
4. **Baseline death rate degrading** - FIXED: Track BEST rate, freeze at warmup end
5. **Teacher recovery to 1.0 stops learning** - FIXED: Cap at 0.85
6. **Agent dying when no food in radius** - FIXED: Adaptive radius (8->12) + food memory

---

## 7. Development Roadmap

### Completed Phases

- **Phase 1**: L2+L3 wired with teacher-student decay
- **Phase 2**: Weather, Herb/Seed/PlantedCrop, save/load
- **Phase 4**: Multi-turn LLM, handoff metrics
- **Phase 5a**: Visual field (ASCII mini-map)
- **Phase 5b**: Event-triggered LLM
- **Phase 5c**: Surplus-faucet (goal pressure)
- **Phase 5d**: Information metabolism (novelty)
- **Phase 5e**: Multi-step planning
- **Phase 5f**: Anti-oscillation
- **Phase 5g**: Stuckness detection

### Pending Phases

**Phase 3: Multi-Agent** (Priority: High)
- Reference: `social_qse_agent_v2.py` (1151 lines)
- Social signals: help/warning/share/teach/follow/avoid
- Teaching/learning between agents
- Competition for resources
- Emergent personality traits

**Macro/Skill System** (Priority: Medium)
- LLM creates named affordances ("get food", "escape storm")
- Policy learns when to trigger macros
- Bridges gap between deliberate planning and reflexive behavior

**Phase 5h: Cognitive Battery** (Priority: Low)
- Benchmarking framework from `cognitive_battery.py`
- Protocol A: Zero-training baseline
- Protocol C1: Context-switch adaptation
- Protocol C2: Memory-cued navigation

---

## 8. Key Design Decisions

1. **QSE entropy -> LLM temperature**: Internal cognitive state modulates creativity
2. **Strategy -> personality**: QSE strategy changes LLM's system prompt
3. **Tool registry**: Structured actions, not free-text (auditable, learnable)
4. **Real death**: Genuine survival pressure drives meaningful behavior
5. **Food migration**: Resources don't regenerate in same spot (anti-camping)
6. **Competence-based decay**: Student must prove competent before taking over
7. **Behavior cloning**: Teacher demonstrations are replayed for offline consolidation
8. **Directional state cues**: Policy needs to know *where* food/hazards are, not just count

---

## 9. Key Parameters

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| Basal metabolism | core.py | 0.002/0.0008 | Energy/hydration drain per tick |
| Crisis zone | core.py | <0.25 | Energy threshold for survival reflex |
| Struggling zone | core.py | <0.40 | Energy threshold for bias toward teacher |
| Teacher decay | core.py | 0.9995 | Per-step decay multiplier |
| Teacher floor | core.py | 0.10 | Minimum teacher probability |
| Teacher cap | core.py | 0.85 | Maximum recovery (student always learns) |
| Warmup samples | core.py | 2000 | Before competence evaluation |
| BC interval | core.py | 200 | Ticks between behavior cloning updates |
| BC batch size | core.py | 64 | Demonstrations per BC update |
| Anti-oscillation window | core.py | 10 | Recent actions tracked |
| Stuckness threshold | core.py | 3 | Unique positions to be "stuck" |
| State dimensions | action_policy.py | 34 | Input features to policy MLP |

---

## 10. Technical Reference

### Dependencies
- Python >= 3.10 (tested on 3.14)
- numpy >= 1.23
- scipy >= 1.10 (for FFT in QSE engine)
- matplotlib >= 3.7
- pygame-ce >= 2.4
- requests >= 2.28
- Ollama (local LLM server)
- pyyaml (optional, for config loading)

### Ollama Setup
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
# Ollama runs on http://localhost:11434
```

### QSE Mathematical Foundation

**Theorem 1 (Symbolic Fields):** Surplus S -> Psi (sigmoid), Phi (linear), Sigma (curvature)

**Theorem 2 (Surplus Dynamics):** dS/dt = gamma*S*(1-S) + beta*laplacian(sigma) + tension*phi - damping*S

**Theorem 3 (Emergent Time):** tau = tau_min + (tau_max - tau_min) * sigmoid(k * (|delta_sigma| - theta))

**Schrodinger Evolution:** Split-step FFT. |psi|^2 entropy drives exploration.
