# emile-Kosmos: Comprehensive Architecture & Development Handoff

## 1. Project Overview

**emile-Kosmos** is a living world simulation where an autonomous agent survives in a procedurally generated environment. The agent's cognition is driven by a **QSE (Quantum Surplus Emergence) wavefunction** that continuously evolves, producing entropy and context dynamics that modulate an **LLM's** creativity and personality in real time.

**Key innovation**: The QSE wavefunction runs at 20Hz between decisions. Shannon entropy from |psi|^2 determines how "creative" the LLM is (high entropy = high temperature = divergent thinking). TD(lambda) learns which cognitive strategies work in which contexts. The agent doesn't just react — it has genuine cognitive dynamics.

**Repository**: https://github.com/Baglecake/emile-Kosmos
**Parent project**: https://github.com/Baglecake/emile-mini (QSE cognitive engine)
**Environment**: macOS, Python 3.14 via pyenv, Ollama for local LLM inference, pygame-ce for rendering

---

## 2. Directory Structure

```
emile-kosmos/
  emile_mini/                    # Vendored QSE cognitive engine (from emile-mini v0.6.0)
    __init__.py                  # Exports EmileAgent, QSEConfig
    config.py                    # QSEConfig dataclass (52 parameters) + YAML loader
    agent.py                     # EmileAgent: main cognitive orchestration loop
    qse_core.py                  # QSEEngine: Schrodinger equation + surplus dynamics
    symbolic.py                  # SymbolicReasoner: computes Psi, Phi, Sigma fields
    memory.py                    # MemoryModule: working + episodic + semantic memory
    context.py                   # ContextModule: context switching with hysteresis
    goal.py                      # GoalModule: simple goal selection (legacy)
    goal_v2.py                   # GoalModuleV2: TD(lambda) strategy selection (active)
    multimodal.py                # Text/image/audio adapters + fusion layer
    --- NOT YET WIRED INTO KOSMOS ---
    embodied_qse_emile.py        # EmbodiedQSEAgent: full 3-layer learned pipeline
    goal_mapper.py               # GoalMapper: Layer 2 (strategy -> goal via TD-lambda)
    action_policy.py             # ActionPolicy: Layer 3 (goal -> action via REINFORCE)
    utils/
      tau_prime.py               # Emergent time calculation helpers
      json_logger.py             # JSONL logging utility
  kosmos/                        # The Kosmos application
    __init__.py
    __main__.py                  # Entry point, CLI argument parsing
    world/
      __init__.py
      grid.py                    # KosmosWorld: biome grid, objects, day/night, seasons
      objects.py                 # Food, Water, Hazard, CraftItem + craft recipes
    agent/
      __init__.py
      core.py                    # KosmosAgent: QSE + tools + LLM + survival
    tools/
      __init__.py
      registry.py                # ToolRegistry: extensible tool system with JSON schemas
      builtins.py                # 8 built-in tools: move, examine, consume, pickup, craft, rest, remember, wait
    llm/
      __init__.py
      ollama.py                  # OllamaReasoner: QSE-modulated LLM with structured tool calls
    render/
      __init__.py
      pygame_render.py           # KosmosRenderer: full pygame visualization
  pyproject.toml
  .python-version                # pyenv: Python 3.14.0
  .gitignore
```

---

## 3. How to Run

```bash
cd ~/emile-kosmos
eval "$(pyenv init -)"
python3 -m kosmos --model phi3:mini --speed 8
```

**Options:**
- `--model MODEL` — Ollama model name (default: llama3.1:8b). Smaller = faster (phi3:mini recommended)
- `--size SIZE` — World grid dimension (default: 30)
- `--speed SPEED` — Ticks per second (default: 8)
- `--seed SEED` — World generation seed

**Controls:** SPACE=pause, UP/DOWN=speed, Q/ESC=quit

**Requirements:** Python 3.10+, numpy, scipy, matplotlib, pygame-ce, requests, Ollama running locally

---

## 4. Architecture Deep Dive

### 4.1 The QSE Cognitive Engine (emile_mini/)

The mathematical core. A 1D quantum field theory simulation that produces emergent cognitive dynamics.

**The cognitive loop (EmileAgent.step):**
1. **Symbolic fields** (symbolic.py): Surplus field S -> compute Psi (presence), Phi (tension), Sigma (curvature)
2. **QSE dynamics** (qse_core.py): Update surplus S with growth/decay/rupture. Evolve quantum wavefunction psi via split-step FFT Schrodinger equation
3. **Entropy** (qse_core.py): Shannon entropy of |psi|^2 probability density -> normalized to [0,1]. High entropy = spread-out wavefunction = uncertain/creative state. Low entropy = peaked wavefunction = focused/certain state
4. **Context** (context.py): Track Sigma EMA. When |delta_sigma| exceeds threshold (with hysteresis + dwell time), shift context. Context is an integer ID representing the agent's "frame of reference"
5. **Strategy** (goal_v2.py): TD(lambda) selects from {explore, exploit, rest, learn, social} based on (context, energy, reward_history). Learns which strategies produce reward in which states
6. **Memory** (memory.py): Working memory (last 10 items), episodic memory (last 1000 structured events), semantic memory (accumulated knowledge)

**Key parameters (config.py):**
- `GRID_SIZE=64` — spatial resolution of the quantum field
- `HBAR=1.0`, `MASS=1.0` — quantum constants
- `S_GAMMA=0.37` — surplus growth rate
- `S_THETA_RUPTURE=0.5` — threshold for surplus rupture events
- `RECONTEXT_THRESHOLD=0.35` — sigma change needed for context switch
- `CONTEXT_MIN_DWELL_STEPS=15` — minimum steps in one context before switching

### 4.2 The Kosmos Agent (kosmos/agent/core.py)

Wraps EmileAgent with survival mechanics, tool use, and LLM reasoning.

**Current tick cycle (KosmosAgent.tick):**
1. Basal metabolism: energy -0.003, hydration -0.001 per tick
2. Death check: energy <= 0 -> die, respawn at center with memories intact, inventory lost
3. QSE step: update wavefunction, get context/entropy/surplus
4. Strategy selection: GoalModuleV2.select_goal(context, energy, entropy)
5. Build situation description (text summary of position, nearby objects, energy)
6. **If LLM available**: Ask LLM to reason about tools (returns structured JSON: {tool, args, thought})
7. **If LLM offline**: Use heuristic decision tree (emergency food-seeking, opportunistic consumption, random exploration)
8. Execute tool via ToolRegistry
9. Compute reward and feed back to GoalModuleV2 + QSE engine

**Survival mechanics:**
- Energy: 0-100%. Depletes via movement (biome-dependent), basal metabolism. Restored by eating food, resting
- Hydration: 0-100%. Depletes slowly. Restored by drinking water
- Death: energy=0. Respawn with memories but lose inventory and crafted items
- Crafting: combine 2 materials -> tool (axe reduces forest cost, rope reduces water cost, sling reduces hazard damage, bowl carries water)

**Inventory:** Up to 6 CraftItems. Lost on death.

### 4.3 The World (kosmos/world/)

**grid.py — KosmosWorld:**
- Biomes generated via smoothed noise (bilinear upscaled 6x6 random grid):
  - PLAINS (v<0.4→0.7): normal movement, food spawns here
  - FOREST (v<0.2→0.4): 1.3x move cost, food+water spawn, rest bonus
  - DESERT (v<0.7→0.85): 1.8x move cost, hazards spawn
  - WATER (v<0.2): 2.5x move cost, water spawns
  - ROCK (v>0.85): 3.0x move cost, hazards spawn
- Day/night cycle: 200 ticks/day. Night (60-100% of cycle) = 1.4x movement cost
- Seasons: 800 ticks/season, 4 seasons. (Currently cosmetic only — not yet affecting spawns)
- Food migration: food rots (decay_rate per tick), new food spawns randomly (never same spot)
- Objects: Food (4 variants: berry/mushroom/fruit/root), Water (puddle), Hazard (thorns/snake/pitfall), CraftItem (stick/stone/fiber/shell)

**objects.py:**
- All objects have `tick()` -> returns False when decayed (removed from world)
- Food decay rates: berry 0.004, mushroom 0.006, fruit 0.002, root 0.001
- Energy values: berry 0.20, mushroom 0.15, fruit 0.35, root 0.12
- Craft recipes: wood+stone=axe, wood+fiber=rope, stone+fiber=sling, wood+shell=bowl

### 4.4 The Tool System (kosmos/tools/)

**registry.py — ToolRegistry:**
- Tools have: name, description, parameters dict, category, bound function
- Categories: action, perception, social, meta
- `schemas()` returns JSON tool descriptions for LLM consumption
- `invoke(name, **kwargs)` calls the bound function, returns {success, result/error}

**builtins.py — 8 built-in tools:**
| Tool | Category | Parameters | What it does |
|------|----------|------------|--------------|
| move | action | direction (n/s/e/w) | Move one cell, apply biome cost, check hazards |
| examine | perception | target | Describe surroundings or specific object |
| pickup | action | item | Take CraftItem from ground to inventory |
| consume | action | item | Eat food or drink water at current position |
| craft | action | item1, item2 | Combine two inventory items via recipe |
| rest | action | — | Recover energy (biome-dependent: forest best, desert worst) |
| remember | meta | query | Keyword search agent's memory |
| wait | action | — | Do nothing |

### 4.5 The LLM Integration (kosmos/llm/ollama.py)

**OllamaReasoner:**
- `reason()`: Sends situation description + tool schemas -> gets structured JSON response: `{tool, args, thought}`
- `narrate()`: Generates short first-person narration of events (max 20 words)
- Temperature mapping: `0.3 + entropy * 1.2` (range 0.3-1.5)
- Strategy -> personality:
  - explore: "You are curious and adventurous. Seek the unknown."
  - exploit: "You are efficient and focused. Get what you need directly."
  - rest: "You are tired and cautious. Conserve energy. Rest if safe."
  - learn: "You are analytical. Examine things. Gather information."
  - social: "You are sociable. Look for others. Communicate."
- Uses `format: "json"` for structured output
- Timeout: 30s for reasoning, 15s for narration
- Falls back to heuristic when Ollama is unreachable
- QSE strategy also filters which tool categories the LLM sees (e.g., explore/learn get perception+meta tools, exploit only gets action)

### 4.6 The Renderer (kosmos/render/pygame_render.py)

**KosmosRenderer:**
- Grid: cells colored by biome, dimmed at night
- Objects: circles (food), diamonds (water), X marks (hazards), plus signs (craft items)
- Agent: yellow circle with directional eye, fading trail (last 80 positions)
- Right panel (280px): strategy (colored), context, time/season, energy bar, hydration bar, entropy bar, LLM temp, stats (food/water/deaths/steps/explored), inventory list, crafted items, thought bubble
- Bottom panel (100px): LLM narration lines (scrolling, word-wrapped)
- Narration triggers: food consumed, crafted item, hazard hit, death, periodic (every 60 ticks)

---

## 5. What's Currently Working

| Component | Status | Notes |
|-----------|--------|-------|
| QSE wavefunction | WORKING | Runs at 20Hz in background thread |
| Entropy computation | WORKING | Shannon entropy from |psi|^2 |
| Context switching | WORKING | With hysteresis and dwell time |
| Strategy selection (L1) | LEARNING | TD(lambda) with eligibility traces |
| LLM tool selection | WORKING | Structured JSON via Ollama |
| LLM narration | WORKING | Async, triggered by events |
| Heuristic fallback | WORKING | When LLM offline |
| World simulation | WORKING | Biomes, day/night, food decay/migration |
| Survival mechanics | WORKING | Energy depletion = death, respawn |
| Crafting | WORKING | 4 recipes, crafted items give bonuses |
| Tool registry | WORKING | 8 tools, extensible |
| Pygame rendering | WORKING | Grid + panel + narration |
| Goal mapping (L2) | NOT WIRED | File exists, not integrated |
| Action policy (L3) | NOT WIRED | File exists, not integrated |
| World persistence | MISSING | No save/load |
| Multi-agent | MISSING | Single agent only |

---

## 6. What's NOT Wired In (Future Integration)

### 6.1 Layer 2: GoalMapper (goal_mapper.py)

**What it does:** Learns which *embodied goals* to pursue given a *strategy*. Instead of a hardcoded mapping (explore -> explore_space, exploit -> seek_food), it learns from experience via TD(lambda).

**How to wire it in:**
- Add `GoalMapper` to `KosmosAgent.__init__`
- After strategy selection, call `goal_mapper.select_goal(strategy, energy, health, food_nearby, shelter_nearby, entropy)` to get a specific goal
- After reward computation, call `goal_mapper.update(reward, ...)`
- The goal would then influence which tools the LLM considers or the heuristic fallback

**Key detail:** Has warm-start from hardcoded rules, so it starts reasonable and improves.

### 6.2 Layer 3: ActionPolicy (action_policy.py)

**What it does:** A small numpy MLP (30-dim input -> 32 hidden -> 7 actions) trained via REINFORCE. Learns which actions produce reward for each goal. Uses entropy-modulated temperature for exploration.

**How to wire it in (teacher-student pattern):**
1. LLM acts as "teacher" — selects actions initially
2. ActionPolicy learns from the rewards those actions produce
3. `teacher_prob` starts at 1.0 and decays toward 0.1
4. Over time, the learned policy takes over from the LLM
5. Critical: needs the warmup fix from emile-mini (200 learned-policy samples before performance gating kicks in)

**The teacher decay system (from embodied_qse_emile.py):**
```python
# Warmup: unconditional decay until learned policy has enough samples
if self._learned_samples < 200:
    self._teacher_prob = max(floor, self._teacher_prob * 0.998)
elif self._learned_reward_ema >= self._heuristic_reward_ema - 0.05:
    self._teacher_prob = max(floor, self._teacher_prob * 0.998)
else:
    # Learned policy underperforming — nudge back toward teacher
    self._teacher_prob = min(1.0, self._teacher_prob / 0.998)
```

### 6.3 Full 3-Layer Pipeline (embodied_qse_emile.py)

Reference implementation of all three layers working together. This file contains:
- `EmbodiedQSEAgent`: extends EmileAgent with the full learned stack
- `SensoriMotorBody`: body schema with proprioception
- `EmbodiedEnvironment`: the original emile-mini grid world

Key patterns to extract for Kosmos integration:
- Teacher decay with warmup (lines ~790-809)
- Reward shaping (dense reward from food, energy delta, exploration bonus)
- Existential pressure detection (loop detection -> force context switch)
- Performance-gated teacher decay (learned EMA vs heuristic EMA comparison)

---

## 7. Known Issues & Bugs

1. **Agent camps food spots**: In the simple emile-mini version, the agent finds food and circles it forever. Kosmos addresses this with food decay/migration and real death, but the heuristic fallback still tends toward local optima.

2. **LLM latency**: Each reasoning call takes 1-5 seconds depending on model. At speed 8+, the agent may fall behind. Phi3:mini is fastest. Consider async queuing so the world keeps ticking while LLM thinks.

3. **Seasons are cosmetic**: The season system exists but doesn't affect spawn rates, food availability, or hazard frequency. Should be wired in for more dynamic gameplay.

4. **No world persistence**: World state is lost on quit. No save/load system.

5. **Narration can queue up**: If LLM is slow, narration requests pile up in background threads. Could overwhelm with many threads.

6. **Heuristic fallback is simplistic**: When LLM is offline, the agent just does basic if/else (low energy -> seek food, otherwise wander). Doesn't use the full tool set.

---

## 8. Development Roadmap

### Phase 1: Wire Learning Layers
- Integrate GoalMapper (L2) into KosmosAgent
- Integrate ActionPolicy (L3) with teacher decay from LLM
- Add teacher_prob display to renderer panel
- Verify the warmup fix works in Kosmos context

### Phase 2: Richer World
- Seasons affect spawns (winter = less food, summer = more hazards)
- Weather events (rain = water spawns, storm = energy cost)
- More object types (medicine, tools, clothing)
- More craft recipes
- World save/load (JSON serialization)

### Phase 3: Multi-Agent
- Multiple KosmosAgents sharing the world
- Social strategy enables communication between agents
- Teaching/learning between agents (social learning from emile-mini)
- Competition for resources

### Phase 4: Better LLM Integration
- Multi-turn reasoning (agent remembers recent reasoning chain)
- LLM as teacher with measured handoff to learned policy
- Async LLM calls so world doesn't block
- Evaluation: track when learned policy outperforms LLM

### Phase 5: UI/UX
- Mouse interaction (click to inspect cells/objects)
- Minimap for large worlds
- Screenshot/recording export
- Dashboard with learning curves
- Configuration UI (tweak parameters live)

---

## 9. Key Design Decisions

1. **QSE entropy -> LLM temperature**: This is the core insight. The agent's internal cognitive state (wavefunction entropy) directly modulates how creative the LLM's responses are. High entropy = spread-out quantum state = uncertain = high temperature = creative/divergent. Low entropy = peaked = certain = low temperature = focused/convergent.

2. **Strategy -> personality**: Rather than just adjusting temperature, the QSE strategy changes the LLM's entire personality via system prompt. This creates qualitatively different behavior modes, not just quantitative variation.

3. **Tool registry**: Actions are structured, not free-text. This makes the system auditable, learnable (Layer 3 can train on discrete action space), and reliable (no parsing free-text LLM output for actions).

4. **Real death**: Unlike emile-mini (energy floor at 0.05), Kosmos has real consequences. This creates genuine survival pressure that drives meaningful exploration and resource-seeking behavior.

5. **Food migration**: Resources don't regenerate in the same spot. This prevents the camping behavior observed in emile-mini's visual world.

6. **Vendored emile-mini**: Core QSE engine is copied into the repo rather than used as a dependency. This makes the repo self-contained and allows independent evolution.

---

## 10. Technical Reference

### Dependencies
- Python >= 3.10 (tested on 3.14)
- numpy >= 1.23
- scipy >= 1.10 (for FFT in QSE engine)
- matplotlib >= 3.7 (for potential future visualizations)
- pygame-ce >= 2.4 (community edition, has Python 3.14 wheels)
- requests >= 2.28 (for Ollama API)
- Ollama (local LLM server, https://ollama.com)
- pyyaml (optional, for config.yaml loading)

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi3:mini      # fastest, good for tool calls
ollama pull llama3.1:8b    # better quality
ollama pull qwen2.5:32b    # best quality (needs ~20GB RAM)

# Ollama runs on http://localhost:11434
```

### QSE Mathematical Foundation
The QSE engine implements three theorems from the emile framework:

**Theorem 1 (Symbolic Fields):** Surplus field S -> Psi (sigmoid activation), Phi (linear activation), Sigma (curvature = Psi - Phi)

**Theorem 2 (Surplus Dynamics):** dS/dt = gamma*S*(1-S) + beta*laplacian(sigma) + tension*phi - damping*S. When S > theta_rupture, rupture occurs (S *= 1 - epsilon).

**Theorem 3 (Emergent Time):** tau = tau_min + (tau_max - tau_min) * sigmoid(k * (|delta_sigma| - theta)). Large changes in sigma slow down time (more processing). Small changes speed up time.

**Schrodinger Evolution:** Split-step FFT method. V(x) = double_well + sigma-coupled dynamic potential. Wavefunction psi evolves under this potential. |psi|^2 is the probability density whose Shannon entropy drives exploration.
