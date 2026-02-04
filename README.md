# émile-Kosmos

A living world simulation where an autonomous agent survives in a procedurally
generated environment. The agent's cognition is driven by a **Quantum Surplus
Emergence (QSE)** wavefunction that modulates an LLM's creativity and
personality in real time.

## The QSE Innovation

Traditional AI agents use fixed decision logic or static LLM prompts.
emile-Kosmos introduces a fundamentally different approach:

- A **quantum wavefunction** evolves continuously (20 Hz) via split-step FFT
  Schrodinger dynamics
- **Shannon entropy** of |psi|^2 determines the LLM's temperature: high entropy
  (spread-out wavefunction) = creative/divergent thinking; low entropy (peaked
  wavefunction) = focused/convergent thinking
- **TD(lambda)** learns which cognitive strategies (explore, exploit, rest,
  learn, social) work in which contexts
- The QSE strategy also shifts the LLM's entire **personality** via system
  prompt, creating qualitatively different behavior modes

The result: an agent with genuine cognitive dynamics, not just reactive
decision-making.

Built on the [emile-mini](https://github.com/Baglecake/emile-mini) QSE
cognitive engine.

## Requirements

- Python >= 3.10
- numpy >= 1.23
- scipy >= 1.10
- pygame-ce >= 2.4
- requests >= 2.28
- [Ollama](https://ollama.com) (local LLM server) -- optional but recommended

## Installation

```bash
git clone https://github.com/Baglecake/emile-kosmos.git
cd emile-kosmos
pip install -e .
```

### Ollama Setup (Optional)

The agent uses a local LLM via Ollama for reasoning and narration. Without
Ollama, the agent falls back to heuristic decision-making.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one)
ollama pull phi3:mini      # Fastest, good for tool calls
ollama pull llama3.1:8b    # Better quality reasoning
```

## Running

```bash
python -m kosmos
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `llama3.1:8b` | Ollama model name |
| `--size SIZE` | `30` | World grid dimension |
| `--speed SPEED` | `8` | Simulation ticks per second |
| `--seed SEED` | random | World generation seed |

Example:

```bash
python -m kosmos --model phi3:mini --size 40 --speed 12
```

## Controls

| Key | Action |
|-----|--------|
| SPACE | Pause / Resume |
| UP / DOWN | Increase / Decrease speed |
| Q / ESC | Quit |

## Architecture Overview

```
emile_mini/           Vendored QSE cognitive engine (v0.6.0)
  qse_core.py         Schrodinger evolution + surplus dynamics
  agent.py            EmileAgent: cognitive orchestration loop
  symbolic.py         Psi/Phi/Sigma field computation
  goal_v2.py          TD(lambda) strategy selection
  context.py          Hysteresis-based context switching
  memory.py           Working + episodic + semantic memory
  config.py           52 tunable QSE parameters

kosmos/               The Kosmos application
  __main__.py          Entry point, CLI args
  world/
    grid.py            Procedural biome grid, day/night, seasons
    objects.py         Food, Water, Hazard, CraftItem + recipes
  agent/
    core.py            KosmosAgent: QSE + tools + LLM + survival
  tools/
    registry.py        Extensible tool system with JSON schemas
    builtins.py        8 built-in tools (move, examine, consume, ...)
  llm/
    ollama.py          QSE-modulated LLM with structured tool calls
  render/
    pygame_render.py   Pygame visualization with info panels
```

### Cognitive Loop (each tick)

1. QSE wavefunction evolves, producing entropy and context dynamics
2. TD(lambda) selects a cognitive strategy (explore/exploit/rest/learn/social)
3. Strategy modulates which tools the LLM sees and its personality
4. LLM reasons about available tools and returns a structured action (or heuristic fallback)
5. Tool executes in the world (move, eat, craft, examine, etc.)
6. Reward feeds back to both TD(lambda) and QSE engine

### The World

- **Biomes**: plains, forest, desert, water, rock -- each with different movement costs
- **Day/night cycle**: 200 ticks per day, night increases movement cost
- **Seasons**: spring/summer/autumn/winter (800 ticks each) affect food and hazard spawn rates
- **Food migration**: resources decay over time and respawn in new locations
- **Crafting**: combine materials (stick + stone = axe, etc.) for survival bonuses
- **Real death**: energy depletion kills the agent; respawn with memories but lose inventory

## Development Roadmap

- **Phase 1**: Wire learning layers (GoalMapper L2, ActionPolicy L3 with
  LLM-as-teacher decay)
- **Phase 2**: Richer world (weather events, more objects, world save/load)
- **Phase 3**: Multi-agent (shared world, social learning, resource competition)
- **Phase 4**: Better LLM integration (multi-turn reasoning, async calls,
  teacher-student handoff metrics)
- **Phase 5**: UI/UX (mouse interaction, minimap, dashboards, live parameter
  tuning)

## License

MIT
