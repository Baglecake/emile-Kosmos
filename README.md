# emile-Kosmos

An autonomous agent simulation combining biologically-inspired cognitive dynamics with large language model reasoning. The agent survives in a procedurally generated world, making decisions through a hybrid architecture that integrates continuous neural field dynamics with structured tool use.

## Overview

emile-Kosmos addresses a fundamental limitation of LLM-based agents: static decision-making. Rather than using fixed prompts or simple rule-based systems, this project implements a **cognitive dynamics layer** that modulates the LLM's behavior in real time based on the agent's internal state and environmental context.

The core innovation is treating cognition as a continuous dynamical system rather than discrete query-response cycles. The agent's "mental state" evolves according to field equations, and this evolution determines both *what* the agent considers and *how* it reasons.

## Cognitive Architecture

The system implements a three-layer decision hierarchy:

### Layer 1: Strategy Selection
A temporal-difference learning module (TD-lambda) selects abstract cognitive strategies: explore, exploit, rest, learn, or social. Strategy selection adapts based on environmental feedback and internal state.

### Layer 2: Goal Mapping
Strategies are translated into concrete embodied goals (seek food, find water, avoid hazards, etc.) through a learned mapping that considers the agent's current resources and nearby affordances.

### Layer 3: Action Policy
Goals are executed through either:
- A learned neural policy trained via behavior cloning from successful demonstrations
- Heuristic reflexes for survival-critical situations
- LLM-guided planning for novel or complex situations

### Cognitive Dynamics Engine

The underlying cognitive engine (vendored from [emile-mini](https://github.com/Baglecake/emile-mini)) models internal state as a probability distribution evolving via Schrodinger-like dynamics:

- **Entropy** of the distribution modulates decision temperature: high entropy leads to exploratory behavior; low entropy leads to focused exploitation
- **Surplus** measures discrepancy between expected and observed states, driving learning and adaptation
- **Curvature** tracks structural patterns in failure, triggering cognitive "ruptures" that force exploration of new strategies

This provides principled answers to questions like "when should the agent explore vs exploit?" and "when should it abandon a failing strategy?"

## Key Features

- **Survival mechanics**: Energy, hydration, day/night cycles, weather events, seasonal variation
- **Tool-based actions**: Move, examine, consume, craft, plant, rest, remember
- **Multi-step planning**: LLM generates action sequences with explicit goals and replan conditions
- **Context-aware plan validation**: Plans are accepted only if the current situation matches the context when they were requested
- **Teacher-student learning**: LLM decisions are used to train the neural policy, with gradual handoff as competence increases
- **Structured logging**: Detailed event logs and metrics for analysis and debugging

## Requirements

- Python 3.10 or later
- numpy, scipy, pygame-ce, requests
- Ollama (optional, for LLM reasoning)

## Installation

```bash
git clone https://github.com/Baglecake/emile-kosmos.git
cd emile-kosmos
pip install -e .
```

### Ollama Setup (Optional)

The agent uses a local LLM via Ollama for reasoning. Without Ollama, the agent operates using heuristic decision-making only.

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

## Usage

```bash
python -m kosmos
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model MODEL` | `llama3.1:8b` | Ollama model name |
| `--size SIZE` | `30` | World grid dimension |
| `--speed SPEED` | `8` | Simulation ticks per second |
| `--seed SEED` | random | World generation seed |
| `--save PATH` | none | Save state on exit |
| `--load PATH` | none | Load state on start |

### Controls

| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| Up/Down | Adjust speed |
| Q/Escape | Quit |

## Project Structure

```
emile_mini/                 Vendored cognitive engine
  qse_core.py               Field dynamics and surplus computation
  goal_v2.py                TD-lambda strategy selection
  goal_mapper.py            Strategy to goal mapping
  symbolic.py               Symbolic field computation

kosmos/                     Application code
  agent/
    core.py                 Main agent loop and decision logic
    action_policy.py        Neural policy (MLP + REINFORCE)
    surplus_tension.py      Surplus/curvature metrics
    demo_buffer.py          Behavior cloning buffer
  world/
    grid.py                 Procedural world generation
    objects.py              Game objects and crafting
    weather.py              Weather system
  llm/
    ollama.py               LLM interface with structured output
  tools/
    registry.py             Tool registration system
    builtins.py             Built-in tool definitions
  render/
    pygame_render.py        Visualization
  logging_config.py         Structured logging
  persistence.py            Save/load functionality
```

## Development Status

### Completed

- Three-layer cognitive architecture (strategy, goal, action)
- Survival mechanics with tuned metabolism
- Weather system (rain, storm, heat wave, fog, wind)
- Crafting and agriculture (herbs, seeds, planted crops)
- Multi-step LLM planning with interrupt handling
- Event-triggered LLM firing based on context changes
- Surplus-tension module for principled exploration/exploitation
- Cognitive integrity metrics (collaboration vs. compromise)
- Context-aware plan validation (state-validity checking)
- Structured logging and metrics output

### In Progress

- Intent system: shifting LLM from micro-planning to goal/quota specification
- Multi-agent scenarios with social dynamics

## Output Files

Each run generates:
- `runs/latest.log`: Human-readable event log
- `runs/latest_metrics.jsonl`: Structured metrics (one JSON object per 10 ticks)

## Documentation

- `archive/docs/ARCHITECTURE.md`: Detailed system architecture
- `archive/docs/ROADMAP_INTENT.md`: Intent system design
- `archive/docs/ROADMAP_SURPLUS_TENSION.md`: Cognitive dynamics specification

## Theoretical Foundations

This project draws on several intersecting research traditions. See `source_notes.md` for detailed summaries.

### Cognition as Situated and Extended

The architecture treats cognition not as isolated computation but as a coupled system spanning agent and environment. This follows Clark and Chalmers' (1998) "extended mind" thesis and Heyes' (2018) argument that distinctively human cognitive mechanisms are "cultural gadgets" constructed through social interaction rather than genetic inheritance.

### Sociogenesis and Shared Intentionality

Tomasello's (1999) account of human cultural origins informs the teacher-student dynamics: the capacity for joint attention and imitative learning enables the "ratchet effect" where cognitive modifications accumulate across interactions. The project explores whether similar dynamics can emerge in human-AI collaboration.

### Constitutive Practice and Sequential Organization

Rawls' (2011) reading of Durkheim and Wittgenstein emphasizes that meaning is constituted through sequential interaction, not retrieved from static representations. The agent's multi-step planning and context-sensitive plan validation reflect this view: plans are accepted or rejected based on whether the current interactional context matches the context of their production.

### Situated Intelligence

Following Ratto's (2025) critique of "artificial general intelligence" as a sociotechnical imaginary, this project pursues what might be called artificial contextual intelligence: systems designed to operate within bounded milieus rather than abstracting away from context. Competence is evaluated relationally, by how well the system coordinates activity within its specific environment.

### Symbolic Behavior as Convention

Santoro et al. (2022) argue that symbolic fluency requires not just pattern recognition but the capacity to construct and modify conventions. The intent system roadmap aims toward this: the LLM specifies goals and constraints (conventions), while the embodied agent learns to satisfy them through grounded interaction.

### Key References

- Clark, A. & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1), 7-19.
- Heyes, C. (2018). *Cognitive gadgets: The cultural evolution of thinking*. Harvard University Press.
- Rawls, A. (2011). Wittgenstein, Durkheim, Garfinkel and Winch. *Journal for the Theory of Social Behaviour*, 41(4), 396-418.
- Ratto, M. (2025). Is intelligence general?
- Santoro, A. et al. (2022). Symbolic behaviour in artificial intelligence. arXiv:2102.03406.
- Shanahan, M. et al. (2023). Role-play with large language models. arXiv:2305.16367.
- Tomasello, M. (1999). *The cultural origins of human cognition*. Harvard University Press.

## License

MIT

## Acknowledgments

Built on the [emile-mini](https://github.com/Baglecake/emile-mini) cognitive engine.
