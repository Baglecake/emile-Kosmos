Kosmos Modules
kosmos/main.py — Entry point that parses CLI args (--speed, --save, --load), initializes the world and agent, starts the pygame renderer, and runs the main tick loop.

kosmos/agent/core.py — The brain of the system. KosmosAgent orchestrates the tick loop: basal metabolism, death/respawn, QSE updates, L1→L2→L3 goal/action selection, teacher-student decision routing, LLM firing, tool execution, reward computation, and all learning updates.

kosmos/agent/action_policy.py — Layer 3 learned policy. A numpy-only MLP (35→hidden→11 actions) trained via REINFORCE with EMA baseline. Encodes agent state into features and maps goals to tool actions. The "student" that learns from the heuristic "teacher."

kosmos/agent/demo_buffer.py — Stores (state, action, reward) demonstrations from the teacher for offline behavior cloning. Enables the student policy to learn from accumulated teacher experience via cross-entropy training every N ticks.

kosmos/agent/surplus_tension.py — Phase 6 QSE metrics module. Computes surplus S (surprise = |expected - observed|), curvature Σ (structural tension), and τ′ (emergent time). Detects "death traps" via high Σ and provides intrinsic reward shaping.

kosmos/llm/ollama.py — LLM interface via Ollama API. Provides reason() for single actions and reason_plan() for multi-step plans. Implements Phase 7 embodied cognition: visceral prompts that describe how hunger/danger feels, and logit biasing to nudge token probabilities based on agent physiology.

kosmos/world/grid.py — The living world simulation. Generates biome maps, manages fixed ResourceNodes that spawn/respawn objects, handles day/night cycles, seasons, weather integration, and provides spatial queries (objects_at, objects_near, move_cost).

kosmos/world/objects.py — Defines all world object types: Food, Water, Hazard, CraftItem, Herb, Seed, PlantedCrop. Also defines Biome enum, biome movement costs, and CRAFT_RECIPES dictionary mapping item pairs to crafted tools.

kosmos/world/weather.py — Weather system with WeatherType enum (CLEAR, RAIN, STORM, HEAT_WAVE, FOG, WIND). WeatherManager handles event lifecycle, cooldowns, intensity curves, and wind direction. Weather affects movement costs, metabolism, and examine radius.

kosmos/tools/builtins.py — Defines the 9 available tools: move, examine, pickup, consume, craft, rest, remember, wait, plant. Each tool has a schema (name, description, parameters, category) used by the LLM and tool registry.

kosmos/tools/registry.py — ToolRegistry that maps tool names to handler functions. Provides schema retrieval filtered by category (navigation, survival, memory, action) and executes tool calls by dispatching to the appropriate agent method.

kosmos/render/pygame_render.py — Pygame-ce visualization. Renders the grid with biome colors, objects, agent position, weather tinting. Side panel shows stats: energy/hydration bars, strategy, goal, zone, plan, novelty, teacher prob, learning metrics, and surplus/tension gauges.

kosmos/persistence.py — Save/load system. Serializes world state (biomes, objects, nodes, tick_count, weather) and agent state (position, energy, inventory, memories, learning weights) to JSON. Handles object type dispatch for proper deserialization.

Emile-Mini Modules (Vendored QSE Engine)
emile_mini/qse_core.py — Core QSE (Quantum Surplus Emergence) calculations. Computes symbolic fields, arousal/valence from physiological state, updates surplus accumulator, and provides the foundational cognitive metrics that drive goal selection.

emile_mini/goal_v2.py — Layer 1 goal selection via GoalModuleV2. Uses TD(λ) learning to select abstract strategies (explore, exploit, rest, learn, social) based on QSE context, entropy, and energy state.

emile_mini/goal_mapper.py — Layer 2 goal mapping. GoalMapper translates abstract L1 strategies into embodied goals (explore_space, seek_food, find_shelter, rest_recover, etc.) via TD(λ) with warm-start initialization.

emile_mini/symbolic.py — SymbolicReasoner for higher-level pattern recognition. Tracks sigma_ema (curvature EMA), detects regime transitions, and provides symbolic interpretation of QSE dynamics.

emile_mini/memory.py — EpisodicMemory for storing and retrieving experiences. Supports similarity-based recall, memory consolidation, and provides relevant memories for LLM context.

emile_mini/config.py — QSEConfig dataclass holding all hyperparameters: learning rates, decay factors, thresholds, feature flags for enabling/disabling components like GoalMapper and learned policy.

emile_mini/context.py — QSEContext dataclass bundling the agent's current cognitive state: arousal, valence, entropy, surplus, strategy — passed between modules as a unified state snapshot.