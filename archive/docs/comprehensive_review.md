Comprehensive Review: emile-Kosmos + emile-cogito Ecosystem
I've thoroughly reviewed all three codebases. Here's what I found:

The Three Systems at a Glance
emile-mini (vendored)	emile-Kosmos	emile-cogito
Purpose	Core QSE engine	Living world simulation	Full cognitive architecture
Scale	~2,500 lines	~3,000 lines	~15,000+ lines
QSE	Numpy FFT Schrodinger	Uses emile-mini's engine	QuTiP-based (qse_core_qutip)
Learning	3-layer RL hierarchy	3-layer + LLM teacher	4 neural FPMs + recursive feedback
LLM	None	Ollama (local, structured JSON)	Gemini API (consciousness expression)
World	20x20 simple grid	30x30 procedural biomes	Abstract info layers (no physical world)
How emile-Kosmos Currently Uses Tools & Guides Reasoning
The reasoning pipeline flows through 4 decision points:

1. QSE wavefunction (qse_core.py) runs at 20Hz, producing:

Shannon entropy of |psi|^2 -> normalized to [0,1]
Surplus mean/variance
Sigma (symbolic curvature) for context switching
2. Strategy selection (goal_v2.py) - GoalModuleV2 uses TD(lambda) to pick one of 5 strategies: explore/exploit/rest/learn/social. The QSE entropy modulates epsilon-greedy exploration.

3. Goal mapping (action_policy.py:encode_kosmos_state) - GoalMapper (now wired) maps strategy to one of 7 embodied goals (seek_food, explore_space, etc.) via TD(lambda) with warm-start.

4. Action selection — this is where the teacher-student split happens (core.py:tick):

Teacher path (probability = _teacher_prob, starts at 1.0): LLM gets a strategy-filtered tool set + personality prompt + situation text. It returns {"tool": "move", "args": {"direction": "north"}, "thought": "..."}. When LLM is busy/unavailable, the heuristic fallback fires.
Student path (probability = 1 - _teacher_prob): The MLP policy (30-dim input -> 32 hidden -> 11 actions) selects an action via softmax with entropy-modulated temperature.
Tool filtering by strategy is subtle but important — in core.py:_strategy_tool_categories, each strategy restricts which tool categories the LLM sees:

explore -> action + perception + meta
exploit -> action only
rest -> action only (rest/wait)
learn -> perception + meta + action
social -> social + perception + action
The LLM prompt (ollama.py:reason) is conditioned by:

Temperature = 0.3 + entropy * 1.2 (range 0.3-1.5)
Personality = strategy-specific system prompt
Multi-turn history = sliding window of 4 exchanges
Last action result = fed back as context
Structured JSON output = format: "json" enforced
Key Differences: emile-cogito vs emile-Kosmos
emile-cogito is architecturally much more ambitious. Here are the patterns that stand out as potentially relevant to Kosmos:

1. Dynamic Parameter Calculation (no hardcoded values)

In emile-cogito, every module queries the platform's get_current_distinction_level() method to dynamically calculate its own parameters. Learning rates, decay rates, thresholds — everything scales with the current consciousness state. Kosmos uses fixed parameters throughout.

2. Metabolic Pressure as Cognitive Driver

emile-cogito's metabolic.py (SurplusDistinctionConsciousness) creates existential pressure — the system must actively maintain its complexity against constant entropic decay. In Kosmos, basal metabolism is a fixed drain (0.003/tick). In cogito, it's a consciousness-responsive regulatory system.

3. Qualia Layer

The qualia.py (QualiaLayer) generates a composite qualitative state vector (valence, arousal, clarity, coherence, attention focus) from system metrics. This feeds into FPM-2's weighting mechanisms. Kosmos has no explicit qualia representation — the closest thing is entropy driving LLM temperature.

4. Consciousness Ecology

consciousness_ecology.py gates access to more complex input streams based on output quality. Higher-quality reasoning unlocks richer inputs. In Kosmos, the world is always fully visible — there's no gating based on cognitive sophistication.

5. LLM as Expression, Not Decision-Maker

The biggest philosophical difference: In Kosmos, the LLM is a decision-maker (teacher) that selects tools. In emile-cogito, the LLM is an expression layer — the ConsciousnessToLinguisticMapper converts internal QSE state into prompts asking Gemini to narrate the system's experience, not to choose actions. The system decides for itself through the 4 FPMs.

6. Multi-Perspective Clock

Each FPM in cogito has its own local clock (tau'). The orchestrator uses these to determine processing priority — which cognitive function is the current bottleneck. Kosmos has a single emergent time tau from the QSE engine.

7. Shared Symbolic Workspace (from cogito proto notes)

The early cogito experiments showed a shared Sigma workspace where multiple agents modulate each other's Phi (resistance). This creates "collective symbolic weather" — emergent moods that influence all agents. This concept maps directly to Phase 3 (multi-agent) in Kosmos's roadmap.

How Kosmos's Tool System Works in Practice
The tool pipeline is straightforward:

builtins.py defines 9 tools as Tool dataclasses (name, description, parameters, category)
registry.py binds functions to tool names, exports JSON schemas for the LLM
The LLM sees filtered schemas based on strategy and returns structured JSON
core.py invokes self.tools.invoke(tool_name, **args) which calls the bound _tool_X() method
The result string feeds back into reward computation and the next LLM prompt
The reward signal shapes all three learning layers:

Energy delta (did energy go up or down?)
Exploration bonus (visited new cells)
Consumption bonus (ate food, drank water)
Crafting bonus
Hazard penalty
Death penalty
Current Gaps / Opportunities
Based on reviewing everything:

The learned MLP takes over too fast — teacher_prob decays from 1.0 to ~0.17 by tick 900 but the MLP hasn't learned survival instincts. The heuristic has hardcoded "eat when energy < 0.3" that the MLP lacks. (This was the issue we discussed before the compact.)

No qualia / consciousness state — emile-cogito shows how a rich internal state vector (valence, arousal, clarity, etc.) can modulate behavior beyond just entropy. Kosmos could benefit from even a simplified version.

No dynamic parameters — all thresholds in Kosmos are hardcoded. emile-cogito's pattern of querying distinction_level for dynamic parameter calculation could make the agent more adaptive.

LLM guides action, not expression — this is a design choice, but emile-cogito's pattern of using the LLM to express/narrate rather than decide is interesting. The QSE engine + learned policies make the actual decisions.

No information gating — the agent always sees everything within examine radius. Cogito's consciousness ecology pattern (better performance = richer inputs) could add depth.

Where would you like to go from here?