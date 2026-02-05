# emile-Kosmos Next Steps Implementation Plan

**Created:** 2026-02-05
**Last Commit:** 7976097 - Complete Phase 5 cogito enhancements + survival tuning
**Status:** Ready for Phase 5f (anti-oscillation) and Phase 5g (stuckness detection)

---

## Current State Summary

The agent is now stable with all Phase 5 features complete:
- Survives 8000+ ticks without intervention
- Learned policy reaches 63% of decisions
- Teacher-student handoff working (l_ema positive ~0.10)
- Multi-step LLM planning operational

**Remaining issues to address:**
- Occasional camping/oscillation behavior (agent revisits same positions)
- "Social" strategy exists but has no implementation (vestigial)

---

## Immediate Tasks (Phase 5f-g)

### Phase 5f: Anti-Oscillation Mechanism

**Source:** `complete_navigation_system_e.py` from original emile-mini repo

**What it does:** Tracks recent actions and applies decay penalties for repetition, encouraging behavioral diversity.

**Implementation in `kosmos/agent/core.py`:**

1. **Add state tracking in `__init__`** (after line ~160):
```python
# 5f: Anti-oscillation (from navigation system)
self._recent_actions: list[str] = []  # last N action names
self._action_repeat_window = 10  # how many actions to track
```

2. **Add helper method** (after `_direction_toward`):
```python
def _get_action_penalty(self, action_name: str) -> float:
    """
    Calculate penalty for repeating an action.

    From complete_navigation_system_e.py: decay penalties for repeated actions
    to encourage behavioral diversity and prevent oscillation.
    """
    if not self._recent_actions:
        return 0.0
    # Count how many times this action appears in recent history
    recent_count = self._recent_actions.count(action_name)
    # Exponential decay penalty: 0.05 * count^1.5
    penalty = 0.05 * (recent_count ** 1.5)
    return min(penalty, 0.5)  # Cap at 0.5 to avoid over-penalizing
```

3. **Track actions after execution** (in `tick()`, after line ~760):
```python
# 5f: Track action for anti-oscillation
self._recent_actions.append(tool_name)
if len(self._recent_actions) > self._action_repeat_window:
    self._recent_actions = self._recent_actions[-self._action_repeat_window:]
```

4. **Apply penalty to reward** (in `tick()`, modify reward calculation around line ~695):
```python
# Apply anti-oscillation penalty
action_penalty = self._get_action_penalty(tool_name)
reward = reward - action_penalty
```

5. **Add to `get_state()`** for debugging:
```python
"action_repeat_penalty": self._get_action_penalty(self.last_action.get("tool", "") if self.last_action else ""),
```

**Verification:**
- Run for 500+ ticks
- Check that repeated actions get penalized in reward
- Agent should show more action diversity over time

---

### Phase 5g: Stuckness Detection

**Source:** `maze_environment.py` from original emile-mini repo

**What it does:** Detects when agent is visiting <= 3 unique positions in recent history, indicating it's "stuck". Triggers exploration or context switch.

**Implementation in `kosmos/agent/core.py`:**

1. **Add state tracking in `__init__`** (after anti-oscillation state):
```python
# 5g: Stuckness detection (from maze_environment.py)
self._stuckness_threshold = 3  # <= this many unique positions = stuck
self._stuckness_window = 20  # positions to consider
self._is_stuck = False
self._stuck_ticks = 0  # how long we've been stuck
```

2. **Add detection method** (after `_get_action_penalty`):
```python
def _check_stuckness(self) -> bool:
    """
    Check if agent is stuck (visiting <= 3 unique positions).

    From maze_environment.py: stuckness detection triggers context-switch
    when agent revisits the same few positions repeatedly.
    """
    if len(self._recent_positions) < self._stuckness_window:
        return False

    recent = self._recent_positions[-self._stuckness_window:]
    unique_positions = len(set(recent))

    was_stuck = self._is_stuck
    self._is_stuck = unique_positions <= self._stuckness_threshold

    if self._is_stuck:
        self._stuck_ticks += 1
    else:
        self._stuck_ticks = 0

    # Log transition
    if self._is_stuck and not was_stuck:
        print(f"[STUCK t={self.total_ticks}] Agent stuck at {unique_positions} unique positions")
    elif not self._is_stuck and was_stuck:
        print(f"[UNSTUCK t={self.total_ticks}] Agent escaped stuckness")

    return self._is_stuck
```

3. **Call in tick() after position tracking** (around line ~740):
```python
# 5g: Check for stuckness
self._check_stuckness()
```

4. **Use stuckness to trigger exploration** (in `_should_fire_llm`, add new trigger):
```python
# 9. Stuckness triggers replanning
if self._is_stuck and self._stuck_ticks >= 5:
    reasons.append("stuck in area")
```

5. **Force exploration in heuristic when stuck** (in `_heuristic_decide`, after consume logic):
```python
# If stuck, force random exploration to break out
if self._is_stuck and self._stuck_ticks >= 10:
    import random
    dirs = ["north", "south", "east", "west"]
    # Prefer direction we haven't gone recently
    recent_dirs = [a.split("_")[1] for a in self._recent_actions if a.startswith("move_")]
    unexplored = [d for d in dirs if d not in recent_dirs[-4:]]
    direction = random.choice(unexplored) if unexplored else random.choice(dirs)
    return {"tool": "move", "args": {"direction": direction},
            "thought": "Breaking out of stuck area."}
```

6. **Add to `get_state()`**:
```python
"is_stuck": self._is_stuck,
"stuck_ticks": self._stuck_ticks,
```

**Verification:**
- Run for 1000+ ticks
- Check console for [STUCK] and [UNSTUCK] messages
- Agent should break out of camping patterns

---

## After 5f-g: Testing and Commit

1. Run `python3 -m kosmos --speed 16` for 2000+ ticks
2. Verify:
   - Agent shows action diversity (not repeating same action many times)
   - [STUCK] events are detected and resolved
   - Death rate remains low (< 3 per 1000 ticks)
   - Learned policy continues to improve
3. Commit: "Add anti-oscillation and stuckness detection from navigation system"

---

## Future Phases (After 5f-g)

### Phase 5h: Cognitive Battery Integration (Optional)

**Source:** `cognitive_battery.py` from original emile-mini

**Purpose:** Benchmarking framework to validate agent cognition with 3 protocols:
- Protocol A: Zero-training baseline
- Protocol C1: Context-switch adaptation
- Protocol C2: Memory-cued navigation

**Implementation:** Create `kosmos/benchmarks/cognitive_battery.py` that adapts the framework for Kosmos's tool-based world.

### Phase 3: Multi-Agent (Future)

**Source:** `social_qse_agent_v2.py` (already in repo root)

**Key components to integrate:**
1. `SocialSignal` class for agent communication
2. `SocialQSEAgent` with teaching/learning
3. `SocialEnvironment` for multi-agent coordination
4. Emergent personality traits

**Prerequisites:**
- Stable single-agent behavior (done)
- World support for multiple agents
- Rendering for multiple agents

---

## File Reference

| File | Purpose |
|------|---------|
| `kosmos/agent/core.py` | Main agent logic, modify for 5f-g |
| `kosmos/render/pygame_render.py` | Add stuck indicator to UI |
| `social_qse_agent_v2.py` | Reference for Phase 3 |
| `ROADMAP.md` | Overall development roadmap |
| `MEMORY.md` (in .claude) | Project memory for context |

---

## Quick Reference: Key Parameters

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `_action_repeat_window` | core.py | 10 | Actions tracked for anti-oscillation |
| `_stuckness_threshold` | core.py | 3 | Unique positions to be "stuck" |
| `_stuckness_window` | core.py | 20 | Positions considered for stuckness |
| `window_size` (novelty) | core.py | 30 | Positions for novelty calc |
| `_teacher_prob` decay | core.py | 0.9995 | Teacher-student decay rate |
| Basal metabolism | core.py | 0.002/0.0008 | Energy/hydration drain per tick |

---

## Resumption Instructions

If context is compacted, read this file first, then:
1. Read `kosmos/agent/core.py` to understand current state
2. Implement Phase 5f (anti-oscillation) following the code snippets above
3. Implement Phase 5g (stuckness detection) following the code snippets above
4. Test with `python3 -m kosmos --speed 16`
5. Commit changes
