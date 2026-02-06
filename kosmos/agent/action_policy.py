"""Learned action policy adapted for Kosmos's tool-based world.

Adapts emile_mini/action_policy.py MLP+REINFORCE architecture with
Kosmos-specific input encoding (grid state) and output actions (tools).

QSE entropy modulates softmax temperature:
  high entropy -> high temperature -> more exploration
  low entropy  -> low temperature  -> more exploitation
"""

import numpy as np

from ..world.objects import Food, Water, Hazard, CraftItem, CRAFT_RECIPES


KOSMOS_ACTIONS = [
    'move_north', 'move_south', 'move_east', 'move_west',
    'examine', 'pickup', 'consume', 'craft', 'rest',
    'remember', 'wait',
]

STRATEGIES = ['explore', 'exploit', 'rest', 'learn', 'social']

EMBODIED_GOALS = [
    'explore_space', 'seek_food', 'find_shelter', 'rest_recover',
    'manipulate_objects', 'social_interact', 'categorize_experience',
]

BIOMES = ['plains', 'forest', 'desert', 'water', 'rock']
TIMES_OF_DAY = ['dawn', 'day', 'dusk', 'night']

KOSMOS_INPUT_DIM = 35  # 34 + sigma_ema (curvature/tension)
KOSMOS_N_ACTIONS = len(KOSMOS_ACTIONS)  # 11


def encode_kosmos_state(
    energy: float,
    hydration: float,
    biome: str,
    time_of_day: str,
    nearby_food: int,
    nearby_water: int,
    nearby_hazard: int,
    nearby_craft: int,
    has_food_here: bool,
    has_water_here: bool,
    has_craft_here: bool,
    has_hazard_here: bool,
    inventory_count: int,
    can_craft: bool,
    strategy: str,
    goal: str,
    entropy: float,
    surplus_mean: float,
    food_dx: float = 0.0,
    food_dy: float = 0.0,
    hazard_dx: float = 0.0,
    hazard_dy: float = 0.0,
    sigma_ema: float = 0.0,
) -> np.ndarray:
    """Encode Kosmos agent state into a fixed-size feature vector (35-dim).

    The directional cues (food_dx/dy, hazard_dx/dy) are normalized relative
    offsets to the nearest food/hazard, ranging from -1 to 1, where:
      - Positive dx means target is to the east
      - Positive dy means target is to the south
      - (0, 0) means no target found within search radius
    """
    state = np.zeros(KOSMOS_INPUT_DIM)

    # Vitals [0-1]
    state[0] = energy
    state[1] = hydration

    # Biome one-hot [2-6]
    if biome in BIOMES:
        state[2 + BIOMES.index(biome)] = 1.0

    # Time of day one-hot [7-10]
    if time_of_day in TIMES_OF_DAY:
        state[7 + TIMES_OF_DAY.index(time_of_day)] = 1.0

    # Nearby object counts (scaled) [11-14]
    state[11] = min(nearby_food, 5) / 5.0
    state[12] = min(nearby_water, 5) / 5.0
    state[13] = min(nearby_hazard, 5) / 5.0
    state[14] = min(nearby_craft, 5) / 5.0

    # Objects at current position [15-18]
    state[15] = float(has_food_here)
    state[16] = float(has_water_here)
    state[17] = float(has_craft_here)
    state[18] = float(has_hazard_here)

    # Inventory [19-20]
    state[19] = min(inventory_count, 10) / 10.0
    state[20] = float(can_craft)

    # Strategy one-hot [21-25]
    if strategy in STRATEGIES:
        state[21 + STRATEGIES.index(strategy)] = 1.0

    # Goal index (scaled) [26]
    if goal in EMBODIED_GOALS:
        state[26] = EMBODIED_GOALS.index(goal) / len(EMBODIED_GOALS)

    # QSE metrics [27-28]
    state[27] = float(np.clip(entropy, 0.0, 1.0))
    state[28] = float(np.clip(surplus_mean, -1.0, 1.0))

    # Directional cues [29-32] - critical for learning to move toward food
    state[29] = float(np.clip(food_dx, -1.0, 1.0))
    state[30] = float(np.clip(food_dy, -1.0, 1.0))
    state[31] = float(np.clip(hazard_dx, -1.0, 1.0))
    state[32] = float(np.clip(hazard_dy, -1.0, 1.0))

    # Survival urgency signal [33] - strong signal for "should eat NOW"
    # Combines low energy with food availability
    should_eat_urgency = 0.0
    if has_food_here and energy < 0.5:
        should_eat_urgency = (0.5 - energy) * 2.0  # 0 at 0.5 energy, 1.0 at 0 energy
    state[33] = should_eat_urgency

    # Curvature/tension signal [34] - indicates structured failure pattern
    # High sigma_ema = agent is in a "death trap" situation
    state[34] = float(np.clip(sigma_ema, 0.0, 1.0))

    return state


class KosmosActionPolicy:
    """Small MLP policy (numpy-only) for Kosmos tool actions.

    Architecture: input_dim -> hidden (tanh) -> n_actions (softmax)
    Training:     REINFORCE with exponential-moving-average baseline
    """

    def __init__(self, input_dim=KOSMOS_INPUT_DIM, hidden_dim=32, lr=0.001,
                 gamma=0.99, baseline_decay=0.95, temperature_base=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = KOSMOS_N_ACTIONS
        self.lr = lr
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self.temperature_base = temperature_base

        # Xavier-like init
        s1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        s2 = np.sqrt(2.0 / (hidden_dim + self.n_actions))
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, self.n_actions) * s2
        self.b2 = np.zeros(self.n_actions)

        # Training state
        self._trajectory = []
        self._baseline = 0.0
        self._total_updates = 0

    def forward(self, state, temperature=1.0):
        """Forward pass through the MLP."""
        z1 = state @ self.W1 + self.b1
        h = np.tanh(z1)
        logits = h @ self.W2 + self.b2
        scaled = logits / max(temperature, 0.01)
        exp_l = np.exp(scaled - np.max(scaled))
        probs = exp_l / (exp_l.sum() + 1e-10)
        return probs, h, z1

    def select_action(self, state_dict: dict, entropy: float = 0.5):
        """Select an action from the policy.

        Args:
            state_dict: keyword args for encode_kosmos_state()
            entropy: QSE entropy for temperature modulation

        Returns:
            (action_name, action_idx, probs)
        """
        state = encode_kosmos_state(**state_dict)
        temperature = self.temperature_base * (0.5 + entropy)
        probs, h, z1 = self.forward(state, temperature)

        action_idx = int(np.random.choice(self.n_actions, p=probs))
        self._trajectory.append({
            'state': state.copy(),
            'action_idx': action_idx,
            'probs': probs.copy(),
            'hidden': h.copy(),
            'z1': z1.copy(),
            'temperature': temperature,
        })
        return KOSMOS_ACTIONS[action_idx], action_idx, probs

    def record_reward(self, reward: float):
        """Attach reward to the most recent trajectory step."""
        if self._trajectory:
            self._trajectory[-1]['reward'] = float(reward)

    def update(self):
        """REINFORCE policy-gradient update over collected trajectory."""
        traj = [t for t in self._trajectory if 'reward' in t]
        if len(traj) < 2:
            self._trajectory.clear()
            return {}

        rewards = np.array([t['reward'] for t in traj])

        # Discounted returns
        returns = np.zeros_like(rewards)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Baseline update
        mean_ret = float(returns.mean())
        self._baseline = (self.baseline_decay * self._baseline
                          + (1 - self.baseline_decay) * mean_ret)

        # Advantages (standardised)
        adv = returns - self._baseline
        std = adv.std()
        if std > 1e-8:
            adv = adv / std

        # Accumulate gradients
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for i, step in enumerate(traj):
            s = step['state']
            a = step['action_idx']
            probs = step['probs']
            h = step['hidden']
            temp = step['temperature']
            A = adv[i]

            # d log pi / d logits = (one_hot - probs) / temperature
            one_hot = np.zeros(self.n_actions)
            one_hot[a] = 1.0
            dl = (one_hot - probs) / temp

            dW2 += np.outer(h, dl) * A
            db2 += dl * A

            dh = dl @ self.W2.T
            dz = dh * (1.0 - h ** 2)
            dW1 += np.outer(s, dz) * A
            db1 += dz * A

        n = len(traj)
        dW1 /= n; db1 /= n; dW2 /= n; db2 /= n

        # Gradient clipping
        total_norm = np.sqrt(
            np.sum(dW1 ** 2) + np.sum(db1 ** 2)
            + np.sum(dW2 ** 2) + np.sum(db2 ** 2))
        if total_norm > 1.0:
            scale = 1.0 / total_norm
            dW1 *= scale; db1 *= scale; dW2 *= scale; db2 *= scale

        # Gradient ascent
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2

        self._total_updates += 1
        stats = {
            'mean_return': mean_ret,
            'baseline': float(self._baseline),
            'grad_norm': float(total_norm),
            'trajectory_length': n,
            'total_updates': self._total_updates,
        }
        self._trajectory.clear()
        return stats

    def save(self, filepath):
        """Save policy weights to .npz file."""
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 baseline=np.array([self._baseline]),
                 total_updates=np.array([self._total_updates]))

    def load(self, filepath):
        """Load policy weights from .npz file."""
        data = np.load(filepath)
        self.W1 = data['W1']; self.b1 = data['b1']
        self.W2 = data['W2']; self.b2 = data['b2']
        self._baseline = float(data['baseline'][0])
        self._total_updates = int(data['total_updates'][0])


def decision_to_action_name(decision: dict) -> str:
    """Extract a granular action name from a decision dict for anti-oscillation.

    For moves, returns "move_north", "move_south", etc. instead of just "move".
    This allows anti-oscillation to penalize N↔S oscillation differently from
    exploring new directions.
    """
    tool = decision.get("tool", "wait")
    args = decision.get("args", {})

    if tool == "move":
        direction = args.get("direction", "north")
        return f"move_{direction}"

    return tool


def action_to_tool_call(action_name: str, agent) -> dict:
    """Convert a KosmosActionPolicy action name to a tool call dict.

    Resolves contextual args (which food/item) greedily from world state.
    """
    if action_name.startswith('move_'):
        direction = action_name.split('_', 1)[1]
        return {"tool": "move", "args": {"direction": direction},
                "thought": f"Policy: move {direction}"}

    if action_name == 'examine':
        return {"tool": "examine", "args": {"target": "surroundings"},
                "thought": "Policy: examine"}

    if action_name == 'pickup':
        for obj in agent.world.objects_at(agent.pos):
            if isinstance(obj, CraftItem):
                return {"tool": "pickup", "args": {"item": obj.name},
                        "thought": f"Policy: pickup {obj.name}"}
        return {"tool": "wait", "args": {}, "thought": "Policy: nothing to pickup"}

    if action_name == 'consume':
        for obj in agent.world.objects_at(agent.pos):
            if isinstance(obj, (Food, Water)):
                return {"tool": "consume", "args": {"item": obj.name},
                        "thought": f"Policy: consume {obj.name}"}
        return {"tool": "wait", "args": {}, "thought": "Policy: nothing to consume"}

    if action_name == 'craft':
        for i, a in enumerate(agent.inventory):
            for b in agent.inventory[i + 1:]:
                key = tuple(sorted([a.craft_tag, b.craft_tag]))
                if key in CRAFT_RECIPES:
                    return {"tool": "craft",
                            "args": {"item1": a.name, "item2": b.name},
                            "thought": f"Policy: craft {a.name}+{b.name}"}
        return {"tool": "wait", "args": {}, "thought": "Policy: can't craft"}

    if action_name == 'rest':
        return {"tool": "rest", "args": {}, "thought": "Policy: rest"}

    if action_name == 'remember':
        return {"tool": "remember", "args": {"query": "danger food"},
                "thought": "Policy: remember"}

    # wait (default)
    return {"tool": "wait", "args": {}, "thought": "Policy: wait"}
