"""
Lightweight learned action policy for the embodied QSE-Émile agent.

Replaces/augments the heuristic _goal_to_embodied_action with a trainable
numpy MLP using REINFORCE with baseline.

QSE entropy modulates the softmax temperature:
  - High entropy (unstable internal state) → high temperature → more exploration
  - Low entropy (stable internal state) → low temperature → more exploitation
"""

import numpy as np


ACTIONS = [
    'move_forward', 'move_backward', 'turn_left', 'turn_right',
    'rest', 'examine', 'forage',
]

# Strategy names matching GoalModuleV2
STRATEGIES = ['explore', 'exploit', 'rest', 'learn', 'social']

INPUT_DIM = 30   # 16 visual + 6 body + 3 QSE + 5 goal one-hot
N_ACTIONS = len(ACTIONS)


def encode_state(visual_field, body_state, qse_metrics, goal_idx, input_dim=INPUT_DIM):
    """Encode observation into fixed-size feature vector."""
    # Visual: summary statistics (16 features)
    if visual_field is not None and visual_field.size > 0:
        v = visual_field
        h, w = v.shape[0], v.shape[1]
        hh, hw = h // 2, w // 2
        visual_features = np.array([
            v.mean(), v.std(),
            v[:, :, 0].mean(), v[:, :, 1].mean(), v[:, :, 2].mean(),
            v.max(), v.min(),
            v[hh, hw, :].mean(),
            v[:hh, :hw].mean(), v[:hh, hw:].mean(),
            v[hh:, :hw].mean(), v[hh:, hw:].mean(),
            np.abs(np.diff(v[:, :, 0], axis=0)).mean(),
            np.abs(np.diff(v[:, :, 0], axis=1)).mean(),
            np.abs(np.diff(v[:, :, 1], axis=0)).mean(),
            np.abs(np.diff(v[:, :, 1], axis=1)).mean(),
        ])
    else:
        visual_features = np.zeros(16)

    # Body state (6 features)
    vel = body_state.velocity if hasattr(body_state, 'velocity') else (0.0, 0.0)
    body_features = np.array([
        float(body_state.energy),
        float(body_state.health),
        np.sin(float(body_state.orientation)),
        np.cos(float(body_state.orientation)),
        float(vel[0]) if hasattr(vel, '__getitem__') else 0.0,
        float(vel[1]) if hasattr(vel, '__getitem__') else 0.0,
    ])

    # QSE metrics (3 features)
    qse_features = np.array([
        float(qse_metrics.get('surplus_mean', 0.0)),
        float(qse_metrics.get('sigma_mean', 0.0)),
        float(qse_metrics.get('normalized_entropy', 0.5)),
    ])

    # Goal encoding: one-hot (5 features)
    goal_onehot = np.zeros(len(STRATEGIES))
    if 0 <= goal_idx < len(STRATEGIES):
        goal_onehot[goal_idx] = 1.0

    state = np.concatenate([visual_features, body_features, qse_features, goal_onehot])

    # Pad or truncate to input_dim
    if len(state) < input_dim:
        state = np.pad(state, (0, input_dim - len(state)))
    elif len(state) > input_dim:
        state = state[:input_dim]

    return state


class ActionPolicy:
    """
    Small MLP policy network (numpy-only).

    Architecture: input_dim → hidden (tanh) → n_actions (softmax)
    Training:     REINFORCE with exponential-moving-average baseline
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=32, lr=0.001,
                 gamma=0.99, baseline_decay=0.95, temperature_base=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = N_ACTIONS
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

    # ------------------------------------------------------------------ #
    #  Forward pass                                                       #
    # ------------------------------------------------------------------ #
    def forward(self, state, temperature=1.0):
        z1 = state @ self.W1 + self.b1
        h = np.tanh(z1)
        logits = h @ self.W2 + self.b2
        scaled = logits / max(temperature, 0.01)
        exp_l = np.exp(scaled - np.max(scaled))
        probs = exp_l / (exp_l.sum() + 1e-10)
        return probs, h, z1

    # ------------------------------------------------------------------ #
    #  Action selection                                                   #
    # ------------------------------------------------------------------ #
    def select_action(self, visual_field, body_state, qse_metrics, goal_idx,
                      entropy=0.5):
        """Return (action_name, intensity, action_idx)."""
        state = encode_state(visual_field, body_state, qse_metrics, goal_idx,
                             self.input_dim)
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
        intensity = float(max(0.3, probs[action_idx]))
        return ACTIONS[action_idx], intensity, action_idx

    def record_reward(self, reward):
        """Attach reward to the most recent trajectory step."""
        if self._trajectory:
            self._trajectory[-1]['reward'] = float(reward)

    # ------------------------------------------------------------------ #
    #  REINFORCE update                                                   #
    # ------------------------------------------------------------------ #
    def update(self):
        """Policy-gradient update over the collected trajectory."""
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
            z1 = step['z1']
            temp = step['temperature']
            A = adv[i]

            # d log π / d logits = (one_hot - probs) / temperature
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

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, filepath):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 baseline=np.array([self._baseline]),
                 total_updates=np.array([self._total_updates]))

    def load(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']; self.b1 = data['b1']
        self.W2 = data['W2']; self.b2 = data['b2']
        self._baseline = float(data['baseline'][0])
        self._total_updates = int(data['total_updates'][0])
