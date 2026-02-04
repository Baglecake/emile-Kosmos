"""
Enhanced Goal Module with Temporal Difference Learning
Replaces simple bandit learning with proper RL
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .config import CONFIG


class GoalModuleV2:
    """
    Proper TD(λ) learning with state representation and eligibility traces.

    Key improvements over goal.py:
    1. State representation: (context, energy_level, recent_reward)
    2. Temporal difference learning with discount factor
    3. Eligibility traces for multi-step credit assignment
    4. Adaptive learning rate
    5. State-grounded goals
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

        # Define state and action spaces
        self.context_bins = 10  # Discretize context 0-9
        self.energy_bins = 5    # Discretize energy [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.reward_bins = 5    # Discretize recent reward [-1, -0.5, 0, 0.5, 1]

        # Available goals/actions
        self.goals = ['explore', 'exploit', 'rest', 'learn', 'social']
        self.n_actions = len(self.goals)

        # Q-table: Q(s, a) where s = (context, energy, reward_hist)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

        # Eligibility traces for TD(λ)
        self.eligibility = defaultdict(lambda: np.zeros(self.n_actions))

        # TD learning parameters
        self.alpha = 0.1          # Learning rate (will be adaptive)
        self.gamma = 0.8          # Discount factor (0.8 for continuous; raise for episodic)
        self.lambda_trace = 0.7   # Eligibility trace decay
        self.epsilon = 0.2        # Exploration rate (modulated by entropy)

        # State tracking
        self.current_state = None
        self.current_action = None
        self.recent_rewards = []
        self.recent_rewards_window = 10

        # Statistics
        self.total_updates = 0
        self.episode_rewards = []
        self.visit_counts = defaultdict(lambda: np.zeros(self.n_actions))

        # Backward-compat: mutable attribute read by embodied code
        self.current_goal = 'explore'

    def _discretize_state(self, context: int, energy: float, reward_history: List[float]) -> Tuple[int, int, int]:
        """
        Convert continuous state to discrete state tuple.

        Args:
            context: Current context ID (already discrete)
            energy: Current energy level [0, 1]
            reward_history: Recent rewards

        Returns:
            (context_bin, energy_bin, reward_bin)
        """
        # Context is already discrete, just clip to bins
        context_bin = min(context, self.context_bins - 1)

        # Discretize energy
        energy_bin = int(np.clip(energy * self.energy_bins, 0, self.energy_bins - 1))

        # Discretize average recent reward
        if len(reward_history) > 0:
            avg_reward = np.mean(reward_history[-self.recent_rewards_window:])
            # Map [-inf, inf] to [0, reward_bins-1]
            avg_reward = np.clip(avg_reward, -1.0, 1.0)  # Clip to reasonable range
            reward_bin = int((avg_reward + 1.0) / 2.0 * (self.reward_bins - 1))
            reward_bin = np.clip(reward_bin, 0, self.reward_bins - 1)
        else:
            reward_bin = self.reward_bins // 2  # Neutral

        return (context_bin, energy_bin, reward_bin)

    def select_goal(
        self,
        context: int,
        energy: float,
        entropy: float = 0.5,
        qse_metrics: Optional[Dict] = None
    ) -> str:
        """
        Select goal using epsilon-greedy with entropy modulation.

        Args:
            context: Current context ID
            energy: Current energy level
            entropy: QSE entropy for exploration modulation
            qse_metrics: Optional QSE metrics for advanced selection

        Returns:
            Selected goal name
        """
        # Discretize state
        state = self._discretize_state(context, energy, self.recent_rewards)
        self.current_state = state

        # Entropy modulates exploration
        effective_epsilon = self.epsilon * entropy

        # Epsilon-greedy selection
        if np.random.rand() < effective_epsilon:
            # Explore: random goal
            goal_idx = np.random.randint(self.n_actions)
        else:
            # Exploit: best goal for this state
            q_values = self.Q[state]

            # Add small noise to break ties
            noise = 1e-6 * np.random.randn(self.n_actions)
            goal_idx = np.argmax(q_values + noise)

        self.current_action = goal_idx
        goal = self.goals[goal_idx]
        self.current_goal = goal

        # Update visit count
        self.visit_counts[state][goal_idx] += 1

        return goal

    def update(
        self,
        reward: float,
        next_context: int,
        next_energy: float,
        done: bool = False
    ) -> Dict:
        """
        TD(λ) update with eligibility traces.

        Args:
            reward: Immediate reward
            next_context: Next context ID
            next_energy: Next energy level
            done: Whether episode is terminal

        Returns:
            Update statistics
        """
        if self.current_state is None or self.current_action is None:
            return {}

        # Track reward
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_rewards_window * 2:
            self.recent_rewards = self.recent_rewards[-self.recent_rewards_window:]

        # Get next state
        next_state = self._discretize_state(next_context, next_energy, self.recent_rewards)

        # Current Q-value
        current_q = self.Q[self.current_state][self.current_action]

        # Compute TD target
        if done:
            td_target = reward
        else:
            next_q_max = np.max(self.Q[next_state])
            td_target = reward + self.gamma * next_q_max

        # TD error
        td_error = td_target - current_q

        # Adaptive learning rate (decrease with visits)
        visits = self.visit_counts[self.current_state][self.current_action]
        adaptive_alpha = self.alpha / (1.0 + 0.01 * visits)

        # Update eligibility trace for current state-action
        self.eligibility[self.current_state][self.current_action] += 1.0

        # Update all Q-values proportional to eligibility
        for state_key in list(self.eligibility.keys()):
            for action_idx in range(self.n_actions):
                if self.eligibility[state_key][action_idx] > 1e-6:
                    # TD(λ) update
                    delta_q = adaptive_alpha * td_error * self.eligibility[state_key][action_idx]
                    self.Q[state_key][action_idx] += delta_q

                    # Decay eligibility trace
                    self.eligibility[state_key][action_idx] *= self.gamma * self.lambda_trace

        # Statistics
        self.total_updates += 1

        stats = {
            'td_error': float(td_error),
            'current_q': float(current_q),
            'target_q': float(td_target),
            'alpha': float(adaptive_alpha),
            'reward': float(reward)
        }

        return stats

    def reset_episode(self):
        """Reset eligibility traces at episode end."""
        self.eligibility.clear()
        self.current_state = None
        self.current_action = None

    def get_value_function(self, context: int, energy: float) -> np.ndarray:
        """
        Get value function V(s) = max_a Q(s,a) for given state.

        Args:
            context: Context ID
            energy: Energy level

        Returns:
            Value for each discretized state
        """
        values = []
        for reward_bin in range(self.reward_bins):
            state = (min(context, self.context_bins - 1),
                    int(energy * self.energy_bins),
                    reward_bin)
            values.append(np.max(self.Q[state]))
        return np.array(values)

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        total_states = len(self.Q)
        total_state_actions = sum(len(q) for q in self.Q.values())

        avg_q = np.mean([np.mean(q) for q in self.Q.values()]) if self.Q else 0.0
        max_q = np.max([np.max(q) for q in self.Q.values()]) if self.Q else 0.0

        return {
            'total_states_visited': total_states,
            'total_updates': self.total_updates,
            'average_q_value': float(avg_q),
            'max_q_value': float(max_q),
            'total_state_actions': total_state_actions,
            'exploration_rate': float(self.epsilon),
            'recent_avg_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        }

    def save_policy(self, filepath: str):
        """Save Q-table to file."""
        import json

        # Convert defaultdict to regular dict for JSON serialization
        q_dict = {str(k): v.tolist() for k, v in self.Q.items()}

        data = {
            'Q': q_dict,
            'goals': self.goals,
            'parameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'lambda': self.lambda_trace,
                'epsilon': self.epsilon
            },
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_policy(self, filepath: str):
        """Load Q-table from file."""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore Q-table
        self.Q.clear()
        for k_str, v_list in data['Q'].items():
            # Parse tuple from string
            k = eval(k_str)
            self.Q[k] = np.array(v_list)

        # Restore parameters
        params = data['parameters']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.lambda_trace = params['lambda']
        self.epsilon = params['epsilon']

    # ------------------------------------------------------------------
    # Backward-compatibility shims so GoalModuleV2 can be used where the
    # old GoalModule (goal.py) was expected.
    # ------------------------------------------------------------------

    def add_goal(self, goal_id: str):
        """No-op.  V2 uses a fixed strategy set."""
        pass

    def select_action(self, qse_metrics: dict) -> str:
        """Legacy API: select goal from a flat metrics dict."""
        context = int(qse_metrics.get('context_id', 0))
        energy = float(qse_metrics.get('energy', 0.5))
        entropy = float(qse_metrics.get('normalized_entropy', 0.5))
        goal = self.select_goal(context, energy, entropy, qse_metrics)
        self.current_goal = goal
        return goal

    def feedback(self, reward: float, next_context: int = None,
                 next_energy: float = None):
        """Legacy API: provide scalar reward.  Delegates to TD(λ) update."""
        if next_context is None:
            next_context = self.current_state[0] if self.current_state else 0
        if next_energy is None:
            next_energy = 0.5
        return self.update(reward, next_context, next_energy, done=False)

    def get_q_values(self) -> dict:
        """Return current Q-values as {goal_name: float}."""
        if self.current_state is not None and self.current_state in self.Q:
            return {g: float(self.Q[self.current_state][i])
                    for i, g in enumerate(self.goals)}
        return {g: 0.0 for g in self.goals}

    def get_history(self) -> list:
        """Compatibility stub."""
        return []

    def reset_exploitation_bias(self):
        """Reset eligibility traces (called by existential-pressure handler)."""
        self.eligibility.clear()


# Backward compatibility wrapper
class GoalModule:
    """
    Wrapper to maintain backward compatibility with existing code.
    Delegates to GoalModuleV2 but maintains old interface.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self._v2 = GoalModuleV2(cfg)
        self.current_goal = 'explore'
        self.goal_q_values = {goal: 0.0 for goal in self._v2.goals}

    def select_goal(self, qse_metrics: Dict) -> str:
        """Legacy interface: select goal from QSE metrics."""
        # Extract relevant state information
        context = qse_metrics.get('context_id', 0)
        # Estimate energy (not in old interface, use default)
        energy = 0.5
        entropy = qse_metrics.get('normalized_entropy', 0.5)

        goal = self._v2.select_goal(context, energy, entropy, qse_metrics)
        self.current_goal = goal

        # Update legacy Q-values for compatibility
        state = self._v2.current_state
        if state in self._v2.Q:
            for i, g in enumerate(self._v2.goals):
                self.goal_q_values[g] = float(self._v2.Q[state][i])

        return goal

    def update_q_value(self, goal: str, reward: float):
        """Legacy interface: update Q-value."""
        # Map to TD update (assumes same state)
        context = 0  # Unknown in legacy interface
        energy = 0.5
        self._v2.update(reward, context, energy, done=False)
