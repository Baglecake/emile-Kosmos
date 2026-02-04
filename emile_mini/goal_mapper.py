"""
GoalMapper: Learned strategy -> embodied goal mapping (Layer 2)

Replaces the hardcoded _strategy_to_embodied_goal() with a TD(lambda)
learner that discovers which embodied goal to pursue given the current
strategy and body state.

State: (strategy_idx, energy_bin, health_bin, food_nearby, shelter_nearby)
       ~300 discrete states
Actions: 7 embodied goals
Reward: same dense reward signal used by Layer 1
"""

import numpy as np
from collections import defaultdict


STRATEGIES = ['explore', 'exploit', 'rest', 'learn', 'social']

EMBODIED_GOALS = [
    'explore_space',
    'seek_food',
    'find_shelter',
    'rest_recover',
    'manipulate_objects',
    'social_interact',
    'categorize_experience',
]


class GoalMapper:
    """TD(lambda) learner for strategy -> embodied goal mapping."""

    def __init__(self, alpha=0.15, gamma=0.8, lambda_trace=0.7,
                 epsilon_base=0.3, warm_start=True):
        self.strategies = list(STRATEGIES)
        self.goals = list(EMBODIED_GOALS)
        self.n_strategies = len(self.strategies)
        self.n_goals = len(self.goals)

        # TD(lambda) parameters
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.epsilon_base = epsilon_base

        # Q-table: state -> array of Q-values (one per goal)
        self.Q = defaultdict(lambda: np.zeros(self.n_goals))
        self.eligibility = defaultdict(lambda: np.zeros(self.n_goals))

        # Current state tracking
        self.current_state = None
        self.current_action = None
        self.total_updates = 0

        if warm_start:
            self._warm_start()

    # ------------------------------------------------------------------
    # Warm start from hardcoded rules
    # ------------------------------------------------------------------
    def _warm_start(self):
        """Seed Q-values so the hardcoded mapping starts as preferred."""
        bonus = 0.3
        for s_idx, strat in enumerate(self.strategies):
            for e_bin in range(5):
                for h_bin in range(3):
                    for food in range(2):
                        for shelter in range(2):
                            state = (s_idx, e_bin, h_bin, food, shelter)
                            energy = (e_bin + 0.5) / 5.0
                            health = (h_bin + 0.5) / 3.0
                            goal = self._hardcoded_goal(strat, energy, health)
                            g_idx = self.goals.index(goal)
                            self.Q[state][g_idx] += bonus

    @staticmethod
    def _hardcoded_goal(strategy, energy, health):
        """The original hardcoded mapping (for reference and warm start)."""
        if strategy == 'rest':
            return 'rest_recover'
        if strategy == 'social':
            return 'social_interact'
        if strategy == 'learn':
            return 'categorize_experience'
        if strategy == 'explore':
            return 'explore_space'
        # exploit
        if energy < 0.4:
            return 'seek_food'
        if health < 0.5:
            return 'find_shelter'
        return 'manipulate_objects'

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------
    def _discretize_state(self, strategy, energy, health,
                          food_nearby, shelter_nearby):
        """Discretize continuous state into Q-table key."""
        s_idx = (self.strategies.index(strategy)
                 if strategy in self.strategies else 0)
        e_bin = min(int(energy * 5), 4)
        h_bin = min(int(health * 3), 2)
        return (s_idx, e_bin, h_bin, int(food_nearby), int(shelter_nearby))

    # ------------------------------------------------------------------
    # Goal selection
    # ------------------------------------------------------------------
    def select_goal(self, strategy, energy, health,
                    food_nearby=False, shelter_nearby=False, entropy=0.5):
        """Select embodied goal using epsilon-greedy with entropy modulation."""
        state = self._discretize_state(
            strategy, energy, health, food_nearby, shelter_nearby)

        # Entropy modulates exploration rate
        epsilon = self.epsilon_base * (0.5 + entropy)
        epsilon = min(epsilon, 0.95)

        if np.random.random() < epsilon:
            action = np.random.randint(self.n_goals)
        else:
            action = int(np.argmax(self.Q[state]))

        # Decay previous eligibility traces
        if self.current_state is not None:
            for s in self.eligibility:
                self.eligibility[s] *= self.gamma * self.lambda_trace

        self.current_state = state
        self.current_action = action
        self.eligibility[state][action] = 1.0

        return self.goals[action]

    # ------------------------------------------------------------------
    # TD(lambda) update
    # ------------------------------------------------------------------
    def update(self, reward, next_strategy, next_energy, next_health,
               next_food_nearby=False, next_shelter_nearby=False, done=False):
        """TD(lambda) update after observing reward."""
        if self.current_state is None:
            return {}

        next_state = self._discretize_state(
            next_strategy, next_energy, next_health,
            next_food_nearby, next_shelter_nearby)

        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.Q[next_state]))

        current_q = float(self.Q[self.current_state][self.current_action])
        td_error = target - current_q

        # Update all eligible states
        for s in list(self.eligibility.keys()):
            self.Q[s] += self.alpha * td_error * self.eligibility[s]
            self.eligibility[s] *= self.gamma * self.lambda_trace
            if np.max(np.abs(self.eligibility[s])) < 1e-6:
                del self.eligibility[s]

        self.total_updates += 1

        return {
            'td_error': float(td_error),
            'current_q': current_q,
            'reward': float(reward),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_statistics(self):
        """Get learning statistics."""
        all_q = list(self.Q.values())
        if all_q:
            all_vals = np.concatenate(all_q)
            return {
                'total_states': len(self.Q),
                'total_updates': self.total_updates,
                'avg_q': float(np.mean(all_vals)),
                'max_q': float(np.max(all_vals)),
            }
        return {
            'total_states': 0, 'total_updates': 0,
            'avg_q': 0.0, 'max_q': 0.0,
        }

    def reset_episode(self):
        """Reset episode state (preserves Q-table)."""
        self.current_state = None
        self.current_action = None
        self.eligibility.clear()

    def get_preferred_goals(self):
        """Return a dict of state -> preferred goal for analysis."""
        result = {}
        for state, qvals in self.Q.items():
            best = int(np.argmax(qvals))
            result[state] = self.goals[best]
        return result
