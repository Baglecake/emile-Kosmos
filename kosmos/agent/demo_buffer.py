"""Demonstration buffer for behavior cloning from teacher (LLM/heuristic).

This implements the "consolidation" step that's missing from the current
architecture. The LLM/heuristic makes good decisions, but they evaporate
after one use. This buffer stores them so the student policy can practice
repeatedly - like human habit formation through repetition.

The key insight from GPT: "consciousness interprets → you act → your
nervous system consolidates patterns offline (sleep, repetition, replay)."
"""

import numpy as np
from collections import deque
from typing import Optional
import json

from .action_policy import encode_kosmos_state, KOSMOS_ACTIONS, KOSMOS_INPUT_DIM


class DemonstrationBuffer:
    """Store teacher demonstrations for offline behavior cloning.

    Each demonstration contains:
    - state: the encoded state vector at decision time
    - action_idx: index into KOSMOS_ACTIONS
    - reward: immediate reward received
    - survival_weight: how many ticks the agent survived after this action
    - source: 'llm', 'heuristic', or 'plan' (for filtering/weighting)
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._survival_tracker: list[int] = []  # Track indices for survival weighting

    def add(self, state_dict: dict, action_name: str, reward: float,
            source: str = "teacher") -> int:
        """Add a demonstration to the buffer.

        Args:
            state_dict: kwargs for encode_kosmos_state()
            action_name: e.g., 'move_north', 'consume', 'wait'
            reward: immediate reward
            source: 'llm', 'heuristic', 'plan'

        Returns:
            Index of the added demo (for survival tracking)
        """
        # Encode state
        state = encode_kosmos_state(**state_dict)

        # Convert action name to index
        if action_name in KOSMOS_ACTIONS:
            action_idx = KOSMOS_ACTIONS.index(action_name)
        else:
            # Handle tool names that need conversion (e.g., 'move' -> default)
            action_idx = KOSMOS_ACTIONS.index('wait')

        demo = {
            "state": state,
            "action_idx": action_idx,
            "reward": reward,
            "survival_ticks": 0,  # Updated later
            "source": source,
        }

        self.buffer.append(demo)
        idx = len(self.buffer) - 1
        self._survival_tracker.append(idx)

        return idx

    def update_survival(self, ticks_survived: int, lookback: int = 50):
        """Update survival weights for recent demonstrations.

        Called periodically to credit demos that led to survival.

        Args:
            ticks_survived: how many ticks since last death
            lookback: how many recent demos to update
        """
        # Update the most recent demos with survival info
        for i in range(max(0, len(self.buffer) - lookback), len(self.buffer)):
            if i < len(self.buffer):
                self.buffer[i]["survival_ticks"] = max(
                    self.buffer[i]["survival_ticks"],
                    ticks_survived
                )

    def on_death(self):
        """Called when agent dies - mark recent demos as leading to death."""
        # Recent demos (last 20) contributed to death - lower their weight
        for i in range(max(0, len(self.buffer) - 20), len(self.buffer)):
            if i < len(self.buffer):
                self.buffer[i]["survival_ticks"] = min(
                    self.buffer[i]["survival_ticks"],
                    5  # Cap survival credit for death-leading actions
                )

    def sample(self, batch_size: int = 32,
               weighted: bool = True) -> list[dict]:
        """Sample a batch of demonstrations.

        Args:
            batch_size: number of demos to sample
            weighted: if True, weight by reward + survival

        Returns:
            List of demo dicts
        """
        if len(self.buffer) == 0:
            return []

        n = min(batch_size, len(self.buffer))

        if weighted and len(self.buffer) > 0:
            # Compute weights: reward + survival bonus
            weights = []
            for demo in self.buffer:
                w = 1.0  # Base weight
                w += max(0, demo["reward"]) * 2.0  # Reward bonus
                w += demo["survival_ticks"] * 0.01  # Survival bonus
                # Bonus for LLM demos (they saw the full context)
                if demo["source"] == "llm":
                    w *= 1.2
                weights.append(max(0.1, w))  # Minimum weight

            weights = np.array(weights)
            weights = weights / weights.sum()

            indices = np.random.choice(len(self.buffer), size=n,
                                       replace=False, p=weights)
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)

        return [self.buffer[i] for i in indices]

    def get_positive_demos(self, min_reward: float = 0.0,
                           min_survival: int = 50) -> list[dict]:
        """Get demos that led to good outcomes.

        Useful for focused behavior cloning on successful actions.
        """
        return [d for d in self.buffer
                if d["reward"] >= min_reward and d["survival_ticks"] >= min_survival]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self._survival_tracker.clear()

    # Persistence
    def save(self, filepath: str):
        """Save buffer to JSON file."""
        data = []
        for demo in self.buffer:
            data.append({
                "state": demo["state"].tolist(),
                "action_idx": demo["action_idx"],
                "reward": demo["reward"],
                "survival_ticks": demo["survival_ticks"],
                "source": demo["source"],
            })
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load buffer from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.buffer.clear()
        for item in data:
            self.buffer.append({
                "state": np.array(item["state"]),
                "action_idx": item["action_idx"],
                "reward": item["reward"],
                "survival_ticks": item["survival_ticks"],
                "source": item["source"],
            })


def behavior_cloning_update(policy, demos: list[dict],
                            learning_rate: float = 0.01) -> dict:
    """Train policy via behavior cloning on demonstration batch.

    This is supervised learning: minimize cross-entropy between
    policy output and teacher action.

    Args:
        policy: KosmosActionPolicy instance
        demos: list of demo dicts from buffer
        learning_rate: learning rate for gradient step

    Returns:
        dict with training stats
    """
    if len(demos) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "n_demos": 0}

    total_loss = 0.0
    correct = 0

    # Accumulate gradients
    dW1 = np.zeros_like(policy.W1)
    db1 = np.zeros_like(policy.b1)
    dW2 = np.zeros_like(policy.W2)
    db2 = np.zeros_like(policy.b2)

    for demo in demos:
        state = demo["state"]
        target_idx = demo["action_idx"]

        # Forward pass
        probs, h, z1 = policy.forward(state, temperature=1.0)

        # Cross-entropy loss
        loss = -np.log(probs[target_idx] + 1e-10)
        total_loss += loss

        # Accuracy
        if np.argmax(probs) == target_idx:
            correct += 1

        # Backward pass (cross-entropy gradient)
        # d loss / d logits = probs - one_hot(target)
        one_hot = np.zeros(policy.n_actions)
        one_hot[target_idx] = 1.0
        dl = probs - one_hot  # Gradient w.r.t. logits

        # Gradient for W2, b2
        dW2 += np.outer(h, dl)
        db2 += dl

        # Backprop through tanh
        dh = dl @ policy.W2.T
        dz = dh * (1.0 - h ** 2)  # tanh derivative

        # Gradient for W1, b1
        dW1 += np.outer(state, dz)
        db1 += dz

    # Average gradients
    n = len(demos)
    dW1 /= n; db1 /= n; dW2 /= n; db2 /= n

    # Gradient clipping
    total_norm = np.sqrt(
        np.sum(dW1 ** 2) + np.sum(db1 ** 2) +
        np.sum(dW2 ** 2) + np.sum(db2 ** 2)
    )
    if total_norm > 1.0:
        scale = 1.0 / total_norm
        dW1 *= scale; db1 *= scale; dW2 *= scale; db2 *= scale

    # Gradient descent (minimize loss)
    policy.W1 -= learning_rate * dW1
    policy.b1 -= learning_rate * db1
    policy.W2 -= learning_rate * dW2
    policy.b2 -= learning_rate * db2

    return {
        "loss": total_loss / n,
        "accuracy": correct / n,
        "n_demos": n,
        "grad_norm": total_norm,
    }
