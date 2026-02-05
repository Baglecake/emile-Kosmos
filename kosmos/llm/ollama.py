"""LLM reasoning via Ollama with QSE-modulated parameters."""

import json
import threading
from collections import deque
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434"


class ConversationHistory:
    """Sliding window of recent exchanges for multi-turn context."""

    def __init__(self, max_turns: int = 4, max_chars: int = 3000):
        self._turns: deque[dict] = deque(maxlen=max_turns * 2)
        self.max_chars = max_chars

    def add_user(self, content: str):
        self._turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        self._turns.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        return list(self._turns)

    def clear(self):
        self._turns.clear()

    def _trim(self):
        """Drop oldest turns if total character count exceeds limit."""
        while len(self._turns) > 2:
            total = sum(len(t["content"]) for t in self._turns)
            if total <= self.max_chars:
                break
            self._turns.popleft()


class OllamaReasoner:
    """
    LLM interface that produces structured tool calls and inner monologue.

    QSE entropy -> temperature (creativity).
    QSE strategy -> personality/focus.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self._lock = threading.Lock()
        self._available = None
        self.history = ConversationHistory(max_turns=4)

    def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            # Try exact match, then prefix match
            if self.model in models:
                self._available = True
            else:
                matches = [m for m in models if self.model in m]
                if matches:
                    self.model = matches[0]
                    self._available = True
                else:
                    self._available = False
            return self._available
        except Exception:
            self._available = False
            return False

    def reason(
        self,
        situation: str,
        tools: list[dict],
        strategy: str,
        entropy: float,
        energy: float,
        inventory: list[str],
        memory_hits: list[str] | None = None,
        last_result: str = "",
    ) -> dict:
        """
        Ask the LLM to decide what to do.

        Returns: {"tool": "tool_name", "args": {...}, "thought": "inner monologue"}
        """
        temperature = 0.3 + entropy * 1.2

        personality = {
            "explore": "You are curious and adventurous. Seek the unknown.",
            "exploit": "You are efficient and focused. Get what you need directly.",
            "rest": "You are tired and cautious. Conserve energy. Rest if safe.",
            "learn": "You are analytical. Examine things. Gather information.",
            "social": "You are sociable. Look for others. Communicate.",
        }.get(strategy, "You are a survivor. Stay alive.")

        tool_list = "\n".join(
            f"- {t['name']}: {t['description']} "
            f"(params: {', '.join(t['parameters'].keys()) if t['parameters'] else 'none'})"
            for t in tools
        )

        inv_str = ", ".join(inventory) if inventory else "empty"
        mem_str = ""
        if memory_hits:
            mem_str = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memory_hits[:3])

        system = (
            f"{personality}\n\n"
            f"You are a small creature trying to survive in a wild world. "
            f"Your energy is {energy:.0%}. Your inventory: [{inv_str}].{mem_str}\n\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Respond ONLY with valid JSON in this exact format:\n"
            f'{{"tool": "tool_name", "args": {{"param": "value"}}, '
            f'"thought": "one sentence inner monologue"}}\n'
            f"Pick the single best action for your current situation."
        )

        # Build multi-turn message list
        user_content = situation
        if last_result:
            user_content = f"Previous action result: {last_result}\n\n{situation}"

        messages = [{"role": "system", "content": system}]

        # Insert conversation history between system and current turn
        with self._lock:
            history = self.history.get_messages()
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "options": {"temperature": float(temperature)},
                    "stream": False,
                    "format": "json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            parsed = self._parse_response(content)

            # Only add to history if parse succeeded (not a fallback)
            if parsed.get("tool") != "examine" or parsed.get("thought") != "Let me look around.":
                with self._lock:
                    self.history.add_user(user_content)
                    self.history.add_assistant(content)

            return parsed
        except requests.ConnectionError:
            return self._fallback(strategy, energy)
        except Exception:
            return self._fallback(strategy, energy)

    def narrate(self, event: str, strategy: str, entropy: float) -> str:
        """Generate a short narration for an event."""
        temperature = 0.4 + entropy * 1.0

        personality = {
            "explore": "You speak with wonder and curiosity.",
            "exploit": "You speak tersely, focused on the task.",
            "rest": "You speak softly, conserving energy.",
            "learn": "You speak analytically, noting details.",
            "social": "You speak warmly, as if to a companion.",
        }.get(strategy, "You observe simply.")

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are a small creature narrating your experience. "
                                f"{personality} "
                                f"Respond with ONE sentence, max 20 words. "
                                f"First person. Present tense. No quotes."
                            ),
                        },
                        {"role": "user", "content": event},
                    ],
                    "options": {"temperature": float(temperature)},
                    "stream": False,
                },
                timeout=15,
            )
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            # Clean up: take first sentence only
            for sep in [".", "!", "?"]:
                if sep in text:
                    text = text[: text.index(sep) + 1]
                    break
            return text[:120]
        except Exception:
            return ""

    def reason_plan(
        self,
        situation: str,
        tools: list[dict],
        strategy: str,
        entropy: float,
        energy: float,
        inventory: list[str],
        memory_hits: list[str] | None = None,
        last_result: str = "",
    ) -> dict:
        """
        Ask the LLM to produce a multi-step plan (3-5 actions).

        Returns: {
            "plan": [{"tool": "...", "args": {...}, "thought": "..."}],
            "goal": "description of what the plan achieves",
            "replan_if": ["condition1", "condition2", ...]
        }
        """
        temperature = 0.3 + entropy * 0.8  # Slightly lower for planning

        personality = {
            "explore": "You are curious and adventurous. Seek the unknown.",
            "exploit": "You are efficient and focused. Get what you need directly.",
            "rest": "You are tired and cautious. Conserve energy. Rest if safe.",
            "learn": "You are analytical. Examine things. Gather information.",
            "social": "You are sociable. Look for others. Communicate.",
        }.get(strategy, "You are a survivor. Stay alive.")

        tool_list = "\n".join(
            f"- {t['name']}: {t['description']} "
            f"(params: {', '.join(t['parameters'].keys()) if t['parameters'] else 'none'})"
            for t in tools
        )

        inv_str = ", ".join(inventory) if inventory else "empty"
        mem_str = ""
        if memory_hits:
            mem_str = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memory_hits[:3])

        system = (
            f"{personality}\n\n"
            f"You are a small creature trying to survive in a wild world. "
            f"Your energy is {energy:.0%}. Your inventory: [{inv_str}].{mem_str}\n\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Create a SHORT PLAN (2-4 steps) to achieve a goal. "
            f"Respond ONLY with valid JSON in this exact format:\n"
            f'{{"plan": [{{"tool": "name", "args": {{}}, "thought": "why"}}], '
            f'"goal": "what the plan achieves", '
            f'"replan_if": ["condition1"]}}\n\n'
            f"Valid replan_if conditions: energy_critical, hazard_nearby, "
            f"goal_changed, inventory_full, target_gone, weather_change\n"
            f"Keep plans short and achievable. Focus on immediate survival needs first."
        )

        user_content = situation
        if last_result:
            user_content = f"Previous action result: {last_result}\n\n{situation}"

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "options": {"temperature": float(temperature)},
                    "stream": False,
                    "format": "json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            return self._parse_plan_response(content, strategy, energy)
        except requests.ConnectionError:
            return self._fallback_plan(strategy, energy)
        except Exception:
            return self._fallback_plan(strategy, energy)

    def _parse_plan_response(self, content: str, strategy: str, energy: float) -> dict:
        """Parse LLM plan response."""
        try:
            data = json.loads(content)
            plan = data.get("plan", [])
            # Validate plan structure
            if not plan or not isinstance(plan, list):
                return self._fallback_plan(strategy, energy)
            # Validate each step
            valid_plan = []
            for step in plan[:5]:  # Max 5 steps
                if isinstance(step, dict) and "tool" in step:
                    valid_plan.append({
                        "tool": step.get("tool", "wait"),
                        "args": step.get("args", {}),
                        "thought": step.get("thought", ""),
                    })
            if not valid_plan:
                return self._fallback_plan(strategy, energy)
            return {
                "plan": valid_plan,
                "goal": data.get("goal", "survive"),
                "replan_if": data.get("replan_if", ["energy_critical", "hazard_nearby"]),
            }
        except (json.JSONDecodeError, KeyError):
            return self._fallback_plan(strategy, energy)

    def _fallback_plan(self, strategy: str, energy: float) -> dict:
        """Fallback plan when LLM is unavailable."""
        if energy < 0.3:
            return {
                "plan": [
                    {"tool": "examine", "args": {"target": "surroundings"}, "thought": "Look for food"},
                    {"tool": "rest", "args": {}, "thought": "Conserve energy"},
                ],
                "goal": "find food and rest",
                "replan_if": ["energy_critical"],
            }
        if strategy == "explore":
            return {
                "plan": [
                    {"tool": "move", "args": {"direction": "north"}, "thought": "Explore north"},
                    {"tool": "examine", "args": {"target": "surroundings"}, "thought": "Survey area"},
                    {"tool": "move", "args": {"direction": "east"}, "thought": "Continue exploring"},
                ],
                "goal": "explore new territory",
                "replan_if": ["hazard_nearby", "goal_changed"],
            }
        return {
            "plan": [
                {"tool": "examine", "args": {"target": "surroundings"}, "thought": "Look around"},
            ],
            "goal": "assess situation",
            "replan_if": ["energy_critical"],
        }

    def _parse_response(self, content: str) -> dict:
        """Parse LLM JSON response into tool call."""
        try:
            data = json.loads(content)
            return {
                "tool": data.get("tool", "wait"),
                "args": data.get("args", {}),
                "thought": data.get("thought", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return self._fallback("explore", 0.5)

    def _fallback(self, strategy: str, energy: float) -> dict:
        """Heuristic fallback when LLM is unavailable."""
        if energy < 0.2:
            return {"tool": "rest", "args": {}, "thought": "I need to rest..."}
        if strategy == "explore":
            dirs = ["north", "south", "east", "west"]
            import random
            return {
                "tool": "move",
                "args": {"direction": random.choice(dirs)},
                "thought": "I should keep exploring.",
            }
        return {"tool": "examine", "args": {"target": "surroundings"}, "thought": "Let me look around."}
