"""Logging configuration for Kosmos simulation runs.

Creates two output files per run:
- runs/latest.log: Human-readable narrative + LLM messages
- runs/latest_metrics.jsonl: Structured metrics every N ticks
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path


# Ensure runs directory exists
RUNS_DIR = Path(__file__).parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Log file paths
LOG_FILE = RUNS_DIR / "latest.log"
METRICS_FILE = RUNS_DIR / "latest_metrics.jsonl"


def setup_logging():
    """Configure logging for a new run. Clears previous log files."""

    # Clear previous log files
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    # Create main logger
    logger = logging.getLogger("kosmos")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler for narrative log (overwrites each run)
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(name)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # LLM-specific logger (child of kosmos)
    llm_logger = logging.getLogger("kosmos.llm")
    llm_logger.setLevel(logging.DEBUG)

    # Metrics logger (child of kosmos)
    metrics_logger = logging.getLogger("kosmos.metrics")
    metrics_logger.setLevel(logging.DEBUG)

    # Write run header
    logger.info(f"=== Kosmos Run Started: {datetime.now().isoformat()} ===")

    return logger


def get_logger(name: str = "kosmos") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_metrics(tick: int, agent_state: dict):
    """
    Write metrics to JSONL file.

    Called every N ticks to capture simulation state for analysis.
    """
    # Extract key metrics
    metrics = {
        "tick": tick,
        "timestamp": datetime.now().isoformat(),
        # Survival
        "energy": agent_state.get("energy", 0),
        "hydration": agent_state.get("hydration", 0),
        "alive": agent_state.get("alive", True),
        "deaths": agent_state.get("deaths", 0),
        # Position and exploration
        "pos": agent_state.get("pos", (0, 0)),
        "cells_visited": agent_state.get("cells_visited", 0),
        # Strategy and goals
        "strategy": agent_state.get("strategy", ""),
        "embodied_goal": agent_state.get("embodied_goal", ""),
        "consciousness_zone": agent_state.get("consciousness_zone", ""),
        # Learning
        "teacher_prob": agent_state.get("teacher_prob", 1.0),
        "decision_source": agent_state.get("decision_source", ""),
        "learned_samples": agent_state.get("learned_samples", 0),
        "learned_ema": agent_state.get("learned_ema", 0),
        "heuristic_ema": agent_state.get("heuristic_ema", 0),
        # Phase 6: Surplus/Tension
        "surplus_ema": agent_state.get("surplus_ema", 0),
        "sigma_ema": agent_state.get("sigma_ema", 0),
        "tau_prime": agent_state.get("tau_prime", 1.0),
        "ruptures": agent_state.get("ruptures", 0),
        "integrity": agent_state.get("integrity", 0),
        "diversity": agent_state.get("diversity", 0.5),
        # Planning
        "plan_goal": agent_state.get("plan_goal", ""),
        "plan_steps_remaining": agent_state.get("plan_steps_remaining", 0),
        "plans_started": agent_state.get("plans_started", 0),
        "plans_completed": agent_state.get("plans_completed", 0),
        # Inventory
        "inventory": agent_state.get("inventory", []),
        "crafted": agent_state.get("crafted", []),
        # World
        "weather": agent_state.get("weather", "clear"),
        "time_of_day": agent_state.get("time_of_day", "day"),
        # Novelty and stuckness
        "novelty": agent_state.get("novelty", 0.5),
        "is_stuck": agent_state.get("is_stuck", False),
    }

    with open(METRICS_FILE, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def log_llm_event(event_type: str, tick: int, **kwargs):
    """
    Log LLM-related events with structured data.

    Event types: FIRE, RECV, ADOPT, EXEC, DONE, INTERRUPT, STALE, ERROR
    """
    logger = logging.getLogger("kosmos.llm")

    msg_parts = [f"t={tick}", f"event={event_type}"]
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "..."
        msg_parts.append(f"{key}={value}")

    logger.info(" | ".join(msg_parts))


def log_death(tick: int, pos: tuple, biome: str, weather: str,
              zone: str, hydration: float, deaths: int):
    """Log death events for post-mortem analysis."""
    logger = logging.getLogger("kosmos")
    logger.warning(
        f"DEATH t={tick} | pos={pos} biome={biome} weather={weather} "
        f"zone={zone} hydration={hydration:.2f} total_deaths={deaths}"
    )


def log_rupture(tick: int, sigma: float, k_eff: float, pos: tuple,
                rupture_count: int):
    """Log rupture events."""
    logger = logging.getLogger("kosmos")
    logger.warning(
        f"RUPTURE t={tick} | Σ={sigma:.2f} k={k_eff:.1f} pos={pos} "
        f"total_ruptures={rupture_count}"
    )
