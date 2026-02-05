#!/usr/bin/env python3
"""Extended test for Phase 5f-g with detailed metrics logging."""

import time
import json
from collections import Counter
from datetime import datetime

from kosmos.world.grid import KosmosWorld
from kosmos.agent.core import KosmosAgent


def run_extended_test(duration_minutes=10, speed=32, output_file="test_results.json"):
    """
    Run an extended test and capture detailed metrics.

    Args:
        duration_minutes: How long to run (in real time)
        speed: Ticks per second
        output_file: Where to save results
    """
    print(f"=== Extended Test: {duration_minutes} minutes at speed {speed} ===")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Initialize
    world = KosmosWorld(size=30, seed=42)
    agent = KosmosAgent(world, model='llama3.1:8b')
    agent.start()

    # Metrics collection
    metrics = {
        "test_info": {
            "duration_minutes": duration_minutes,
            "speed": speed,
            "start_time": datetime.now().isoformat(),
            "seed": 42,
        },
        "summary": {},
        "time_series": [],  # Periodic snapshots
        "events": {
            "deaths": [],
            "stuck_events": [],
            "unstuck_events": [],
            "crafts": [],
        },
        "action_counts": Counter(),
        "decision_source_counts": Counter(),
        "zone_time": Counter(),
        "strategy_time": Counter(),
    }

    # Tracking variables
    start_time = time.time()
    target_duration = duration_minutes * 60
    tick_interval = 1.0 / speed

    last_stuck = False
    last_report_tick = 0
    report_interval = 500  # Report every N ticks

    total_ticks = 0

    print(f"Running for {duration_minutes} minutes...")
    print(f"Progress reports every {report_interval} ticks")
    print()

    try:
        while (time.time() - start_time) < target_duration:
            # Tick world and agent
            world.tick()
            result = agent.tick()
            total_ticks += 1

            # Get current state
            state = agent.get_state()

            # Track action
            action = result.get("tool", "unknown")
            metrics["action_counts"][action] += 1

            # Track decision source
            metrics["decision_source_counts"][state["decision_source"]] += 1

            # Track zone and strategy time
            metrics["zone_time"][state["consciousness_zone"]] += 1
            metrics["strategy_time"][state["strategy"]] += 1

            # Detect events
            # Death
            if result.get("tool") == "death":
                metrics["events"]["deaths"].append({
                    "tick": total_ticks,
                    "pos": state["pos"],
                    "zone": state["consciousness_zone"],
                    "energy": state["energy"],
                })

            # Stuck transitions
            if state["is_stuck"] and not last_stuck:
                metrics["events"]["stuck_events"].append({
                    "tick": total_ticks,
                    "pos": state["pos"],
                    "novelty": state["novelty"],
                })
            elif not state["is_stuck"] and last_stuck:
                metrics["events"]["unstuck_events"].append({
                    "tick": total_ticks,
                })
            last_stuck = state["is_stuck"]

            # Craft events
            if "Crafted" in str(result.get("result", "")):
                metrics["events"]["crafts"].append({
                    "tick": total_ticks,
                    "result": result.get("result", ""),
                })

            # Periodic snapshot
            if total_ticks - last_report_tick >= report_interval:
                elapsed = time.time() - start_time
                snapshot = {
                    "tick": total_ticks,
                    "elapsed_seconds": round(elapsed, 1),
                    "energy": round(state["energy"], 3),
                    "hydration": round(state["hydration"], 3),
                    "deaths": agent.deaths,
                    "zone": state["consciousness_zone"],
                    "strategy": state["strategy"],
                    "teacher_prob": round(state["teacher_prob"], 4),
                    "learned_samples": state["learned_samples"],
                    "learned_ema": round(state["learned_ema"], 4),
                    "heuristic_ema": round(state["heuristic_ema"], 4),
                    "novelty": round(state["novelty"], 3),
                    "goal_satisfaction": round(state["goal_satisfaction"], 3),
                    "is_stuck": state["is_stuck"],
                    "stuck_ticks": state["stuck_ticks"],
                    "action_penalty": round(state["action_repeat_penalty"], 3),
                    "cells_visited": state["cells_visited"],
                    "food_eaten": state["food_eaten"],
                    "crafted": state["crafted"],
                }
                metrics["time_series"].append(snapshot)

                # Print progress
                pct = (elapsed / target_duration) * 100
                print(f"[{pct:5.1f}%] t={total_ticks:5d} | energy={state['energy']:.2f} "
                      f"deaths={agent.deaths} zone={state['consciousness_zone']:12s} "
                      f"teacher_prob={state['teacher_prob']:.3f} learned={state['learned_samples']} "
                      f"novelty={state['novelty']:.2f} stuck={state['is_stuck']}")

                last_report_tick = total_ticks

            # Pace to target speed (approximate)
            # Skip sleep to run as fast as possible for testing
            # time.sleep(tick_interval)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        agent.stop()

    # Calculate summary
    elapsed_total = time.time() - start_time
    metrics["summary"] = {
        "total_ticks": total_ticks,
        "total_time_seconds": round(elapsed_total, 1),
        "effective_speed": round(total_ticks / elapsed_total, 1),
        "total_deaths": agent.deaths,
        "deaths_per_1000_ticks": round(agent.deaths / (total_ticks / 1000), 2) if total_ticks > 0 else 0,
        "final_energy": round(agent.energy, 3),
        "final_teacher_prob": round(agent._teacher_prob, 4),
        "final_learned_samples": agent._learned_samples,
        "final_learned_ema": round(agent._learned_reward_ema, 4),
        "final_heuristic_ema": round(agent._heuristic_reward_ema, 4),
        "cells_visited": len(agent.cells_visited),
        "food_eaten": agent.food_eaten,
        "water_drunk": agent.water_drunk,
        "items_crafted": len(agent.crafted),
        "crafted_items": agent.crafted,
        "stuck_events": len(metrics["events"]["stuck_events"]),
        "unstuck_events": len(metrics["events"]["unstuck_events"]),
    }

    # Convert Counters to dicts for JSON
    metrics["action_counts"] = dict(metrics["action_counts"])
    metrics["decision_source_counts"] = dict(metrics["decision_source_counts"])
    metrics["zone_time"] = dict(metrics["zone_time"])
    metrics["strategy_time"] = dict(metrics["strategy_time"])

    # Calculate percentages
    if total_ticks > 0:
        metrics["action_percentages"] = {
            k: round(v / total_ticks * 100, 1)
            for k, v in metrics["action_counts"].items()
        }
        metrics["decision_source_percentages"] = {
            k: round(v / total_ticks * 100, 1)
            for k, v in metrics["decision_source_counts"].items()
        }
        metrics["zone_percentages"] = {
            k: round(v / total_ticks * 100, 1)
            for k, v in metrics["zone_time"].items()
        }
        metrics["strategy_percentages"] = {
            k: round(v / total_ticks * 100, 1)
            for k, v in metrics["strategy_time"].items()
        }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("EXTENDED TEST COMPLETE")
    print("=" * 60)
    print()
    print(f"Duration: {elapsed_total/60:.1f} minutes ({total_ticks} ticks)")
    print(f"Effective speed: {metrics['summary']['effective_speed']:.1f} ticks/sec")
    print()
    print("--- Survival ---")
    print(f"Deaths: {agent.deaths} ({metrics['summary']['deaths_per_1000_ticks']:.2f} per 1000 ticks)")
    print(f"Final energy: {agent.energy:.2f}")
    print(f"Food eaten: {agent.food_eaten}")
    print(f"Water drunk: {agent.water_drunk}")
    print()
    print("--- Learning ---")
    print(f"Teacher prob: {agent._teacher_prob:.4f}")
    print(f"Learned samples: {agent._learned_samples}")
    print(f"Learned EMA: {agent._learned_reward_ema:.4f}")
    print(f"Heuristic EMA: {agent._heuristic_reward_ema:.4f}")
    print()
    print("--- Phase 5f-g ---")
    print(f"Stuck events: {len(metrics['events']['stuck_events'])}")
    print(f"Unstuck events: {len(metrics['events']['unstuck_events'])}")
    print()
    print("--- Actions ---")
    for action, pct in sorted(metrics.get("action_percentages", {}).items(),
                               key=lambda x: -x[1]):
        print(f"  {action}: {pct:.1f}%")
    print()
    print("--- Decision Sources ---")
    for src, pct in sorted(metrics.get("decision_source_percentages", {}).items(),
                           key=lambda x: -x[1]):
        print(f"  {src}: {pct:.1f}%")
    print()
    print("--- Zones ---")
    for zone, pct in sorted(metrics.get("zone_percentages", {}).items(),
                            key=lambda x: -x[1]):
        print(f"  {zone}: {pct:.1f}%")
    print()
    print("--- Strategies ---")
    for strat, pct in sorted(metrics.get("strategy_percentages", {}).items(),
                              key=lambda x: -x[1]):
        print(f"  {strat}: {pct:.1f}%")
    print()
    print(f"Results saved to: {output_file}")
    print()

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extended test for emile-Kosmos")
    parser.add_argument("--minutes", type=int, default=10, help="Test duration in minutes")
    parser.add_argument("--speed", type=int, default=32, help="Target ticks per second")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output file")
    args = parser.parse_args()

    run_extended_test(
        duration_minutes=args.minutes,
        speed=args.speed,
        output_file=args.output,
    )
