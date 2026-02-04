"""
Embodied QSE-Émile: Sensorimotor Grid World
(v2)
Where QSE dynamics shape both perception and action through:
- Visual field integration with surplus dynamics
- Body schema formation through proprioception
- Context-dependent perceptual interpretation
- Emergent categorization through experience
- Memory formation through embodied interaction
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

# Import existing QSE-Émile modules
from emile_mini.agent import EmileAgent
from emile_mini.config import QSEConfig, CONFIG
from emile_mini.qse_core import calculate_emergent_time

# Optional multimodal import
try:
    from emile_mini.multimodal import ModalityFeature
except Exception:
    ModalityFeature = None

# JSONL writer
try:
    from emile_mini.utils.jsonl_logger import JSONLWriter
except Exception:
    JSONLWriter = None  # will guard at runtime

from emile_mini.utils.tau_prime import calculate_tau_prime, compute_delta_sigma_emergent


@dataclass
class BodyState:
    """Complete body state information"""
    position: Tuple[int, int]  # (x, y) in grid
    orientation: float  # 0-2π radians
    velocity: Tuple[float, float]  # movement vector
    energy: float  # metabolic energy level
    size: float  # body size for collision detection
    health: float  # overall body condition


class SensoriMotorBody:
    """Embodied agent with sensors, actuators, and body schema"""

    def __init__(self, initial_position=(5, 5), vision_range=3):
        # Let type checkers know this exists; set at runtime by agent
        self.cfg: Optional[QSEConfig] = None

        # Body state
        self.state = BodyState(
            position=initial_position,
            orientation=0.0,
            velocity=(0.0, 0.0),
            energy=1.0,
            size=0.8,
            health=1.0
        )

        # Sensory capabilities
        self.vision_range = vision_range
        self.vision_field = np.zeros((2*vision_range+1, 2*vision_range+1))
        self.proprioception = np.zeros(6)  # [pos_x, pos_y, orient, vel_x, vel_y, energy]

        # Body schema (learned through experience)
        self.body_schema = {
            'boundaries': deque(maxlen=100),  # learned body boundaries
            'affordances': defaultdict(list),  # what actions are possible where
            'sensorimotor_mappings': defaultdict(list)  # perception-action correlations
        }

        # Action repertoire
        self.actions = {
            'move_forward': self._move_forward,
            'turn_left': self._turn_left,
            'turn_right': self._turn_right,
            'move_backward': self._move_backward,
            'rest': self._rest,
            'examine': self._examine,
            'forage': self._forage  # NEW: Forage action
        }

    def _forage(self, intensity=1.0):
        """Forage for resources at current location"""
        # This will be handled by the environment
        return (0, 0)

    def _move_forward(self, intensity=1.0):
        """Move forward in current orientation"""
        dx = np.cos(self.state.orientation) * intensity
        dy = np.sin(self.state.orientation) * intensity
        return (dx, dy)

    def _move_backward(self, intensity=0.5):
        """Move backward from current orientation"""
        dx = -np.cos(self.state.orientation) * intensity
        dy = -np.sin(self.state.orientation) * intensity
        return (dx, dy)

    def _turn_left(self, intensity=1.0):
        """Turn counterclockwise"""
        self.state.orientation += np.pi/4 * intensity
        self.state.orientation = self.state.orientation % (2 * np.pi)
        return (0, 0)

    def _turn_right(self, intensity=1.0):
        """Turn clockwise"""
        self.state.orientation -= np.pi/4 * intensity
        self.state.orientation = self.state.orientation % (2 * np.pi)
        return (0, 0)

    def _rest(self, intensity=1.0):
        """Rest to recover energy"""
        self.state.energy = min(1.0, self.state.energy + 0.1 * intensity)
        return (0, 0)

    def _examine(self, intensity=1.0):
        """Focus attention on current location"""
        # This will be used for detailed perception
        return (0, 0)

    def execute_action(self, action_name: str, intensity: float = 1.0):
        """Execute embodied action with config-based energy management.
        - Uses costs from self.cfg if available; otherwise QSEConfig defaults.
        - 'rest' recovers energy (negative cost).
        """
        if action_name not in getattr(self, "actions", {}):
            return (0, 0)

        # Perform the embodied action
        movement = self.actions[action_name](intensity)

        # Get config object
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            try:
                from emile_mini.config import QSEConfig  # packaged path
            except Exception:
                from emile_mini.config import QSEConfig   # script path (fallback)
            cfg = QSEConfig()

        # Energy parameters (with safe defaults)
        move_cost       = getattr(cfg, "ENERGY_MOVE_COST", 0.005)
        turn_cost       = getattr(cfg, "ENERGY_TURN_COST", 0.002)
        rest_recovery   = getattr(cfg, "ENERGY_REST_RECOVERY", 0.04)
        examine_cost    = getattr(cfg, "ENERGY_EXAMINE_COST", 0.001)
        basal_cost      = getattr(cfg, "ENERGY_BASAL_COST", 0.004)
        min_floor       = getattr(cfg, "ENERGY_MIN_FLOOR", 0.05)

        energy_map = {
            "move_forward":  move_cost,
            "move_backward": move_cost,
            "turn_left":     turn_cost,
            "turn_right":    turn_cost,
            "rest":         -rest_recovery,   # recover energy
            "examine":       examine_cost,
            "forage":        examine_cost,
        }

        cost = float(energy_map.get(action_name, turn_cost)) * float(intensity)
        cost += basal_cost  # metabolism: always costs energy to exist
        new_energy = float(self.state.energy) - cost

        # Clamp between min floor and a soft max (1.2 matches the rest of your code)
        try:
            import numpy as _np
            self.state.energy = float(_np.clip(new_energy, min_floor, 1.2))
        except Exception:
            self.state.energy = max(min_floor, min(1.2, new_energy))

        self.state.velocity = movement
        return movement

    def update_proprioception(self):
        """Update internal body state sensing"""
        self.proprioception = np.array([
            self.state.position[0] / 20.0,  # normalized position
            self.state.position[1] / 20.0,
            self.state.orientation / (2 * np.pi),
            self.state.velocity[0],
            self.state.velocity[1],
            self.state.energy
        ])

    def update_body_schema(self, collision_occurred, environment_feedback):
        """Learn body boundaries and affordances through experience"""

        # Record boundary learning
        if collision_occurred:
            self.body_schema['boundaries'].append({
                'position': self.state.position,
                'orientation': self.state.orientation,
                'collision_type': environment_feedback.get('collision_type', 'unknown')
            })

        # Learn affordances (what's possible in different contexts)
        if environment_feedback.get('affordances'):
            context = f"pos_{self.state.position}_orient_{int(self.state.orientation * 4 / np.pi)}"
            self.body_schema['affordances'][context].append(environment_feedback['affordances'])

        # Build sensorimotor mappings
        if environment_feedback.get('sensory_change'):
            mapping = {
                'action': environment_feedback.get('last_action'),
                'sensory_before': environment_feedback.get('sensory_before'),
                'sensory_after': environment_feedback.get('sensory_after'),
                'outcome': environment_feedback.get('outcome')
            }
            context = f"energy_{int(self.state.energy * 10)}"
            self.body_schema['sensorimotor_mappings'][context].append(mapping)


class EmbodiedEnvironment:
    """Rich sensorimotor environment for embodied cognition"""

    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.objects = {}
        self.textures = np.random.random((size, size))  # Visual texture map
        self.resource_cells = set()  # NEW: Track forageable resources
        self.create_rich_environment()

        # Environment dynamics
        self.time_step = 0
        self.day_night_cycle = 0.0
        self.weather = 'clear'

        # Resource respawn queue: list of (respawn_step, obj_type, properties)
        self._respawn_queue = []
        self._respawn_delay = 80  # steps before a consumed resource reappears
        self._forage_respawn_delay = 40

        # Tracking
        self.agent_trail = deque(maxlen=1000)

    def create_rich_environment(self):
        """Create environment with diverse objects and textures"""

        # Walls around perimeter
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Scatter various objects
        objects_to_place = [
            ('food', 0.3, 5),      # Nutritious objects
            ('water', 0.5, 3),     # Hydration sources
            ('shelter', 0.8, 2),   # Safe resting spots
            ('obstacle', 1.0, 8),  # Barriers
            ('tool', 0.2, 4),      # Manipulable objects
            ('social', 0.1, 2),    # Other entities
        ]

        for obj_type, grid_value, count in objects_to_place:
            for _ in range(count):
                while True:
                    x, y = np.random.randint(2, self.size-2, 2)
                    if self.grid[x, y] == 0:  # Empty space
                        self.grid[x, y] = grid_value
                        self.objects[(x, y)] = {
                            'type': obj_type,
                            'properties': self._generate_object_properties(obj_type),
                            'discovered': False,
                            'interaction_count': 0
                        }
                        break

        # NEW: Add forageable resource patches
        for _ in range(8):  # 8 forage patches
            x, y = np.random.randint(2, self.size-2, 2)
            if self.grid[x, y] == 0:
                self.resource_cells.add((x, y))

    def _generate_object_properties(self, obj_type):
        """Generate rich properties for objects"""
        base_properties = {
            'food': {'nutrition': np.random.uniform(0.1, 0.3), 'taste': np.random.choice(['sweet', 'bitter', 'neutral'])},
            'water': {'purity': np.random.uniform(0.8, 1.0), 'temperature': np.random.choice(['cold', 'warm'])},
            'shelter': {'comfort': np.random.uniform(0.6, 1.0), 'size': np.random.choice(['small', 'medium', 'large'])},
            'obstacle': {'hardness': np.random.uniform(0.7, 1.0), 'height': np.random.choice(['low', 'medium', 'high'])},
            'tool': {'utility': np.random.uniform(0.2, 0.8), 'weight': np.random.choice(['light', 'medium', 'heavy'])},
            'social': {'friendliness': np.random.uniform(-0.5, 0.5), 'activity': np.random.choice(['resting', 'moving', 'feeding'])}
        }
        return base_properties.get(obj_type, {})

    def cell_has_resource(self, pos) -> bool:
        """Check if position has forageable resources"""
        return pos in self.resource_cells

    def consume_resource(self, pos) -> bool:
        """Consume resource at position (removes it)"""
        if pos in self.resource_cells:
            self.resource_cells.remove(pos)
            # Schedule respawn at a random empty cell
            self._respawn_queue.append(
                (self.time_step + self._forage_respawn_delay, 'forage', None))
            return True
        return False

    def _consume_object(self, pos):
        """Remove an object and schedule it to respawn elsewhere."""
        if pos not in self.objects:
            return
        obj = self.objects.pop(pos)
        self.grid[pos[0], pos[1]] = 0.0  # clear grid cell
        self._respawn_queue.append(
            (self.time_step + self._respawn_delay,
             obj['type'], obj['properties']))

    def _tick_respawns(self):
        """Check if any consumed resources should reappear."""
        still_pending = []
        for entry in self._respawn_queue:
            respawn_step, obj_type, props = entry
            if self.time_step < respawn_step:
                still_pending.append(entry)
                continue
            # Find a random empty interior cell
            for _ in range(50):
                x = np.random.randint(2, self.size - 2)
                y = np.random.randint(2, self.size - 2)
                if self.grid[x, y] == 0 and (x, y) not in self.objects:
                    if obj_type == 'forage':
                        self.resource_cells.add((x, y))
                    else:
                        grid_val = {'food': 0.3, 'water': 0.5,
                                    'shelter': 0.8}.get(obj_type, 0.2)
                        self.grid[x, y] = grid_val
                        new_props = props if props else self._generate_object_properties(obj_type)
                        self.objects[(x, y)] = {
                            'type': obj_type,
                            'properties': new_props,
                            'discovered': False,
                            'interaction_count': 0,
                        }
                    break
        self._respawn_queue = still_pending

    def get_visual_field(self, body, context_filter=None):
        """Generate visual perception based on body position and context"""
        x, y = body.state.position
        vision_range = body.vision_range

        # Base visual field
        vision_field = np.zeros((2*vision_range+1, 2*vision_range+1, 3))  # RGB channels

        for i in range(-vision_range, vision_range+1):
            for j in range(-vision_range, vision_range+1):
                world_x, world_y = x + i, y + j

                if 0 <= world_x < self.size and 0 <= world_y < self.size:
                    # Distance-based acuity
                    distance = np.sqrt(i*i + j*j)
                    acuity = max(0.1, 1.0 - distance / vision_range)

                    # Base color from grid value
                    grid_val = self.grid[world_x, world_y]
                    texture_val = self.textures[world_x, world_y]

                    # Apply context-dependent filtering
                    if context_filter == 'food_seeking':
                        # Enhance food-related features
                        if (world_x, world_y) in self.objects and self.objects[(world_x, world_y)]['type'] == 'food':
                            grid_val *= 1.5
                    elif context_filter == 'safety_seeking':
                        # Enhance shelter and highlight obstacles
                        if (world_x, world_y) in self.objects:
                            obj_type = self.objects[(world_x, world_y)]['type']
                            if obj_type == 'shelter':
                                grid_val *= 1.3
                            elif obj_type == 'obstacle':
                                grid_val *= 1.2

                    # RGB encoding
                    vision_field[i+vision_range, j+vision_range, 0] = grid_val * acuity
                    vision_field[i+vision_range, j+vision_range, 1] = texture_val * acuity
                    vision_field[i+vision_range, j+vision_range, 2] = (grid_val + texture_val) * 0.5 * acuity

        # Add environmental effects
        lighting = 0.5 + 0.5 * np.sin(self.day_night_cycle)  # Day/night cycle
        vision_field *= lighting

        return vision_field

    def step(self, body, action_name, action_intensity=1.0):
        """Execute embodied step in environment"""

        old_position = body.state.position
        sensory_before = self.get_visual_field(body).flatten()

        # Execute action
        movement = body.execute_action(action_name, action_intensity)

        # Apply movement to environment (round so intensity 0.6-1.0 still moves 1 cell)
        new_x = round(old_position[0] + movement[0])
        new_y = round(old_position[1] + movement[1])

        # Collision detection with bounds safety
        in_bounds = (0 <= new_x < self.size) and (0 <= new_y < self.size)
        collision_occurred = False
        environment_feedback = {
            'sensory_before': sensory_before,
            'last_action': action_name
        }

        if in_bounds and (self.grid[new_x, new_y] == 0 or self.grid[new_x, new_y] < 0.4):
            # Valid movement
            body.state.position = (new_x, new_y)
            self.agent_trail.append((new_x, new_y))
        else:
            # Collision (out-of-bounds or blocked cell)
            collision_occurred = True
            if not in_bounds:
                environment_feedback['collision_type'] = 'wall'
            else:
                environment_feedback['collision_type'] = 'wall' if self.grid[new_x, new_y] >= 0.8 else 'object'

        # Current position after any movement/collision resolution
        current_pos = body.state.position

        # Forage action handling: use global CONFIG, not a fresh QSEConfig()
        if action_name == 'forage':
            if self.cell_has_resource(current_pos):
                if self.consume_resource(current_pos):
                    forage_min = getattr(CONFIG, 'ENERGY_FORAGE_REWARD_MIN', 0.08)
                    forage_max = getattr(CONFIG, 'ENERGY_FORAGE_REWARD_MAX', 0.16)
                    energy_gain = float(np.random.uniform(forage_min, forage_max))
                    body.state.energy = min(1.0, body.state.energy + energy_gain)
                    environment_feedback['outcome'] = f"foraged resources (+{energy_gain:.2f} energy)"
                    environment_feedback['consumed'] = True
                    environment_feedback['reward'] = 0.1
                else:
                    environment_feedback['outcome'] = "no resources to forage"
            else:
                environment_feedback['outcome'] = "no resources here"

        # Object interactions - FIXED to be more responsive
        if current_pos in self.objects:
            obj = self.objects[current_pos]
            obj['discovered'] = True
            obj['interaction_count'] += 1

            # Object-specific interactions
            if obj['type'] == 'food':
                if action_name == 'examine':
                    nutrition = obj['properties']['nutrition']
                    body.state.energy = min(1.0, body.state.energy + nutrition)
                    environment_feedback['outcome'] = f"consumed {obj['properties']['taste']} food (+{nutrition:.2f} energy)"
                    self._consume_object(current_pos)  # food is finite
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['taste']} food"

            elif obj['type'] == 'water':
                if action_name == 'examine':
                    body.state.health = min(1.0, body.state.health + 0.1)
                    body.state.energy = min(1.0, body.state.energy + 0.05)
                    environment_feedback['outcome'] = f"drank {obj['properties']['temperature']} water"
                    self._consume_object(current_pos)  # water is finite
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['temperature']} water"

            elif obj['type'] == 'shelter':
                if action_name in ['rest', 'examine']:
                    comfort = obj['properties']['comfort']
                    body.state.energy = min(1.0, body.state.energy + comfort * 0.3)
                    environment_feedback['outcome'] = f"rested in {obj['properties']['size']} shelter (+{comfort*0.3:.2f} energy)"
                else:
                    environment_feedback['outcome'] = f"found {obj['properties']['size']} shelter"

            elif obj['type'] == 'tool':
                environment_feedback['outcome'] = f"found {obj['properties']['weight']} tool"

            elif obj['type'] == 'social':
                environment_feedback['outcome'] = f"encountered {obj['properties']['activity']} entity"

        # ALSO check nearby objects (within 1 step)
        elif action_name == 'examine':
            x, y = current_pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nearby_pos = (x + dx, y + dy)
                    if nearby_pos in self.objects and not self.objects[nearby_pos]['discovered']:
                        obj = self.objects[nearby_pos]
                        obj['discovered'] = True
                        environment_feedback['outcome'] = f"spotted {obj['type']} nearby"
                        break

        # Update body and environment
        body.update_proprioception()
        sensory_after = self.get_visual_field(body).flatten()
        environment_feedback['sensory_after'] = sensory_after

        # FIX: Add sensory change detection
        sensory_change_magnitude = np.linalg.norm(sensory_after - sensory_before)
        environment_feedback['sensory_change'] = sensory_change_magnitude > 0.1  # Threshold for significant change

        # Learn affordances
        available_actions = self._get_available_actions(body)
        environment_feedback['affordances'] = available_actions

        body.update_body_schema(collision_occurred, environment_feedback)

        # Environment dynamics
        self.time_step += 1
        self.day_night_cycle += 0.1
        self._tick_respawns()

        return environment_feedback

    def _get_available_actions(self, body):
        """Determine what actions are possible from current position"""
        affordances = []
        x, y = body.state.position

        # Movement affordances
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                self.grid[new_x, new_y] < 0.5):
                affordances.append(f"move_to_{dx}_{dy}")

        # Object affordances
        if (x, y) in self.objects:
            obj_type = self.objects[(x, y)]['type']
            if obj_type == 'food':
                affordances.append('consume')
            elif obj_type == 'water':
                affordances.append('drink')
            elif obj_type == 'shelter':
                affordances.append('rest_sheltered')
            elif obj_type == 'tool':
                affordances.append('manipulate')

        return affordances


class EmbodiedQSEAgent(EmileAgent):
    """QSE-Émile agent with embodied cognition capabilities"""
    def __init__(self, config=QSEConfig()):
        super().__init__(config)
        self._prev_sigma_mean: float | None = None
        self._prev_S_mean: float | None = None
        self._taup_state = {"ema_dsigma": 0.0}
        self._tau_prime_prev: float = float(getattr(self.cfg, "TAU_PRIME_MAX", 1.0))
        # NEW: keep last sigma array for τc; and phasic dwell state
        self._prev_sigma_arr = None
        self._strong_rupture_dwell_count = 0
        self._strong_rupture_state = False
        # Embodied components
        self.body = SensoriMotorBody()
        self.body.cfg = self.cfg  # propagate active config to body
        self.perceptual_categories = defaultdict(list)
        self.embodied_memories = deque(maxlen=1000)

        # Override base-class goal module with TD(λ) learner
        from emile_mini.goal_v2 import GoalModuleV2
        self.goal = GoalModuleV2(config)

        # Learned goal mapper (Layer 2: strategy -> embodied goal)
        self._goal_mapper = None
        if getattr(config, 'GOAL_MAPPER_LEARNED', True):
            from emile_mini.goal_mapper import GoalMapper
            self._goal_mapper = GoalMapper()

        # Novelty tracking for dense reward
        self._visited_positions = set()
        self._prev_energy = getattr(self.body.state, 'energy', 1.0)

        # Optional learned action policy (gated by config)
        self._action_policy = None
        self._teacher_prob = 1.0  # start fully heuristic
        self._learned_reward_ema = 0.0
        self._heuristic_reward_ema = 0.0
        self._learned_samples = 0  # count of learned-policy actions taken
        self._used_learned = False  # which chose this step's action
        if getattr(config, 'LEARNED_POLICY_ENABLED', False):
            from emile_mini.action_policy import ActionPolicy
            self._action_policy = ActionPolicy(
                hidden_dim=getattr(config, 'POLICY_HIDDEN_DIM', 32),
                lr=getattr(config, 'POLICY_LR', 0.001),
                gamma=getattr(config, 'POLICY_GAMMA', 0.99),
            )

    def receive_memory_cue(self, cue: dict) -> bool:
        """Process navigation and other memory cues."""
        try:
            ctype = cue.get("type")
            if ctype == "navigation_cue":
                # Persist target so action selection can choose to bias behavior (optional)
                self._nav_target_quadrant = cue.get("target_quadrant", None)
                # Store cue for traceability
                if hasattr(self, "memory"):
                    self.memory.store(cue, tags={"type": "navigation_cue"})
                # Map to an existing goal (keeps it simple)
                self.goal.current_goal = "explore_space"
                return True
            return False
        except Exception:
            # Keep runner resilient even if a cue is malformed
            return False

    # ------------------------------------------------------------------
    # Strategy → embodied goal mapping
    # ------------------------------------------------------------------
    def _strategy_to_embodied_goal(self, strategy: str) -> str:
        """Map GoalModuleV2 abstract strategy to a concrete embodied goal
        using current body state as context."""
        if strategy == 'rest':
            return 'rest_recover'
        if strategy == 'social':
            return 'social_interact'
        if strategy == 'learn':
            return 'categorize_experience'
        if strategy == 'explore':
            return 'explore_space'
        # 'exploit' -- choose based on needs
        energy = self.body.state.energy
        health = self.body.state.health
        if energy < 0.4:
            return 'seek_food'
        if health < 0.5:
            return 'find_shelter'
        return 'manipulate_objects'

    def _largest_cluster_1d(self, mask: np.ndarray) -> int:
        """Size of largest contiguous True run in a 1D mask."""
        if mask.size == 0 or not np.any(mask):
            return 0
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                if cur > best: best = cur
            else:
                cur = 0
        return best

    def embodied_step(self, environment, dt=0.01):
        """
        Complete embodied cognitive step with FIXED memory storage
        """

        # 1. Perception: Get sensory information
        context_id = self.context.get_current()
        context_filter = self._get_perceptual_filter(context_id)
        visual_field = environment.get_visual_field(self.body, context_filter)

        # 2. Symbolic reasoning
        mm = None
        if getattr(self.cfg, 'MULTIMODAL_ENABLED', False) and ModalityFeature is not None and getattr(self, 'mm_image', None) is not None:
            w = self._dynamic_weights() if hasattr(self, '_dynamic_weights') else {}
            feats = [
                ModalityFeature('vision', self.mm_image.encode(visual_field), w.get('vision', 1.0)),
                ModalityFeature('proprio', np.array([
                    self.body.state.position[0], self.body.state.position[1],
                    self.body.state.orientation, self.body.state.energy
                ], dtype=float), w.get('proprio', 1.0))
            ]
            mm = feats
        sigma = self.symbolic.step(self.qse.S, modality_features=mm)

        # NEW: compute τc from sigma used to drive the core
        tau_current_now = calculate_emergent_time(sigma, self._prev_sigma_arr, self.cfg)
        self._prev_sigma_arr = np.copy(sigma)

        # NEW: rupture metrics from current sigma
        thr = float(self.cfg.S_THETA_RUPTURE)
        rupture_mask = np.abs(sigma) > thr
        rupture_fraction = float(np.mean(rupture_mask))
        tonic_rupture_active = bool(rupture_fraction > 0.0)

        # Make phasic selective: spatial threshold + dwell
        MIN_FRAC_STRONG = 0.20  # tighter
        DWELL_STRONG = 3        # longer
        strong_now = rupture_fraction >= MIN_FRAC_STRONG
        if strong_now:
            self._strong_rupture_dwell_count += 1
            if self._strong_rupture_dwell_count >= DWELL_STRONG:
                self._strong_rupture_state = True
        else:
            self._strong_rupture_dwell_count = max(0, self._strong_rupture_dwell_count - 1)
            if self._strong_rupture_dwell_count == 0:
                self._strong_rupture_state = False

        phasic_rupture_active = bool(self._strong_rupture_state)
        largest_cluster_size = int(self._largest_cluster_1d(rupture_mask))
        total_ruptured_cells = int(np.sum(rupture_mask))

        # Use τ′ from previous step to modulate effective dt
        dt_eff = float(dt) * float(self._tau_prime_prev)

        # 3. QSE Processing with emergent time coupling
        cognitive_metrics = self.qse.step(sigma, dt_eff)

        # Extract fields for τ′ update
        sigma_mean = float(cognitive_metrics.get('sigma_mean', 0.0))
        S_mean = float(cognitive_metrics.get('surplus_mean', 0.0))
        phasic = int(phasic_rupture_active)  # use our phasic flag

        # Emergent Δσ and τ′ for the NEXT step
        delta_sigma_eff, self._taup_state = compute_delta_sigma_emergent(
            sigma_t=sigma_mean,
            sigma_prev=self._prev_sigma_mean,
            S_t=S_mean,
            S_prev=self._prev_S_mean,
            phasic_active=phasic,
            cfg=self.cfg,
            state=self._taup_state,
        )
        tau_prime_now = calculate_tau_prime(delta_sigma_eff, self.cfg)
        self._prev_sigma_mean = sigma_mean
        self._prev_S_mean = S_mean
        self._tau_prime_prev = float(tau_prime_now)

        # Make τ′ and dt_eff visible; attach rupture metrics
        cognitive_metrics["tau_prime"] = float(tau_prime_now)
        cognitive_metrics["dt_eff"] = float(dt_eff)
        cognitive_metrics["tonic_rupture_active"] = tonic_rupture_active
        cognitive_metrics["phasic_rupture_active"] = phasic_rupture_active
        cognitive_metrics["largest_cluster_size"] = largest_cluster_size
        cognitive_metrics["total_ruptured_cells"] = total_ruptured_cells


        # 4. Context management with QSE feedback
        distinction_level = abs(
            self.symbolic.get_sigma_ema() if hasattr(self.symbolic, 'get_sigma_ema')
            else cognitive_metrics.get('sigma_mean', 0.0)
        )
        old_context = self.context.get_current()
        self.context.update({'distinction_level': distinction_level})
        new_context = self.context.get_current()

        # 5. Goal selection via TD(λ) + strategy-to-goal mapping
        entropy = float(cognitive_metrics.get('normalized_entropy', 0.5))
        strategy = self.goal.select_goal(
            new_context, self.body.state.energy, entropy, cognitive_metrics)

        # Layer 2: map strategy to embodied goal (learned or hardcoded)
        if self._goal_mapper is not None:
            food_nearby = self._detect_object_type(visual_field, 'food')
            shelter_nearby = self._detect_object_type(visual_field, 'shelter')
            embodied_goal = self._goal_mapper.select_goal(
                strategy, self.body.state.energy, self.body.state.health,
                food_nearby, shelter_nearby, entropy)
        else:
            embodied_goal = self._strategy_to_embodied_goal(strategy)

        # 6. Action selection (learned policy or heuristic)
        self._prev_energy = float(self.body.state.energy)
        goal_idx = self.goal.current_action if self.goal.current_action is not None else 0
        self._used_learned = (self._action_policy is not None
                              and np.random.random() > self._teacher_prob)
        if self._used_learned:
            action_name, action_intensity, _ = self._action_policy.select_action(
                visual_field, self.body.state, cognitive_metrics, goal_idx,
                entropy=entropy)
        else:
            action_name, action_intensity = self._goal_to_embodied_action(
                embodied_goal, visual_field, environment)

        # 7. Execute action in environment
        environment_feedback = environment.step(self.body, action_name, action_intensity)

        # 8. Learn from embodied experience
        self._update_embodied_learning(visual_field, environment_feedback, cognitive_metrics)

        # 9. Dense reward + TD(λ) updates for Layer 1 and Layer 2
        reward = self._calculate_embodied_reward(environment_feedback, embodied_goal)
        self.goal.update(reward, new_context, self.body.state.energy, done=False)
        if self._goal_mapper is not None:
            food_now = self._detect_object_type(
                environment.get_visual_field(self.body), 'food')
            shelter_now = self._detect_object_type(
                environment.get_visual_field(self.body), 'shelter')
            self._goal_mapper.update(
                reward, strategy, self.body.state.energy,
                self.body.state.health, food_now, shelter_now)

        # 10. Learned-policy bookkeeping (if enabled)
        if self._action_policy is not None:
            self._action_policy.record_reward(reward)
            update_interval = getattr(self.cfg, 'POLICY_UPDATE_INTERVAL', 50)
            if not hasattr(self, '_policy_step_count'):
                self._policy_step_count = 0
            self._policy_step_count += 1
            if self._policy_step_count % update_interval == 0:
                self._action_policy.update()

            # Performance-gated teacher decay with warmup
            ema_alpha = 0.02
            if self._used_learned:
                self._learned_reward_ema = ((1 - ema_alpha) * self._learned_reward_ema
                                            + ema_alpha * reward)
                self._learned_samples += 1
            else:
                self._heuristic_reward_ema = ((1 - ema_alpha) * self._heuristic_reward_ema
                                              + ema_alpha * reward)
            decay = getattr(self.cfg, 'POLICY_TEACHER_DECAY', 0.998)
            floor = getattr(self.cfg, 'POLICY_TEACHER_MIN', 0.1)
            warmup = 200  # unconditional decay until learned policy has enough samples
            if self._learned_samples < warmup:
                # Warmup: always decay so learned policy gets a chance to act
                self._teacher_prob = max(floor, self._teacher_prob * decay)
            elif self._learned_reward_ema >= self._heuristic_reward_ema - 0.05:
                self._teacher_prob = max(floor, self._teacher_prob * decay)
            else:
                # Learned policy underperforming — nudge back toward heuristic
                self._teacher_prob = min(1.0, self._teacher_prob / decay)
        current_goal = embodied_goal  # for memory entry below

        # 11. Structured episodic memory
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1

        memory_entry = {
            'step': self.step_counter,
            'position': self.body.state.position,
            'energy': float(self.body.state.energy),
            'context': int(new_context),
            'strategy': strategy,
            'goal': current_goal,
            'action': action_name,
            'reward': float(reward),
            'surplus': float(cognitive_metrics.get('surplus_mean', 0)),
            'sigma_mean_raw': float(sigma_mean),
            'sigma_mean_ema': float(self.symbolic.get_sigma_ema()),
            'context_switched': new_context != old_context,
            'embodied_outcome': environment_feedback.get('outcome', 'none')
        }
        self.memory.store(memory_entry, tags={'type': 'episodic', 'embodied': True})

        # 12. history and return
        return {
            'cognitive_metrics': cognitive_metrics,
            'action': action_name,
            'action_intensity': action_intensity,
            'environment_feedback': environment_feedback,
            'reward': reward,
            'body_state': self.body.state,
            'context': new_context,
            'context_switched': new_context != old_context,
            'distinction_level': abs(self.symbolic.get_sigma_ema()),

            # analysis-friendly flat fields
            'qse_influence': float(abs(self.symbolic.get_sigma_ema())),
            'q_value_change': float(reward),
            'decision_events': 1 if new_context != old_context else 0,
            'tau_current': float(tau_current_now),
            'tau_prime': float(tau_prime_now),
            'sigma_mean': float(sigma_mean),
            'context_changed': int(new_context != old_context),
            'strategy': str(strategy),
            'goal': str(current_goal),
        }

    def _get_perceptual_filter(self, context_id):
        """Context-dependent perceptual filtering"""
        context_filters = {
            0: None,  # Default perception
            1: 'food_seeking',  # Enhance food-related features
            2: 'safety_seeking',  # Focus on shelter/obstacles
            3: 'exploration',  # Enhance novel features
            4: 'social',  # Focus on other entities
        }
        return context_filters.get(context_id, None)

    def _integrate_sensorimotor_input(self, visual_field, proprioception):
        """Integrate sensory input into QSE-compatible format"""

        # Flatten and normalize visual input
        visual_flat = visual_field.flatten()
        visual_normalized = (visual_flat - visual_flat.mean()) / (visual_flat.std() + 1e-8)

        # Combine with proprioception
        proprioception_normalized = (proprioception - 0.5) * 2  # Center around 0

        # Create integrated sensorimotor vector
        sensorimotor_vector = np.concatenate([
            visual_normalized[:32],  # Sample of visual input
            proprioception_normalized
        ])

        return sensorimotor_vector

    def _goal_to_embodied_action(self, goal, visual_field, environment=None):
        """
        Enhanced embodied action selection with forage mechanics and curiosity fix.
        Converts cognitive goals into embodied actions with energy management.
        """
        import numpy as np

        # --- CURIOSITY (softened: only when examining is goal-relevant) ---
        vision_range = self.body.vision_range
        center_vision = visual_field[vision_range, vision_range, :]
        if (goal in ('seek_food', 'find_energy_source', 'manipulate_objects',
                     'categorize_experience')
                and np.random.random() < 0.5
                and np.mean(center_vision) > 0.6):
            return 'examine', 1.0
        # --- END CURIOSITY ---

        # ENHANCED ENERGY MANAGEMENT with forage mechanics
        if self.body.state.energy < 0.25:
            if goal in ["rest_recover", "seek_food", "conserve_energy"]:
                if np.random.random() < 0.6:
                    return 'forage', 1.0
                else:
                    return 'rest', 1.0
            elif goal in ["seek_food", "find_energy_source"]:
                return 'forage', 1.0
            else:
                if np.random.random() < 0.5:
                    return 'forage', 1.0
                else:
                    return 'rest', 1.0

        # BASE GOAL LOGIC with forage integration
        if goal == "explore_space":
            rand = np.random.random()
            if rand < 0.2:
                return 'forage', 0.8
            elif rand < 0.5:
                return 'examine', 1.0
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.8

        elif goal in ["seek_food", "find_energy_source"]:
            if self._detect_object_type(visual_field, 'food'):
                return 'examine', 1.0
            else:
                rand = np.random.random()
                if rand < 0.3:
                    return 'forage', 1.0
                elif rand < 0.7:
                    return 'examine', 1.0
                else:
                    action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                    return action, 0.6

        elif goal == "find_shelter":
            if self._detect_object_type(visual_field, 'shelter'):
                return 'examine', 1.0
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.5

        elif goal in ["rest_recover", "conserve_energy"]:
            if np.random.random() < 0.7:
                return 'rest', 1.0
            else:
                return 'forage', 1.0

        elif goal == "manipulate_objects":
            return 'examine', 1.0

        elif goal == "seek_nourishment":  # Social agent goal
            if self._detect_object_type(visual_field, 'food'):
                return 'examine', 1.0
            else:
                if np.random.random() < 0.4:
                    return 'forage', 1.0
                else:
                    action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                    return action, 0.7

        elif goal == "test_unknown":  # Social agent goal
            rand = np.random.random()
            if rand < 0.4:
                return 'examine', 1.0
            elif rand < 0.6:
                return 'forage', 0.8
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.6

        else:
            rand = np.random.random()
            if rand < 0.3:
                return 'examine', 0.8
            elif rand < 0.4:
                return 'forage', 0.6
            else:
                action = np.random.choice(['move_forward', 'turn_left', 'turn_right'])
                return action, 0.7

    def _detect_object_type(self, visual_field, object_type):
        """FIXED: Better object detection in visual field"""

        # More sensitive detection thresholds
        threshold_map = {
            'food': 0.25,      # Lower thresholds
            'water': 0.45,
            'shelter': 0.75,
            'obstacle': 0.9
        }

        threshold = threshold_map.get(object_type, 0.5)

        # Check multiple visual channels
        detected = (np.any(visual_field[:, :, 0] > threshold) or
                   np.any(visual_field[:, :, 1] > threshold) or
                   np.any(visual_field[:, :, 2] > threshold))

        return detected

    def _update_embodied_learning(self, visual_field, environment_feedback, cognitive_metrics):
        """Learn from embodied experience"""

        # Create embodied memory entry
        memory_entry = {
            'timestamp': len(self.embodied_memories),
            'visual_snapshot': visual_field.copy(),
            'body_state': self.body.state,
            'context': self.context.get_current(),
            'goal': self.goal.current_goal,
            'environment_feedback': environment_feedback,
            'surplus_mean': cognitive_metrics.get('surplus_mean', 0),
            'sigma_mean': cognitive_metrics.get('sigma_mean', 0)
        }

        self.embodied_memories.append(memory_entry)

        # Update perceptual categories based on experience
        if environment_feedback.get('outcome'):
            outcome = environment_feedback['outcome']
            visual_signature = np.mean(visual_field, axis=(0, 1))  # Simple visual signature

            self.perceptual_categories[outcome].append(visual_signature)

        # Store in regular memory system too
        self.memory.store(
            {
                'embodied_experience': memory_entry,
                'outcome': environment_feedback.get('outcome', 'exploration')
            },
            tags={'type': 'episodic', 'embodied': True}
        )

    def _calculate_embodied_reward(self, environment_feedback, current_goal):
        """Dense reward combining survival, novelty, goal completion, and energy."""
        reward = 0.0
        outcome = environment_feedback.get('outcome', '') or ''

        # 1. Survival: small tick reward for each step alive
        reward += 0.01

        # 2. Energy maintenance — continuous, steep penalty below 0.3
        energy = self.body.state.energy
        if energy > 0.6:
            reward += 0.05
        elif energy < 0.15:
            reward -= 0.5          # near-death: dominates all other signals
        elif energy < 0.3:
            reward -= 0.3 * (0.3 - energy) / 0.15   # scales 0 → -0.3

        # 3. Energy change bonus
        energy_delta = energy - getattr(self, '_prev_energy', energy)
        if energy_delta > 0:
            reward += 0.5 * energy_delta

        # 4. Food/resource consumption — always rewarded, bigger when hungry
        if 'consumed' in outcome:
            hunger_bonus = max(0.0, 1.0 - energy)  # 0 at full, 1.0 at empty
            if current_goal in ('seek_food', 'find_energy_source'):
                reward += 1.5 + hunger_bonus
            else:
                reward += 1.0 + hunger_bonus
        elif current_goal == 'find_shelter' and 'rested' in outcome:
            reward += 0.6
        elif current_goal == 'rest_recover' and 'rested' in outcome:
            reward += 0.3
        elif current_goal == 'explore_space':
            reward += 0.1

        # 5. Novelty: visiting a new grid cell
        pos = self.body.state.position
        pos_key = (int(round(pos[0])), int(round(pos[1])))
        if pos_key not in self._visited_positions:
            reward += 0.15
            self._visited_positions.add(pos_key)

        # 6. Forage success
        if 'foraged' in outcome:
            reward += 0.6

        # 7. Collision penalty
        if environment_feedback.get('collision_type'):
            reward -= 0.15

        return reward


def run_embodied_experiment(steps=1000, visualize=True, output_dir: Optional[Path] = None,
                            save_jsonl: bool = True, seed: Optional[int] = None) -> dict:
    """Run embodied QSE-Émile experiment with FIXED parameters"""

    print("🤖 EMBODIED QSE-ÉMILE EXPERIMENT (FIXED VERSION)")
    print("=" * 50)
    print("Simulating embodied cognition in sensorimotor grid world")
    print("FIXES: Lower energy costs, better object detection, context switching enabled")

    # Seeding for reproducibility
    if seed is not None:
        np.random.seed(int(seed))

    # Create environment and agent
    environment = EmbodiedEnvironment(size=20)
    agent = EmbodiedQSEAgent()

    print(f"Environment created with {len(environment.objects)} objects")
    # Print the actual recontext thresholds in use
    print(f"Context thresholds: hi={agent.cfg.RECONTEXT_THRESHOLD}, "
          f"lo={agent.cfg.RECONTEXT_THRESHOLD - agent.cfg.RECONTEXT_HYSTERESIS}")

    # Output directory
    if output_dir is None:
        output_dir = Path("runs") / f"embodied_emile_{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL writer
    jsonl_writer = None
    if save_jsonl:
        if JSONLWriter is None:
            print("[warn] JSONLWriter not available; skipping JSONL logging.")
        else:
            jsonl_writer = JSONLWriter(output_dir / "qse_agent_dynamics_embodied_emile.jsonl")

    # Tracking
    trajectory = []
    context_switches = []
    object_discoveries = []
    embodied_metrics = {
        'energy_over_time': [],
        'health_over_time': [],
        'context_over_time': [],
        'goal_over_time': [],
        'surplus_over_time': [],
        'sigma_over_time': []
    }

    print(f"Running {steps} embodied steps...")

    dt = 0.01
    prev_reward_val: Optional[float] = None

    for step in range(steps):
        if step % 200 == 0:  # Less frequent updates for longer runs
            print(f"  Step {step}/{steps} - Energy: {agent.body.state.energy:.3f}, Discoveries: {len(object_discoveries)}")

        # Record pre-step state
        old_context = agent.context.get_current()

        # Execute embodied step
        result = agent.embodied_step(environment, dt=dt)

        reward_val = float(result.get("reward", 0.0))
        if prev_reward_val is None:
            qvc_proxy = 0.0
        else:
            qvc_proxy = float(abs(reward_val - prev_reward_val))
        prev_reward_val = reward_val

        # Track trajectory and metrics
        trajectory.append(agent.body.state.position)
        embodied_metrics['energy_over_time'].append(agent.body.state.energy)
        embodied_metrics['health_over_time'].append(agent.body.state.health)
        embodied_metrics['context_over_time'].append(result['context'])
        embodied_metrics['goal_over_time'].append(result.get('goal', None))
        embodied_metrics['surplus_over_time'].append(result['cognitive_metrics'].get('surplus_mean', 0))
        embodied_metrics['sigma_over_time'].append(result['cognitive_metrics'].get('sigma_mean', 0))

        # Track context switches
        if result['context'] != old_context:
            context_switches.append({
                'step': step,
                'position': agent.body.state.position,
                'old_context': old_context,
                'new_context': result['context'],
                'goal': result.get('goal', agent.goal.current_goal),
                'body_energy': agent.body.state.energy
            })

        # Track object discoveries
        if result['environment_feedback'].get('outcome'):
            object_discoveries.append({
                'step': step,
                'position': agent.body.state.position,
                'outcome': result['environment_feedback']['outcome'],
                'context': result['context']
            })

        # JSONL logging (per step)
        if jsonl_writer is not None:
            cm = result['cognitive_metrics']
            pos_x, pos_y = agent.body.state.position
            jsonl_writer.write({
                "step": int(step),
                "timestamp": float(step * dt),
                "dt": float(dt),

                # QSE dynamics
                "tau_current": float(result.get("tau_current", cm.get("tau_current", 0.01))),
                "tau_prime": float(result.get("tau_prime", cm.get("tau_prime", 1.0))),
                "dt_eff": float(cm.get("dt_eff", dt)),
                "surplus_mean": float(cm.get("surplus_mean", 0.0)),
                "sigma_mean": float(cm.get("sigma_mean", 0.0)),

                # NEW: rupture fields to power rupture-only analysis
                "tonic_rupture_active": bool(cm.get("tonic_rupture_active", False)),
                "phasic_rupture_active": bool(cm.get("phasic_rupture_active", False)),
                "largest_cluster_size": int(cm.get("largest_cluster_size", 0)),
                "total_ruptured_cells": int(cm.get("total_ruptured_cells", 0)),

                # Decision/context
                "decision_triggered": int(result.get("decision_events", 0)),
                "context": int(result.get("context", 0)),
                "context_changed": int(result.get("context_changed", int(result.get("context", 0) != 0))),

                # Learning/change
                "q_value_change_proxy": float(qvc_proxy),
                "reward": reward_val,

                # Behavior state
                "action": str(result.get("action", "")),
                "position_x": float(pos_x),
                "position_y": float(pos_y),
                "orientation": float(agent.body.state.orientation),
                "energy": float(agent.body.state.energy),
                "health": float(agent.body.state.health),
                "regime": "embodied",
                "goal": str(result.get("goal", "")) if result.get("goal", None) is not None else "",
                "outcome": str(result['environment_feedback'].get('outcome', '')),
            })

    print(f"Experiment complete!")
    print(f"Context switches: {len(context_switches)}")
    print(f"Object discoveries: {len(object_discoveries)}")
    print(f"Final energy: {agent.body.state.energy:.3f}")

    # Close writer
    if jsonl_writer is not None:
        jsonl_writer.close()

    # Analysis and visualization
    if visualize:
        create_embodied_visualization(environment, agent, trajectory, context_switches,
                                      object_discoveries, embodied_metrics, output_dir=output_dir)

    results = {
        'environment': environment,
        'agent': agent,
        'trajectory': trajectory,
        'context_switches': context_switches,
        'object_discoveries': object_discoveries,
        'embodied_metrics': embodied_metrics,
        'total_steps': steps,
        'output_dir': str(output_dir),
        'jsonl_path': str(output_dir / "qse_agent_dynamics_embodied_emile.jsonl") if save_jsonl else None,
    }

    print_embodied_analysis(results)

    return results


def create_embodied_visualization(environment, agent, trajectory, context_switches,
                                  object_discoveries, embodied_metrics, output_dir: Optional[Path] = None):
    """Create comprehensive visualization of embodied cognition"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Embodied QSE-Émile: Sensorimotor Cognition', fontsize=16, fontweight='bold')

    # 1. Environment map with trajectory
    ax1 = axes[0, 0]

    # Show environment
    im1 = ax1.imshow(environment.grid, cmap='terrain', alpha=0.7)

    # Plot trajectory
    if trajectory:
        traj_x = [pos[1] for pos in trajectory]
        traj_y = [pos[0] for pos in trajectory]
        ax1.plot(traj_x, traj_y, 'r-', alpha=0.6, linewidth=2, label='Trajectory')

    # Mark context switches
    if context_switches:
        switch_x = [sw['position'][1] for sw in context_switches]
        switch_y = [sw['position'][0] for sw in context_switches]
        ax1.scatter(switch_x, switch_y, c='purple', s=50, marker='^',
                   label=f'Context Switches ({len(context_switches)})', alpha=0.8)

    # Mark object discoveries
    if object_discoveries:
        disc_x = [obj['position'][1] for obj in object_discoveries]
        disc_y = [obj['position'][0] for obj in object_discoveries]
        ax1.scatter(disc_x, disc_y, c='gold', s=80, marker='*',
                   label=f'Discoveries ({len(object_discoveries)})', alpha=0.9)

    # Mark final position
    final_pos = agent.body.state.position
    ax1.scatter(final_pos[1], final_pos[0], c='red', s=100, marker='X',
               label='Final Position', edgecolors='white', linewidth=2)

    ax1.set_title('Embodied Exploration & Discovery')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Body state over time
    ax2 = axes[0, 1]
    steps_arr = range(len(embodied_metrics['energy_over_time']))

    ax2.plot(steps_arr, embodied_metrics['energy_over_time'], 'g-', label='Energy', linewidth=2)
    ax2.plot(steps_arr, embodied_metrics['health_over_time'], 'b-', label='Health', linewidth=2)

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Body State (0-1)')
    ax2.set_title('Body State Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. QSE dynamics in embodied context
    ax3 = axes[0, 2]

    ax3.plot(steps_arr, embodied_metrics['surplus_over_time'], 'r-', label='Surplus', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps_arr, embodied_metrics['sigma_over_time'], 'orange', label='Sigma', alpha=0.7)

    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Surplus', color='red')
    ax3_twin.set_ylabel('Sigma', color='orange')
    ax3.set_title('QSE Dynamics During Embodied Experience')
    ax3.grid(True, alpha=0.3)

    # 4. Context switches over time
    ax4 = axes[1, 0]

    if context_switches:
        switch_steps = [sw['step'] for sw in context_switches]
        switch_contexts = [sw['new_context'] for sw in context_switches]

        ax4.scatter(switch_steps, switch_contexts, c='purple', s=60, alpha=0.7)
        ax4.plot(steps_arr, embodied_metrics['context_over_time'], 'k-', alpha=0.5, linewidth=1)

    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Context ID')
    ax4.set_title('Context Evolution During Embodied Experience')
    ax4.grid(True, alpha=0.3)

    # 5. Discovery timeline
    ax5 = axes[1, 1]

    if object_discoveries:
        discovery_types = [obj['outcome'] for obj in object_discoveries]
        discovery_steps = [obj['step'] for obj in object_discoveries]

        # Count discoveries by type
        from collections import Counter
        disc_counts = Counter(discovery_types)

        # Plot discovery events
        for i, (discovery_type, count) in enumerate(disc_counts.items()):
            type_steps = [step for step, outcome in zip(discovery_steps, discovery_types)
                         if outcome == discovery_type]
            ax5.scatter(type_steps, [i] * len(type_steps),
                       label=f'{discovery_type} ({count})', s=50, alpha=0.7)

        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Discovery Type')
        ax5.set_title('Object Discovery Timeline')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No discoveries recorded', ha='center', va='center',
                transform=ax5.transAxes)
        ax5.set_title('Object Discovery Timeline')

    ax5.grid(True, alpha=0.3)

    # 6. Learning progression
    ax6 = axes[1, 2]

    # Show perceptual category formation
    if agent.perceptual_categories:
        category_counts = {cat: len(examples) for cat, examples in agent.perceptual_categories.items()}
        categories = list(category_counts.keys())
        counts = list(category_counts.values())

        bars = ax6.bar(range(len(categories)), counts, alpha=0.7, color='skyblue')
        ax6.set_xticks(range(len(categories)))
        ax6.set_xticklabels(categories, rotation=45, ha='right')
        ax6.set_ylabel('Examples Learned')
        ax6.set_title('Perceptual Category Formation')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No categories formed', ha='center', va='center',
                transform=ax6.transAxes)
        ax6.set_title('Perceptual Category Formation')

    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = Path(output_dir) / 'embodied_qse_emile_simulation.png' if output_dir else Path('embodied_qse_emile_simulation.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass

    print(f"📊  Embodied cognition visualization saved as '{out_png.name}'")


def print_embodied_analysis(results):
    """Print analysis of embodied cognition experiment"""

    print(f"\n🧠  EMBODIED COGNITION ANALYSIS")
    print("=" * 50)

    agent = results['agent']
    context_switches = results['context_switches']
    object_discoveries = results['object_discoveries']
    trajectory = results['trajectory']

    # Basic metrics
    print(f"Total Steps: {results['total_steps']}")
    print(f"Context Switches: {len(context_switches)}")
    print(f"Object Discoveries: {len(object_discoveries)}")
    print(f"Unique Positions Visited: {len(set(trajectory))}")

    # Body state analysis
    final_energy = agent.body.state.energy
    final_health = agent.body.state.health
    print(f"Final Energy: {final_energy:.3f}")
    print(f"Final Health: {final_health:.3f}")

    # Perceptual learning
    categories_learned = len(agent.perceptual_categories)
    total_examples = sum(len(examples) for examples in agent.perceptual_categories.values())
    print(f"Perceptual Categories Formed: {categories_learned}")
    print(f"Total Perceptual Examples: {total_examples}")

    if agent.perceptual_categories:
        print("Categories learned:")
        for category, examples in agent.perceptual_categories.items():
            print(f"  - {category}: {len(examples)} examples")

    # Context switching analysis
    if context_switches:
        context_reasons = []
        for switch in context_switches:
            if switch['body_energy'] < 0.3:
                context_reasons.append('low_energy')
            elif 'consumed' in str(switch.get('goal', '')):
                context_reasons.append('food_seeking')
            else:
                context_reasons.append('exploration')

        from collections import Counter
        reason_counts = Counter(context_reasons)
        print("Context switch triggers:")
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count} switches")

    # Discovery analysis
    if object_discoveries:
        discovery_contexts = [disc['context'] for disc in object_discoveries]
        context_discovery_rate = {}
        for context_id in set(discovery_contexts):
            rate = discovery_contexts.count(context_id) / len(object_discoveries)
            context_discovery_rate[context_id] = rate

        print("Discovery rate by context:")
        for context_id, rate in sorted(context_discovery_rate.items()):
            print(f"  - Context {context_id}: {rate:.2%}")

    # Embodied learning insights
    print(f"\n🔍  EMBODIED INSIGHTS:")

    # Body schema development
    boundary_experiences = len(agent.body.body_schema['boundaries'])
    affordance_contexts = len(agent.body.body_schema['affordances'])
    sensorimotor_mappings = len(agent.body.body_schema['sensorimotor_mappings'])

    print(f"✅  Body Schema Development:")
    print(f"   - Boundary experiences: {boundary_experiences}")
    print(f"   - Affordance contexts learned: {affordance_contexts}")
    print(f"   - Sensorimotor mappings: {sensorimotor_mappings}")

    # Context-driven perception
    context_switches_near_discoveries = 0
    for switch in context_switches:
        for discovery in object_discoveries:
            if abs(switch['step'] - discovery['step']) <= 5:
                context_switches_near_discoveries += 1
                break

    if context_switches and object_discoveries:
        context_discovery_correlation = context_switches_near_discoveries / len(context_switches)
        print(f"✅  Context-Discovery Correlation: {context_discovery_correlation:.2%}")

    # Exploration efficiency
    exploration_efficiency = len(set(trajectory)) / len(trajectory)
    print(f"✅  Exploration Efficiency: {exploration_efficiency:.3f}")

    print(f"\n🎯  EMBODIED COGNITION SUCCESS METRICS:")
    if categories_learned > 0:
        print(f"✅  Emergent categorization achieved ({categories_learned} categories)")
    if boundary_experiences > 5:
        print(f"✅  Body schema formation in progress ({boundary_experiences} experiences)")
    if len(context_switches) > 0:
        print(f"✅  Context-dependent perception active ({len(context_switches)} switches)")
    if object_discoveries:
        print(f"✅  Meaningful world interaction ({len(object_discoveries)} discoveries)")


def main():
    """Run embodied QSE-Émile experiment via CLI"""

    ap = argparse.ArgumentParser(description="Run embodied_qse_emile with per-step JSONL logging.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", type=str, default=None, help="Output directory for artifacts (JSONL/PNG).")
    ap.add_argument("--no-visualize", action="store_true", help="Disable plotting.")
    ap.add_argument("--no-jsonl", action="store_true", help="Do not write per-step JSONL.")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else None
    visualize = not args.no_visualize
    save_jsonl = not args.no_jsonl

    results = run_embodied_experiment(
        steps=args.steps,
        visualize=visualize,
        output_dir=out_dir,
        save_jsonl=save_jsonl,
        seed=args.seed,
    )
    return results


if __name__ == "__main__":
    _ = main()

