"""Pygame renderer for Kosmos world."""

import pygame
import numpy as np
import threading
from typing import Optional

from ..world.grid import KosmosWorld
from ..world.objects import (
    Biome, BIOME_COLORS, Food, Water, Hazard, CraftItem, WorldObject,
)
from ..agent.core import KosmosAgent


# Color palette
BG = (12, 12, 16)
PANEL_BG = (18, 18, 24)
TEXT_DIM = (100, 100, 110)
TEXT_MED = (160, 160, 170)
TEXT_BRIGHT = (220, 220, 230)
BAR_BG = (30, 30, 38)
GRID_LINE = (25, 25, 32)

STRATEGY_COLORS = {
    "explore": (80, 180, 220),
    "exploit": (220, 160, 40),
    "rest": (100, 180, 100),
    "learn": (180, 120, 220),
    "social": (220, 120, 160),
}

AGENT_COLOR = (240, 220, 60)
AGENT_EYE = (20, 20, 30)
TRAIL_COLOR = (60, 55, 30)


class KosmosRenderer:
    """Pygame visualization for the Kosmos world."""

    def __init__(
        self,
        world: KosmosWorld,
        agent: KosmosAgent,
        cell_size: int = 20,
    ):
        self.world = world
        self.agent = agent
        self.cell_size = cell_size

        # Layout
        grid_px = world.size * cell_size
        self.panel_width = 280
        self.narration_height = 100
        self.width = grid_px + self.panel_width
        self.height = grid_px + self.narration_height

        # State
        self.running = False
        self.paused = False
        self.speed = 8  # ticks per second
        self.trail: list[tuple] = []
        self.max_trail = 80

        # Narration
        self.narration_lines: list[str] = []
        self.narration_timer = 0
        self._narrate_lock = threading.Lock()
        self._narrate_semaphore = threading.Semaphore(2)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("emile-Kosmos")
        self.font_sm = pygame.font.SysFont("menlo", 11)
        self.font_md = pygame.font.SysFont("menlo", 13)
        self.font_lg = pygame.font.SysFont("menlo", 16, bold=True)
        self.font_title = pygame.font.SysFont("menlo", 18, bold=True)
        self.clock = pygame.time.Clock()

    def run(self):
        """Main render loop."""
        self.init_pygame()
        self.running = True
        self.agent.start()

        tick_accum = 0.0

        while self.running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            if not self.paused:
                tick_accum += dt * self.speed
                while tick_accum >= 1.0:
                    tick_accum -= 1.0
                    self._game_tick()

            self._draw()
            pygame.display.flip()

        self.agent.stop()
        pygame.quit()

    def _handle_key(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_UP:
            self.speed = min(60, self.speed + 2)
        elif key == pygame.K_DOWN:
            self.speed = max(1, self.speed - 2)
        elif key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False

    def _game_tick(self):
        """One simulation tick."""
        # World evolves
        self.world.tick()

        # Agent acts
        result = self.agent.tick()

        # Trail
        self.trail.append(self.agent.pos)
        if len(self.trail) > self.max_trail:
            self.trail = self.trail[-self.max_trail:]

        # Trigger narration on interesting events
        self.narration_timer += 1
        tool = result.get("tool", "")
        res_str = str(result.get("result", ""))

        should_narrate = False
        event_desc = ""

        if tool == "death":
            should_narrate = True
            event_desc = f"I have died. Death #{self.agent.deaths}."
        elif "Ate" in res_str:
            should_narrate = True
            event_desc = f"Found and consumed food. {res_str}"
        elif "Crafted" in res_str:
            should_narrate = True
            event_desc = f"Created something new. {res_str}"
        elif "Ouch" in res_str:
            should_narrate = True
            event_desc = f"Encountered danger. {res_str}"
        elif self.narration_timer >= 60:
            should_narrate = True
            state = self.agent.get_state()
            event_desc = (
                f"Wandering through {state['time_of_day']} in {self.world.biomes[self.agent.pos].value}. "
                f"Energy {state['energy']:.0%}. Strategy: {state['strategy']}."
            )
            self.narration_timer = 0

        if should_narrate and self.agent.use_llm:
            self._request_narration(event_desc)

    def _request_narration(self, event: str):
        """Request LLM narration in background thread (max 2 concurrent)."""
        if not self._narrate_semaphore.acquire(blocking=False):
            return  # Already 2 narrations in flight, skip
        def _do():
            try:
                text = self.agent.llm.narrate(event, self.agent.strategy, self.agent.entropy)
                if text:
                    with self._narrate_lock:
                        self.narration_lines.append(text)
                        if len(self.narration_lines) > 6:
                            self.narration_lines = self.narration_lines[-6:]
            finally:
                self._narrate_semaphore.release()
        threading.Thread(target=_do, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #
    def _draw(self):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_objects()
        self._draw_trail()
        self._draw_agent()
        self._draw_panel()
        self._draw_narration()

    def _draw_grid(self):
        """Draw biome-colored grid."""
        cs = self.cell_size
        for r in range(self.world.size):
            for c in range(self.world.size):
                biome = self.world.biomes[r, c]
                color = BIOME_COLORS.get(biome, (30, 30, 30))
                # Night dimming
                if self.world.is_night:
                    color = tuple(max(0, int(v * 0.5)) for v in color)
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

    def _draw_objects(self):
        """Draw world objects."""
        cs = self.cell_size
        for pos, objs in self.world.objects.items():
            r, c = pos
            cx = c * cs + cs // 2
            cy = r * cs + cs // 2
            for obj in objs:
                color = obj.color
                if self.world.is_night:
                    color = tuple(max(0, int(v * 0.6)) for v in color)
                radius = max(2, cs // 4)
                if isinstance(obj, Food):
                    pygame.draw.circle(self.screen, color, (cx, cy), radius)
                elif isinstance(obj, Water):
                    # Diamond
                    pts = [(cx, cy - radius), (cx + radius, cy),
                           (cx, cy + radius), (cx - radius, cy)]
                    pygame.draw.polygon(self.screen, color, pts)
                elif isinstance(obj, Hazard):
                    # X shape
                    pygame.draw.line(self.screen, color,
                                    (cx - radius, cy - radius),
                                    (cx + radius, cy + radius), 2)
                    pygame.draw.line(self.screen, color,
                                    (cx + radius, cy - radius),
                                    (cx - radius, cy + radius), 2)
                elif isinstance(obj, CraftItem):
                    # Plus
                    pygame.draw.line(self.screen, color,
                                    (cx, cy - radius), (cx, cy + radius), 2)
                    pygame.draw.line(self.screen, color,
                                    (cx - radius, cy), (cx + radius, cy), 2)

    def _draw_trail(self):
        """Draw fading agent trail."""
        cs = self.cell_size
        n = len(self.trail)
        for i, (r, c) in enumerate(self.trail):
            alpha = int(40 * (i + 1) / n) if n > 0 else 0
            color = tuple(min(255, v + alpha) for v in TRAIL_COLOR)
            rect = pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4)
            s = pygame.Surface((cs - 4, cs - 4), pygame.SRCALPHA)
            s.fill((*color, alpha + 20))
            self.screen.blit(s, rect.topleft)

    def _draw_agent(self):
        """Draw the agent."""
        cs = self.cell_size
        r, c = self.agent.pos
        cx = c * cs + cs // 2
        cy = r * cs + cs // 2
        radius = max(4, cs // 3)

        # Body
        color = AGENT_COLOR if self.agent.alive else (100, 60, 60)
        pygame.draw.circle(self.screen, color, (cx, cy), radius)

        # Eye (direction-aware)
        offsets = {"north": (0, -2), "south": (0, 2),
                   "east": (2, 0), "west": (-2, 0)}
        dx, dy = offsets.get(self.agent.facing, (2, 0))
        pygame.draw.circle(self.screen, AGENT_EYE,
                           (cx + dx, cy + dy), max(1, radius // 3))

        # Dead indicator
        if not self.agent.alive:
            pygame.draw.line(self.screen, (200, 50, 50),
                             (cx - radius, cy - radius),
                             (cx + radius, cy + radius), 2)
            pygame.draw.line(self.screen, (200, 50, 50),
                             (cx + radius, cy - radius),
                             (cx - radius, cy + radius), 2)

    def _draw_panel(self):
        """Draw the right info panel."""
        grid_px = self.world.size * self.cell_size
        panel_x = grid_px + 8
        y = 10

        # Background
        pygame.draw.rect(self.screen, PANEL_BG,
                         (grid_px, 0, self.panel_width, self.height))

        state = self.agent.get_state()

        # Title
        self._text(panel_x, y, "emile-Kosmos", self.font_title, TEXT_BRIGHT)
        y += 26

        # Status
        alive_str = "ALIVE" if state["alive"] else "DEAD"
        alive_color = (100, 200, 100) if state["alive"] else (200, 60, 60)
        self._text(panel_x, y, alive_str, self.font_md, alive_color)
        y += 20

        # Strategy
        strat = state["strategy"]
        strat_color = STRATEGY_COLORS.get(strat, TEXT_MED)
        self._text(panel_x, y, f"Strategy: {strat}", self.font_md, strat_color)
        y += 18

        # Context
        self._text(panel_x, y, f"Context: {state['context']}", self.font_sm, TEXT_DIM)
        y += 16

        # Time
        self._text(panel_x, y, f"{state['time_of_day']} | {state['season']}",
                   self.font_sm, TEXT_DIM)
        y += 20

        # Energy bar
        self._draw_bar(panel_x, y, "Energy", state["energy"],
                       (60, 180, 60), (180, 40, 40))
        y += 22

        # Hydration bar
        self._draw_bar(panel_x, y, "Hydration", state["hydration"],
                       (60, 120, 220), (120, 80, 40))
        y += 22

        # Entropy bar
        self._draw_bar(panel_x, y, "Entropy", state["entropy"],
                       (180, 120, 220), (60, 60, 80))
        y += 22

        # LLM temp
        temp = 0.3 + state["entropy"] * 1.2
        self._text(panel_x, y, f"LLM Temp: {temp:.2f}", self.font_sm, TEXT_DIM)
        y += 16
        llm_str = f"LLM: {self.agent.llm.model}" if state["use_llm"] else "LLM: offline (heuristic)"
        self._text(panel_x, y, llm_str, self.font_sm,
                   (80, 160, 80) if state["use_llm"] else (160, 80, 80))
        y += 22

        # Stats
        self._text(panel_x, y, "-- Stats --", self.font_sm, TEXT_MED)
        y += 16
        stats = [
            f"Food: {state['food_eaten']}  Water: {state['water_drunk']}",
            f"Deaths: {state['deaths']}  Steps: {state['steps']}",
            f"Explored: {state['cells_visited']} cells",
            f"Tick: {state['total_ticks']}  Speed: {self.speed}/s",
        ]
        for s in stats:
            self._text(panel_x, y, s, self.font_sm, TEXT_DIM)
            y += 14

        y += 8

        # Inventory
        self._text(panel_x, y, "-- Inventory --", self.font_sm, TEXT_MED)
        y += 16
        if state["inventory"]:
            for item in state["inventory"][:6]:
                self._text(panel_x + 4, y, item, self.font_sm, (180, 160, 100))
                y += 13
        else:
            self._text(panel_x + 4, y, "(empty)", self.font_sm, TEXT_DIM)
            y += 13

        if state["crafted"]:
            y += 4
            self._text(panel_x, y, "Crafted:", self.font_sm, TEXT_MED)
            y += 14
            for item in state["crafted"]:
                self._text(panel_x + 4, y, item, self.font_sm, (200, 180, 80))
                y += 13

        y += 10

        # Thought bubble
        thought = state.get("thought", "")
        if thought:
            self._text(panel_x, y, "-- Thought --", self.font_sm, TEXT_MED)
            y += 16
            # Word-wrap thought
            words = thought.split()
            line = ""
            for w in words:
                test = line + " " + w if line else w
                if self.font_sm.size(test)[0] < self.panel_width - 20:
                    line = test
                else:
                    self._text(panel_x + 4, y, line, self.font_sm, (180, 200, 220))
                    y += 13
                    line = w
            if line:
                self._text(panel_x + 4, y, line, self.font_sm, (180, 200, 220))
                y += 13

        # Controls at bottom
        ctrl_y = self.height - self.narration_height - 50
        self._text(panel_x, ctrl_y, "SPACE:pause  UP/DN:speed", self.font_sm, TEXT_DIM)
        self._text(panel_x, ctrl_y + 13, "Q:quit", self.font_sm, TEXT_DIM)
        if self.paused:
            self._text(panel_x, ctrl_y + 28, "|| PAUSED", self.font_md, (220, 180, 60))

    def _draw_narration(self):
        """Draw the bottom narration panel."""
        grid_px = self.world.size * self.cell_size
        y_start = grid_px
        pygame.draw.rect(self.screen, PANEL_BG,
                         (0, y_start, self.width, self.narration_height))
        pygame.draw.line(self.screen, (40, 40, 50),
                         (0, y_start), (self.width, y_start))

        y = y_start + 6
        with self._narrate_lock:
            lines = list(self.narration_lines)

        # Show last few lines that fit
        visible = lines[-(self.narration_height // 16):]
        for line in visible:
            # Word-wrap
            words = line.split()
            current = ""
            for w in words:
                test = current + " " + w if current else w
                if self.font_sm.size(test)[0] < self.width - 20:
                    current = test
                else:
                    self._text(8, y, current, self.font_sm, (160, 180, 200))
                    y += 14
                    current = w
                    if y > y_start + self.narration_height - 14:
                        break
            if current and y <= y_start + self.narration_height - 14:
                self._text(8, y, current, self.font_sm, (160, 180, 200))
                y += 14

    def _draw_bar(self, x, y, label, value, color_full, color_empty):
        """Draw a labeled progress bar."""
        self._text(x, y, f"{label}: {value:.0%}", self.font_sm, TEXT_DIM)
        bar_x = x + 100
        bar_w = self.panel_width - 120
        bar_h = 10
        pygame.draw.rect(self.screen, BAR_BG, (bar_x, y + 2, bar_w, bar_h))
        fill_w = int(bar_w * max(0, min(1, value)))
        # Interpolate color
        t = max(0, min(1, value))
        color = tuple(int(color_empty[i] * (1 - t) + color_full[i] * t) for i in range(3))
        pygame.draw.rect(self.screen, color, (bar_x, y + 2, fill_w, bar_h))

    def _text(self, x, y, text, font, color):
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
