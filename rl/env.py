"""
SailingEnv — gymnasium environment wrapping the sailing physics.

State space:  486 discrete states
  9 signed TWA zones  (which tack + how close to wind)
× 3 speed buckets
× 3 trim buckets
× 2 tack-penalty flags
× 3 x-zone buckets   (LEFT | CENTRE | RIGHT on the course)

Why signed TWA?   The agent must distinguish starboard tack (twa > 0) from
                  port tack (twa < 0) to know *which way* to turn.

Why x-zone?       Without position the agent can never learn *when* to tack.
                  "I'm in the groove on starboard tack near the right boundary"
                  → tack now. That decision is impossible without x_zone.

Action space: 4 discrete  (steer left / right / sheet in / ease out)

Phase 2 (DQN): swap _encode() for continuous_obs() and the Q-table for a net.
Everything else — env interface, reward, physics — stays identical.
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from physics import (
    get_initial_state, step as phys_step,
    signed_twa, get_trim_status,
    NO_GO_HALF_ANGLE,
)

# ── course layout (must match js/game.js) ────────────────────────────────────
CANVAS_W   = 800
CANVAS_H   = 750
COURSE_CX  = CANVAS_W / 2          # 400 — centre of start/finish lines
LINE_HALF  = 150                    # half-width → lines run x ∈ [250, 550]
START_Y    = 650
FINISH_Y   = 80

# ── simulation timing ─────────────────────────────────────────────────────────
PHYS_DT   = 0.05   # physics sub-step (seconds)
SUBSTEPS  = 3      # sub-steps per RL action → 0.15 s per action
MAX_STEPS = 1500

# ── discrete state dimensions ─────────────────────────────────────────────────
#
#  Signed TWA zone (9):
#    0  IN_NO_GO                  |twa| < 40°
#    1  S_TIGHT   starboard  40–50°     (just above no-go)
#    2  S_GROOVE  starboard  50–68°     ← VMG sweet spot
#    3  S_REACH   starboard  68–100°
#    4  S_BROAD   starboard  >100°
#    5  P_TIGHT   port       40–50°
#    6  P_GROOVE  port       50–68°
#    7  P_REACH   port       68–100°
#    8  P_BROAD   port       >100°
#
#  Speed (3):  SLOW (<0.6) | MID (0.6–1.3) | FAST (>1.3)
#  Trim  (3):  EASE | OK | SHEET
#  Tacking (2): not in penalty | in penalty
#  x-zone (3): LEFT (<300) | CENTRE (300–500) | RIGHT (>500)
#
N_TWA   = 9
N_SPEED = 3
N_TRIM  = 3
N_TACK  = 2
N_XZONE = 3
N_STATES = N_TWA * N_SPEED * N_TRIM * N_TACK * N_XZONE   # 486

TWA_ZONE_NAMES  = ['NO_GO',
                   'S_TIGHT', 'S_GROOVE', 'S_REACH', 'S_BROAD',
                   'P_TIGHT', 'P_GROOVE', 'P_REACH', 'P_BROAD']
SPEED_BKT_NAMES = ['SLOW', 'MID', 'FAST']
TRIM_BKT_NAMES  = ['EASE', 'OK', 'SHEET']
TACK_NAMES      = ['', 'TACK']
XZONE_NAMES     = ['LEFT', 'CTR', 'RIGHT']

STATE_NAMES = []
for _tz in range(N_TWA):
    for _s in range(N_SPEED):
        for _tr in range(N_TRIM):
            for _tk in range(N_TACK):
                for _x in range(N_XZONE):
                    parts = [TWA_ZONE_NAMES[_tz], SPEED_BKT_NAMES[_s],
                             TRIM_BKT_NAMES[_tr], XZONE_NAMES[_x]]
                    if _tk: parts.append('TACK')
                    STATE_NAMES.append(' | '.join(parts))

# ── discrete actions ──────────────────────────────────────────────────────────
ACTION_CONTROLS = [
    {'rudder': -1.0, 'sheet_delta':  0.0},   # 0  STEER_LEFT
    {'rudder':  1.0, 'sheet_delta':  0.0},   # 1  STEER_RIGHT
    {'rudder':  0.0, 'sheet_delta':  0.4},   # 2  SHEET_IN
    {'rudder':  0.0, 'sheet_delta': -0.4},   # 3  EASE_OUT
]
ACTION_NAMES = ['STEER_LEFT', 'STEER_RIGHT', 'SHEET_IN', 'EASE_OUT']


# ── environment ───────────────────────────────────────────────────────────────

class SailingEnv(gym.Env):
    """
    Upwind sailing race.
    Boat starts below the start line. Must cross start (within x bounds),
    then reach the finish line (also within x bounds) at the top.
    """
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space      = spaces.Discrete(len(ACTION_CONTROLS))
        self.boat     = None
        self.steps    = 0
        self._started = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.boat     = get_initial_state()
        self.steps    = 0
        self._started = False
        return self._encode(), {}

    def step(self, action):
        controls = ACTION_CONTROLS[action]
        for _ in range(SUBSTEPS):
            self.boat = phys_step(self.boat, controls, PHYS_DT)
        self.steps += 1

        # Start line: must cross within the marked x bounds (same as the game)
        if not self._started and self._on_line(START_Y):
            self._started = True

        off_canvas = (self.boat['x'] < -50 or self.boat['x'] > CANVAS_W + 50
                      or self.boat['y'] < -50)

        reward     = self._reward(off_canvas)
        terminated = off_canvas or (self._started and self._on_line(FINISH_Y))
        truncated  = self.steps >= MAX_STEPS

        info = {'boat': dict(self.boat), 'started': self._started, 'step': self.steps}
        return self._encode(), reward, terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _on_line(self, line_y: float) -> bool:
        """
        True if the boat has crossed line_y AND is within the marked course width.
        Mirrors boatCrossedLine() in js/game.js.
        """
        return (self.boat['y'] <= line_y
                and abs(self.boat['x'] - COURSE_CX) <= LINE_HALF)

    # ── state encoding ────────────────────────────────────────────────────────

    def _encode(self) -> int:
        """Map continuous boat state → integer in [0, N_STATES)."""
        twa = signed_twa(self.boat['heading'])
        twa_deg = math.degrees(twa)
        abs_deg = abs(twa_deg)

        # Signed TWA zone
        if abs_deg < 40:
            tz = 0                          # no-go
        elif twa_deg > 0:                   # starboard tack
            if   abs_deg < 50:  tz = 1
            elif abs_deg < 68:  tz = 2
            elif abs_deg < 100: tz = 3
            else:               tz = 4
        else:                               # port tack
            if   abs_deg < 50:  tz = 5
            elif abs_deg < 68:  tz = 6
            elif abs_deg < 100: tz = 7
            else:               tz = 8

        # Speed bucket
        spd = self.boat['speed']
        sb = 0 if spd < 0.6 else (1 if spd < 1.3 else 2)

        # Trim bucket
        tb = {'EASE SAIL': 0, 'TRIM OK': 1, 'OVERSHEET': 2}[get_trim_status(self.boat)]

        # Tacking-penalty flag
        tkb = int(self.boat['tacking'])

        # x-zone: where is the boat on the course left↔right?
        x = self.boat['x']
        xb = 0 if x < 300 else (1 if x <= 500 else 2)

        # Row-major encoding
        idx = (tz  * (N_SPEED * N_TRIM * N_TACK * N_XZONE)
             + sb  * (N_TRIM  * N_TACK * N_XZONE)
             + tb  * (N_TACK  * N_XZONE)
             + tkb * N_XZONE
             + xb)
        return idx

    # ── reward ────────────────────────────────────────────────────────────────

    def _reward(self, off_canvas: bool) -> float:
        """
        Dense VMG reward drives the agent upwind every step.
        Boundary penalty teaches the agent to stay on the course.
        Finish bonus makes completing the race worth it.
        """
        if off_canvas:
            return -20.0   # hard penalty for leaving the course area

        twa = signed_twa(self.boat['heading'])

        # Core: upwind progress
        vmg = self.boat['speed'] * math.cos(abs(twa))
        r   = vmg

        # Penalise no-go zone
        if abs(twa) < NO_GO_HALF_ANGLE:
            r -= 0.5

        # Finish bonus
        if self._started and self._on_line(FINISH_Y):
            r += 50.0

        return r

    # ── continuous observation (Phase 2 — DQN) ───────────────────────────────

    def continuous_obs(self) -> np.ndarray:
        """Flat float32 vector for the neural network in Phase 2."""
        twa = signed_twa(self.boat['heading'])
        return np.array([
            (self.boat['x'] - COURSE_CX) / LINE_HALF,   # ±1 at course edges
            self.boat['y'] / CANVAS_H,
            math.cos(self.boat['heading']),
            math.sin(self.boat['heading']),
            twa / math.pi,
            self.boat['speed'] / 4.0,
            self.boat['sheet_pct'],
            self.boat['tack_penalty_timer'] / 2.5,
        ], dtype=np.float32)
