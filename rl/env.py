"""
SailingEnv — a gymnasium environment wrapping the sailing physics.

State space:  90 discrete states  (5 TWA zones × 3 speeds × 3 trims × 2 tack flags)
Action space: 4 discrete actions  (steer left / right / sheet in / ease out)

The discrete encoding means we can store Q(s,a) in a 90×4 numpy table.
When we outgrow this (Phase 2), we swap the encoder for the raw continuous
vector and the table for a neural network — everything else stays the same.
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from physics import (
    get_initial_state, step as phys_step,
    signed_twa, get_trim_status, get_vmg,
    NO_GO_HALF_ANGLE,
)

# ── course layout (matches game constants) ────────────────────────────────────
CANVAS_W  = 800
CANVAS_H  = 750
START_Y   = 650
FINISH_Y  = 80
LINE_HALF = 150   # half-width of start/finish lines

# ── simulation timing ─────────────────────────────────────────────────────────
PHYS_DT   = 0.05   # physics sub-step (seconds)
SUBSTEPS  = 3      # sub-steps per RL action → 0.15 s simulated time per action
MAX_STEPS = 1500   # episode cut-off (~225 simulated seconds)

# ── discrete state dimensions ─────────────────────────────────────────────────
#
#  TWA zone  (5): IN_NO_GO | TIGHT | GROOVE | REACHING | BROAD
#  Speed     (3): SLOW | MID | FAST
#  Trim      (3): EASE SAIL | TRIM OK | OVERSHEET
#  Tacking   (2): no | yes
#
N_TWA   = 5
N_SPEED = 3
N_TRIM  = 3
N_TACK  = 2
N_STATES = N_TWA * N_SPEED * N_TRIM * N_TACK   # 90

TWA_ZONE_NAMES  = ['NO_GO',   'TIGHT', 'GROOVE', 'REACH',  'BROAD']
SPEED_BKT_NAMES = ['SLOW',    'MID',   'FAST']
TRIM_BKT_NAMES  = ['EASE',    'OK',    'SHEET']
TACK_NAMES      = ['',        'TACK']

# Build human-readable label for each of the 90 states
STATE_NAMES = []
for _t in range(N_TWA):
    for _s in range(N_SPEED):
        for _tr in range(N_TRIM):
            for _tk in range(N_TACK):
                parts = [TWA_ZONE_NAMES[_t], SPEED_BKT_NAMES[_s], TRIM_BKT_NAMES[_tr]]
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
    Start below the start line (y=650). Cross it, then reach the finish (y=80).
    Reward = VMG upwind each step + bonus on crossing finish.
    """
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space      = spaces.Discrete(len(ACTION_CONTROLS))
        self.boat   = None
        self.steps  = 0
        self._started = False   # has the boat crossed the start line?

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.boat     = get_initial_state()
        self.steps    = 0
        self._started = False
        return self._encode(), {}

    def step(self, action):
        controls = ACTION_CONTROLS[action]

        # Run several physics sub-steps per RL action for smoother dynamics
        for _ in range(SUBSTEPS):
            self.boat = phys_step(self.boat, controls, PHYS_DT)

        self.steps += 1

        # Update start-line flag once the boat crosses upward
        if not self._started and self.boat['y'] <= START_Y:
            self._started = True

        reward     = self._reward()
        terminated = self._started and self.boat['y'] <= FINISH_Y
        truncated  = self.steps >= MAX_STEPS

        info = {
            'boat':    dict(self.boat),
            'started': self._started,
            'step':    self.steps,
        }
        return self._encode(), reward, terminated, truncated, info

    # ── state encoding ────────────────────────────────────────────────────────

    def _encode(self):
        """Map continuous boat state → single integer in [0, N_STATES)."""
        twa         = signed_twa(self.boat['heading'])
        abs_twa_deg = math.degrees(abs(twa))

        # TWA zone
        if   abs_twa_deg < 40:  twa_z = 0   # in no-go
        elif abs_twa_deg < 50:  twa_z = 1   # tight, just above no-go edge
        elif abs_twa_deg < 68:  twa_z = 2   # groove — VMG sweet spot
        elif abs_twa_deg < 100: twa_z = 3   # reaching
        else:                   twa_z = 4   # broad / running

        # Speed bucket
        spd = self.boat['speed']
        if   spd < 0.6:  spd_b = 0
        elif spd < 1.3:  spd_b = 1
        else:            spd_b = 2

        # Trim bucket
        trim_b = {'EASE SAIL': 0, 'TRIM OK': 1, 'OVERSHEET': 2}[
            get_trim_status(self.boat)
        ]

        # Tacking flag
        tack_b = int(self.boat['tacking'])

        # Encode as a single index (row-major)
        return (twa_z * N_SPEED * N_TRIM * N_TACK
                + spd_b * N_TRIM * N_TACK
                + trim_b * N_TACK
                + tack_b)

    # ── reward ────────────────────────────────────────────────────────────────

    def _reward(self):
        """
        Dense reward: VMG upwind every step.
        This gives the agent a signal at every timestep, not just on finish.

        The no-go penalty teaches the no-go constraint.
        The finish bonus makes completing the race worth it.
        """
        twa = signed_twa(self.boat['heading'])

        # Core reward: how fast are we making upwind progress?
        vmg = self.boat['speed'] * math.cos(abs(twa))
        r   = vmg

        # Penalise the no-go zone — being there is always wrong
        if abs(twa) < NO_GO_HALF_ANGLE:
            r -= 0.5

        # Large bonus for crossing the finish line
        if self._started and self.boat['y'] <= FINISH_Y:
            r += 50.0

        return r

    # ── continuous observation (for Phase 2 DQN) ─────────────────────────────

    def continuous_obs(self):
        """
        Returns a flat numpy vector of the continuous state.
        Not used by the Q-table, but ready for the DQN phase.
        """
        twa = signed_twa(self.boat['heading'])
        return np.array([
            self.boat['x']    / CANVAS_W,
            self.boat['y']    / CANVAS_H,
            math.cos(self.boat['heading']),
            math.sin(self.boat['heading']),
            twa / math.pi,
            self.boat['speed']    / 4.0,
            self.boat['sheet_pct'],
            self.boat['tack_penalty_timer'] / 2.5,
        ], dtype=np.float32)
