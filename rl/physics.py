"""
Direct Python port of js/physics.js — identical maths, no DOM, no canvas.

All constants match the browser game so the RL agent trains on the same
physics the player experiences.
"""

import math

# ── constants (mirror of PHYSICS object in physics.js) ──────────────────────
TWO_PI               = 2 * math.pi
NO_GO_HALF_ANGLE     = math.radians(40)   # dead zone either side of true wind
DRAG                 = 0.055              # quadratic drag; terminal ≈ sqrt(MAX_DRIVE/DRAG)
MAX_DRIVE            = 0.35              # peak force at beam reach (~90° TWA)
TURN_RATE            = 3.0              # rad/s at full rudder
TACK_SPEED_FACTOR    = 0.40             # speed fraction kept on a tack
TACK_PENALTY_DURATION = 2.5            # seconds of reduced drive post-tack
LEEWAY_COEFF         = 0.05             # cosmetic sideways slip
MASS                 = 1.0
PIXELS_PER_UNIT      = 50              # canvas px per speed-unit per second

# Wind comes FROM the top of the canvas (90° in math coords = "up").
TRUE_WIND_DIR = math.radians(90)

# ── state ────────────────────────────────────────────────────────────────────

def get_initial_state():
    return {
        'x':                  400.0,
        'y':                  700.0,    # below start line
        'vx':                 0.0,
        'vy':                 0.0,
        'heading':            math.radians(40),  # ~50° TWA starboard — drives immediately
        'speed':              0.0,
        'sheet_pct':          0.80,
        'rudder':             0.0,
        'tack_penalty_timer': 0.0,
        'last_twa':           None,
        'tacking':            False,
    }

# ── core physics step ─────────────────────────────────────────────────────────

def step(state, controls, dt):
    """
    Pure function: (state, controls, dt) -> new_state.
    controls = {'rudder': -1..1, 'sheet_delta': float}
    """
    s = dict(state)   # shallow copy — all values are scalars

    # Apply controls
    s['rudder']    = max(-1.0, min(1.0, controls.get('rudder', 0.0)))
    s['sheet_pct'] = max(0.0,  min(1.0, s['sheet_pct'] + controls.get('sheet_delta', 0.0) * dt))

    twa     = signed_twa(s['heading'])
    abs_twa = abs(twa)

    # ── Tack detection (sign flip of TWA) ────────────────────────────────────
    last = s['last_twa']
    if last is not None and last != 0 and math.copysign(1, twa) != math.copysign(1, last):
        s['speed']             *= TACK_SPEED_FACTOR
        s['tack_penalty_timer'] = TACK_PENALTY_DURATION
        s['tacking']            = True
    if s['tack_penalty_timer'] <= 0:
        s['tacking'] = False
    s['last_twa']           = twa
    s['tack_penalty_timer'] = max(0.0, s['tack_penalty_timer'] - dt)

    # ── Drive force ───────────────────────────────────────────────────────────
    drive = 0.0
    if abs_twa > NO_GO_HALF_ANGLE:
        drive = _drive_from_twa(abs_twa, s['sheet_pct'])
        if s['tack_penalty_timer'] > 0:
            ramp   = 1.0 - s['tack_penalty_timer'] / TACK_PENALTY_DURATION
            drive *= ramp * ramp

    # ── Speed (net force = drive − drag) ─────────────────────────────────────
    drag       = DRAG * s['speed'] ** 2
    s['speed'] = max(0.0, s['speed'] + (drive - drag) / MASS * dt)

    # ── Position ──────────────────────────────────────────────────────────────
    leeway_angle = (
        math.copysign(LEEWAY_COEFF * s['speed'], twa)
        if abs_twa < math.radians(100) else 0.0
    )
    move_angle = s['heading'] + leeway_angle
    s['vx']    = math.cos(move_angle) * s['speed']
    s['vy']    = math.sin(move_angle) * s['speed']

    s['x'] +=  s['vx'] * dt * PIXELS_PER_UNIT
    s['y'] -= s['vy'] * dt * PIXELS_PER_UNIT   # canvas y is inverted

    # ── Heading ───────────────────────────────────────────────────────────────
    s['heading']  = (s['heading'] + s['rudder'] * TURN_RATE * dt) % TWO_PI

    return s

# ── helpers ───────────────────────────────────────────────────────────────────

def signed_twa(heading):
    """
    Signed True Wind Angle. Positive = starboard tack, negative = port tack.
    Range (-π, π).
    """
    twa = TRUE_WIND_DIR - heading
    # Normalise to (-π, π) — equivalent to JS while-loops
    return (twa + math.pi) % TWO_PI - math.pi


def _drive_from_twa(abs_twa, sheet_pct):
    """
    Drive force as a function of absolute TWA and sheet trim.
    Peaks at ~90° TWA (beam reach). Creates a real VMG tradeoff upwind.
    """
    ng   = NO_GO_HALF_ANGLE
    beam = math.pi / 2

    if abs_twa <= beam:
        # Quarter-sine ramp: 0 at no-go edge → 1.0 at beam reach
        t         = (abs_twa - ng) / (beam - ng)
        raw_drive = math.sin(t * math.pi / 2)
    else:
        # Gentle decay: beam reach → dead run
        t         = (abs_twa - beam) / (math.pi - beam)
        raw_drive = 1.0 - 0.4 * t

    # Optimal sheet: tight close-hauled (0.90) → well eased on a run (→0.25)
    optimal_sheet    = max(0.15, 0.90 - 0.65 * ((abs_twa - ng) / (math.pi - ng)))
    trim_error       = abs(sheet_pct - optimal_sheet)
    trim_efficiency  = max(0.15, 1.0 - 3.0 * trim_error ** 2)

    return MAX_DRIVE * raw_drive * trim_efficiency


def get_trim_status(state):
    """Returns 'EASE SAIL', 'TRIM OK', or 'OVERSHEET'."""
    abs_twa      = abs(signed_twa(state['heading']))
    optimal      = max(0.15, 0.90 - 0.65 * ((abs_twa - NO_GO_HALF_ANGLE) / (math.pi - NO_GO_HALF_ANGLE)))
    diff         = state['sheet_pct'] - optimal
    if diff >  0.12: return 'OVERSHEET'
    if diff < -0.12: return 'EASE SAIL'
    return 'TRIM OK'


def get_vmg(state):
    """Velocity Made Good toward the upwind mark (positive = making progress)."""
    twa = signed_twa(state['heading'])
    return state['speed'] * math.cos(abs(twa))
