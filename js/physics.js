// Sailing physics — pure JS, no DOM, no canvas.
// Coordinate system: x = right, y = up (screen y is inverted in renderer).
// Wind blows FROM the top of the screen, so true wind direction = 270° (blowing "south" = downward).
// Angles: 0° = right, 90° = up, measured counter-clockwise. Boat heading 90° = pointing upwind.

const TWO_PI = Math.PI * 2;

// --- constants (tune these for feel) ---
const DEGREES = Math.PI / 180;

const PHYSICS = {
  // No-go zone: boat cannot point within this many degrees of true wind
  NO_GO_HALF_ANGLE: 42 * DEGREES,   // ±42° from wind = 84° no-go zone

  // Peak drive angle: TWA where drive is maximum (close-hauled optimal)
  PEAK_TWA: 45 * DEGREES,

  // Drag coefficient
  DRAG: 0.018,

  // Max sail drive force (tuned so terminal speed feels satisfying)
  MAX_DRIVE: 0.28,

  // Boat turning rate (radians per second per unit rudder at speed 1)
  TURN_RATE: 1.1,

  // Minimum speed before rudder has meaningful effect
  MIN_STEER_SPEED: 0.3,

  // Speed at which tack penalty kicks in (crossing through the wind)
  TACK_SPEED_FACTOR: 0.35,   // speed reduced to this fraction on tack
  TACK_PENALTY_DURATION: 2.8, // seconds of reduced drive after a tack

  // Leeway: sideways slip coefficient (fraction of forward speed)
  LEEWAY_COEFF: 0.06,

  // Mass (affects acceleration feel)
  MASS: 1.0,
};

// True wind blows FROM this direction (in our coordinate system).
// 90° = wind coming from "top" (upward on screen), boat must beat upwind.
const TRUE_WIND_DIR = 90 * DEGREES; // wind FROM 90° → blows toward 270°
const TRUE_WIND_SPEED = 1.0;        // normalised; all speeds relative to this

export function getInitialState() {
  return {
    x: 400,          // pixels, centre of canvas
    y: 650,          // near bottom of canvas
    vx: 0,
    vy: 0,
    heading: 90 * DEGREES,  // pointing straight upwind to start
    speed: 0,
    sheetPct: 0.7,   // mainsheet 0=fully eased, 1=fully sheeted
    rudder: 0,       // -1=full port, 0=centre, +1=full starboard
    tackPenaltyTimer: 0,
    lastTWA: null,   // used to detect tack (sign change of TWA)
    tacking: false,
  };
}

// Pure step function — takes state + controls, returns new state.
// dt in seconds. controls = { rudder: -1..1, sheetDelta: number }
export function step(state, controls, dt) {
  const s = { ...state };

  // --- Apply controls ---
  s.rudder = Math.max(-1, Math.min(1, controls.rudder));
  s.sheetPct = Math.max(0, Math.min(1, s.sheetPct + (controls.sheetDelta || 0) * dt));

  // --- True Wind Angle (TWA) ---
  // Wind blows FROM TRUE_WIND_DIR. Wind vector direction = TRUE_WIND_DIR + π.
  const twa = signedTWA(s.heading);

  // --- Tack detection (sign change of TWA) ---
  if (s.lastTWA !== null && Math.sign(twa) !== Math.sign(s.lastTWA) && s.lastTWA !== 0) {
    s.speed *= PHYSICS.TACK_SPEED_FACTOR;
    s.tackPenaltyTimer = PHYSICS.TACK_PENALTY_DURATION;
    s.tacking = true;
  } else if (s.tackPenaltyTimer <= 0) {
    s.tacking = false;
  }
  s.lastTWA = twa;
  s.tackPenaltyTimer = Math.max(0, s.tackPenaltyTimer - dt);

  // --- Drive force ---
  const absTWA = Math.abs(twa);
  let drive = 0;

  if (absTWA > PHYSICS.NO_GO_HALF_ANGLE) {
    // Sail drive curve: peak at PEAK_TWA, falls off toward beam/run and toward no-go
    drive = driveFromTWA(absTWA, s.sheetPct);
    if (s.tackPenaltyTimer > 0) {
      // Ramp drive back in over the penalty window
      const ramp = 1 - (s.tackPenaltyTimer / PHYSICS.TACK_PENALTY_DURATION);
      drive *= ramp * ramp;
    }
  }

  // --- Drag ---
  const drag = PHYSICS.DRAG * s.speed * s.speed;

  // --- Acceleration along heading ---
  const netForce = (drive - drag) / PHYSICS.MASS;
  s.speed = Math.max(0, s.speed + netForce * dt);

  // --- Leeway (sideways slip, away from wind, only when beating) ---
  // Positive leeway when on starboard tack (twa > 0), negative on port.
  const leewayAngle = (absTWA < 100 * DEGREES)
    ? Math.sign(twa) * PHYSICS.LEEWAY_COEFF * s.speed
    : 0;

  const moveAngle = s.heading + leewayAngle;
  s.vx = Math.cos(moveAngle) * s.speed;
  s.vy = Math.sin(moveAngle) * s.speed;

  s.x += s.vx * dt * 60; // scale to pixel-friendly units
  s.y -= s.vy * dt * 60; // canvas y is inverted

  // --- Steering ---
  const steerEffect = Math.min(s.speed / PHYSICS.MIN_STEER_SPEED, 1);
  s.heading += s.rudder * PHYSICS.TURN_RATE * steerEffect * dt;
  s.heading = ((s.heading % TWO_PI) + TWO_PI) % TWO_PI;

  return s;
}

// Signed TWA: positive = starboard tack, negative = port tack.
// Returns value in (-π, π).
function signedTWA(heading) {
  // Wind comes FROM TRUE_WIND_DIR. Relative to boat heading:
  let twa = TRUE_WIND_DIR - heading;
  // Normalise to (-π, π)
  while (twa > Math.PI) twa -= TWO_PI;
  while (twa < -Math.PI) twa += TWO_PI;
  return twa;
}

// Drive force as a function of absolute TWA and sheet trim.
// Shape: zero at NO_GO_HALF_ANGLE, peak near PEAK_TWA, gradual decay to run.
function driveFromTWA(absTWA, sheetPct) {
  const ng = PHYSICS.NO_GO_HALF_ANGLE;
  const pk = PHYSICS.PEAK_TWA;

  let rawDrive;
  if (absTWA <= pk) {
    // Ramp up from no-go edge to peak
    const t = (absTWA - ng) / (pk - ng);
    rawDrive = t * t;
  } else {
    // Decay from peak toward run (broad reach / dead run reduce drive in this model)
    const t = (absTWA - pk) / (Math.PI - pk);
    rawDrive = 1 - 0.55 * t; // partial drive even on a run
  }

  // Sheet trim efficiency: optimal trim varies by point of sail.
  // Simplified: close-hauled wants ~0.85 sheet, reaching wants ~0.6.
  const optimalSheet = 0.85 - 0.35 * ((absTWA - ng) / (Math.PI - ng));
  const trimError = Math.abs(sheetPct - optimalSheet);
  const trimEfficiency = Math.max(0.2, 1 - 2.5 * trimError * trimError);

  return PHYSICS.MAX_DRIVE * rawDrive * trimEfficiency;
}

// --- Observation vector for RL (flat numeric array) ---
export function getObservation(state, raceState) {
  const twa = signedTWA(state.heading);
  return [
    state.x / 800,
    state.y / 800,
    Math.cos(state.heading),
    Math.sin(state.heading),
    twa / Math.PI,
    state.speed / 5,
    state.sheetPct,
    state.tackPenaltyTimer / PHYSICS.TACK_PENALTY_DURATION,
    raceState?.timeToStart ?? 0,
    raceState?.distToFinish ?? 0,
  ];
}

// Exported for renderer/HUD
export { signedTWA, TRUE_WIND_DIR, TRUE_WIND_SPEED, PHYSICS };
