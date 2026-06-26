// Sailing physics — pure JS, no DOM, no canvas.
// Coordinate system: x = right, y = up (math). Canvas y is inverted in renderer.
// Wind comes FROM the top of the canvas (TRUE_WIND_DIR = 90°, blows downward).
// Boat heading 90° = bow pointing straight into the wind (the no-go zone).

const TWO_PI = Math.PI * 2;
const DEG = Math.PI / 180;

const PHYSICS = {
  // No-go half-angle: zero drive within this of the wind direction
  NO_GO_HALF_ANGLE: 40 * DEG,

  // Drag (quadratic). Terminal speed ≈ sqrt(MAX_DRIVE / DRAG)
  DRAG: 0.016,

  // Peak drive force (at ~90° TWA = beam reach, best trim)
  MAX_DRIVE: 0.32,

  // Turning rate radians/s at full rudder (scales with a speed factor)
  TURN_RATE: 1.2,

  // Tack penalty
  TACK_SPEED_FACTOR: 0.38,
  TACK_PENALTY_DURATION: 2.8,

  // Cosmetic leeway: tiny sideways slip when beating
  LEEWAY_COEFF: 0.05,

  MASS: 1.0,

  // Canvas pixels per speed-unit per second (keeps movement frame-rate independent)
  PIXELS_PER_UNIT: 65,
};

// Wind FROM 90° (top of canvas). Blows toward 270° (downward on screen).
const TRUE_WIND_DIR = 90 * DEG;

export function getInitialState() {
  return {
    x: 400,
    y: 700,             // below start line — gives a few seconds to reach it
    vx: 0,
    vy: 0,
    // 40° heading → ~50° TWA → starboard tack, solidly outside no-go, driving immediately
    heading: 40 * DEG,
    speed: 0,
    sheetPct: 0.80,     // reasonable default for close-hauled
    rudder: 0,
    tackPenaltyTimer: 0,
    lastTWA: null,
    tacking: false,
  };
}

// Pure step function — no side effects, no DOM.
// controls = { rudder: -1..1, sheetDelta: number (fraction/s) }
export function step(state, controls, dt) {
  const s = { ...state };

  // Apply controls
  s.rudder   = Math.max(-1, Math.min(1, controls.rudder  ?? 0));
  s.sheetPct = Math.max(0,  Math.min(1, s.sheetPct + (controls.sheetDelta ?? 0) * dt));

  // True Wind Angle: + = starboard tack, - = port tack
  const twa    = signedTWA(s.heading);
  const absTWA = Math.abs(twa);

  // Tack detection (sign flip of TWA)
  if (s.lastTWA !== null && s.lastTWA !== 0 && Math.sign(twa) !== Math.sign(s.lastTWA)) {
    s.speed *= PHYSICS.TACK_SPEED_FACTOR;
    s.tackPenaltyTimer = PHYSICS.TACK_PENALTY_DURATION;
    s.tacking = true;
  }
  if (s.tackPenaltyTimer <= 0) s.tacking = false;
  s.lastTWA = twa;
  s.tackPenaltyTimer = Math.max(0, s.tackPenaltyTimer - dt);

  // Drive force
  let drive = 0;
  if (absTWA > PHYSICS.NO_GO_HALF_ANGLE) {
    drive = driveFromTWA(absTWA, s.sheetPct);
    if (s.tackPenaltyTimer > 0) {
      // Ramp drive back in quadratically after a tack
      const ramp = 1 - s.tackPenaltyTimer / PHYSICS.TACK_PENALTY_DURATION;
      drive *= ramp * ramp;
    }
  }

  // Speed update (drag is quadratic)
  const drag = PHYSICS.DRAG * s.speed * s.speed;
  s.speed = Math.max(0, s.speed + ((drive - drag) / PHYSICS.MASS) * dt);

  // Leeway: small lateral slip only when close-hauled
  const leewayAngle = (absTWA < 100 * DEG)
    ? Math.sign(twa) * PHYSICS.LEEWAY_COEFF * s.speed
    : 0;

  const moveAngle = s.heading + leewayAngle;
  s.vx = Math.cos(moveAngle) * s.speed;
  s.vy = Math.sin(moveAngle) * s.speed;

  // Position (frame-rate independent)
  s.x +=  s.vx * dt * PHYSICS.PIXELS_PER_UNIT;
  s.y -= s.vy * dt * PHYSICS.PIXELS_PER_UNIT;  // canvas y inverted

  // Steering: small minimum effect at zero speed so the player isn't stranded
  const steerEffect = 0.25 + 0.75 * Math.min(s.speed / 2.0, 1.0);
  s.heading += s.rudder * PHYSICS.TURN_RATE * steerEffect * dt;
  s.heading = ((s.heading % TWO_PI) + TWO_PI) % TWO_PI;

  return s;
}

// Signed TWA. Range (-π, π). Positive = starboard tack.
function signedTWA(heading) {
  let twa = TRUE_WIND_DIR - heading;
  while (twa >  Math.PI) twa -= TWO_PI;
  while (twa < -Math.PI) twa += TWO_PI;
  return twa;
}

// Drive force as a function of TWA and sheet trim.
//
// Drive peaks at ~90° TWA (beam reach). This creates a real VMG tradeoff:
//   - Close-hauled (~45° TWA): slow speed but good angle → moderate VMG
//   - Beam reach (90° TWA):    fast speed but perpendicular → zero VMG upwind
//   - Optimal VMG groove:      ~50–60° TWA (player has to find it)
//
function driveFromTWA(absTWA, sheetPct) {
  const ng   = PHYSICS.NO_GO_HALF_ANGLE;  // 40° — no-go edge
  const beam = Math.PI / 2;               // 90° — peak drive

  let rawDrive;
  if (absTWA <= beam) {
    // Quarter-sine ramp: 0 at no-go edge, 1.0 at beam reach
    const t = (absTWA - ng) / (beam - ng);
    rawDrive = Math.sin(t * Math.PI / 2);
  } else {
    // Gentle decay from beam reach to dead run
    const t = (absTWA - beam) / (Math.PI - beam);
    rawDrive = 1.0 - 0.4 * t;
  }

  // Optimal sheet: tight close-hauled (0.90), eased reaching/running (→0.25)
  const optimalSheet = Math.max(0.15, 0.90 - 0.65 * ((absTWA - ng) / (Math.PI - ng)));
  const trimError    = Math.abs(sheetPct - optimalSheet);
  const trimEfficiency = Math.max(0.15, 1.0 - 3.0 * trimError * trimError);

  return PHYSICS.MAX_DRIVE * rawDrive * trimEfficiency;
}

// For trim feedback on the HUD
export function getTrimStatus(state) {
  const absTWA = Math.abs(signedTWA(state.heading));
  const ng = PHYSICS.NO_GO_HALF_ANGLE;
  const optimalSheet = Math.max(0.15, 0.90 - 0.65 * ((absTWA - ng) / (Math.PI - ng)));
  const diff = state.sheetPct - optimalSheet;
  if (diff >  0.12) return 'OVERSHEET';
  if (diff < -0.12) return 'EASE SAIL';
  return 'TRIM OK';
}

// Flat observation vector for RL
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
    raceState?.timeToStart  ?? 0,
    raceState?.distToFinish ?? 0,
  ];
}

export { signedTWA, TRUE_WIND_DIR, PHYSICS };
