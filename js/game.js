// Race state machine — wraps physics with start/finish logic.

const CANVAS_W = 800;
const CANVAS_H = 750;

const START_Y   = 620;   // y-pixel of start line
const FINISH_Y  = 100;   // y-pixel of finish line
const LINE_HALF = 150;   // half-width of both lines

export function createRaceState() {
  return {
    phase: 'pre-start',   // 'pre-start' | 'early-start' | 'racing' | 'finished'
    timeToStart: 60,      // countdown seconds
    raceTime: 0,
    startLeft:   { x: CANVAS_W / 2 - LINE_HALF, y: START_Y },
    startRight:  { x: CANVAS_W / 2 + LINE_HALF, y: START_Y },
    finishLeft:  { x: CANVAS_W / 2 - LINE_HALF, y: FINISH_Y },
    finishRight: { x: CANVAS_W / 2 + LINE_HALF, y: FINISH_Y },
    distToFinish: 0,
    earlyStartPenalty: 0,
  };
}

// Returns updated race state given boat state and dt.
export function updateRace(rs, boatState, dt) {
  const r = { ...rs };
  r.distToFinish = Math.max(0, boatState.y - FINISH_Y);

  if (r.phase === 'pre-start') {
    r.timeToStart = Math.max(0, r.timeToStart - dt);

    // Check if boat has crossed start line early
    if (boatCrossedLine(boatState, r.startLeft, r.startRight) && r.timeToStart > 0) {
      r.phase = 'early-start';
    }

    if (r.timeToStart <= 0) {
      // Gun fired
      if (boatCrossedLine(boatState, r.startLeft, r.startRight)) {
        r.phase = 'racing';
        r.raceTime = 0;
      } else {
        r.phase = 'pre-start'; // still waiting to cross
        r.timeToStart = 0;     // gun fired, now racing as soon as they cross
      }
    }
  } else if (r.phase === 'early-start') {
    r.timeToStart = Math.max(0, r.timeToStart - dt);
    // Return to below start line to clear the penalty
    if (!boatCrossedLine(boatState, r.startLeft, r.startRight)) {
      r.phase = 'pre-start';
    }
  } else if (r.phase === 'racing') {
    r.raceTime += dt;
    if (boatCrossedLine(boatState, r.finishLeft, r.finishRight)) {
      r.phase = 'finished';
    }
  }
  // 'finished' is terminal

  return r;
}

// True if boat's y is above (screen-upward from) the line and x is within bounds.
function boatCrossedLine(boat, left, right) {
  return boat.y <= left.y && boat.x >= left.x && boat.x <= right.x;
}

export { CANVAS_W, CANVAS_H };
