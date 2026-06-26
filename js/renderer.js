import { signedTWA, TRUE_WIND_DIR, PHYSICS } from './physics.js';

const DEGREES = Math.PI / 180;

export function createRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  function draw(state, raceState) {
    // Background
    ctx.fillStyle = '#1a3a5c';
    ctx.fillRect(0, 0, W, H);

    drawWaterTexture(ctx, W, H);
    drawCourse(ctx, raceState);
    drawBoat(ctx, state);
    drawHUD(ctx, state, raceState, W, H);
  }

  return { draw };
}

function drawWaterTexture(ctx, W, H) {
  // Subtle grid of wave lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let y = 0; y < H; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
}

function drawCourse(ctx, rs) {
  if (!rs) return;

  // Start line between two buoys
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(rs.startLeft.x, rs.startLeft.y);
  ctx.lineTo(rs.startRight.x, rs.startRight.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Start buoys
  drawBuoy(ctx, rs.startLeft,  '#ff4444');
  drawBuoy(ctx, rs.startRight, '#ff4444');

  // Finish line
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(rs.finishLeft.x, rs.finishLeft.y);
  ctx.lineTo(rs.finishRight.x, rs.finishRight.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Finish buoys
  drawBuoy(ctx, rs.finishLeft,  '#ffcc00');
  drawBuoy(ctx, rs.finishRight, '#ffcc00');

  // Chequered flag at top centre
  drawChequeredFlag(ctx, (rs.finishLeft.x + rs.finishRight.x) / 2 - 20, rs.finishLeft.y - 40);

  // Countdown or race status
  if (rs.phase === 'pre-start') {
    const t = Math.ceil(rs.timeToStart);
    ctx.fillStyle = t <= 5 ? '#ff4444' : '#ffffff';
    ctx.font = 'bold 32px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`START IN ${t}s`, 400, rs.startLeft.y + 35);
  } else if (rs.phase === 'early-start') {
    ctx.fillStyle = '#ff2200';
    ctx.font = 'bold 24px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('OCS — RETURN!', 400, rs.startLeft.y + 35);
  } else if (rs.phase === 'racing') {
    ctx.fillStyle = '#aaffaa';
    ctx.font = '18px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`RACE TIME  ${formatTime(rs.raceTime)}`, 400, rs.startLeft.y + 35);
  } else if (rs.phase === 'finished') {
    ctx.fillStyle = '#ffff00';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`FINISHED!  ${formatTime(rs.raceTime)}`, 400, H / 2);
  }
}

function drawBuoy(ctx, pos, color) {
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawChequeredFlag(ctx, x, y) {
  const size = 8;
  const cols = 5, rows = 3;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? '#ffffff' : '#000000';
      ctx.fillRect(x + c * size, y + r * size, size, size);
    }
  }
}

function drawBoat(ctx, state) {
  ctx.save();
  ctx.translate(state.x, state.y);
  // Canvas y is inverted; heading 90° should point up
  ctx.rotate(-state.heading + Math.PI / 2);

  const twa = signedTWA(state.heading);
  const inNoGo = Math.abs(twa) < PHYSICS.NO_GO_HALF_ANGLE;

  // Hull — small triangle
  ctx.beginPath();
  ctx.moveTo(0, -14);   // bow
  ctx.lineTo(-6, 8);    // port stern
  ctx.lineTo(6, 8);     // starboard stern
  ctx.closePath();
  ctx.fillStyle = inNoGo ? '#ff8844' : '#e8e8e8';
  ctx.fill();
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Mast dot
  ctx.beginPath();
  ctx.arc(0, -4, 2, 0, Math.PI * 2);
  ctx.fillStyle = '#555';
  ctx.fill();

  // Boom/sail — line showing sail angle
  const sailAngle = getSailAngle(twa, state.sheetPct);
  ctx.save();
  ctx.rotate(sailAngle);
  ctx.strokeStyle = inNoGo ? '#ff884488' : '#ffffffcc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, -4);
  ctx.lineTo(0, 10);
  ctx.stroke();
  ctx.restore();

  ctx.restore();
}

// Returns the boom angle relative to boat centreline (visual only)
function getSailAngle(twa, sheetPct) {
  // On starboard tack (twa > 0): boom goes to port (-ve angle)
  // On port tack (twa < 0): boom goes to starboard (+ve angle)
  const side = twa >= 0 ? -1 : 1;
  // Sheet pulls boom toward centreline; ease pushes it out
  const maxAngle = 0.9; // radians (~52°)
  return side * maxAngle * (1 - sheetPct * 0.7);
}

function drawHUD(ctx, state, rs, W, H) {
  const twa = signedTWA(state.heading);
  const twaDeg = (twa / Math.PI * 180).toFixed(0);
  const inNoGo = Math.abs(twa) < PHYSICS.NO_GO_HALF_ANGLE;
  const vmg = computeVMG(state);

  // Wind arrow (top-right)
  drawWindArrow(ctx, W - 60, 60);

  // Info panel (top-left)
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(8, 8, 190, 120);

  ctx.font = '13px monospace';
  ctx.textAlign = 'left';

  const lines = [
    `SPD  ${state.speed.toFixed(2)}`,
    `VMG  ${vmg.toFixed(2)}`,
    `TWA  ${twaDeg}°  ${tack(twa)}`,
    `SAIL ${(state.sheetPct * 100).toFixed(0)}%`,
    inNoGo ? '⚠ NO-GO ZONE' : (state.tacking ? '↻ TACKING...' : ''),
  ];

  lines.forEach((line, i) => {
    ctx.fillStyle = (i === 4 && inNoGo) ? '#ff8844' : '#ccffcc';
    ctx.fillText(line, 14, 28 + i * 18);
  });

  // Controls reminder (bottom)
  ctx.fillStyle = 'rgba(0,0,0,0.4)';
  ctx.fillRect(0, H - 26, W, 26);
  ctx.fillStyle = '#888';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('A/D — steer   ↑/↓ — sheet in/out', W / 2, H - 9);
}

function drawWindArrow(ctx, cx, cy) {
  ctx.save();
  ctx.translate(cx, cy);
  // Wind comes FROM top → arrow points downward (direction it blows toward)
  ctx.rotate(Math.PI); // pointing down = wind blowing downward = from top

  ctx.strokeStyle = '#aaddff';
  ctx.fillStyle = '#aaddff';
  ctx.lineWidth = 2;

  // Shaft
  ctx.beginPath();
  ctx.moveTo(0, -22);
  ctx.lineTo(0, 14);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(0, 22);
  ctx.lineTo(-7, 10);
  ctx.lineTo(7, 10);
  ctx.closePath();
  ctx.fill();

  ctx.restore();

  ctx.fillStyle = '#aaddff';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('WIND', cx, cy + 42);
}

function computeVMG(state) {
  // VMG upwind = speed * cos(|TWA|)
  const twa = signedTWA(state.heading);
  return state.speed * Math.cos(Math.abs(twa));
}

function tack(twa) {
  return twa >= 0 ? 'STBD' : 'PORT';
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}
