import { signedTWA, TRUE_WIND_DIR, PHYSICS, getTrimStatus } from './physics.js';

const DEG = Math.PI / 180;

export function createRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  function draw(state, raceState) {
    ctx.fillStyle = '#1a3a5c';
    ctx.fillRect(0, 0, W, H);
    drawWater(ctx, W, H);
    drawWindStreaks(ctx, W);
    drawCourse(ctx, raceState, W, H);
    drawBoat(ctx, state);
    drawHUD(ctx, state, raceState, W, H);
  }

  return { draw };
}

function drawWater(ctx, W, H) {
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let y = 0; y < H; y += 45) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
}

// Small downward arrows along top edge to show wind direction unambiguously
function drawWindStreaks(ctx, W) {
  ctx.fillStyle = 'rgba(170,210,255,0.18)';
  for (let x = 60; x < W - 40; x += 90) {
    drawMiniArrow(ctx, x, 18);
  }
}

function drawMiniArrow(ctx, x, y) {
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x, y + 14);
  ctx.moveTo(x - 4, y + 8); ctx.lineTo(x, y + 14); ctx.lineTo(x + 4, y + 8);
  ctx.strokeStyle = 'rgba(170,210,255,0.35)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawCourse(ctx, rs, W, H) {
  if (!rs) return;

  // Start line
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(rs.startLeft.x, rs.startLeft.y);
  ctx.lineTo(rs.startRight.x, rs.startRight.y);
  ctx.stroke();
  ctx.setLineDash([]);
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
  drawBuoy(ctx, rs.finishLeft,  '#ffcc00');
  drawBuoy(ctx, rs.finishRight, '#ffcc00');

  const flagX = (rs.finishLeft.x + rs.finishRight.x) / 2 - 20;
  drawChequeredFlag(ctx, flagX, rs.finishLeft.y - 36);

  // Race status text
  drawRaceStatus(ctx, rs, W);
}

function drawRaceStatus(ctx, rs, W) {
  ctx.textAlign = 'center';
  if (rs.phase === 'pre-start') {
    const t = Math.ceil(rs.timeToStart);
    ctx.fillStyle = t <= 3 ? '#ff4444' : '#ffffff';
    ctx.font = 'bold 28px monospace';
    ctx.fillText(
      rs.timeToStart <= 0 ? 'GO! — CROSS THE LINE' : `START IN  ${t}s`,
      W / 2, rs.startLeft.y + 36
    );
  } else if (rs.phase === 'early-start') {
    ctx.fillStyle = '#ff2200';
    ctx.font = 'bold 22px monospace';
    ctx.fillText('OCS — RETURN TO START!', W / 2, rs.startLeft.y + 36);
  } else if (rs.phase === 'racing') {
    ctx.fillStyle = '#aaffaa';
    ctx.font = '16px monospace';
    ctx.fillText(`RACE TIME  ${formatTime(rs.raceTime)}`, W / 2, rs.startLeft.y + 36);
  } else if (rs.phase === 'finished') {
    ctx.fillStyle = '#ffff00';
    ctx.font = 'bold 32px monospace';
    ctx.fillText(`FINISHED!`, W / 2, 380);
    ctx.font = '22px monospace';
    ctx.fillText(formatTime(rs.raceTime), W / 2, 420);
  }
}

function drawBuoy(ctx, pos, color) {
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawChequeredFlag(ctx, x, y) {
  const s = 8, cols = 5, rows = 3;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = (r + c) % 2 === 0 ? '#fff' : '#000';
      ctx.fillRect(x + c * s, y + r * s, s, s);
    }
}

function drawBoat(ctx, state) {
  ctx.save();
  ctx.translate(state.x, state.y);
  // Heading 90° = up in math coords. Canvas rotation: subtract from π/2 for correct orientation.
  ctx.rotate(-state.heading + Math.PI / 2);

  const twa   = signedTWA(state.heading);
  const inNoGo = Math.abs(twa) < PHYSICS.NO_GO_HALF_ANGLE;

  // Hull
  ctx.beginPath();
  ctx.moveTo(0, -15);   // bow
  ctx.lineTo(-6, 9);
  ctx.lineTo(6, 9);
  ctx.closePath();
  ctx.fillStyle   = inNoGo ? '#cc5533' : '#ddeeff';
  ctx.strokeStyle = '#223344';
  ctx.lineWidth = 1;
  ctx.fill();
  ctx.stroke();

  // Mast
  ctx.beginPath();
  ctx.arc(0, -3, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = '#445';
  ctx.fill();

  // Boom + sail — amber/orange so it reads clearly against the water
  const sailAngle = getSailAngle(twa, state.sheetPct);
  ctx.save();
  ctx.rotate(sailAngle);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = inNoGo ? 'rgba(255,140,40,0.4)' : '#f4a030';
  ctx.beginPath();
  ctx.moveTo(0, -3);   // mast pivot
  ctx.lineTo(0, 12);   // boom end
  ctx.stroke();
  // Small triangle fill for the sail body
  ctx.beginPath();
  ctx.moveTo(0, -3);
  ctx.lineTo(0, 12);
  ctx.lineTo(sailAngle > 0 ? 5 : -5, 2);
  ctx.closePath();
  ctx.fillStyle = inNoGo ? 'rgba(255,140,40,0.15)' : 'rgba(244,160,48,0.35)';
  ctx.fill();
  ctx.restore();

  ctx.restore();
}

// Boom angle relative to boat centreline.
// On stbd tack (twa>0): boom goes port (negative angle). Port tack: opposite.
// Fully sheeted → boom near centre; fully eased → boom well out.
function getSailAngle(twa, sheetPct) {
  const side = twa >= 0 ? -1 : 1;
  const maxAngle = 1.25; // radians (~72°) fully eased
  return side * maxAngle * (1.0 - sheetPct * 0.75);
}

function drawHUD(ctx, state, rs, W, H) {
  const twa    = signedTWA(state.heading);
  const twaDeg = (twa / (Math.PI / 180)).toFixed(0);
  const inNoGo = Math.abs(twa) < PHYSICS.NO_GO_HALF_ANGLE;
  const vmg    = state.speed * Math.cos(Math.abs(twa));
  const trim   = getTrimStatus(state);

  // Wind label top-right
  ctx.fillStyle = 'rgba(170,210,255,0.85)';
  ctx.font = 'bold 13px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('WIND ↓ (from top)', W - 12, 30);

  // Info panel top-left
  ctx.fillStyle = 'rgba(0,0,0,0.60)';
  ctx.fillRect(8, 8, 200, 140);

  ctx.textAlign = 'left';
  ctx.font = '13px monospace';

  const trimColor = trim === 'TRIM OK' ? '#aaffaa' : '#ffcc44';

  const lines = [
    { text: `SPD  ${state.speed.toFixed(2)}`,           color: '#ccffcc' },
    { text: `VMG  ${vmg.toFixed(2)}`,                   color: '#aaffcc' },
    { text: `TWA  ${twaDeg}°  (${twa >= 0 ? 'STBD' : 'PORT'})`, color: '#ccffcc' },
    { text: `SAIL ${(state.sheetPct * 100).toFixed(0)}%  ${trim}`, color: trimColor },
    { text: inNoGo ? '⚠ NO-GO ZONE'
           : state.tacking ? '↻ TACKING...' : '',
      color: inNoGo ? '#ff7744' : '#ffcc44' },
  ];

  lines.forEach((l, i) => {
    if (!l.text) return;
    ctx.fillStyle = l.color;
    ctx.fillText(l.text, 14, 30 + i * 20);
  });

  // Controls bar at bottom
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fillRect(0, H - 28, W, 28);
  ctx.fillStyle = '#778899';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('A / D — steer left / right     ↑ / ↓ — sheet in / ease', W / 2, H - 10);
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}
