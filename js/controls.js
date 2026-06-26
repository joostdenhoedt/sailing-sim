// Keyboard → action mapping. Returns a controls object each frame.

const SHEET_RATE = 0.4;   // fraction per second when key held
const RUDDER_RATE = 2.0;  // rudder moves to target per second (instant in this simple version)

export function createControls() {
  const keys = new Set();

  window.addEventListener('keydown', e => {
    keys.add(e.code);
    e.preventDefault();
  });
  window.addEventListener('keyup', e => keys.delete(e.code));

  // Returns controls object for physics.step()
  function sample(dt) {
    let rudder = 0;
    if (keys.has('KeyA') || keys.has('ArrowLeft'))  rudder = -1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) rudder =  1;

    let sheetDelta = 0;
    if (keys.has('ArrowUp'))   sheetDelta =  SHEET_RATE; // sheet in (trim)
    if (keys.has('ArrowDown')) sheetDelta = -SHEET_RATE; // ease out

    return { rudder, sheetDelta };
  }

  return { sample };
}
