// Mask editor. The mask is drawn on an off-screen canvas of the picture's own size (white = regenerate), shown over the
// picture as a translucent layer, and exported as a black-and-white PNG of that same size: the server crops the picture
// and the mask to the chosen ratio together, so they must share one frame.
// The canvases live outside Alpine's reactive state (MASK): proxies around DOM objects break their methods.

const MASK = { mc: null, view: null, key: null, undo: [], redo: [], drawing: null, hover: null, pending: null };
const UNDO_LIMIT = 12;   // every step is a full copy of the mask: 12 of a 1920 x 1200 picture is about 110 MB

function maskMixin() {
  return {
    mask: { tool: 'brush', size: 48, show: true, empty: true, canUndo: false, canRedo: false },

    // The picture the mask belongs to: the init image of Inpaint, or image 1 of an Edit that works inside it.
    get maskSlot() {
      if (this.mode === 'Inpaint') return 'init';
      if (this.mode === 'Edit' && this.form.editInside) return 'ed0';
      return null;
    },
    get maskPic() { return this.maskSlot ? this.pics[this.maskSlot] || null : null; },

    // Called by x-effect whenever the picture changes: a new picture starts with an empty mask.
    maskSetup(pic, view) {
      if (!pic || !view) return;
      const key = pic.url;
      if (MASK.key === key && MASK.view === view) return;
      MASK.key = key; MASK.view = view;
      MASK.mc = document.createElement('canvas'); MASK.mc.width = pic.w; MASK.mc.height = pic.h;
      view.width = pic.w; view.height = pic.h;
      MASK.undo = []; MASK.redo = []; MASK.drawing = null;
      this.mask.size = Math.max(8, Math.round(Math.min(pic.w, pic.h) / 20));
      this.maskSync();
      if (MASK.pending) { const b = MASK.pending; MASK.pending = null; this.maskLoadBlob(b); }
    },
    // A mask to apply once the picture is there (reusing an old job: the picture is loaded first, then its mask).
    maskLater(blob) { MASK.key = null; MASK.pending = blob; },

    maskSync() {
      const ctx = MASK.mc.getContext('2d');
      const data = ctx.getImageData(0, 0, MASK.mc.width, MASK.mc.height).data;
      let empty = true;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 8) { empty = false; break; }
      this.mask.empty = empty; this.mask.canUndo = MASK.undo.length > 0; this.mask.canRedo = MASK.redo.length > 0;
      this.maskDraw();
    },
    maskDraw() {
      const view = MASK.view; if (!view || !MASK.mc) return;
      const ctx = view.getContext('2d');
      ctx.clearRect(0, 0, view.width, view.height);
      if (this.mask.show) { ctx.globalAlpha = 0.55; ctx.drawImage(MASK.mc, 0, 0); ctx.globalAlpha = 1; }
      const d = MASK.drawing, lw = Math.max(2, view.width / 400);
      ctx.lineWidth = lw;
      if (d && d.tool === 'rect') {
        ctx.fillStyle = 'rgba(255,255,255,.35)'; ctx.strokeStyle = '#fff';
        ctx.fillRect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0); ctx.strokeRect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0);
      }
      if (MASK.hover && this.mask.tool !== 'rect') {
        ctx.strokeStyle = '#8ab4ff'; ctx.beginPath(); ctx.arc(MASK.hover.x, MASK.hover.y, this.mask.size / 2, 0, Math.PI * 2); ctx.stroke();
      }
    },
    maskPoint(ev) {
      const r = MASK.view.getBoundingClientRect();
      return { x: (ev.clientX - r.left) * MASK.view.width / r.width, y: (ev.clientY - r.top) * MASK.view.height / r.height };
    },
    maskSnapshot() {
      MASK.undo.push(MASK.mc.getContext('2d').getImageData(0, 0, MASK.mc.width, MASK.mc.height));
      if (MASK.undo.length > UNDO_LIMIT) MASK.undo.shift();
      MASK.redo = [];
    },
    maskStroke(from, to) {
      const ctx = MASK.mc.getContext('2d');
      ctx.globalCompositeOperation = this.mask.tool === 'eraser' ? 'destination-out' : 'source-over';
      ctx.strokeStyle = ctx.fillStyle = '#fff'; ctx.lineCap = ctx.lineJoin = 'round'; ctx.lineWidth = this.mask.size;
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';
    },
    maskDown(ev) {
      if (!MASK.mc || ev.button > 0) return;
      ev.target.setPointerCapture(ev.pointerId);
      const p = this.maskPoint(ev);
      this.maskSnapshot();
      MASK.drawing = this.mask.tool === 'rect' ? { tool: 'rect', x0: p.x, y0: p.y, x1: p.x, y1: p.y } : { tool: 'pen', last: p };
      if (MASK.drawing.tool === 'pen') this.maskStroke(p, p);
      this.maskDraw();
    },
    maskMove(ev) {
      if (!MASK.mc) return;
      const p = this.maskPoint(ev); MASK.hover = p;
      const d = MASK.drawing;
      if (d && d.tool === 'pen') { this.maskStroke(d.last, p); d.last = p; }
      else if (d) { d.x1 = p.x; d.y1 = p.y; }
      this.maskDraw();
    },
    maskUp() {
      const d = MASK.drawing; if (!d) return;
      MASK.drawing = null;
      if (d.tool === 'rect') {
        const ctx = MASK.mc.getContext('2d');
        ctx.globalCompositeOperation = this.mask.tool === 'eraser' ? 'destination-out' : 'source-over';
        ctx.fillStyle = '#fff'; ctx.fillRect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0);
      }
      this.maskSync();
    },
    maskLeave() { MASK.hover = null; this.maskDraw(); },

    maskUndo() {
      if (!MASK.undo.length) return;
      const ctx = MASK.mc.getContext('2d');
      MASK.redo.push(ctx.getImageData(0, 0, MASK.mc.width, MASK.mc.height));
      ctx.putImageData(MASK.undo.pop(), 0, 0); this.maskSync();
    },
    maskRedo() {
      if (!MASK.redo.length) return;
      const ctx = MASK.mc.getContext('2d');
      MASK.undo.push(ctx.getImageData(0, 0, MASK.mc.width, MASK.mc.height));
      ctx.putImageData(MASK.redo.pop(), 0, 0); this.maskSync();
    },
    maskClear() { if (!MASK.mc) return; this.maskSnapshot(); MASK.mc.getContext('2d').clearRect(0, 0, MASK.mc.width, MASK.mc.height); this.maskSync(); },
    maskInvert() {
      if (!MASK.mc) return;
      this.maskSnapshot();
      const ctx = MASK.mc.getContext('2d'), img = ctx.getImageData(0, 0, MASK.mc.width, MASK.mc.height), d = img.data;
      for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = 255 - d[i + 3]; }
      ctx.putImageData(img, 0, 0); this.maskSync();
    },

    // A mask PNG: brightness becomes opacity, so white = regenerate, like the server reads it.
    async maskLoadBlob(blob) {
      if (!MASK.mc) { this.maskLater(blob); return; }
      const url = URL.createObjectURL(blob);
      try {
        const img = await new Promise((resolve, reject) => {
          const i = new Image(); i.onload = () => resolve(i); i.onerror = () => reject(new Error('that file is not a picture')); i.src = url;
        });
        this.maskSnapshot();
        const tmp = document.createElement('canvas'); tmp.width = MASK.mc.width; tmp.height = MASK.mc.height;
        const t = tmp.getContext('2d'); t.drawImage(img, 0, 0, tmp.width, tmp.height);
        const data = t.getImageData(0, 0, tmp.width, tmp.height), d = data.data;
        for (let i = 0; i < d.length; i += 4) { const lum = (d[i] + d[i + 1] + d[i + 2]) / 3; d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = lum; }
        MASK.mc.getContext('2d').putImageData(data, 0, 0);
        this.maskSync();
      } catch (e) { this.flash(e.message); }
      URL.revokeObjectURL(url);
    },
    maskFileChosen(ev) { const f = ev.target.files[0]; ev.target.value = ''; if (f) this.maskLoadBlob(f); },

    // The mask as a PNG: black where it is kept, white where it is regenerated.
    maskExport() {
      const out = document.createElement('canvas'); out.width = MASK.mc.width; out.height = MASK.mc.height;
      const ctx = out.getContext('2d'); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, out.width, out.height); ctx.drawImage(MASK.mc, 0, 0);
      return new Promise(resolve => out.toBlob(resolve, 'image/png'));
    },
  };
}
