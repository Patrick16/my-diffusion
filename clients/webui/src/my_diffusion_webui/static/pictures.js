// Input pictures: init image, references. A "slot" is a named place for one picture (see slotDefs).
// The picture's bytes live outside Alpine's reactive state (BLOBS): a Blob wrapped in a reactive proxy cannot be used.
// Alpine only holds what the page shows: {url, name, w, h}.

const BLOBS = new Map();   // slot -> Blob

function picturesMixin() {
  return {
    pics: {},                  // slot -> {url, name, w, h}
    activeSlot: null,          // the slot the pointer is over: where a pasted picture goes
    galleryPick: null,         // slot id while the "pick from the gallery" window is open
    maxRefs: 2, editLora: 'krea2_identity_edit_v1_2', editLoraWeight: 1,   // from GET /config (admins), else these defaults

    // What each mode needs. required: the request is refused without it.
    get slotDefs() {
      const subj = this.form.refMode === 'subjects';
      const refTag = i => (subj || i > 0 ? `Image ${i + 1} · subject` : 'Image 1 · scene');
      switch (this.mode) {
        case 'Img2img': return [{ id: 'init', label: 'Init image', tag: 'Init', required: true }];
        case 'Inpaint': {
          const s = [{ id: 'init', label: 'Init image', tag: 'Init', required: true }];
          if (this.maxRefs >= 2) s.push({ id: 'inpRef', label: 'Add a reference (optional)', tag: 'Image 2', required: false });
          return s;
        }
        case 'Edit': return Array.from({ length: this.maxRefs }, (_, i) => ({ id: 'ed' + i, label: 'Reference ' + (i + 1), tag: refTag(i), required: i === 0 }));
        case 'Outpaint': {
          const s = [{ id: 'op0', label: 'Picture to extend (reference 1)', tag: 'Reference 1', required: true }];
          if (this.maxRefs >= 2) s.push({ id: 'op1', label: 'Second reference: a person or object (optional)', tag: 'Reference 2', required: false });
          return s;
        }
        default: return [];
      }
    },

    async setPicture(slot, blob, name) {
      const url = URL.createObjectURL(blob);
      let dims;
      try {
        dims = await new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve([img.naturalWidth, img.naturalHeight]);
          img.onerror = () => reject(new Error('that file is not a picture the browser can read'));
          img.src = url;
        });
      } catch (e) { URL.revokeObjectURL(url); throw e; }
      this.clearPicture(slot);
      BLOBS.set(slot, blob);
      this.pics = { ...this.pics, [slot]: { url, name: name || slot, w: dims[0], h: dims[1] } };
    },
    clearPicture(slot) {
      const p = this.pics[slot];
      if (!p) return;
      URL.revokeObjectURL(p.url); BLOBS.delete(slot);
      const next = { ...this.pics }; delete next[slot]; this.pics = next;
    },
    async setFromFile(slot, file) {
      try {
        if (!file || !file.type.startsWith('image/')) throw new Error('that is not a picture');
        await this.setPicture(slot, file, file.name);
      } catch (e) { this.flash(e.message); }
    },
    fileChosen(slot, ev) { const f = ev.target.files[0]; ev.target.value = ''; return this.setFromFile(slot, f); },
    onDrop(slot, ev) { ev.preventDefault(); return this.setFromFile(slot, ev.dataTransfer.files[0]); },
    onPaste(ev) {
      const item = [...(ev.clipboardData ? ev.clipboardData.items : [])].find(i => i.type.startsWith('image/'));
      if (!item || this.route.page !== 'create' || !this.slotDefs.length) return;
      const slot = (this.activeSlot && this.slotDefs.some(s => s.id === this.activeSlot) ? this.activeSlot : null)
        || (this.slotDefs.find(s => !this.pics[s.id]) || this.slotDefs[0]).id;
      ev.preventDefault();
      this.setFromFile(slot, item.getAsFile());
    },

    // A picture the server already has: a result (image) or an input of an earlier job (inputs/<name>).
    async pictureFromUrl(slot, url, name) {
      const r = await fetch(url);
      if (!r.ok) throw new Error('the picture is not available any more');
      await this.setPicture(slot, await r.blob(), name);
    },
    openGalleryPick(slot) { this.galleryPick = slot; },
    async pickFromJob(job) {
      const slot = this.galleryPick; this.galleryPick = null;
      try { await this.pictureFromUrl(slot, `/api/result/${job.job_id}/image`, `${job.job_id.slice(0, 8)}.png`); }
      catch (e) { this.flash(e.message); }
    },
    get pickable() { return this.jobList.filter(j => j.status === 'done').slice(0, 40); },

    toDataUrl(blob) {
      return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result); r.onerror = () => reject(new Error('cannot read the picture'));
        r.readAsDataURL(blob);
      });
    },
    encode(slot) { return this.toDataUrl(BLOBS.get(slot)); },

    // The picture's frame after the server crops it to a ratio: the kept part as fractions of the picture.
    frameStyle(f) { return f ? { left: f.x * 100 + '%', top: f.y * 100 + '%', width: f.w * 100 + '%', height: f.h * 100 + '%' } : {}; },
    cropFrame(pic, ratio) {
      if (!pic || !ratio || ratio === 'auto') return null;
      const [rw, rh] = ratio.split(':').map(Number);
      const a = pic.w / pic.h, t = rw / rh;
      if (Math.abs(a - t) < 0.001) return null;
      return t < a ? { x: (1 - t / a) / 2, y: 0, w: t / a, h: 1 } : { x: 0, y: (1 - a / t) / 2, w: 1, h: a / t };
    },
  };
}
