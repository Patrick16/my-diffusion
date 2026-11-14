// The web UI's logic: one Alpine component for the whole page. All calls go to this UI's own server
// (/api/... is forwarded to the my-diffusion API, /ui/... is the UI's own settings); the API key never reaches here.

const MODES = ['Text2img', 'Img2img', 'Inpaint', 'Edit', 'Outpaint'];
const FORM_KEY = 'my-diffusion-webui.form.v1';
const PAGE = 50; // gallery page size

const DEFAULT_FORM = () => ({
  prompt: '', negative: '', resolution: PREFS.resolution || '720', ratio: PREFS.ratio || '16:9',
  seed: '', lockSeed: false,
  steps: null, cfg: null, cfgStart: 0, cfgEnd: 1, shift: '',   // null / '' = automatic (the checkpoint's defaults)
  loras: [],                                                    // [{name, weight}]
  // Img2img / Inpaint
  strengthI2I: 0.6, strengthInp: 1.0, feather: 2, iters: 0,
  // Edit
  refMode: 'scene', cutBg: true, editInside: false,
  // Outpaint
  margins: { top: 0, bottom: 0, left: 0, right: 25 }, keepOriginal: true, outCutBg: false,
  // The server's edit LoRA is added by the server for every request with references; a value here changes it for this request
  editLoraWeight: null, editLoraOff: false,
});

// Objects with getters (get x() {...}) must be combined by their property descriptors: a plain spread would run the getters once.
function merge(...parts) {
  const out = {};
  for (const p of parts) Object.defineProperties(out, Object.getOwnPropertyDescriptors(p));
  return out;
}

// Seconds an edit takes on a 16 GB card, from the guide (image-modes.md): [one reference, two references]
const EDIT_SECONDS = { '720': [30, null], '1080': [80, 110], '1440': [170, null] };

// File names with folders ("people/anna.safetensors") -> the rows of a folder tree. A filter opens every folder it matches in.
function buildTreeRows(names, filter, openMap) {
  const needle = (filter || '').trim().toLowerCase(), tree = {};
  for (const name of names) {
    if (needle && !name.toLowerCase().includes(needle)) continue;
    let node = tree; const parts = name.split('/');
    parts.forEach((part, i) => {
      if (i === parts.length - 1) node[part] = { file: name };
      else node = (node[part] = node[part] && !node[part].file ? node[part] : {});
    });
  }
  const rows = [], count = n => Object.values(n).reduce((t, c) => t + (c.file ? 1 : count(c)), 0);
  const walk = (node, depth, prefix) => {
    const keys = Object.keys(node).sort((x, y) => (!!node[x].file - !!node[y].file) || x.localeCompare(y));
    for (const key of keys) {
      const child = node[key];
      if (child.file) { rows.push({ kind: 'file', depth, label: key.replace(/\.(safetensors|gguf)$/i, ''), name: child.file }); continue; }
      const path = prefix + key + '/', open = !!needle || !!openMap[path];
      rows.push({ kind: 'dir', depth, label: key, path, open, hint: count(child) + ' files' });
      if (open) walk(child, depth + 1, path);
    }
  };
  walk(tree, 0, '');
  return rows;
}

function app() {
  return merge(picturesMixin(), maskMixin(), settingsMixin(), {
    // ---------------------------------------------------------------- shared state
    route: { page: 'create', id: null },
    conn: { api_url: '', has_key: false, user_id: '' },
    connForm: { api_url: '', api_key: '', user_id: '' },
    connTest: null, connBusy: false,
    health: null, apiDown: null,       // apiDown: the reason, when the API cannot be reached
    jobs: {},                          // id -> job (the API's JobInfo), kept fresh by the list poll and the event streams
    jobsLoaded: false, exhausted: false, limit: PAGE, watchers: {},
    now: Date.now() / 1000,
    toast: '', zoom: false,

    // ---------------------------------------------------------------- create
    modes: MODES, mode: 'Text2img',
    resolutions: {}, loraFiles: [], ckpt: { current: null, available: [], defaults: {} },
    form: DEFAULT_FORM(),
    currentId: null, submitting: false, formError: '',
    enhancing: false, beforeEnhance: null,
    picker: null,                      // 'checkpoint' | 'lora' | null
    pickFilter: '', pickOpen: {}, ckptBusy: false,

    // ---------------------------------------------------------------- gallery / detail
    filters: { status: 'all', mode: 'all', q: '' },
    statusFilters: ['all', 'done', 'running', 'queued', 'failed', 'cancelled'],
    modeFilters: ['all', 'text', 'img2img', 'inpaint', 'edit', 'outpaint'],

    // ================================================================ start
    async init() {
      this.loadPrefs(); this.limit = this.prefs.pageSize; this.mode = this.prefs.mode; this.form = DEFAULT_FORM();
      this.restoreForm();
      this.$watch('form', () => this.saveForm());
      window.addEventListener('hashchange', () => this.readRoute());
      window.addEventListener('paste', ev => this.onPaste(ev));
      this.readRoute();
      setInterval(() => { this.now = Date.now() / 1000; }, 1000);
      await this.loadConnection();
      this.loopHealth();
      this.loopJobs();
      this.loadReference();
    },

    readRoute() {
      const [page, id] = location.hash.replace(/^#\/?/, '').split('/');
      this.route = { page: ['create', 'gallery', 'job', 'settings'].includes(page) ? page : 'create', id: id || null };
      if (this.route.page === 'job' && this.route.id && !this.jobs[this.route.id]) this.fetchJob(this.route.id);
      window.scrollTo(0, 0);
      if (this.route.page === 'settings') { this.loadConfig(); this.loadListen(); }
    },

    // ================================================================ talking to the API
    async api(path, { method = 'GET', body } = {}) {
      let response;
      try {
        response = await fetch('/api/' + path, {
          method, headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
        });
      } catch (e) {
        throw new Error('the web UI server does not answer');
      }
      if (!response.ok) {
        let detail = response.statusText;
        try { const j = await response.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not JSON */ }
        const err = new Error(detail); err.status = response.status; throw err;
      }
      return response.json();
    },

    flash(text) { this.toast = text; clearTimeout(this._toastTimer); this._toastTimer = setTimeout(() => { this.toast = ''; }, 4000); },

    // ---------------------------------------------------------------- connection
    async loadConnection() {
      this.conn = await (await fetch('/ui/connection')).json();
      this.connForm = { api_url: this.conn.api_url, api_key: '', user_id: this.conn.user_id };
    },
    async saveConnection() {
      this.connBusy = true; this.connTest = null;
      try {
        const body = { api_url: this.connForm.api_url, user_id: this.connForm.user_id };
        if (this.connForm.api_key !== '') body.api_key = this.connForm.api_key;   // an empty field keeps the saved key
        const r = await fetch('/ui/connection', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) { const j = await r.json(); throw new Error(Array.isArray(j.detail) ? j.detail.map(d => d.msg).join('; ') : j.detail); }
        await this.loadConnection();
        await this.testConnection();
        this.refreshHealth(); this.refreshJobs(); this.loadReference();
      } catch (e) { this.connTest = { error: e.message }; }
      this.connBusy = false;
    },
    async clearKey() {
      await fetch('/ui/connection', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ api_key: '' }) });
      await this.loadConnection(); this.testConnection();
    },
    async testConnection() {
      this.connBusy = true;
      try { this.connTest = await (await fetch('/ui/connection/test', { method: 'POST' })).json(); }
      catch (e) { this.connTest = { error: e.message }; }
      this.connBusy = false;
    },

    // ---------------------------------------------------------------- health (the top bar)
    async refreshHealth() {
      try { this.health = await this.api('health'); this.apiDown = null; }
      catch (e) { this.health = null; this.apiDown = e.message; }
    },
    loopHealth() { this.refreshHealth().finally(() => setTimeout(() => this.loopHealth(), this.activeJobs.length ? 3000 : 8000)); },

    get ready() { return this.health && this.health.status === 'ok'; },
    get queueIdle() {
      const q = this.health && this.health.queue;
      return !!q && q.waiting === 0 && q.slots.every(s => !s.job);
    },
    get statusPill() {
      if (this.apiDown) return { dot: '#ff9aa8', text: 'API not reachable' };
      if (!this.health) return { dot: '#9aa3b2', text: 'Connecting…' };
      const h = this.health;
      if (h.status === 'error') return { dot: '#ff9aa8', text: 'Models failed to load' };
      if (h.status === 'loading') return { dot: '#f5a35c', text: 'Models loading…' };
      const d = this.ckptDefaults();
      return { dot: '#5fd39a', text: `Models ready · ${this.stem(h.checkpoint)}${d ? ' · ' + d.steps + ' steps' : ''}` };
    },
    get queuePill() {
      const q = this.health && this.health.queue;
      if (!q) return null;
      const running = q.slots.filter(s => s.job).length;
      return { dot: running || q.waiting ? '#f5a35c' : '#5fd39a', text: running || q.waiting ? `${running} running · ${q.waiting} waiting` : 'Queue is empty' };
    },
    ckptDefaults() { return this.health && this.ckpt.defaults ? this.ckpt.defaults[this.health.kind] : null; },

    // ---------------------------------------------------------------- reference data: sizes, LoRAs, checkpoints
    async loadReference() {
      try { this.resolutions = await this.api('resolutions'); } catch (e) { /* the pill already says the API is down */ }
      if (!(this.form.resolution in this.resolutions)) this.form.resolution = Object.keys(this.resolutions)[0] || '720';
      try { this.loraFiles = (await this.api('loras')).map(n => n.replace(/\.safetensors$/i, '')); } catch (e) { this.loraFiles = []; }
      try { this.ckpt = await this.api('checkpoints'); } catch (e) { /* needs a key; shown in Settings */ }
      try {   // admins only: how many references fit and which LoRA the server adds to an edit
        const cfg = await this.api('config'), val = n => (cfg.settings.find(x => x.name === n) || {}).value;
        this.maxRefs = val('max_references') || this.maxRefs; this.editLora = val('edit_lora') || this.editLora;
        this.editLoraWeight = val('edit_lora_weight') ?? this.editLoraWeight;
      } catch (e) { /* not an admin: the defaults are the server's defaults */ }
    },
    get tiers() { return Object.keys(this.resolutions); },
    // 'auto' (the exact proportions of image 1) exists only for an edit in scene mode that has an image 1 to take them from
    get autoAllowed() {
      if (this.mode !== 'Edit' || this.form.refMode !== 'scene') return false;
      return this.form.editInside ? !!this.pics.ed1 : !!this.pics.ed0;
    },
    get ratio() { return this.form.ratio === 'auto' && !this.autoAllowed ? '16:9' : this.form.ratio; },
    get ratios() {
      const names = Object.keys(this.resolutions[this.form.resolution] || {});
      return this.mode === 'Edit' && this.form.refMode === 'scene' ? ['auto', ...names] : names;
    },
    get showRatios() { return this.mode !== 'Outpaint'; },
    // The pixel budget of a tier is the area of its 1:1 frame (16:9 at 720 is 1280 x 720 = 960 x 960).
    budget(tier) { const t = this.resolutions[tier] || {}, d = t['1:1'] || Object.values(t)[0]; return d ? d.width * d.height : 0; },
    frameFor(aspect, tier) {
      const b = this.budget(tier), r16 = v => Math.max(16, Math.round(v / 16) * 16);
      return b ? { w: r16(Math.sqrt(b * aspect)), h: r16(Math.sqrt(b / aspect)) } : null;
    },
    get sizeNow() {
      let frame = null;
      if (this.mode === 'Outpaint') frame = this.outFrame && this.outFrame.size;
      else if (this.ratio === 'auto') { const p = this.pics.ed0; frame = p ? this.frameFor(p.w / p.h, this.form.resolution) : null; }
      else { const d = (this.resolutions[this.form.resolution] || {})[this.ratio]; frame = d ? { w: d.width, h: d.height } : null; }
      return frame ? `${frame.w} × ${frame.h}` : '';
    },
    ratioBox(label) {
      const d = (this.resolutions[this.form.resolution] || {})[label === 'auto' ? '1:1' : label];
      if (!d) return '';
      const k = 26 / Math.max(d.width, d.height);
      return `width:${(d.width * k).toFixed(1)}px;height:${(d.height * k).toFixed(1)}px`;
    },
    // Outpaint: the frame is the picture plus the margins (in percent of its own size), drawn at the tier's pixel budget.
    get outFrame() {
      const p = this.pics.op0; if (!p) return null;
      const m = this.form.margins, n = k => Math.min(300, Math.max(0, Number(m[k]) || 0));
      const fw = p.w * (1 + (n('left') + n('right')) / 100), fh = p.h * (1 + (n('top') + n('bottom')) / 100);
      return { size: this.frameFor(fw / fh, this.form.resolution), aspect: fw / fh,
               box: { x: p.w * n('left') / 100 / fw, y: p.h * n('top') / 100 / fh, w: p.w / fw, h: p.h / fh },
               any: ['top', 'bottom', 'left', 'right'].some(k => n(k) > 0) };
    },
    // Big edits run, but slowly; the server decides by its edit_limits policy.
    get editImages() {
      if (this.mode === 'Edit') return this.slotDefs.filter(s => this.pics[s.id]).length;
      if (this.mode === 'Outpaint') return 1 + (this.pics.op1 ? 1 : 0);
      if (this.mode === 'Inpaint' && this.pics.inpRef) return 2;
      return 0;
    },
    get editWarning() {
      const n = this.editImages; if (!n) return '';
      const secs = (EDIT_SECONDS[this.form.resolution] || [])[n - 1];
      if (!(this.form.resolution === '1080' || this.form.resolution === '1440' || n >= 2 && this.form.resolution !== '480')) return '';
      const policy = this.health && this.health.edit_limits;
      return `Large edit${secs ? `: about ${secs} s on a 16 GB GPU` : ': it will take a while'}; the network runs with part of its blocks streamed from RAM. ` +
        (policy === 'strict' ? 'The server policy is "strict": it may refuse this size.' : policy === 'warn' ? 'The server accepts it (policy "warn") and adds a warning to the job.' : '');
    },
    // The edit LoRA the server adds by itself: shown as an "auto" row, sent only when changed for this request.
    get editLoraVisible() { return this.mode === 'Edit' || this.mode === 'Outpaint' || (this.mode === 'Inpaint' && !!this.pics.inpRef); },
    get editLoraShown() { return this.form.editLoraOff ? 0 : (this.form.editLoraWeight ?? this.editLoraWeight); },

    // ---------------------------------------------------------------- the form is remembered between visits
    saveForm() { if (!this.prefs.rememberForm) return; try { localStorage.setItem(FORM_KEY, JSON.stringify(this.form)); } catch (e) { /* private window: fine */ } },
    restoreForm() {
      if (!this.prefs.rememberForm) return;
      try { const s = JSON.parse(localStorage.getItem(FORM_KEY)); if (s) this.form = { ...DEFAULT_FORM(), ...s }; } catch (e) { /* ignore */ }
    },

    // ---------------------------------------------------------------- prompt: lines starting with # are the user's own notes
    promptToSend() {
      return this.form.prompt.split('\n').filter(l => !l.trimStart().startsWith('#')).join('\n').trim();
    },
    async enhance() {
      const idea = this.promptToSend();
      if (!idea) { this.formError = 'Write a prompt first (lines starting with # are not used).'; return; }
      this.formError = ''; this.enhancing = true;
      try {
        const r = await this.api('enhance-prompt', { method: 'POST', body: { prompt: idea } });
        this.beforeEnhance = this.form.prompt;
        const comment = idea.split('\n').map(l => '# ' + l).join('\n');
        this.form.prompt = comment + '\n' + r.enhanced_prompt;
      } catch (e) { this.formError = 'Enhance failed: ' + e.message; }
      this.enhancing = false;
    },
    undoEnhance() { if (this.beforeEnhance !== null) { this.form.prompt = this.beforeEnhance; this.beforeEnhance = null; } },

    // ---------------------------------------------------------------- file picker (checkpoints, LoRAs): a folder tree with a filter
    openPicker(kind) { this.picker = this.picker === kind ? null : kind; this.pickFilter = ''; this.pickOpen = {}; },
    pickerRows() {
      return buildTreeRows(this.picker === 'checkpoint' ? this.ckpt.available : this.loraFiles, this.pickFilter, this.pickOpen);
    },
    toggleDir(path) { this.pickOpen = { ...this.pickOpen, [path]: !this.pickOpen[path] }; },
    pickFile(row) {
      if (this.picker === 'lora') {
        if (!this.form.loras.some(l => l.name === row.name.replace(/\.safetensors$/i, ''))) this.form.loras.push({ name: row.name.replace(/\.safetensors$/i, ''), weight: 1 });
        this.picker = null;
      }
    },
    loraAdded(name) { return this.form.loras.some(l => l.name === name.replace(/\.safetensors$/i, '')); },
    async switchCheckpoint(name) {
      if (!this.queueIdle) return;
      this.ckptBusy = true;
      try {
        await this.api('checkpoints/load', { method: 'POST', body: { name } });
        this.ckpt = await this.api('checkpoints'); await this.refreshHealth();
        this.flash('Checkpoint switched: ' + this.stem(name)); this.picker = null;
      } catch (e) { this.flash('Switch failed: ' + e.message); }
      this.ckptBusy = false;
    },

    // ================================================================ generate
    async buildRequest() {
      const f = this.form, m = this.mode;
      const body = { prompt: this.promptToSend(), resolution: f.resolution };
      if (m !== 'Outpaint') body.aspect_ratio = this.ratio;
      if (!body.prompt) throw new Error('Write a prompt first (lines starting with # are not used).');
      if (f.negative.trim()) body.negative_prompt = f.negative.trim();
      if (String(f.seed).trim() !== '') {
        body.seed = Number(f.seed);
        if (!Number.isInteger(body.seed) || body.seed < 0) throw new Error('The seed must be a whole number.');
      }
      if (f.steps !== null) body.steps = f.steps;
      if (f.cfg !== null) body.guidance_scale = f.cfg;
      body.guidance_start = f.cfgStart; body.guidance_end = f.cfgEnd;
      if (String(f.shift).trim() !== '') body.shift = parseFloat(f.shift);
      const loras = f.loras.map(l => ({ name: l.name, weight: Number(l.weight) }));
      if (this.editLoraVisible && (f.editLoraOff || f.editLoraWeight !== null) && !loras.some(l => l.name === this.editLora)) {
        loras.push({ name: this.editLora, weight: f.editLoraOff ? 0 : Number(f.editLoraWeight) });
      }
      if (loras.length) body.loras = loras;

      const need = (slot, text) => { if (!this.pics[slot]) throw new Error(text); };
      const withMask = async () => {
        if (this.mask.empty) throw new Error('Paint the area to regenerate on the picture (white = regenerated).');
        body.mask = await this.toDataUrl(await this.maskExport()); body.feather = Number(f.feather);
        body.strength = Number(f.strengthInp);
        if (f.iters) body.inpaint_iterations = f.iters;
      };
      if (m === 'Img2img') {
        need('init', 'Add an init image.');
        body.init_image = await this.encode('init'); body.strength = Number(f.strengthI2I);
      } else if (m === 'Inpaint') {
        need('init', 'Add an init image.');
        body.init_image = await this.encode('init'); await withMask();
        if (this.pics.inpRef) { body.references = [await this.encode('inpRef')]; body.reference_mode = 'scene'; }
      } else if (m === 'Edit') {
        need('ed0', 'Add reference 1 (image 1).');
        const ids = this.slotDefs.map(s => s.id).filter(id => this.pics[id]);
        let refs = ids;
        if (f.editInside) {
          if (f.refMode !== 'scene') throw new Error('Editing inside image 1 needs the Scene reference mode.');
          body.init_image = await this.encode('ed0'); await withMask(); refs = ids.slice(1);
        }
        if (refs.length) body.references = await Promise.all(refs.map(id => this.encode(id)));
        body.reference_mode = f.refMode;
        if (f.cutBg) body.remove_background = 'subjects';
      } else if (m === 'Outpaint') {
        need('op0', 'Add the picture to extend.');
        if (!this.outFrame.any) throw new Error('Set a margin above 0 on at least one side.');
        const mg = f.margins, n = k => Math.min(300, Math.max(0, Number(mg[k]) || 0));
        body.references = [await this.encode('op0')]; if (this.pics.op1) body.references.push(await this.encode('op1'));
        body.reference_mode = 'scene';
        body.outpaint = { top: n('top'), bottom: n('bottom'), left: n('left'), right: n('right') };
        body.keep_original = f.keepOriginal;
        if (f.outCutBg && this.pics.op1) body.remove_background = 'subjects';
      }
      return body;
    },
    async generate() {
      this.formError = ''; this.submitting = true;
      try {
        const job = await this.api('generate', { method: 'POST', body: await this.buildRequest() });
        this.setJob(job); this.currentId = job.job_id; this.watch(job.job_id);
        if (!this.form.lockSeed) this.form.seed = '';   // a typed seed is used once, unless it is locked
      } catch (e) { this.formError = e.status === 429 ? 'The queue is full: ' + e.message : e.message; }
      this.submitting = false;
    },
    dice() { this.form.seed = String(Math.floor(Math.random() * 2 ** 31)); },
    async cancel(id) {
      try { this.setJob(await this.api('cancel/' + id, { method: 'POST' })); } catch (e) { this.flash('Cancel failed: ' + e.message); }
    },

    // ================================================================ jobs: the list poll and the event streams
    setJob(job) { this.jobs = { ...this.jobs, [job.job_id]: job }; },
    async fetchJob(id) { try { this.setJob(await this.api('result/' + id)); } catch (e) { this.flash('Job not found: ' + e.message); } },
    async refreshJobs() {
      try {
        const list = await this.api('jobs?limit=' + this.limit);
        const next = { ...this.jobs };
        for (const j of list) next[j.job_id] = (this.watchers[j.job_id] && next[j.job_id] && !['done', 'failed', 'cancelled'].includes(j.status)) ? { ...j, ...this.freshest(next[j.job_id], j) } : j;
        this.jobs = next; this.jobsLoaded = true; this.exhausted = list.length < this.limit;
        for (const j of list) if (['queued', 'running'].includes(j.status)) this.watch(j.job_id);
      } catch (e) { /* the health pill reports the problem */ }
    },
    freshest(a, b) { return (a.step || 0) > (b.step || 0) ? a : b; },
    loopJobs() { this.refreshJobs().finally(() => setTimeout(() => this.loopJobs(), this.activeJobs.length ? (this.prefs.updates === 'polling' ? this.prefs.pollSec * 1000 : 4000) : 10000)); },
    loadMore() { this.limit += this.prefs.pageSize; this.refreshJobs(); },

    // The progress of a waiting or running job comes as server-sent events. The browser's EventSource cannot be used
    // for it in general (no headers), but here the key is added by our own server, so a plain fetch stream is enough.
    async watch(id) {
      if (this.watchers[id] || this.prefs.updates === 'polling') return;   // polling mode: the list poll alone keeps the state
      this.watchers[id] = true;
      try {
        const response = await fetch(`/api/result/${id}/events`, { headers: { Accept: 'text/event-stream' } });
        if (!response.ok || !response.body) throw new Error('no stream');
        const reader = response.body.getReader(), decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let cut;
          while ((cut = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
            const data = block.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('');
            if (data) this.setJob(JSON.parse(data));
          }
        }
      } catch (e) { /* the list poll keeps the state right if a stream breaks */ }
      delete this.watchers[id];
      this.refreshJobs();
    },

    get jobList() { return Object.values(this.jobs).sort((a, b) => b.created_at - a.created_at); },
    get activeJobs() { return this.jobList.filter(j => j.status === 'queued' || j.status === 'running'); },
    get runningJobs() { return this.activeJobs.filter(j => j.status === 'running'); },
    get waitingJobs() { return this.activeJobs.filter(j => j.status === 'queued').sort((a, b) => a.position - b.position); },
    get current() {
      return (this.currentId && this.jobs[this.currentId]) || this.runningJobs[0] || this.waitingJobs[0] || this.jobList[0] || null;
    },
    get recent() { return this.jobList.filter(j => j.status === 'done').slice(0, 8); },

    // ---------------------------------------------------------------- gallery
    modeOf(j) {
      if (j.result && j.result.mode) {
        const m = j.result.mode;
        if (m === 'edit') return j.result.outpaint ? 'outpaint' : 'edit';
        return m === 'text2image' ? 'text' : m;
      }
      const k = Object.keys(j.inputs || {});
      if (k.includes('mask')) return 'inpaint';
      if (k.includes('init')) return 'img2img';
      if (k.some(n => n.startsWith('ref'))) return (j.params.outpaint && Object.values(j.params.outpaint).some(v => v)) ? 'outpaint' : 'edit';
      return 'text';
    },
    get filtered() {
      const q = this.filters.q.trim().toLowerCase();
      return this.jobList.filter(j =>
        (this.filters.status === 'all' || j.status === this.filters.status) &&
        (this.filters.mode === 'all' || this.modeOf(j) === this.filters.mode) &&
        (!q || (j.params.prompt || '').toLowerCase().includes(q)));
    },
    get doneCount() { return this.jobList.filter(j => j.status === 'done').length; },
    tileMeta(j) {
      const r = j.result;
      if (j.status === 'done' && r) return `${r.width}×${r.height} · seed ${r.seed} · ${this.seconds(j)}`;
      if (j.status === 'running') return `step ${j.step || 0} / ${j.total_steps || '?'}`;
      if (j.status === 'queued') return `waiting · position ${j.position}`;
      return j.status;
    },
    seconds(j) { return j.started_at && j.finished_at ? Math.round(j.finished_at - j.started_at) + ' s' : ''; },
    aspect(j) { const r = j.result; return r ? `${r.width}/${r.height}` : '16/9'; },

    // ---------------------------------------------------------------- detail
    get detail() { return this.route.id ? this.jobs[this.route.id] : null; },
    get detailIndex() { return this.filtered.findIndex(j => j.job_id === this.route.id); },
    go(id) { location.hash = '#/job/' + id; },
    step(delta) {
      const done = this.filtered.filter(j => j.status === 'done');
      const i = done.findIndex(j => j.job_id === this.route.id), n = done[i + delta];
      if (n) this.go(n.job_id);
    },
    // Puts a job's settings and (while the server still has them) its input pictures back into the form.
    async reuse(j, newSeed) {
      const p = j.params, kind = this.modeOf(j), f = DEFAULT_FORM();
      const mode = { text: 'Text2img', img2img: 'Img2img', inpaint: 'Inpaint', edit: 'Edit', outpaint: 'Outpaint' }[kind];
      const loras = [];
      for (const l of p.loras || []) {
        if (l.name !== this.editLora) loras.push({ name: l.name, weight: l.weight });
        else if (l.weight === 0) f.editLoraOff = true;
        else if (l.weight !== this.editLoraWeight) f.editLoraWeight = l.weight;
      }
      Object.assign(f, {
        prompt: p.prompt || '', negative: p.negative_prompt || '', resolution: String(p.resolution || '720'),
        ratio: p.aspect_ratio || '16:9', seed: newSeed ? '' : String(j.result ? j.result.seed : (p.seed ?? '')),
        steps: p.steps ?? null, cfg: p.guidance_scale ?? null, cfgStart: p.guidance_start ?? 0, cfgEnd: p.guidance_end ?? 1,
        shift: p.shift ?? '', loras,
        strengthI2I: kind === 'img2img' ? (p.strength ?? 0.6) : 0.6, strengthInp: kind === 'inpaint' ? (p.strength ?? 1) : 1,
        feather: p.feather ?? 2, iters: p.inpaint_iterations || 0,
        refMode: p.reference_mode || 'scene', cutBg: p.remove_background === 'subjects', outCutBg: p.remove_background === 'subjects',
        keepOriginal: !!p.keep_original,
      });
      if (p.outpaint) f.margins = { top: p.outpaint.top, bottom: p.outpaint.bottom, left: p.outpaint.left, right: p.outpaint.right };
      this.form = f; this.mode = mode; location.hash = '#/create';

      const slots = { img2img: { init: 'init' }, inpaint: { init: 'init', ref1: 'inpRef' }, edit: { ref1: 'ed0', ref2: 'ed1' }, outpaint: { ref1: 'op0', ref2: 'op1' } }[kind] || {};
      for (const slot of ['init', 'inpRef', 'ed0', 'ed1', 'op0', 'op1']) this.clearPicture(slot);
      const missing = [];
      for (const [name, url] of Object.entries(j.inputs || {})) {
        if (name === 'mask' || !slots[name]) continue;
        try { await this.pictureFromUrl(slots[name], '/api' + url, name); } catch (e) { missing.push(name); }
      }
      if (kind === 'inpaint' && j.inputs && j.inputs.mask && this.pics.init) {
        try {
          const r = await fetch('/api' + j.inputs.mask);
          if (!r.ok) throw new Error('expired');
          const blob = await r.blob();
          await this.$nextTick(); await this.$nextTick();   // the editor sets itself up for the new picture first
          await this.maskLoadBlob(blob);
        } catch (e) { missing.push('mask'); }
      }
      if (missing.length) this.flash('Settings loaded. These pictures are gone from the server (kept 7 days): ' + missing.join(', ') + '. Add them again.');
    },
    async useAsInit(j) {
      this.mode = 'Img2img'; location.hash = '#/create';
      try { await this.pictureFromUrl('init', `/api/result/${j.job_id}/image`, j.job_id.slice(0, 8) + '.png'); } catch (e) { this.flash(e.message); }
    },
    async useAsReference(j) {
      this.mode = 'Edit'; location.hash = '#/create';
      const slot = ['ed0', 'ed1'].slice(0, this.maxRefs).find(id => !this.pics[id]) || 'ed0';
      try { await this.pictureFromUrl(slot, `/api/result/${j.job_id}/image`, j.job_id.slice(0, 8) + '.png'); } catch (e) { this.flash(e.message); }
    },
    copy(text) { navigator.clipboard.writeText(String(text)).then(() => this.flash('Copied'), () => this.flash('Copy failed')); },

    // ================================================================ small formatters
    stem(name) { return (name || '').replace(/\.(safetensors|gguf)$/i, ''); },
    pct(j) { return j && j.total_steps ? Math.round(((j.step || 0) / j.total_steps) * 100) : 0; },
    ago(t) {
      if (!t) return '';
      const s = Math.max(0, Math.round(this.now - t));
      if (s < 60) return s + ' s ago';
      if (s < 3600) return Math.round(s / 60) + ' min ago';
      if (s < 86400) return Math.round(s / 3600) + ' h ago';
      return new Date(t * 1000).toLocaleDateString();
    },
    when(t) { return t ? new Date(t * 1000).toLocaleString() : ''; },
    statusColor(s) { return { done: '#8fe0b8', failed: '#ff9aa8', cancelled: '#c3c9d6' }[s] || '#f5c98a'; },
    kv(obj) { return Object.entries(obj || {}); },
    interval(r) { return r ? r.guidance_interval.join(' – ') : ''; },
  });
}
