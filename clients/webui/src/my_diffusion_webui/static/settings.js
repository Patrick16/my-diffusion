// The Settings page. The server's part is built from GET /config (every setting says its type, limits, description and
// how a change takes effect), so a setting added to the server shows up here without a change in this file.
// Changes are collected as a draft and sent together with PATCH /config; the server checks all or nothing.

const PREFS = {};   // this web UI's own preferences (kept in the browser), read by DEFAULT_FORM and the job loops
const PREFS_KEY = 'my-diffusion-webui.prefs.v1';
const PREFS_DEFAULT = { updates: 'live', pollSec: 2, pageSize: 50, rememberForm: true, mode: 'Text2img', resolution: '720', ratio: '16:9' };

const GROUPS = [
  { id: 'paths', title: 'Folders', lead: 'On the server\'s disk. Browse shows only the allowed roots: the model folders in use and the folders the server lists as extra roots' },
  { id: 'models', title: 'Models', lead: '' },
  { id: 'performance', title: 'Performance', lead: '' },
  { id: 'editing', title: 'Editing', lead: 'References, outpainting and the size policy for them' },
  { id: 'queue', title: 'Queue', lead: 'Limits are checked when a job is submitted; jobs already waiting stay in the queue' },
  { id: 'server', title: 'Server', lead: '' },
];
const APPLY_TAG = { live: 'live', reload_models: 'reloads models', restart: 'restart' };

function settingsMixin() {
  return {
    cfg: null, cfgError: '', cfgBusy: false,
    draft: {}, localErrors: {}, serverErrors: {}, applyNote: null,
    groups: GROUPS,
    spick: null,                    // the file picker open on a setting row: {name, names, filter, open}
    fsb: null,                      // the folder browser: {name, data, error}
    prefs: { ...PREFS_DEFAULT },
    listen: { host: '127.0.0.1', port: 8080, lan_address: null, mode: 'local' },
    listenNote: '',

    // ---------------------------------------------------------------- this web UI's preferences
    loadPrefs() {
      try { Object.assign(this.prefs, JSON.parse(localStorage.getItem(PREFS_KEY)) || {}); } catch (e) { /* defaults */ }
      Object.assign(PREFS, this.prefs);
    },
    setPref(key, value) {
      this.prefs[key] = value; Object.assign(PREFS, this.prefs);
      try { localStorage.setItem(PREFS_KEY, JSON.stringify(this.prefs)); } catch (e) { /* private window */ }
      if (key === 'pageSize') { this.limit = value; this.refreshJobs(); }
    },
    resetPrefs() {
      for (const k of Object.keys(this.prefs)) this.prefs[k] = PREFS_DEFAULT[k];
      Object.assign(PREFS, this.prefs); try { localStorage.removeItem(PREFS_KEY); } catch (e) { /* ignore */ }
      this.flash('Interface preferences are back to their defaults');
    },
    async loadListen() {
      try {
        const l = await (await fetch('/ui/listen')).json();
        this.listen = { ...l, mode: ['127.0.0.1', 'localhost', '::1'].includes(l.host) ? 'local' : (l.host === '0.0.0.0' ? 'network' : 'custom') };
      } catch (e) { /* the page still works */ }
    },
    setListenMode(mode) {
      this.listen.mode = mode;
      if (mode === 'local') this.listen.host = '127.0.0.1';
      if (mode === 'network') this.listen.host = '0.0.0.0';
    },
    async saveListen() {
      const r = await fetch('/ui/listen', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ host: this.listen.host, port: Number(this.listen.port) }) });
      if (!r.ok) { this.listenNote = 'Not saved: the address or port is not valid.'; return; }
      this.listenNote = 'Saved. It takes effect when the web UI is started again.'; this.loadListen();
    },

    // ---------------------------------------------------------------- the server's settings
    async loadConfig() {
      this.cfgBusy = true; this.cfgError = '';
      try { this.cfg = await this.api('config'); }
      catch (e) { this.cfg = null; this.cfgError = e.status === 403 ? 'These settings are for admin clients. The key in Connection belongs to a client without admin rights.' : e.message; }
      this.cfgBusy = false;
    },
    items(group) { return this.cfg ? this.cfg.settings.filter(s => s.group === group) : []; },
    label(s) { const t = s.name.replace(/_/g, ' '); return t[0].toUpperCase() + t.slice(1); },
    tags(s) {
      const t = [];
      if (!s.editable) t.push({ text: 'read-only', cls: '' });
      else t.push({ text: APPLY_TAG[s.apply], cls: s.apply === 'live' ? 'live' : s.apply === 'restart' ? 'rst' : 'rel' });
      if (s.source !== 'default') t.push({ text: s.source, cls: '' });
      if (s.name in this.draft) t.push({ text: 'changed', cls: 'chg' });
      if (s.pending_restart) t.push({ text: 'pending', cls: 'chg' });
      return t;
    },
    // Which control a setting gets.
    control(s) {
      if (!s.editable) return 'readonly';
      if (s.name === 'checkpoint' || s.name === 'edit_lora') return 'file';
      if (s.type === 'boolean') return 'bool';
      if (s.choices) return 'choice';
      if (s.type === 'directory') return 'dir';
      if (s.name === 'text_encoder') return 'dir';
      if (s.type === 'integer' || s.type === 'number') return 'number';
      return s.type === 'json' ? 'json' : 'text';
    },
    cur(s) { return s.name in this.draft ? this.draft[s.name] : s.value; },
    // What a field shows: JSON for lists and the like, an empty string for "not set".
    shown(s) {
      const v = this.cur(s);
      if (v === null || v === undefined) return '';
      return typeof v === 'object' ? JSON.stringify(v) : String(v);
    },
    // Raw text from a field -> the value to send. Throws a message when it cannot be one.
    coerce(s, raw) {
      if (s.type === 'boolean' || s.choices) return raw;
      const text = String(raw).trim();
      if (text === '' && s.nullable) return null;
      if (s.type === 'integer' || s.type === 'number') {
        const n = Number(text);
        if (text === '' || !Number.isFinite(n)) throw new Error('a number is needed');
        if (s.type === 'integer' && !Number.isInteger(n)) throw new Error('a whole number is needed');
        return n;
      }
      if (s.type === 'json') { try { return JSON.parse(text); } catch (e) { throw new Error('not valid JSON'); } }
      return String(raw);
    },
    edit(s, raw) {
      const errors = { ...this.localErrors }; delete errors[s.name];
      const sv = { ...this.serverErrors }; delete sv[s.name]; this.serverErrors = sv;
      let value;
      try { value = this.coerce(s, raw); } catch (e) { errors[s.name] = e.message; this.draft = { ...this.draft, [s.name]: raw }; this.localErrors = errors; return; }
      this.localErrors = errors;
      const next = { ...this.draft };
      if (JSON.stringify(value) === JSON.stringify(s.value)) delete next[s.name]; else next[s.name] = value;
      this.draft = next;
    },
    get draftNames() { return Object.keys(this.draft); },
    discard() { this.draft = {}; this.localErrors = {}; this.serverErrors = {}; this.applyNote = null; },
    errorOf(s) { return this.localErrors[s.name] || this.serverErrors[s.name] || ''; },

    // PATCH /config: all or nothing. dry: only check.
    async sendChange(body, dry) {
      this.cfgBusy = true; this.applyNote = null; this.serverErrors = {};
      try {
        const r = await fetch('/api/config', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, dry_run: dry }) });
        const j = await r.json();
        if (r.status === 422) { this.serverErrors = j.errors || {}; this.applyNote = { bad: true, text: j.detail }; }
        else if (!r.ok) this.applyNote = { bad: true, text: typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) };
        else {
          this.cfg = j.config;
          if (!dry) { this.draft = {}; this.localErrors = {}; await this.afterConfigChange(); }
          this.applyNote = { bad: false, dry, applied: j.applied, restart: j.restart_required, unchanged: j.unchanged, reloaded: j.reloaded, warnings: j.warnings };
        }
      } catch (e) { this.applyNote = { bad: true, text: e.message }; }
      this.cfgBusy = false;
    },
    applyChanges(dry) {
      if (Object.keys(this.localErrors).length) { this.applyNote = { bad: true, text: 'Fix the marked values first.' }; return; }
      return this.sendChange({ set: this.draft }, dry);
    },
    async resetSetting(s) {
      const d = { ...this.draft }; delete d[s.name]; this.draft = d;
      await this.sendChange({ reset: [s.name] }, false);
    },
    // Some settings change what the Create page shows (checkpoints, LoRAs, how many references fit).
    async afterConfigChange() { await this.refreshHealth(); await this.loadReference(); },

    // ---------------------------------------------------------------- file pickers on a setting row
    openFilePick(s) {
      if (this.spick && this.spick.name === s.name) { this.spick = null; return; }
      const names = s.name === 'checkpoint' ? this.ckpt.available : this.loraFiles;
      this.spick = { name: s.name, names, filter: '', open: {} };
    },
    spickRows() { return this.spick ? buildTreeRows(this.spick.names, this.spick.filter, this.spick.open) : []; },
    toggleSpick(path) { this.spick.open = { ...this.spick.open, [path]: !this.spick.open[path] }; },
    chooseFile(s, row) { this.edit(s, s.name === 'edit_lora' ? row.name.replace(/\.safetensors$/i, '') : row.name); this.spick = null; },

    // ---------------------------------------------------------------- folder browser (GET /fs)
    async openBrowse(s) {
      if (this.fsb && this.fsb.name === s.name) { this.fsb = null; return; }
      this.fsb = { name: s.name, data: null, error: '' };
      const start = s.name === 'text_encoder' ? this.siblingFolder('models_dir', 'text_encoders') : String(this.cur(s) || '');
      await this.browseTo(start);
    },
    siblingFolder(setting, sub) {
      const m = this.cfg && this.cfg.settings.find(x => x.name === setting);
      return m ? String(this.cur(m)).replace(/[\\/]+$/, '') + (String(m.value).includes('\\') ? '\\' : '/') + sub : '';
    },
    async browseTo(path) {
      try { this.fsb.data = await this.api('fs' + (path ? '?path=' + encodeURIComponent(path) : '')); this.fsb.error = ''; }
      catch (e) {
        this.fsb.error = e.message;
        if (path) { try { this.fsb.data = await this.api('fs'); } catch (e2) { /* the error above says it */ } }   // a bad start: show the roots
      }
    },
    useFolder(s) {
      const p = this.fsb && this.fsb.data && this.fsb.data.path; if (!p) return;
      this.edit(s, s.name === 'text_encoder' ? p.split(/[\\/]/).filter(Boolean).pop() : p);
      this.fsb = null;
    },
  };
}
