/**
 * shop.js — In-game Armory: Weapons, Aircraft, LocalStorage with anti-tamper
 *
 * Anti-cheat: save data is signed with a djb2 hash + secret key.
 * If the stored checksum doesn't match, the save resets to default.
 */

// ── Secret (split to make naive inspection harder) ────────────────────────────
const _K = ['ps_', 'gm_', '4r', 'm0', 'ry_', 'K3y', '_v1'];
const _SECRET = _K.join('');

function _djb2(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++)
        h = (Math.imul(h, 33) ^ str.charCodeAt(i)) >>> 0;
    return h.toString(36);
}

// ── XOR cipher helpers (file encryption) ─────────────────────────────────────
// Produces binary output that is unreadable as plain text.
// Security level: same as the signature — deters casual inspection.
const _FILE_MAGIC = [0x50, 0x53, 0x56, 0x31]; // 'PSV1'

function _xorEncrypt(jsonStr) {
    const key   = new TextEncoder().encode(_SECRET);
    const bytes = new TextEncoder().encode(jsonStr);
    const out   = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++)
        out[i] = bytes[i] ^ key[i % key.length];
    return out;
}

function _xorDecrypt(buffer) {
    const magic = new Uint8Array(buffer, 0, 4);
    for (let i = 0; i < 4; i++)
        if (magic[i] !== _FILE_MAGIC[i]) throw new Error('Bad magic');
    const key     = new TextEncoder().encode(_SECRET);
    const payload = new Uint8Array(buffer, 4);
    const plain   = new Uint8Array(payload.length);
    for (let i = 0; i < payload.length; i++)
        plain[i] = payload[i] ^ key[i % key.length];
    return new TextDecoder().decode(plain);
}

// ── Weapon Catalog ────────────────────────────────────────────────────────────
// fireRate = shoot cooldown frames (lower = faster)
// speed    = bullet pixels/frame
// range    = fraction of screen height (1.0 = full screen, 0.3 = short)
// spread   = bullets per shot
const WEAPONS = {
    m2browning: {
        id: 'm2browning', name: 'M2 Browning', icon: '🔫',
        desc: '.50 cal HMG — Reliable starter',
        damage: 1, fireRate: 18, speed: 16, range: 1.0, spread: 1,
        color: '#FFD700', cost: 0,
    },
    hispano: {
        id: 'hispano', name: 'Hispano', icon: '🔫',
        desc: '20mm cannon — Balanced & punchy',
        damage: 2, fireRate: 13, speed: 17, range: 1.0, spread: 1,
        color: '#00E5FF', cost: 150,
    },
    defa: {
        id: 'defa', name: 'DEFA 30mm', icon: '🔫',
        desc: 'Revolver cannon ~1500rpm — High damage',
        damage: 3, fireRate: 10, speed: 18, range: 1.0, spread: 1,
        color: '#FF9800', cost: 300,
    },
    gsh: {
        id: 'gsh', name: 'GSh-30-1', icon: '🔫',
        desc: '30mm ~1800rpm — Fast & lethal',
        damage: 4, fireRate: 8, speed: 18, range: 1.0, spread: 1,
        color: '#FF6B35', cost: 500,
    },
    vulcan: {
        id: 'vulcan', name: 'M61 Vulcan', icon: '🌀',
        desc: '20mm gatling 6000rpm — Bullet hose',
        damage: 1, fireRate: 3, speed: 20, range: 1.0, spread: 3,
        color: '#CE93D8', cost: 700,
    },
    ironbeam: {
        id: 'ironbeam', name: 'Iron Beam', icon: '⚡',
        desc: 'DEW Laser — Instant & lethal, short range',
        damage: 8, fireRate: 5, speed: 40, range: 0.35, spread: 1,
        color: '#FF1744', isLaser: true, cost: 1200,
    },
};

// ── Aircraft Catalog ──────────────────────────────────────────────────────────
// speed  = px/frame movement
// lives  = starting lives
// width  = hitbox width (smaller = harder to hit = better)
const PLANES = {
    tiger: {
        id: 'tiger', name: 'Tiger', icon: '🛡️',
        desc: 'F-5 Tiger — Light & affordable',
        speed: 5, lives: 3, width: 38, cost: 0, col: '#90CAF9',
    },
    fulcrum: {
        id: 'fulcrum', name: 'Fulcrum', icon: '✈',
        desc: 'MiG-29 — Nimble frontline fighter',
        speed: 5.5, lives: 3, width: 36, cost: 200, col: '#4FC3F7',
    },
    falcon: {
        id: 'falcon', name: 'Falcon', icon: '🦅',
        desc: 'F-16 Fighting Falcon — Agile & lethal',
        speed: 6, lives: 3, width: 34, cost: 400, col: '#81D4FA',
    },
    flanker: {
        id: 'flanker', name: 'Flanker', icon: '🔱',
        desc: 'Su-27 — Heavy brawler, extra durability',
        speed: 5.5, lives: 4, width: 42, cost: 600, col: '#F48FB1',
    },
    rafale: {
        id: 'rafale', name: 'Rafale', icon: '🇫🇷',
        desc: 'Dassault Rafale — Speed & agility',
        speed: 6.5, lives: 3, width: 33, cost: 900, col: '#A5D6A7',
    },
    raptor: {
        id: 'raptor', name: 'Raptor', icon: '👻',
        desc: 'F-22 Raptor — Stealth apex predator',
        speed: 7, lives: 5, width: 30, cost: 1500, col: '#FFE082',
    },
};

// ── GameStore ─────────────────────────────────────────────────────────────────
class GameStore {
    constructor() { this.state = null; this.load(); }

    _sign(data) {
        return _djb2(JSON.stringify(data) + _SECRET);
    }

    load() {
        try {
            const raw = localStorage.getItem('pshooter_v1');
            if (!raw) { this._init(); return; }
            const wrapped = JSON.parse(raw);
            const data    = JSON.parse(wrapped.d);

            // ── Step 1: Signature check (detects naive localStorage edits) ──────
            if (this._sign(data) !== wrapped.s) {
                console.warn('[Store] Tampered save — signature mismatch. Resetting.');
                this._init(); return;
            }

            // ── Step 2: Type & structure validation ───────────────────────
            if (typeof data.points !== 'number' || typeof data.score !== 'number' ||
                !Array.isArray(data.ow) || !Array.isArray(data.op) ||
                typeof data.ew !== 'string' || typeof data.ep !== 'string') {
                console.warn('[Store] Corrupt structure. Resetting.');
                this._init(); return;
            }

            // ── Step 3: Plausibility check ────────────────────────────
            // Points earned can ONLY come from score. Max: floor(score/10) + small buffer.
            // This is the key check that stops someone who edits points directly.
            const maxLegitPoints = Math.floor(data.score / 10);
            if (data.points > maxLegitPoints + 5) {  // +5 = rounding buffer
                console.warn('[Store] Points exceed possible earn rate. Resetting.');
                this._init(); return;
            }

            // ── Step 4: Clamp to safe ranges ───────────────────────────
            data.points = Math.max(0, Math.floor(data.points));
            data.score  = Math.max(0, Math.floor(data.score));

            // ── Step 5: Validate owned items are real catalog IDs ─────────────
            data.ow = data.ow.filter(id => WEAPONS[id]);
            data.op = data.op.filter(id => PLANES[id]);
            if (!data.ow.includes('m2browning')) data.ow.unshift('m2browning'); // starter always present
            if (!data.op.includes('tiger'))      data.op.unshift('tiger');
            if (!WEAPONS[data.ew]) data.ew = 'm2browning';
            if (!PLANES[data.ep])  data.ep = 'tiger';

            this.state = data;
        } catch(e) {
            console.warn('[Store] Parse error. Resetting.');
            this._init();
        }
    }

    save() {
        const d = JSON.stringify(this.state);
        localStorage.setItem('pshooter_v1', JSON.stringify({ d, s: this._sign(this.state) }));
    }

    _init() {
        this.state = { points: 0, score: 0, ow: ['m2browning'], ew: 'm2browning', op: ['tiger'], ep: 'tiger' };
        this.save();
    }

    addScore(amount) {
        amount = Math.max(0, Math.floor(amount));
        this.state.score  += amount;
        this.state.points += Math.floor(amount / 10);
        // Do NOT call save() here — called explicitly at gameover/menu to avoid
        // synchronous localStorage writes every kill frame (causes freeze).
    }

    buy(type, id) {
        const catalog = type === 'weapon' ? WEAPONS : PLANES;
        const listKey = type === 'weapon' ? 'ow' : 'op';
        const item = catalog[id];
        if (!item || this.state[listKey].includes(id)) return false;
        if (this.state.points < item.cost) return false;
        this.state.points -= item.cost;
        this.state[listKey].push(id);
        this.save();
        return true;
    }

    equip(type, id) {
        const listKey = type === 'weapon' ? 'ow' : 'op';
        const equipKey = type === 'weapon' ? 'ew' : 'ep';
        if (!this.state[listKey].includes(id)) return false;
        this.state[equipKey] = id;
        this.save();
        return true;
    }

    getWeapon() { return WEAPONS[this.state.ew] || WEAPONS.m2browning; }
    getPlane()  { return PLANES[this.state.ep]  || PLANES.tiger; }

    /**
     * Export save as a portable base64 string.
     * The code contains the same signed payload stored in localStorage,
     * so it passes all the same anti-tamper checks on import.
     */
    exportCode() {
        this.save(); // ensure latest data is in state
        const d = JSON.stringify(this.state);
        const wrapped = JSON.stringify({ d, s: this._sign(this.state) });
        return btoa(wrapped);
    }

    /**
     * Import a save code produced by exportCode().
     * Runs the FULL validation pipeline (signature + plausibility).
     * Returns { ok: true } or { ok: false, reason: string }.
     */
    importCode(code) {
        try {
            const wrapped = JSON.parse(atob(code.trim()));
            const data    = JSON.parse(wrapped.d);

            if (this._sign(data) !== wrapped.s)
                return { ok: false, reason: 'Invalid code (signature mismatch).' };

            if (typeof data.points !== 'number' || typeof data.score !== 'number' ||
                !Array.isArray(data.ow) || !Array.isArray(data.op) ||
                typeof data.ew !== 'string' || typeof data.ep !== 'string')
                return { ok: false, reason: 'Invalid code (corrupt data).' };

            const maxLegit = Math.floor(data.score / 10);
            if (data.points > maxLegit + 5)
                return { ok: false, reason: 'Invalid code (implausible points).' };

            data.points = Math.max(0, Math.floor(data.points));
            data.score  = Math.max(0, Math.floor(data.score));
            data.ow = data.ow.filter(id => WEAPONS[id]);
            data.op = data.op.filter(id => PLANES[id]);
            if (!data.ow.includes('m2browning')) data.ow.unshift('m2browning');
            if (!data.op.includes('tiger'))      data.op.unshift('tiger');
            if (!WEAPONS[data.ew]) data.ew = 'm2browning';
            if (!PLANES[data.ep])  data.ep = 'tiger';

            this.state = data;
            this.save();
            return { ok: true };
        } catch(e) {
            return { ok: false, reason: 'Invalid code (parse error).' };
        }
    }

    /**
     * Download save as an XOR-encrypted binary file (.pshooter).
     * File format: [PSV1 magic (4 bytes)] + [XOR-encrypted signed JSON]
     * The file is unreadable as plain text — looks like binary data.
     */
    downloadSave() {
        this.save();
        const wrapped = JSON.stringify({ d: JSON.stringify(this.state), s: this._sign(this.state) });
        const encrypted = _xorEncrypt(wrapped);

        // Prepend magic header
        const file = new Uint8Array(4 + encrypted.length);
        file.set(_FILE_MAGIC, 0);
        file.set(encrypted, 4);

        const blob = new Blob([file], { type: 'application/octet-stream' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url;
        a.download = `pshooter_save_${Date.now()}.pshooter`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Import save from an ArrayBuffer read from a .pshooter file.
     * Decrypts XOR, verifies signature, validates plausibility.
     * Returns { ok: true } or { ok: false, reason: string }.
     */
    importFile(buffer) {
        try {
            const jsonStr = _xorDecrypt(buffer);               // throws if bad magic
            const wrapped = JSON.parse(jsonStr);
            const data    = JSON.parse(wrapped.d);

            if (this._sign(data) !== wrapped.s)
                return { ok: false, reason: 'File is corrupted or tampered.' };

            if (typeof data.points !== 'number' || typeof data.score !== 'number' ||
                !Array.isArray(data.ow) || !Array.isArray(data.op) ||
                typeof data.ew !== 'string' || typeof data.ep !== 'string')
                return { ok: false, reason: 'File has invalid structure.' };

            const maxLegit = Math.floor(data.score / 10);
            if (data.points > maxLegit + 5)
                return { ok: false, reason: 'File data is implausible (modified?).' };

            data.points = Math.max(0, Math.floor(data.points));
            data.score  = Math.max(0, Math.floor(data.score));
            data.ow = data.ow.filter(id => WEAPONS[id]);
            data.op = data.op.filter(id => PLANES[id]);
            if (!data.ow.includes('m2browning')) data.ow.unshift('m2browning');
            if (!data.op.includes('tiger'))      data.op.unshift('tiger');
            if (!WEAPONS[data.ew]) data.ew = 'm2browning';
            if (!PLANES[data.ep])  data.ep = 'tiger';

            this.state = data;
            this.save();
            return { ok: true };
        } catch(e) {
            return { ok: false, reason: 'Invalid file format (not a .pshooter file?).' };
        }
    }
}

const store = new GameStore();

// ── Shop UI ───────────────────────────────────────────────────────────────────
let shopTab = 'weapons';
let shopZones = [];
let shopMsg   = '';
let shopMsgTimer = 0;
let importInput = '';  // text typed into the import field

function drawShop(ctx, W, H) {
    shopZones = [];

    // Overlay
    ctx.fillStyle = 'rgba(2,3,20,0.96)';
    ctx.fillRect(0, 0, W, H);

    // Header
    ctx.textAlign = 'center';
    ctx.fillStyle = '#FFD700';
    ctx.font = 'bold 28px monospace';
    ctx.fillText('⚙ ARMORY', W / 2, 42);

    ctx.fillStyle = '#aaa';
    ctx.font = '14px monospace';
    ctx.fillText(`💰 ${store.state.points} pts   |   Score: ${store.state.score}`, W / 2, 66);

    // Close button
    _zone(shopZones, 'close', null, W - 80, 14, 64, 28);
    _btn(ctx, 'CLOSE', W - 80, 14, 64, 28, '#F44336', false);

    // Tabs
    const tabs = [
        { id: 'weapons', label: '\uD83D\uDD2B WEAPONS' },
        { id: 'planes',  label: '\u2708 AIRCRAFT' },
        // { id: 'data',    label: '\uD83D\uDCBE DATA' },
    ];
    tabs.forEach((t, i) => {
        const tx = 130 + i * 180, ty = 82, tw = 160, th = 28;
        _zone(shopZones, 'tab', t.id, tx - tw / 2, ty, tw, th);
        _btn(ctx, t.label, tx - tw / 2, ty, tw, th, shopTab === t.id ? '#1565C0' : '#1a237e', shopTab === t.id);
    });

    // Grid or DATA tab
    if (shopTab === 'data') {
        _drawDataTab(ctx, W, H);
    } else {
        const items = shopTab === 'weapons' ? Object.values(WEAPONS) : Object.values(PLANES);
        const cols = 3, cardW = 238, cardH = 148, gx = 20, gy = 124, gap = 12;
        items.forEach((item, idx) => {
            const col = idx % cols, row = Math.floor(idx / cols);
            const cx = gx + col * (cardW + gap);
            const cy = gy + row * (cardH + gap);
            _drawCard(ctx, item, cx, cy, cardW, cardH);
            _zone(shopZones, shopTab === 'weapons' ? 'weapon' : 'plane', item.id, cx, cy, cardW, cardH);
        });
    }

    // Feedback message
    if (shopMsgTimer > 0) {
        shopMsgTimer--;
        ctx.textAlign = 'center';
        ctx.fillStyle = shopMsg.startsWith('\u2705') ? '#4CAF50' : '#F44336';
        ctx.font = 'bold 15px monospace';
        ctx.fillText(shopMsg, W / 2, H - 16);
    }

    ctx.textAlign = 'left';
}

function _drawDataTab(ctx, W, H) {
    const cx = W / 2;
    ctx.textAlign = 'center';

    // Title
    ctx.fillStyle = '#90CAF9'; ctx.font = 'bold 18px monospace';
    ctx.fillText('Cross-Browser Save Transfer', cx, 110);
    ctx.fillStyle = '#777'; ctx.font = '12px monospace';
    ctx.fillText('Transfer your progress via Text Code or Encrypted File.', cx, 130);

    // --- TEXT SECTION ---
    ctx.fillStyle = '#FFD700'; ctx.font = 'bold 14px monospace';
    ctx.fillText('\uD83D\uDCDD TEXT CODE', cx, 165);

    // Export Code
    const code = store.exportCode();
    const shortCode = code.length > 50 ? code.slice(0, 47) + '...' : code;
    ctx.fillStyle = '#0a1628';
    ctx.beginPath(); ctx.roundRect(cx - 340, 175, 500, 34, 6); ctx.fill();
    ctx.strokeStyle = '#1a3a6a'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = '#80CBC4'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
    ctx.fillText(shortCode, cx - 330, 197);
    ctx.textAlign = 'center';

    _zone(shopZones, 'copy_code', null, cx + 180, 173, 160, 38);
    _btn(ctx, '\uD83D\uDCCB Copy Code', cx + 180, 173, 160, 38, '#1565C0', false);

    // Import Code
    ctx.fillStyle = importInput ? '#0a1628' : '#060d1a';
    ctx.beginPath(); ctx.roundRect(cx - 340, 220, 500, 34, 6); ctx.fill();
    ctx.strokeStyle = '#2a4a8a'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = importInput ? '#80CBC4' : '#444'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
    ctx.fillText(importInput || 'Paste code here...', cx - 330, 242);
    ctx.textAlign = 'center';

    _zone(shopZones, 'paste_import', null, cx + 180, 218, 160, 38);
    _btn(ctx, '\uD83D\uDCCB Paste & Import', cx + 180, 218, 160, 38, '#2E7D32', false);

    // --- FILE SECTION ---
    ctx.fillStyle = '#FFD700'; ctx.font = 'bold 14px monospace';
    ctx.fillText('\uD83D\uDCC1 ENCRYPTED FILE (.pshooter)', cx, 290);

    // Download File Button
    _zone(shopZones, 'download_file', null, cx - 170, 305, 160, 38);
    _btn(ctx, '\uD83D\uDCE5 Download Save', cx - 170, 305, 160, 38, '#4527A0', false);

    // Upload File Button
    _zone(shopZones, 'upload_file', null, cx + 10, 305, 160, 38);
    _btn(ctx, '\uD83D\uDCE4 Upload Save', cx + 10, 305, 160, 38, '#2E7D32', false);

    // Stats
    ctx.fillStyle = '#555'; ctx.font = '12px monospace';
    ctx.fillText(`Current: ${store.state.points} pts  |  Total Score: ${store.state.score}  |  Owned: ${store.state.ow.length}W, ${store.state.op.length}P`, cx, 375);

    ctx.fillStyle = '#3a3a5a'; ctx.font = '11px monospace';
    ctx.fillText('Encrypted files are unreadable binary. Codes are Base64. Both use the same signature.', cx, 400);

    ctx.textAlign = 'left';
}

function _drawCard(ctx, item, cx, cy, cw, ch) {
    const isWeapon  = shopTab === 'weapons';
    const ownedList = isWeapon ? store.state.ow : store.state.op;
    const equipped  = isWeapon ? store.state.ew : store.state.ep;
    const owned     = ownedList.includes(item.id);
    const isEquip   = equipped === item.id;

    // Card bg
    ctx.fillStyle = isEquip ? '#0d1f4a' : owned ? '#0a1628' : '#080f1e';
    ctx.strokeStyle = isEquip ? '#2196F3' : owned ? '#1a3a6a' : '#121c30';
    ctx.lineWidth = isEquip ? 2 : 1;
    ctx.beginPath(); ctx.roundRect(cx, cy, cw, ch, 8); ctx.fill(); ctx.stroke();

    // Name
    ctx.fillStyle = isEquip ? '#90CAF9' : '#e0e0e0';
    ctx.font = `bold 13px monospace`;
    ctx.textAlign = 'left';
    ctx.fillText(`${item.icon}  ${item.name}`, cx + 10, cy + 20);

    // Desc
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.fillText(item.desc, cx + 10, cy + 36);

    // Stat bars
    if (isWeapon) {
        _statBar(ctx, cx + 10, cy + 50, cw - 20, 'DMG', item.damage / 8);
        _statBar(ctx, cx + 10, cy + 66, cw - 20, 'SPD', item.speed / 40);
        _statBar(ctx, cx + 10, cy + 82, cw - 20, 'RNG', item.range);
        _statBar(ctx, cx + 10, cy + 98, cw - 20, 'ROF', 1 - (item.fireRate / 18));
    } else {
        _statBar(ctx, cx + 10, cy + 50, cw - 20, 'SPD', item.speed / 7);
        _statBar(ctx, cx + 10, cy + 66, cw - 20, 'HP ', item.lives / 5);
        _statBar(ctx, cx + 10, cy + 82, cw - 20, 'AGI', 1 - (item.width - 30) / 14);
    }

    // Action button
    const bx = cx + 10, by = cy + ch - 30, bw2 = cw - 20, bh = 22;
    if (isEquip) {
        _btn(ctx, '✓ EQUIPPED', bx, by, bw2, bh, '#1565C0', true);
    } else if (owned) {
        _btn(ctx, 'EQUIP', bx, by, bw2, bh, '#2E7D32', false);
    } else {
        const canAfford = store.state.points >= item.cost;
        _btn(ctx, `💰 ${item.cost} pts — BUY`, bx, by, bw2, bh, canAfford ? '#4CAF50' : '#555', false);
    }
}

function _statBar(ctx, x, y, w, label, frac) {
    frac = Math.max(0, Math.min(1, frac));
    ctx.fillStyle = '#555';
    ctx.fillText(label, x, y + 9);
    const bx = x + 36, bw = w - 40;
    ctx.fillStyle = '#222'; ctx.fillRect(bx, y, bw, 8);
    ctx.fillStyle = `hsl(${frac * 120}, 80%, 50%)`;
    ctx.fillRect(bx, y, bw * frac, 8);
    ctx.font = '10px monospace'; ctx.fillStyle = '#777';
}

function _btn(ctx, label, x, y, w, h, bg, active) {
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 5); ctx.fill();
    if (active) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke(); }
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, x + w / 2, y + h / 2 + 4);
    ctx.textAlign = 'left';
}

function _zone(zones, type, id, x, y, w, h) {
    zones.push({ type, id, x, y, w, h });
}

function handleShopClick(mx, my) {
    for (const z of shopZones) {
        if (mx < z.x || mx > z.x + z.w || my < z.y || my > z.y + z.h) continue;
        if (z.type === 'close') { return 'close'; }
        if (z.type === 'tab')   { shopTab = z.id; importInput = ''; return 'tab'; }

        // DATA tab actions
        if (z.type === 'copy_code') {
            const code = store.exportCode();
            navigator.clipboard.writeText(code).then(() => {
                shopMsg = '\u2705 Code copied to clipboard!'; shopMsgTimer = 120;
            }).catch(() => {
                prompt('Copy this save code:', code);
            });
            return 'action';
        }
        if (z.type === 'paste_import') {
            navigator.clipboard.readText().then(text => {
                importInput = text.trim();
                const result = store.importCode(importInput);
                if (result.ok) {
                    shopMsg = '\u2705 Save imported! Progress restored.'; shopMsgTimer = 180;
                    importInput = '';
                } else {
                    shopMsg = '\u274C ' + result.reason; shopMsgTimer = 180;
                }
            }).catch(() => {
                const pasted = prompt('Paste your save code:');
                if (pasted) {
                    importInput = pasted.trim();
                    const result = store.importCode(importInput);
                    if (result.ok) {
                        shopMsg = '\u2705 Save imported! Progress restored.'; shopMsgTimer = 180;
                        importInput = '';
                    } else {
                        shopMsg = '\u274C ' + result.reason; shopMsgTimer = 180;
                    }
                }
            });
            return 'action';
        }
        if (z.type === 'clear_import') { importInput = ''; return 'action'; }

        // File actions
        if (z.type === 'download_file') {
            store.downloadSave();
            shopMsg = '\u2705 Save file downloaded!'; shopMsgTimer = 120;
            return 'action';
        }
        if (z.type === 'upload_file') {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pshooter';
            input.onchange = e => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = re => {
                    const res = store.importFile(re.target.result);
                    if (res.ok) {
                        shopMsg = '\u2705 File imported! Progress restored.'; shopMsgTimer = 180;
                    } else {
                        shopMsg = '\u274C ' + res.reason; shopMsgTimer = 180;
                    }
                };
                reader.readAsArrayBuffer(file);
            };
            input.click();
            return 'action';
        }

        if (z.type === 'weapon' || z.type === 'plane') {
            const owned = z.type === 'weapon' ? store.state.ow : store.state.op;
            if (owned.includes(z.id)) {
                store.equip(z.type === 'weapon' ? 'weapon' : 'plane', z.id);
                shopMsg = '\u2705 Equipped!'; shopMsgTimer = 90;
            } else {
                const ok = store.buy(z.type === 'weapon' ? 'weapon' : 'plane', z.id);
                if (ok) {
                    store.equip(z.type === 'weapon' ? 'weapon' : 'plane', z.id);
                    shopMsg = '\u2705 Purchased & equipped!'; shopMsgTimer = 90;
                } else {
                    shopMsg = '\u274C Not enough points!'; shopMsgTimer = 90;
                }
            }
            return 'action';
        }
    }
    return null;
}
