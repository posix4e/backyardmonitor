let spots = []; // [{id,name,polygon:[[x,y],...]}]
let selectedSpotId = null;
const DEFAULT_SPOT = {
    w: 120,
    h: 200
};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasWrap = document.getElementById('canvasWrap');
// Spots panel elements
const spotSearchInput = document.getElementById('spot_search');
const spotListEl = document.getElementById('spot_list');
const spotEditor = document.getElementById('spot_editor');
const spotEditIdEl = document.getElementById('spot_edit_id');
// Spot editing for name/min/stable moved to spot detail page
const spotSizeEl = document.getElementById('spot_size');
let baseW = 0,
    baseH = 0; // native frame size from backend
const initW = canvas.width,
    initH = canvas.height; // initial canvas size (shrink target)
let expanded = false; // false => 640x360, true => full native size
let hoverSpotId = null; // track hover for selection hint
// Pointer/touch drag state
let dragging = false;
let dragSpotId = null;
let draggingVertexIndex = null; // if not null, dragging a specific vertex of selected spot
let dragStart = null; // {x,y} in native coords
let dragStartPoly = null; // deep copy of polygon at start
let downOnEmpty = false; // distinguish tap-to-add

function screenToNative(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const sx = baseW ? (baseW / canvas.width) : 1;
    const sy = baseH ? (baseH / canvas.height) : 1;
    const x = Math.round((clientX - rect.left) * sx);
    const y = Math.round((clientY - rect.top) * sy);
    return {
        x,
        y
    };
}

function nativeToScreen(x, y) {
    const sx = baseW ? (canvas.width / baseW) : 1;
    const sy = baseH ? (canvas.height / baseH) : 1;
    return {
        x: x * sx,
        y: y * sy
    };
}

function polyBBox(poly) {
    if (!poly || !poly.length) return null;
    const xs = poly.map(p => p[0]);
    const ys = poly.map(p => p[1]);
    const x1 = Math.min(...xs),
        x2 = Math.max(...xs),
        y1 = Math.min(...ys),
        y2 = Math.max(...ys);
    return {
        x1,
        y1,
        x2,
        y2
    };
}

function hitSpotAt(x, y, tolPx = 0) {
    // tolPx in screen px; convert to native units using average scale
    const sx = baseW ? (baseW / canvas.width) : 1;
    const sy = baseH ? (baseH / canvas.height) : 1;
    const tol = Math.max(0, Math.round((tolPx * (sx + sy)) / 2));
    for (let i = (spots || []).length - 1; i >= 0; i--) {
        const s = spots[i];
        const bb = polyBBox(s.polygon);
        const insidePoly = pointInPoly([x, y], s.polygon);
        const insideBBox = bb ? (x >= (bb.x1 - tol) && x <= (bb.x2 + tol) && y >= (bb.y1 - tol) && y <= (bb.y2 + tol)) : false;
        if (insidePoly || insideBBox) return s;
    }
    return null;
}

function vertexNear(s, x, y, tolPx) {
    if (!s || !s.polygon) return -1;
    // compare in screen space for tolerance
    const p = nativeToScreen(x, y);
    for (let i = 0; i < s.polygon.length; i++) {
        const v = s.polygon[i];
        const vs = nativeToScreen(v[0], v[1]);
        const dx = vs.x - p.x,
            dy = vs.y - p.y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= (tolPx * tolPx)) return i;
    }
    return -1;
}

function _distToSegSq(px, py, x1, y1, x2, y2) {
    const vx = x2 - x1,
        vy = y2 - y1;
    const wx = px - x1,
        wy = py - y1;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return (px - x1) * (px - x1) + (py - y1) * (py - y1);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return (px - x2) * (px - x2) + (py - y2) * (py - y2);
    const b = c1 / c2;
    const bx = x1 + b * vx,
        by = y1 + b * vy;
    return (px - bx) * (px - bx) + (py - by) * (py - by);
}

function edgeNear(s, x, y, tolPx) {
    if (!s || !s.polygon || s.polygon.length < 2) return -1;
    // work in screen space
    const p = nativeToScreen(x, y);
    for (let i = 0; i < s.polygon.length; i++) {
        const a = s.polygon[i];
        const b = s.polygon[(i + 1) % s.polygon.length];
        const as = nativeToScreen(a[0], a[1]);
        const bs = nativeToScreen(b[0], b[1]);
        const d2 = _distToSegSq(p.x, p.y, as.x, as.y, bs.x, bs.y);
        if (d2 <= tolPx * tolPx) return i; // edge between i and i+1
    }
    return -1;
}

function applySize() {
    // If expanded and we know native size, use it; otherwise stay at initial size
    if (expanded && baseW && baseH) {
        canvas.width = baseW;
        canvas.height = baseH;
    } else {
        canvas.width = initW;
        canvas.height = initH;
    }
    const btn = document.getElementById('sizeBtn');
    if (btn) btn.textContent = expanded ? 'Shrink' : 'Expand';
}

async function control(action) {
    const bStart = document.getElementById('btnStart');
    const bStop = document.getElementById('btnStop');
    try {
        if (bStart) bStart.disabled = true;
        if (bStop) bStop.disabled = true;
        const res = await fetch('/api/control?action=' + encodeURIComponent(action), {
            method: 'POST'
        });
        if (!res.ok) {
            const txt = await res.text();
            alert('Control failed: ' + txt);
        }
        // give the worker a moment to spin up/down
        await new Promise(r => setTimeout(r, 400));
        await refresh();
    } catch (e) {
        console.error(e);
        alert('Control error: ' + (e && e.message ? e.message : e));
    } finally {
        if (bStart) bStart.disabled = false;
        if (bStop) bStop.disabled = false;
    }
}
async function refresh() {
    const status = await (await fetch('/api/status')).json();
    document.getElementById('status').innerText = `running=${status.running} last_ts=${status.last_ts} size=${status.width}x${status.height}`;
    if (status.width && status.height) {
        baseW = status.width;
        baseH = status.height;
    }
    // Apply the current sizing preference
    applySize();
}

function rectPolygon(cx, cy, w, h) {
    const x0 = cx - w / 2,
        y0 = cy - h / 2;
    return [
        [x0, y0],
        [x0 + w, y0],
        [x0 + w, y0 + h],
        [x0, y0 + h]
    ];
}

function pointInPoly(pt, poly) {
    let [x, y] = pt;
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0],
            yi = poly[i][1];
        const xj = poly[j][0],
            yj = poly[j][1];
        const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

function deleteSelected() {
    if (!selectedSpotId) return;
    spots = spots.filter(sp => sp.id !== selectedSpotId);
    selectedSpotId = null;
    renderSpotList();
    fillSpotEditor();
    draw();
}

function rotateSelected(deg) {
    if (!selectedSpotId) return;
    const s = spots.find(sp => sp.id === selectedSpotId);
    if (!s || !s.polygon || s.polygon.length < 3) return;
    const xs = s.polygon.map(p => p[0]);
    const ys = s.polygon.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const ang = (deg * Math.PI) / 180;
    const ca = Math.cos(ang),
        sa = Math.sin(ang);
    s.polygon = s.polygon.map(([px, py]) => {
        const dx = px - cx,
            dy = py - cy;
        const rx = dx * ca - dy * sa;
        const ry = dx * sa + dy * ca;
        return [cx + rx, cy + ry];
    });
    fillSpotEditor();
    draw();
}

// Name editing removed from main page; handled on spot detail

function selectedBBox() {
    const s = spots.find(sp => sp.id === selectedSpotId);
    if (!s || !s.polygon || s.polygon.length < 3) return null;
    const xs = s.polygon.map(p => p[0]);
    const ys = s.polygon.map(p => p[1]);
    const x1 = Math.min(...xs),
        x2 = Math.max(...xs),
        y1 = Math.min(...ys),
        y2 = Math.max(...ys);
    return {
        x: x1,
        y: y1,
        w: x2 - x1,
        h: y2 - y1,
        name: s.name || s.id
    };
}

// Spots panel rendering and editing
function renderSpotList() {
    if (!spotListEl) return;
    const q = (spotSearchInput && spotSearchInput.value || '').toLowerCase();
    const items = (spots || []).filter(sp => {
        const t = ((sp.id || '') + ' ' + (sp.name || '')).toLowerCase();
        return !q || t.includes(q);
    });
    spotListEl.innerHTML = '';
    for (const sp of items) {
        const div = document.createElement('div');
        div.className = 'list-item' + (sp.id === selectedSpotId ? ' active' : '');
        div.innerHTML = `<span class="muted" style="font-size:12px;">${sp.id}</span> <span>${(sp.name||sp.id)}</span>`;
        div.onclick = () => {
            selectedSpotId = sp.id;
            fillSpotEditor();
            draw();
            renderSpotList();
        };
        div.ondblclick = () => {
            const url = `/static/spot.html?spot=${encodeURIComponent(sp.id)}`;
            window.open(url, '_blank');
        };
        spotListEl.appendChild(div);
    }
}

function fillSpotEditor() {
    if (!spotEditor) return;
    const s = spots.find(sp => sp.id === selectedSpotId);
    if (!s) {
        spotEditor.style.display = 'none';
        return;
    }
    spotEditor.style.display = 'block';
    if (spotEditIdEl) spotEditIdEl.textContent = s.id;
    const box = selectedBBox();
    if (spotSizeEl) spotSizeEl.textContent = box ? `${Math.round(box.w)}×${Math.round(box.h)} px` : '';
}

function openSpotDetails() {
    if (!selectedSpotId) return;
    const url = `/static/spot.html?spot=${encodeURIComponent(selectedSpotId)}`;
    window.open(url, '_blank');
}

function compareSelected() {
    if (!selectedSpotId) return;
    try {
        const box = selectedBBox();
        const r = box ? bboxToScreen(Math.round(box.x), Math.round(box.y), Math.round(box.x + box.w), Math.round(box.y + box.h)) : {
            x: 0,
            y: 0,
            w: 0,
            h: 0
        };
        const sid = selectedSpotId;
        const prevUrl = (window.spotPrev && window.spotPrev[sid]) || null;
        const curUrl = `/api/spot.jpg?id=${encodeURIComponent(sid)}&w=240&ts=${Date.now()}`;
        const prevSig = (window.spotPrevSig && window.spotPrevSig[sid]) || null;
        const curSig = (window.lastSigMap && window.lastSigMap[sid]) || null;
        showComparisonPopup(sid, r, prevUrl, curUrl, prevSig, curSig);
    } catch (err) {
        console.error(err);
    }
}

function toggleSize() {
    expanded = !expanded;
    applySize();
    draw();
}

// Pointer/touch + mouse unified interactions
function updateHover(clientX, clientY) {
    const {
        x,
        y
    } = screenToNative(clientX, clientY);
    const hit = hitSpotAt(x, y, 6);
    const newHover = hit ? hit.id : null;
    if (newHover !== hoverSpotId) {
        hoverSpotId = newHover;
        canvasWrap.style.cursor = hoverSpotId ? 'pointer' : 'default';
        draw();
    }
}

canvas.addEventListener('pointerdown', (e) => {
    try {
        canvas.setPointerCapture(e.pointerId);
    } catch (err) {}
    downOnEmpty = false;
    const {
        x,
        y
    } = screenToNative(e.clientX, e.clientY);
    const isTouch = e.pointerType === 'touch';
    const hit = hitSpotAt(x, y, isTouch ? 16 : 6);
    if (hit) {
        selectedSpotId = hit.id;
        fillSpotEditor();
        renderSpotList();
        const s = spots.find(sp => sp.id === hit.id);
        // Shift+click near edge to insert a new vertex
        if (e.shiftKey && s) {
            const ei = edgeNear(s, x, y, isTouch ? 24 : 10);
            if (ei >= 0) {
                s.polygon.splice(ei + 1, 0, [x, y]);
                draw();
                e.preventDefault();
                return;
            }
        }
        // If near a vertex, drag that vertex
        const vi = s ? vertexNear(s, x, y, isTouch ? 20 : 10) : -1;
        if (vi >= 0) {
            dragging = true;
            dragSpotId = hit.id;
            draggingVertexIndex = vi;
            dragStart = null;
            dragStartPoly = null;
        } else {
            // Drag whole shape
            dragging = true;
            dragSpotId = hit.id;
            dragStart = {
                x,
                y
            };
            dragStartPoly = s && s.polygon ? s.polygon.map(p => [p[0], p[1]]) : null;
        }
    } else {
        // Track that we pressed on empty space to allow tap-to-add on pointerup
        downOnEmpty = true;
    }
    e.preventDefault();
}, {
    passive: false
});

canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
        updateHover(e.clientX, e.clientY);
        return;
    }
    if (!dragSpotId) return;
    const {
        x,
        y
    } = screenToNative(e.clientX, e.clientY);
    const s = spots.find(sp => sp.id === dragSpotId);
    if (!s) return;
    if (draggingVertexIndex !== null && draggingVertexIndex >= 0) {
        s.polygon[draggingVertexIndex] = [x, y];
        draw();
    } else if (dragStart && dragStartPoly) {
        const dx = x - dragStart.x;
        const dy = y - dragStart.y;
        s.polygon = dragStartPoly.map(p => [p[0] + dx, p[1] + dy]);
        draw();
    }
    e.preventDefault();
}, {
    passive: false
});

function endPointer(e) {
    const wasDragging = dragging;
    const wasDownOnEmpty = downOnEmpty;
    const {
        x,
        y
    } = screenToNative(e.clientX, e.clientY);
    // Reset drag state first
    dragging = false;
    dragSpotId = null;
    draggingVertexIndex = null;
    dragStart = null;
    dragStartPoly = null;
    try {
        canvas.releasePointerCapture(e.pointerId);
    } catch (err) {}
    // If it was a tap on empty, add a new spot
    if (!wasDragging && wasDownOnEmpty) {
        const id = 'spot_' + (spots.length + 1);
        let poly = rectPolygon(x, y, DEFAULT_SPOT.w, DEFAULT_SPOT.h);
        spots.push({
            id,
            name: id,
            polygon: poly
        });
        selectedSpotId = id;
        fillSpotEditor();
        renderSpotList();
        renderZonesView();
        draw();
    } else if (!wasDragging) {
        // It was a tap over an existing spot: select it
        const hit = hitSpotAt(x, y, e.pointerType === 'touch' ? 16 : 6);
        if (hit) {
            selectedSpotId = hit.id;
            fillSpotEditor();
            renderSpotList();
            draw();
        }
    }
    e.preventDefault();
}
canvas.addEventListener('pointerup', endPointer, {
    passive: false
});
canvas.addEventListener('pointercancel', endPointer, {
    passive: false
});

// Maintain hover feedback for mouse users when not dragging
canvas.addEventListener('mouseleave', () => {
    hoverSpotId = null;
    canvasWrap.style.cursor = 'default';
    draw();
});


async function draw() {
    // draw latest frame
    const img = new Image();
    let retries = 0;
    img.onload = () => {
        try {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        } catch (e) {}
        // overlay spots as rectangles
        if (spots && spots.length) {
            ctx.strokeStyle = '#44aaff';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(68,170,255,0.12)';
            for (const s of spots) {
                if (!s.polygon || s.polygon.length < 4) continue;
                const sx = baseW ? (canvas.width / baseW) : 1;
                const sy = baseH ? (canvas.height / baseH) : 1;
                const p0 = s.polygon[0];
                ctx.beginPath();
                ctx.moveTo(p0[0] * sx, p0[1] * sy);
                for (let i = 1; i < s.polygon.length; i++) {
                    const p = s.polygon[i];
                    ctx.lineTo(p[0] * sx, p[1] * sy);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.fill();
                if (s.id === selectedSpotId) {
                    ctx.strokeStyle = '#ffaa00';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([6, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.strokeStyle = '#44aaff';
                    // draw vertex handles
                    try {
                        ctx.save();
                        for (let i = 0; i < s.polygon.length; i++) {
                            const v = s.polygon[i];
                            const vx = v[0] * sx,
                                vy = v[1] * sy;
                            ctx.beginPath();
                            ctx.fillStyle = '#fff';
                            ctx.strokeStyle = '#ff8800';
                            ctx.lineWidth = 2;
                            ctx.arc(vx, vy, 5, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.stroke();
                        }
                        ctx.restore();
                    } catch (e) {}
                }
            }
        }
        fillSpotEditor();
    };
    img.onerror = () => {
        // quick retry once to reduce visible grey on transient errors
        if (retries < 1) {
            retries++;
            setTimeout(() => {
                img.src = '/frame.jpg?ts=' + Date.now();
            }, 200);
        }
    };
    img.src = '/frame.jpg?ts=' + Date.now();
}

// Mouse click remains for old browsers; pointer handlers will handle most cases
canvas.addEventListener('click', async (e) => {
    const {
        x,
        y
    } = screenToNative(e.clientX, e.clientY);
    const hit = hitSpotAt(x, y, 6);
    if (hit) {
        selectedSpotId = hit.id;
        fillSpotEditor();
        renderSpotList();
    } else {
        const id = 'spot_' + (spots.length + 1);
        let poly = rectPolygon(x, y, DEFAULT_SPOT.w, DEFAULT_SPOT.h);
        spots.push({
            id,
            name: id,
            polygon: poly
        });
        selectedSpotId = id;
        fillSpotEditor();
        renderSpotList();
    }
    renderZonesView();
    draw();
});

// Double-click to open spot details page
canvas.addEventListener('dblclick', (e) => {
    const {
        x,
        y
    } = screenToNative(e.clientX, e.clientY);
    const hit = hitSpotAt(x, y, 8);
    if (hit) {
        const url = `/static/spot.html?spot=${encodeURIComponent(hit.id)}`;
        window.open(url, '_blank');
    }
});

document.addEventListener('keydown', (e) => {
    if (!selectedSpotId) return;
    const s = spots.find(sp => sp.id === selectedSpotId);
    if (!s) return;
    const step = (e.shiftKey ? 10 : 5);
    if (e.key === 'Delete' || e.key === 'Backspace') {
        spots = spots.filter(sp => sp.id !== selectedSpotId);
        selectedSpotId = null;
        renderZonesView();
        renderSpotList();
        fillSpotEditor();
        draw();
        e.preventDefault();
        return;
    }
    let dx = 0,
        dy = 0;
    if (e.key === 'ArrowLeft') dx = -step;
    else if (e.key === 'ArrowRight') dx = step;
    else if (e.key === 'ArrowUp') dy = -step;
    else if (e.key === 'ArrowDown') dy = step;
    if (dx || dy) {
        s.polygon = s.polygon.map(p => [p[0] + dx, p[1] + dy]);
        renderZonesView();
        draw();
        e.preventDefault();
    }
});

async function saveZones() {
    const payload = {
        spots
    };
    await fetch('/api/zones', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    alert('Saved');
    renderSpotList();
}

async function loadZones() {
    const res = await fetch('/api/zones');
    const z = await res.json();
    spots = z.spots || [];
    renderZonesView();
    renderSpotList();
    fillSpotEditor();
}

function clearSpots() {
    spots = [];
    selectedSpotId = null;
    renderSpotList();
    fillSpotEditor();
    draw();
}

function renderZonesView() {
    /* we keep homepage minimal; could add JSON preview if desired */
}

// Slow down frame refresh to reduce flicker while editing
if (window.__bm_drawInt) {
    try {
        clearInterval(window.__bm_drawInt);
    } catch (e) {}
}
window.__bm_drawInt = setInterval(() => {
    draw();
}, 1000);
// Track last signature per spot for change visualization
let lastSigMap = {};
// Cache previous image URL per spot from aggregated API
let spotPrev = {};
let spotPrevSig = {};
async function refreshSpotPrev() {
    try {
        const j = await (await fetch('/api/spot_recent?per_spot=2&scan_limit=200')).json();
        const m = {};
        const m2 = {};
        for (const it of (j.items || [])) {
            if (it.prev_url) m[it.spot_id] = it.prev_url;
            if (it.prev_sig) m2[it.spot_id] = it.prev_sig;
        }
        spotPrev = m;
        spotPrevSig = m2;
        try {
            window.spotPrev = spotPrev;
            window.spotPrevSig = spotPrevSig;
        } catch (e) {}
    } catch (e) {
        /* ignore */
    }
}
refreshSpotPrev();
if (window.__bm_recentInt) {
    try {
        clearInterval(window.__bm_recentInt);
    } catch (e) {}
}
window.__bm_recentInt = setInterval(refreshSpotPrev, 10000);
// Poll spot stats to detect changes and show effect
if (window.__bm_statInt) {
    try {
        clearInterval(window.__bm_statInt);
    } catch (e) {}
}
window.__bm_statInt = setInterval(async () => {
    try {
        const stats = await (await fetch('/api/spot_stats')).json();
        const items = (stats && stats.spots) || [];
        for (const st of items) {
            const prev = lastSigMap[st.id] || '';
            const cur = st.sig || '';
            if (prev && cur && prev !== cur) {
                try {
                    showSpotChangeEffect(st);
                } catch (e) {
                    console.error(e);
                }
            }
            if (cur) lastSigMap[st.id] = cur;
            try {
                window.lastSigMap = lastSigMap;
            } catch (e) {}
        }
    } catch (e) {
        /* ignore transient errors */
    }
}, 2000);

function bboxToScreen(x1, y1, x2, y2) {
    if (!baseW || !baseH) return null;
    const sx = canvas.width / baseW,
        sy = canvas.height / baseH;
    const rx1 = Math.round(x1 * sx),
        ry1 = Math.round(y1 * sy);
    const rw = Math.round((x2 - x1) * sx),
        rh = Math.round((y2 - y1) * sy);
    return {
        x: rx1,
        y: ry1,
        w: rw,
        h: rh
    };
}

function showSpotChangeEffect(st) {
    // st has x1,y1,x2,y2 from /api/spot_stats
    const rect = bboxToScreen(st.x1, st.y1, st.x2, st.y2);
    if (!rect || rect.w <= 4 || rect.h <= 4) return;
    const el = document.createElement('div');
    el.className = 'pulse-indicator';
    el.style.left = rect.x + 'px';
    el.style.top = rect.y + 'px';
    el.style.width = Math.max(12, rect.w) + 'px';
    el.style.height = Math.max(12, rect.h) + 'px';
    canvasWrap.appendChild(el);
    // Setup auto-remove handlers first so we can safely reference cleanup/t below
    let removed = false;
    const cleanup = () => {
        if (!removed) {
            removed = true;
            el.remove();
        }
    };
    let t = setTimeout(cleanup, 4000);
    el.addEventListener('mouseenter', () => {
        try {
            clearTimeout(t);
        } catch (e) {}
    });
    el.addEventListener('mouseleave', () => {
        t = setTimeout(cleanup, 1500);
    });

    // Show change badge (computed from last signature vs current)
    try {
        const prevSig = (lastSigMap && lastSigMap[st.id]) || '';
        const curSig = st.sig || '';
        const d = hamming(prevSig, curSig);
        if (d !== null) {
            const badge = document.createElement('div');
            badge.className = 'delta-badge';
            badge.textContent = `Δ ${d}`;
            el.appendChild(badge);
        }
    } catch (e) {}
    // Compare UI is handled in the single spot toolbar (no separate star overlay)
}

function hamming(a, b) {
    if (!a || !b || a.length !== b.length) return null;
    let d = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) d++;
    }
    return d;
}
async function showComparisonPopup(spotId, rect, prevUrl = null, curUrl = null, prevSig = null, curSig = null) {
    // remove existing popup
    document.querySelectorAll('.compare-popup').forEach(n => n.remove());
    // fetch previous event (most recent) for this spot
    if (!prevUrl) {
        try {
            const j = await (await fetch(`/api/spot_history?spot_id=${encodeURIComponent(spotId)}&limit=2`)).json();
            if (j && j.items && j.items.length >= 2) {
                const prev = j.items[1];
                prevUrl = (prev && (prev.crop || prev.thumb)) || null;
            } else if (j && j.items && j.items.length === 1) {
                const prev = j.items[0];
                prevUrl = (prev && (prev.crop || prev.thumb)) || null;
            }
        } catch {}
    }
    if (!curUrl) {
        curUrl = `/api/spot.jpg?id=${encodeURIComponent(spotId)}&ts=${Date.now()}`;
    }
    const curSigEff = curSig || (lastSigMap && lastSigMap[spotId]) || '';
    const deltaBits = (prevSig && curSigEff && prevSig.length === curSigEff.length) ? hamming(prevSig, curSigEff) : null;
    const pop = document.createElement('div');
    pop.className = 'compare-popup';
    pop.style.left = Math.max(8, Math.min(canvas.width - 480, rect.x)) + 'px';
    pop.style.top = Math.max(8, Math.min(canvas.height - 260, rect.y)) + 'px';
    pop.innerHTML = `
          <button class="compare-close" title="Close">&times;</button>
          <div style="font-weight:600; margin-bottom:6px;">Spot ${spotId} change</div>
          <div class="muted" style="margin-bottom:6px;">prev: ${prevSig?prevSig:'-'} &bull; current: ${curSigEff?curSigEff:'-'} ${deltaBits!==null?('&bull; Δ: '+deltaBits):''}</div>
          <div class="row">
            <div>
              <div class="muted" style="margin-bottom:4px;">Previous</div>
              <img src="${prevUrl || ''}" alt="previous" onerror="this.style.display='none'"/>
            </div>
            <div>
              <div class="muted" style="margin-bottom:4px;">Current</div>
              <img src="${curUrl}" alt="current"/>
            </div>
          </div>`;
    const close = () => {
        try {
            pop.remove();
        } catch (e) {}
    };
    pop.querySelector('.compare-close').addEventListener('click', close);
    // Close on outside click
    const onDoc = (e) => {
        if (!pop.contains(e.target)) {
            close();
            document.removeEventListener('mousedown', onDoc);
        }
    };
    document.addEventListener('mousedown', onDoc);
    canvasWrap.appendChild(pop);
}
// Generic data panel helpers
async function loadZonesJson() {
    try {
        const z = await (await fetch('/api/zones')).json();
        const ta = document.getElementById('zones_json');
        if (ta) ta.value = JSON.stringify(z, null, 2);
    } catch (e) {
        console.error(e);
    }
}
async function saveZonesJson() {
    try {
        const ta = document.getElementById('zones_json');
        const obj = JSON.parse(ta.value || '{}');
        await fetch('/api/zones', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(obj)
        });
        // also update overlay spots so canvas reflects saved data
        if (obj && Array.isArray(obj.spots)) {
            spots = obj.spots;
            selectedSpotId = null;
            draw();
        }
        alert('Zones saved');
    } catch (e) {
        alert('Invalid JSON or save failed');
        console.error(e);
    }
}

function renderEventsList(items) {
    const el = document.getElementById('events_list');
    const safe = (t) => String(t || '').replace(/[&<>]/g, (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;"
    } [c] || c));
    el.innerHTML = (items || []).map((it) => {
        const ts = new Date((it.ts || 0) * 1000).toLocaleString();
        const hasSpot = !!(it && it.meta && it.meta.spot_id);
        const isSpotChange = String(it && it.kind || '').toLowerCase() === 'spot_change';
        const showWhy = hasSpot && isSpotChange;
        const sig = !!(it && it.meta && it.meta.significant);
        const sigBadge = sig && isSpotChange ? `<span class=\"badge\" style=\"background:#0b6; color:#fff; padding:2px 6px; border-radius:10px; font-size:11px;\">Significant</span>` : '';
        return `<div style=\"display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #f2f2f2;\">` +
            `<span style=\"width:56px;\">#${it.id}</span>` +
            `<span style=\"flex:1;\">${safe(it.kind)}</span>` + sigBadge +
            `<span class=\"muted\" style=\"flex:1;\">${safe(ts)}</span>` +
            (showWhy ? `<button data-eid=\"${it.id}\" class=\"ev_why\">Why?</button>` : '') +
            `<button data-eid=\"${it.id}\" class=\"ev_edit\">Edit</button>` +
            `<button data-eid=\"${it.id}\" class=\"ev_delete\" style=\"color:#b00000;\">Delete</button>` +
            `</div>`
    }).join('');
    el.querySelectorAll('.ev_edit').forEach(btn => {
        btn.addEventListener('click', () => editEvent(btn.getAttribute('data-eid')));
    });
    el.querySelectorAll('.ev_why').forEach(btn => {
        btn.addEventListener('click', () => explainEvent(btn.getAttribute('data-eid')));
    });
    el.querySelectorAll('.ev_delete').forEach(btn => {
        btn.addEventListener('click', async () => {
            const id = btn.getAttribute('data-eid');
            if (!confirm(`Delete event #${id}?`)) return;
            try {
                await fetch(`/api/events/${id}`, {
                    method: 'DELETE'
                });
            } catch (e) {
                console.error(e);
                alert('Delete failed');
                return;
            }
            await loadEvents();
            try {
                await loadImagesSummary();
            } catch (e) {}
            try {
                await loadThumbnails();
            } catch (e) {}
        });
    });
}
async function loadEvents() {
    try {
        const limEl = document.getElementById('events_limit');
        const lim = limEl ? parseInt(limEl.value || '50', 10) : 50;
        const j = await (await fetch(`/api/events?limit=${lim}`)).json();
        let items = j.items || [];
        const sigOnly = !!(document.getElementById('events_sig_only') && document.getElementById('events_sig_only').checked);
        if (sigOnly) {
            items = items.filter(it => String(it.kind || '').toLowerCase() === 'spot_change' && !!(it.meta && it.meta.significant));
        }
        renderEventsList(items);
    } catch (e) {
        console.error(e);
    }
}
async function editEvent(id) {
    try {
        const ev = await (await fetch(`/api/events/${id}`)).json();
        document.getElementById('event_editor').style.display = 'block';
        document.getElementById('ev_id').textContent = `#${ev.id}`;
        document.getElementById('ev_kind').value = ev.kind || '';
        document.getElementById('ev_ts').value = ev.ts || '';
        document.getElementById('ev_meta').value = JSON.stringify(ev.meta || {}, null, 2);
        document.getElementById('event_editor').setAttribute('data-id', String(ev.id));
    } catch (e) {
        console.error(e);
    }
}
async function explainEvent(id) {
    try {
        const ev = await (await fetch(`/api/events/${encodeURIComponent(String(id))}`)).json();
        const meta = ev && ev.meta ? ev.meta : {};
        const spotId = meta.spot_id || '';
        // Build a simple modal with LLM details (if present)
        const wrap = document.createElement('div');
        wrap.style.position = 'fixed';
        wrap.style.left = '0';
        wrap.style.top = '0';
        wrap.style.right = '0';
        wrap.style.bottom = '0';
        wrap.style.background = 'rgba(0,0,0,0.35)';
        wrap.style.zIndex = '1000';
        const card = document.createElement('div');
        card.style.position = 'absolute';
        card.style.left = '50%';
        card.style.top = '50%';
        card.style.transform = 'translate(-50%, -50%)';
        card.style.width = 'min(720px, 92vw)';
        card.style.maxHeight = '80vh';
        card.style.overflow = 'auto';
        card.style.background = '#fff';
        card.style.border = '1px solid #ddd';
        card.style.borderRadius = '10px';
        card.style.boxShadow = '0 10px 28px rgba(0,0,0,0.25)';
        card.style.padding = '12px';
        const mk = (k, v) => `<div style="display:flex; gap:8px;"><div style="width:180px; color:#666;">${k}</div><div style="flex:1; white-space:pre-wrap; font-family: ui-monospace, monospace;">${escapeHtml(String(v || ''))}</div></div>`;
        const llm = {
            status: meta.llm_status || '',
            significant: meta.significant,
            reason: meta.llm_reason || '',
            provider: meta.llm_provider || '',
            model: meta.llm_model || '',
            prompt: meta.llm_prompt || '',
            response: meta.llm_response || '',
            error: meta.llm_error || '',
        };
        const hasLLM = !!(meta && (meta.llm_status || meta.llm_attempted));
        const body = `
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <div style="font-weight:600;">Event #${ev.id} · ${escapeHtml(ev.kind || '')}</div>
            <div class="muted" style="margin-left:auto;">${new Date((ev.ts||0)*1000).toLocaleString()}</div>
          </div>
          <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start;">
            <div>
              <div class="muted" style="margin-bottom:4px;">Thumb</div>
              <img src="/api/event_image?id=${ev.id}&kind=thumb" style="max-width:260px; border:1px solid #eee; border-radius:8px;"/>
            </div>
            <div style="flex:1; min-width:280px;">
              <div style="font-weight:600; margin-bottom:6px;">LLM Assessment</div>
              ${hasLLM ? (
                mk('Status', llm.status || '-') +
                mk('Significant', (llm.significant != null ? String(!!llm.significant) : '-')) +
                (llm.reason ? mk('Reason', llm.reason) : '') +
                (llm.error ? mk('Error', llm.error) : '') +
                (llm.provider ? mk('Provider', llm.provider) : '') +
                (llm.model ? mk('Model', llm.model) : '') +
                (llm.prompt ? mk('Prompt', llm.prompt) : '') +
                (llm.response ? mk('Response', llm.response) : '')
              ) : '<div class="muted">No LLM assessment recorded yet.</div>'}
              <div style="margin-top:8px; display:flex; gap:8px;">
                ${spotId ? `<a href="/static/spot.html?spot=${encodeURIComponent(spotId)}&event=${encodeURIComponent(String(ev.id))}" target="_blank">Open spot timeline ↗</a>` : ''}
                <button id="btn_llm_queue">Analyze now</button>
                <a href="/static/significance.html?event=${encodeURIComponent(String(ev.id))}" target="_blank">Open significance view ↗</a>
                <button id="btn_close">Close</button>
              </div>
            </div>
          </div>`;
        card.innerHTML = body;
        wrap.appendChild(card);
        document.body.appendChild(wrap);
        const close = () => { try { wrap.remove(); } catch (e) {} };
        card.querySelector('#btn_close').addEventListener('click', close);
        card.querySelector('#btn_llm_queue').addEventListener('click', async () => {
            try {
                const r = await fetch(`/api/llm/queue_event?id=${encodeURIComponent(String(ev.id))}`, { method: 'POST' });
                if (!r.ok) {
                    alert('Queue failed');
                    return;
                }
                // brief wait then refresh modal content
                setTimeout(async () => {
                    try {
                        const ev2 = await (await fetch(`/api/events/${encodeURIComponent(String(id))}`)).json();
                        const m2 = ev2 && ev2.meta ? ev2.meta : {};
                        const upd = [];
                        upd.push(mk('Status', m2.llm_status || '-'));
                        upd.push(mk('Significant', (m2.significant != null ? String(!!m2.significant) : '-')));
                        if (m2.llm_reason) upd.push(mk('Reason', m2.llm_reason));
                        if (m2.llm_error) upd.push(mk('Error', m2.llm_error));
                        if (m2.llm_provider) upd.push(mk('Provider', m2.llm_provider));
                        if (m2.llm_model) upd.push(mk('Model', m2.llm_model));
                        if (m2.llm_prompt) upd.push(mk('Prompt', m2.llm_prompt));
                        if (m2.llm_response) upd.push(mk('Response', m2.llm_response));
                        const sec = card.querySelectorAll('div')[1];
                        // naive replace: rebuild the meta section
                        const repl = document.createElement('div');
                        repl.innerHTML = `<div style=\"font-weight:600; margin-bottom:6px;\">LLM Assessment</div>` + (upd.join('') || '<div class="muted">No LLM assessment recorded yet.</div>') +
                                         `<div style=\"margin-top:8px; display:flex; gap:8px;\">${spotId ? `<a href=\"/static/spot.html?spot=${encodeURIComponent(spotId)}&event=${encodeURIComponent(String(ev.id))}\" target=\"_blank\">Open spot timeline ↗</a>` : ''}<button id=\"btn_llm_queue\">Analyze now</button> <button id=\"btn_close\">Close</button></div>`;
                        card.replaceChild(repl, sec);
                        repl.querySelector('#btn_close').addEventListener('click', close);
                        repl.querySelector('#btn_llm_queue').addEventListener('click', async () => { /* prevent losing listener; no-op to avoid recursion */ });
                    } catch (e) {}
                }, 1200);
            } catch (e) {
                console.error(e);
                alert('Queue failed');
            }
        });
    } catch (e) {
        console.error(e);
        alert('Unable to load event');
    }
}
async function saveEvent() {
    try {
        const ed = document.getElementById('event_editor');
        const id = ed.getAttribute('data-id');
        const kind = document.getElementById('ev_kind').value || '';
        const ts = parseFloat(document.getElementById('ev_ts').value || '');
        const meta = JSON.parse(document.getElementById('ev_meta').value || '{}');
        const payload = {
            kind,
            meta
        };
        if (!isNaN(ts)) payload.ts = ts;
        const res = await fetch(`/api/events/${id}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            alert('Save failed');
            return;
        }
        await loadEvents();
        alert('Event saved');
    } catch (e) {
        alert('Invalid meta JSON or save failed');
        console.error(e);
    }
}
// Tabs
function escapeHtml(s) {
    return String(s || '').replace(/[&<>]/g, c => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;"
    } [c] || c));
}

function tsStr(t) {
    try {
        return new Date((t || 0) * 1000).toLocaleString();
    } catch (e) {
        return String(t || '');
    }
}
async function refreshBaseline() {
    try {
        const el = document.getElementById('baseline_info');
        if (!el) return;
        const j = await (await fetch('/api/compare/baseline')).json();
        const ts = j && j.baseline_ts ? new Date(j.baseline_ts * 1000).toLocaleString() : 'none';
        el.textContent = `Baseline: ${ts}`;
    } catch (e) {
        console.error(e);
    }
}
async function setBaselineNow() {
    try {
        await fetch('/api/compare/baseline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        refreshBaseline();
    } catch (e) {
        console.error(e);
    }
}
async function clearBaseline() {
    try {
        await fetch('/api/compare/clear', {
            method: 'POST'
        });
        refreshBaseline();
    } catch (e) {
        console.error(e);
    }
}
async function loadPerception() {
    try {
        const j = await (await fetch('/api/config')).json();
        const sel = document.getElementById('detector_method');
        const phash = document.getElementById('phash_params');
        const rdiff = document.getElementById('roidiff_params');
        const b = document.getElementById('phash_bits');
        const m = document.getElementById('phash_ms');
        const rd_t = document.getElementById('roi_diff_threshold');
        const rd_a = document.getElementById('roi_diff_alpha');
        const rd_min = document.getElementById('roi_diff_min_pixels');
        const rd_cd = document.getElementById('roi_diff_cooldown_ms');
        const dm = (j.DETECTOR_METHOD || 'phash');
        if (sel) sel.value = dm;
        if (b) b.value = j.PHASH_MIN_BITS != null ? j.PHASH_MIN_BITS : 14;
        if (m) m.value = j.PHASH_STABLE_MS != null ? j.PHASH_STABLE_MS : 1200;
        if (rd_t) rd_t.value = j.ROI_DIFF_THRESHOLD != null ? j.ROI_DIFF_THRESHOLD : 0.02;
        if (rd_a) rd_a.value = j.ROI_DIFF_ALPHA != null ? j.ROI_DIFF_ALPHA : 0.05;
        if (rd_min) rd_min.value = j.ROI_DIFF_MIN_PIXELS != null ? j.ROI_DIFF_MIN_PIXELS : 600;
        if (rd_cd) rd_cd.value = j.ROI_DIFF_COOLDOWN_MS != null ? j.ROI_DIFF_COOLDOWN_MS : 0;
        if (phash && rdiff) {
            const showPhash = dm === 'phash';
            phash.style.display = showPhash ? 'flex' : 'none';
            rdiff.style.display = showPhash ? 'none' : 'flex';
        }
        sel.addEventListener('change', () => {
            const v = sel.value;
            if (phash && rdiff) {
                const showPhash = v === 'phash';
                phash.style.display = showPhash ? 'flex' : 'none';
                rdiff.style.display = showPhash ? 'none' : 'flex';
            }
        });
    } catch (e) {
        console.error(e);
    }
}
async function savePerception() {
    try {
        const sel = document.getElementById('detector_method');
        const dm = sel ? sel.value : 'phash';
        const payload = {
            DETECTOR_METHOD: dm
        };
        const m = parseInt(document.getElementById('phash_ms').value || '1200', 10);
        if (!isFinite(m) || m < 0) {
            alert('Stable window must be a non-negative number');
            return;
        }
        payload.PHASH_STABLE_MS = m;
        if (dm === 'phash') {
            const b = parseInt(document.getElementById('phash_bits').value || '14', 10);
            payload.PHASH_MIN_BITS = b;
        } else {
            const rd_t = parseFloat(document.getElementById('roi_diff_threshold').value || '0.02');
            const rd_a = parseFloat(document.getElementById('roi_diff_alpha').value || '0.05');
            const rd_min = parseInt(document.getElementById('roi_diff_min_pixels').value || '600', 10);
            const rd_cd = parseInt(document.getElementById('roi_diff_cooldown_ms').value || '0', 10);
            payload.ROI_DIFF_THRESHOLD = rd_t;
            payload.ROI_DIFF_ALPHA = rd_a;
            payload.ROI_DIFF_MIN_PIXELS = rd_min;
            payload.ROI_DIFF_COOLDOWN_MS = rd_cd;
        }
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            alert('Save failed');
            return;
        }
        try {
            const out = await res.json();
            if (out && out.warning) {
                alert('Applied with note: ' + out.warning);
            } else {
                alert('Perception settings applied');
            }
        } catch (e) {
            alert('Perception settings applied');
        }
    } catch (e) {
        console.error(e);
    }
}
async function deleteEvent() {
    try {
        const ed = document.getElementById('event_editor');
        const id = ed.getAttribute('data-id');
        if (!id) return;
        if (!confirm('Delete this event?')) return;
        const res = await fetch(`/api/events/${id}`, {
            method: 'DELETE'
        });
        if (!res.ok) {
            alert('Delete failed');
            return;
        }
        ed.style.display = 'none';
        ed.removeAttribute('data-id');
        await loadEvents();
    } catch (e) {
        console.error(e);
    }
}
async function loadImagesSummary() {
    try {
        const j = await (await fetch('/api/images/summary')).json();
        const el = document.getElementById('images_summary');
        if (el) el.textContent = `files=${j.files}  referenced=${j.referenced}  orphans=${j.orphans}  size=${(j.bytes/1048576).toFixed(1)} MB`;
    } catch (e) {
        console.error(e);
    }
}
async function cleanupOrphans() {
    try {
        if (!confirm('Delete orphan .jpg files that are not referenced by any event?')) return;
        const res = await fetch('/api/images/cleanup', {
            method: 'POST'
        });
        const j = await res.json();
        alert(`Deleted ${j.deleted} files`);
        loadImagesSummary();
    } catch (e) {
        console.error(e);
    }
}

async function loadThumbnails() {
    try {
        const sigOnly = !!(document.getElementById('thumb_sig_only') && document.getElementById('thumb_sig_only').checked);
        const j = await (await fetch(`/api/thumbnails?limit=60&significant_only=${sigOnly ? '1' : '0'}`)).json();
        const grid = document.getElementById('thumb_grid');
        const fmt = (t) => new Date((t || 0) * 1000).toLocaleString();
        grid.innerHTML = (j.items || []).map((it) => {
            const badge = it.significant ? `<span class="badge" style="position:absolute; left:6px; top:6px; background:#0b6; color:#fff; padding:2px 6px; border-radius:10px; font-size:11px;">Significant</span>` : '';
            return `<div style="position:relative; display:inline-block;">` +
                `<a href="${it.full}" target="_blank" title="#${it.event_id} - ${fmt(it.ts)}">` +
                `<img src="${it.thumb}" style="max-width:140px; border-radius:6px; border:1px solid #eee;"/>` +
                `</a>` +
                badge +
                `<button class="thumb_delete" data-eid="${it.event_id}" title="Delete" style="position:absolute; right:4px; top:4px; background:#fff; border:1px solid #ccc; border-radius:10px; width:22px; height:22px; line-height:1; color:#b00000; cursor:pointer;">×</button>` +
                `</div>`;
        }).join('');
        grid.querySelectorAll('.thumb_delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.preventDefault();
                e.stopPropagation();
                const id = btn.getAttribute('data-eid');
                if (!confirm(`Delete event #${id} and its images?`)) return;
                try {
                    await fetch(`/api/events/${id}`, {
                        method: 'DELETE'
                    });
                } catch (err) {
                    console.error(err);
                    alert('Delete failed');
                    return;
                }
                await loadThumbnails();
                try {
                    await loadEvents();
                } catch (err) {}
                try {
                    await loadImagesSummary();
                } catch (err) {}
            });
        });
    } catch (e) {
        console.error(e);
    }
}
refresh();
loadZones();
loadZonesJson();
loadEvents();
loadImagesSummary();
loadPerception();
refreshBaseline();
async function clearAllEvents() {
    try {
        if (!confirm('Delete all events? This cannot be undone.')) return;
        let bulk = false;
        try {
            const r = await fetch('/api/events/clear_all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            bulk = r.ok;
        } catch (e) {}
        if (!bulk) {
            const j = await (await fetch('/api/events?limit=10000')).json();
            const ids = (j.items || []).map(it => it.id);
            for (const id of ids) {
                try {
                    await fetch(`/api/events/${id}`, {
                        method: 'DELETE'
                    });
                } catch (e) {}
            }
        }
        try {
            await fetch('/api/images/cleanup', {
                method: 'POST'
            });
        } catch (e) {}
        await loadEvents();
        alert('Events cleared');
    } catch (e) {
        console.error(e);
        alert('Clear failed');
    }
}
