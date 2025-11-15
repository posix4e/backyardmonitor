function q(key) {
    const u = new URLSearchParams(location.search);
    return u.get(key) || '';
}

function fmtTs(t) {
    try {
        return new Date((t || 0) * 1000).toLocaleString();
    } catch (e) {
        return String(t || '');
    }
}

async function loadExplain(eventId) {
    if (!eventId) return;
    try {
        const r = await fetch(`/api/event_explain?id=${encodeURIComponent(eventId)}`);
        if (!r.ok) return;
        const j = await r.json();
        const ex = j && j.explain ? j.explain : {};
        const lines = [];
        if (ex.spot_id) lines.push(`spot_id: ${ex.spot_id}`);
        if (ex.delta_bits != null) lines.push(`delta_bits: ${ex.delta_bits}`);
        if (ex.min_bits_used != null) lines.push(`min_bits_used: ${ex.min_bits_used}`);
        if (ex.stable_ms_config != null) lines.push(`stable_ms: ${ex.stable_ms_config}`);
        if (typeof ex.meets_threshold === 'boolean') lines.push(`meets_threshold: ${ex.meets_threshold}`);
        try {
            // Enrich with LLM view by fetching event directly
            const ev = await (await fetch(`/api/events/${encodeURIComponent(String(eventId))}`)).json();
            const m = ev && ev.meta ? ev.meta : {};
            if (m.llm_status || m.llm_attempted) {
                lines.push('');
                lines.push('[LLM]');
                lines.push(`status: ${m.llm_status || '-'}`);
                if (m.significant != null) lines.push(`significant: ${!!m.significant}`);
                if (m.llm_reason) lines.push(`reason: ${m.llm_reason}`);
                if (m.llm_error) lines.push(`error: ${m.llm_error}`);
                if (m.llm_provider) lines.push(`provider: ${m.llm_provider}`);
                if (m.llm_model) lines.push(`model: ${m.llm_model}`);
            }
        } catch (e) {}
        document.getElementById('explain').textContent = lines.join('\n');
    } catch (e) {
        /* ignore */
    }
}

async function load() {
    let spotId = q('spot');
    const eventId = q('event');
    // If spot is not provided, derive it from the event meta
    let ev = null;
    if (eventId) {
        try {
            ev = await (await fetch(`/api/events/${encodeURIComponent(eventId)}`)).json();
        } catch (e) {}
        if (!spotId && ev && ev.meta && ev.meta.spot_id) {
            spotId = ev.meta.spot_id;
        }
    }
    document.getElementById('spot_id').textContent = spotId || '-';
    if (!spotId) {
        alert('Missing spot (could not derive from event)');
        return;
    }
    try {
        await loadSpotConfig(spotId);
    } catch (e) {}
    if (eventId) {
        await loadExplain(eventId);
    } // only when provided

    // Load spot history (newest first)
    const lim = parseInt((document.getElementById('limit').value || '60'), 10);
    const sigOnly = !!(document.getElementById('sig_only') && document.getElementById('sig_only').checked);
    const hist = await (await fetch(`/api/spot_history?spot_id=${encodeURIComponent(spotId)}&limit=${encodeURIComponent(String(lim))}&significant_only=${sigOnly ? '1' : '0'}`)).json();
    const items = (hist && hist.items) ? hist.items : [];
    const grid = document.getElementById('timeline');
    if (!items.length) {
        // Fallback: scan recent general events and filter by spot_id
        try {
            const j = await (await fetch('/api/events?limit=500')).json();
            const all = (j && j.items) ? j.items : [];
            let filt = all.filter(it => String((it.meta && it.meta.spot_id) || '') === String(spotId) && String(it.kind || '').toLowerCase() === 'spot_change');
            if (sigOnly) filt = filt.filter(it => !!(it.meta && it.meta.significant));
            if (!filt.length) {
                grid.innerHTML = '<div class="muted">No events for this spot yet.</div>';
                return;
            }
            // Map to same shape as spot_history items
            const mapped = filt.map(it => ({
                id: it.id,
                ts: it.ts,
                spot_id: spotId,
                thumb: `/api/event_image?id=${it.id}&kind=thumb`,
                full: `/api/event_image?id=${it.id}&kind=full`,
                crop: `/api/event_image?id=${it.id}&kind=crop`
            }));
            // Continue with mapped list
            await renderTimeline(spotId, mapped, eventId);
            return;
        } catch (e) {
            grid.innerHTML = '<div class="muted">No events for this spot yet.</div>';
            return;
        }
    }
    grid.innerHTML = '<div class="muted">Loading timeline…</div>';
    await renderTimeline(spotId, items, eventId);
}

async function loadSpotConfig(spotId) {
    try {
        const z = await (await fetch('/api/zones')).json();
        const sp = (z.spots || []).find(s => String(s.id) === String(spotId));
        const nameEl = document.getElementById('spot_name');
        if (nameEl) nameEl.value = (sp && sp.name) ? sp.name : '';
        const catEl = document.getElementById('spot_category');
        if (catEl) catEl.value = (sp && sp.category) ? sp.category : '';
        const minEl = document.getElementById('spot_override_thresh');
        const stEl = document.getElementById('spot_stable_ms');
        if (minEl) minEl.value = (sp && sp.min_bits != null) ? sp.min_bits : '';
        if (stEl) stEl.value = (sp && sp.stable_ms != null) ? sp.stable_ms : '';
    } catch (e) {
        /* ignore */
    }
}

async function saveSpotConfig() {
    const spotId = document.getElementById('spot_id').textContent || '';
    const nameEl = document.getElementById('spot_name');
    const newName = nameEl ? String(nameEl.value || '').trim() : '';
    const catEl = document.getElementById('spot_category');
    const newCat = catEl ? String(catEl.value || '').trim() : '';
    const minEl = document.getElementById('spot_override_thresh');
    const stEl = document.getElementById('spot_stable_ms');
    const minBits = (minEl && String(minEl.value).trim() !== '') ? parseInt(minEl.value, 10) : null;
    const stableMs = (stEl && String(stEl.value).trim() !== '') ? parseInt(stEl.value, 10) : null;
    if (!spotId) {
        alert('Missing spot id');
        return;
    }
    try {
        const zones = await (await fetch('/api/zones')).json();
        let updated = false;
        const spots = Array.isArray(zones.spots) ? zones.spots.map(s => {
            if (String(s.id) === String(spotId)) {
                updated = true;
                const next = {
                    ...s,
                    name: newName || s.id
                };
                if (newCat) next.category = newCat;
                else delete next.category;
                if (minBits !== null && !Number.isNaN(minBits)) next.min_bits = minBits;
                else delete next.min_bits;
                if (stableMs !== null && !Number.isNaN(stableMs)) next.stable_ms = stableMs;
                else delete next.stable_ms;
                return next;
            }
            return s;
        }) : [];
        if (!updated) {
            alert('Spot not found in zones');
            return;
        }
        const payload = {
            gate: zones.gate || null,
            spots
        };
        const res = await fetch('/api/zones', {
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
        alert('Settings saved');
    } catch (e) {
        console.error(e);
        alert('Failed to save name');
    }
}

async function renderTimeline(spotId, items, eventId) {
    const grid = document.getElementById('timeline');
    // Enrich items with delta bits via /api/events/{id}
    const details = await Promise.all(items.map(async (it) => {
        try {
            const r = await fetch(`/api/events/${encodeURIComponent(String(it.id))}`);
            if (!r.ok) return it;
            const evd = await r.json();
            it.delta_bits = (evd && evd.meta) ? evd.meta.delta_bits : null;
            return it;
        } catch (e) {
            return it;
        }
    }));
    grid.innerHTML = details.map((it, i) => renderRow(spotId, details, i, String(eventId))).join('');
    if (eventId) {
        const target = document.getElementById('ev_' + String(eventId));
        if (target) {
            target.classList.add('highlight');
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }
}

function renderRow(spotId, list, idx, highlightId) {
    const it = list[idx];
    const prev = (idx + 1 < list.length) ? list[idx + 1] : null; // older
    const thumb = it.thumb || it.crop || it.full;
    const curFull = it.full || thumb;
    const prevThumb = prev ? (prev.thumb || prev.crop || prev.full) : null;
    const delta = (it.delta_bits != null) ? `<span class="badge">Δ ${it.delta_bits}</span>` : '';
    const sigBadge = it.significant ? `<span class="badge" style="background:#0b6; color:#fff; padding:2px 6px; border-radius:10px; font-size:11px;">Significant</span>` : '';
    const idAttr = `id="ev_${String(it.id)}"`;
    const cmp = prev && prevThumb ? `
          <div class="cmp" style="display:none;">
            <div class="row" style="gap:8px;">
              <div><div class="muted">Prev</div><a href="${prev.full || prevThumb}" target="_blank"><img src="${prevThumb}"/></a></div>
              <div><div class="muted">This</div><a href="${curFull}" target="_blank"><img src="${thumb}"/></a></div>
            </div>
          </div>
        ` : '';
    return `
          <div class="rowitem" ${idAttr}>
            <a href="${curFull}" target="_blank" title="#${it.id}"><img src="${thumb}"/></a>
            <div class="meta">
              <div><span class="badge">#${it.id}</span> ${delta} ${sigBadge} <span class="muted">${fmtTs(it.ts)}</span></div>
              <div class="actions">
                <a href="/static/spot.html?spot=${encodeURIComponent(spotId)}&event=${encodeURIComponent(String(it.id))}">Link</a>
                <a href="${curFull}" target="_blank">Full</a>
                ${prev && prevThumb ? `<a href="#" onclick="return false;">Compare with previous ↓</a>` : ''}
                <a href="#" class="ev_delete" data-eid="${String(it.id)}" style="color:#b00000;">Delete</a>
              </div>
              ${cmp}
            </div>
          </div>
        `;
}

document.addEventListener('click', async (e) => {
    // toggle compare section on anchor text click
    const a = e.target.closest('a');
    if (!a) return;
    if (a.textContent && a.textContent.indexOf('Compare with previous') !== -1) {
        e.preventDefault();
        const wrap = a.closest('.rowitem');
        if (!wrap) return;
        const cmp = wrap.querySelector('.cmp');
        if (!cmp) return;
        cmp.style.display = (cmp.style.display === 'none' || !cmp.style.display) ? '' : 'none';
        return;
    }
    if (a.classList.contains('ev_delete')) {
        e.preventDefault();
        const id = a.getAttribute('data-eid');
        if (!id) return;
        if (!confirm(`Delete event #${id} and its images?`)) return;
        try {
            await fetch(`/api/events/${encodeURIComponent(String(id))}`, {
                method: 'DELETE'
            });
        } catch (err) {
            console.error(err);
            alert('Delete failed');
            return;
        }
        try {
            await load();
        } catch (err) {}
        return;
    }
});

const reloadBtn = document.getElementById('reload_btn');
if (reloadBtn) {
    reloadBtn.addEventListener('click', load);
}
const saveCfgBtn = document.getElementById('save_cfg_btn');
if (saveCfgBtn) {
    saveCfgBtn.addEventListener('click', saveSpotConfig);
}

async function clearSpotHistory() {
    try {
        const spotId = document.getElementById('spot_id').textContent || '';
        if (!spotId) return;
        if (!confirm(`Delete all history for spot ${spotId}? This cannot be undone.`)) return;
        console.log('[spot] clear history start', spotId);
        let ok = false,
            t = '';
        try {
            const res = await fetch('/api/events/clear_spot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    spot_id: spotId
                })
            });
            ok = res.ok;
            t = await res.text();
            if (!ok) console.warn('POST clear_spot failed', t);
        } catch (e) {
            console.warn('POST clear_spot error', e);
        }
        if (!ok) {
            try {
                const res2 = await fetch(`/api/events/clear_spot?spot_id=${encodeURIComponent(spotId)}`);
                ok = res2.ok;
                t = await res2.text();
                if (!ok) console.warn('GET clear_spot failed', t);
            } catch (e) {
                console.warn('GET clear_spot error', e);
            }
        }
        if (!ok) {
            alert('Clear failed: ' + t);
            return;
        }
        await load();
        alert('Spot history cleared');
    } catch (err) {
        console.error(err);
        alert('Failed to clear spot history');
    }
}

load();
