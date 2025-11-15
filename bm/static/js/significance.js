function q(key) {
  const u = new URLSearchParams(location.search);
  return u.get(key) || '';
}

function kindFromSrc(srcKey) {
  if (!srcKey) return null;
  const m = String(srcKey).replace(/^image_/, '');
  if (m === 'thumb' || m === 'crop' || m === 'full') return m;
  return null;
}

function fmtTs(t) {
  try { return new Date((t || 0) * 1000).toLocaleString(); } catch (e) { return String(t||''); }
}

async function load() {
  const id = q('event');
  if (!id) { alert('Missing ?event='); return; }
  const statusEl = document.getElementById('run_status');
  const ev = await (await fetch(`/api/events/${encodeURIComponent(String(id))}`)).json();
  const meta = ev && ev.meta ? ev.meta : {};
  // Fill header
  const ml = document.getElementById('meta_line');
  if (ml) ml.textContent = `#${ev.id} · ${ev.kind || ''} · ${fmtTs(ev.ts)}`;
  // Resolve previous event
  let prevId = meta.llm_prev_event_id || null;
  let prev = null;
  if (prevId) {
    try { prev = await (await fetch(`/api/events/${encodeURIComponent(String(prevId))}`)).json(); } catch (e) {}
  }
  const prevKind = kindFromSrc(meta.llm_prev_image_source) || 'thumb';
  const curKind = kindFromSrc(meta.llm_image_source) || 'thumb';
  const prevUrl = prev ? `/api/event_image?id=${encodeURIComponent(String(prev.id))}&kind=${prevKind}` : '';
  const curUrl = `/api/event_image?id=${encodeURIComponent(String(ev.id))}&kind=${curKind}`;
  // Images
  const ip = document.getElementById('img_prev');
  const ic = document.getElementById('img_cur');
  if (ip && prevUrl) ip.src = prevUrl;
  if (ic) ic.src = curUrl;
  const pi = document.getElementById('prev_info');
  if (pi) pi.textContent = prev ? `#${prev.id} (${prevKind}) · ${fmtTs(prev.ts)}` : 'None';
  const ci = document.getElementById('cur_info');
  if (ci) ci.textContent = `#${ev.id} (${curKind}) · ${fmtTs(ev.ts)}`;
  const im = document.getElementById('img_mask');
  if (im) {
    if (meta.image_mask) {
      im.src = `/data/frames/${encodeURIComponent(String(meta.image_mask))}`;
    } else {
      im.parentElement.style.display = 'none';
    }
  }
  // LLM details
  const sig = !!meta.significant;
  const sigB = document.getElementById('sig_badge');
  if (sigB) {
    sigB.textContent = sig ? 'Significant' : 'Not significant';
    sigB.className = `badge ${sig ? 'good' : 'warn'}`;
  }
  const stB = document.getElementById('status_badge');
  if (stB) {
    const st = meta.llm_status || '-';
    stB.textContent = `Status: ${st}`;
    stB.className = `badge ${st === 'done' ? 'good' : 'warn'}`;
  }
  const prov = document.getElementById('prov');
  if (prov) prov.textContent = `${meta.llm_provider || ''} / ${meta.llm_model || ''}`;
  const reason = document.getElementById('reason');
  if (reason) reason.textContent = meta.llm_reason || meta.llm_error || '';
  const prompt = document.getElementById('prompt');
  if (prompt) prompt.textContent = meta.llm_prompt || '';
  const response = document.getElementById('response');
  if (response) response.textContent = meta.llm_response || '';
  // Analysis
  const an = [];
  const method = (meta.method || '').toLowerCase();
  if (method === 'roi_diff') {
    an.push('[ROI Diff]');
    if (meta.ratio != null) an.push(`ratio: ${meta.ratio}`);
    if (meta.roi_diff_threshold != null) an.push(`threshold_ratio: ${meta.roi_diff_threshold}`);
    if (meta.roi_diff_min_pixels != null) an.push(`min_pixels: ${meta.roi_diff_min_pixels}`);
    if (meta.changed_pixels != null) an.push(`changed_pixels: ${meta.changed_pixels}`);
    if (meta.roi_pixels != null) an.push(`roi_pixels: ${meta.roi_pixels}`);
    if (meta.stable_ms_used != null) an.push(`stable_ms_used: ${meta.stable_ms_used}`);
    if (meta.state) an.push(`state: ${meta.state}`);
  }
  if (meta.prev_sig || meta.new_sig || meta.delta_bits != null) {
    an.push('[pHash]');
    if (meta.prev_sig) an.push(`prev_sig: ${meta.prev_sig}`);
    if (meta.new_sig) an.push(`new_sig: ${meta.new_sig}`);
    if (meta.delta_bits != null) an.push(`delta_bits: ${meta.delta_bits}`);
    if (meta.phash_min_bits_used != null) an.push(`min_bits_used: ${meta.phash_min_bits_used}`);
    if (meta.stable_ms_used != null) an.push(`stable_ms_used: ${meta.stable_ms_used}`);
  }
  const anEl = document.getElementById('analysis');
  if (anEl) anEl.textContent = an.join('\n');

  // Wire re-run button
  const btn = document.getElementById('btn_rerun');
  if (btn) {
    btn.addEventListener('click', async () => {
      try {
        statusEl.textContent = 'Queuing analysis…';
        const r = await fetch(`/api/llm/queue_event?id=${encodeURIComponent(String(id))}`, { method: 'POST' });
        if (!r.ok) {
          statusEl.textContent = 'Queue failed';
          return;
        }
        // Poll until llm_status is done/error
        const start = Date.now();
        const timeoutMs = 20000;
        async function poll() {
          const ev2 = await (await fetch(`/api/events/${encodeURIComponent(String(id))}`)).json();
          const m2 = ev2 && ev2.meta ? ev2.meta : {};
          if (m2.llm_status && m2.llm_status !== 'started') {
            statusEl.textContent = 'Completed';
            location.reload();
            return;
          }
          if (Date.now() - start > timeoutMs) {
            statusEl.textContent = 'Timed out waiting for result';
            return;
          }
          setTimeout(poll, 800);
        }
        setTimeout(poll, 800);
      } catch (e) {
        statusEl.textContent = 'Error';
      }
    });
  }
}

load();
