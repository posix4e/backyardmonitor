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
}

load();

