// ── Session thread ID ─────────────────────────────────────────────────────────
// Persisted per browser tab via sessionStorage so each tab has an isolated
// multi-turn conversation while the same tab keeps full context.
let sessionThreadId = sessionStorage.getItem('ag_thread_id');
if (!sessionThreadId) {
  sessionThreadId = crypto.randomUUID();
  sessionStorage.setItem('ag_thread_id', sessionThreadId);
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chat       = document.getElementById('chat');
const form       = document.getElementById('chatForm');
const question   = document.getElementById('question');
const trace      = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

// ── Markdown-lite renderer ────────────────────────────────────────────────────
// Supports: **bold**, *italic*, `inline code`, ```code blocks```,
//           # headings, - / * bullet lists, numbered lists, and line breaks.
function escapeHtml(s = '') {
  return s.replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderMarkdown(raw = '') {
  // Protect code blocks first (triple backtick)
  const codeBlocks = [];
  let md = raw.replace(/```([\s\S]*?)```/g, (_, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return `\x00CODE${idx}\x00`;
  });

  // Inline code
  md = md.replace(/`([^`]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`);

  // Headings
  md = md.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  md = md.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold & italic
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g,     '<em>$1</em>');

  // Bullet lists (- or *)
  md = md.replace(/^[ \t]*[-*] (.+)$/gm, '<li>$1</li>');
  md = md.replace(/(<li>[\s\S]*?<\/li>)/g, m => `<ul>${m}</ul>`);

  // Numbered lists
  md = md.replace(/^[ \t]*\d+\. (.+)$/gm, '<li>$1</li>');

  // Line breaks → <br> (but not inside block-level tags we already created)
  md = md.replace(/\n/g, '<br>');

  // Restore code blocks
  md = md.replace(/\x00CODE(\d+)\x00/g, (_, i) => codeBlocks[+i]);

  return md;
}

// ── Message rendering ─────────────────────────────────────────────────────────
function addMessage(role, text, source = '', citations = []) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const citeHtml = citations.length
    ? `<div class="citations"><strong>Sources</strong><br>${
        citations.map(c => c.url
          ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`
          : escapeHtml(c.title)
        ).join('<br>')
      }</div>`
    : '';

  const bodyHtml = role === 'assistant' ? renderMarkdown(text) : escapeHtml(text).replace(/\n/g, '<br>');

  wrap.innerHTML = `
    <div class="avatar">${role === 'assistant' ? 'AI' : 'You'}</div>
    <div class="bubble">
      ${bodyHtml}
      ${source ? `<div class="answer-source">Source: ${escapeHtml(source)}</div>` : ''}
      ${citeHtml}
    </div>`;

  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
  trace.innerHTML = items.length
    ? items.map(x => `<div class="trace-item">${escapeHtml(x)}</div>`).join('')
    : '<div class="empty">No trace.</div>';
}

// ── Agent call ────────────────────────────────────────────────────────────────
async function askAgent(q) {
  addMessage('user', q);
  question.value = '';
  renderTrace(['Running LangGraph workflow…']);
  sourceUsed.textContent = 'Running';
  const btn = form.querySelector('button');
  btn.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, thread_id: sessionThreadId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');

    // If the server echoes back the thread_id (new session bootstrap), persist it
    if (data.thread_id) {
      sessionThreadId = data.thread_id;
      sessionStorage.setItem('ag_thread_id', sessionThreadId);
    }

    addMessage('assistant', data.answer, data.source_used, data.citations || []);
    renderTrace(data.trace || []);
    sourceUsed.textContent = data.source_used;
  } catch (e) {
    addMessage('assistant', `Error: ${e.message}`);
    renderTrace(['Request failed']);
    sourceUsed.textContent = 'Error';
  } finally {
    btn.disabled = false;
  }
}

// ── Form & examples ───────────────────────────────────────────────────────────
form.addEventListener('submit', e => { e.preventDefault(); const q = question.value.trim(); if (q) askAgent(q); });
document.querySelectorAll('.example').forEach(b => b.addEventListener('click', () => askAgent(b.textContent.trim())));

// ── Upload modal ──────────────────────────────────────────────────────────────
const modal = document.getElementById('uploadModal');
document.getElementById('openUpload').onclick  = () => modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick = () => modal.classList.add('hidden');

document.getElementById('uploadBtn').onclick = async () => {
  const file   = document.getElementById('fileInput').files[0];
  const key    = document.getElementById('adminKey').value;
  const status = document.getElementById('uploadStatus');
  if (!file) { status.textContent = 'Choose a file first.'; return; }
  status.textContent = 'Indexing document…';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/ingest', { method: 'POST', headers: { 'X-Admin-Key': key }, body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');
    status.textContent = `✅ Indexed ${d.file}: ${d.chunks} chunks.`;
  } catch (e) {
    status.textContent = `❌ Error: ${e.message}`;
  }
};
