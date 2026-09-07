const qs = (selector) => document.querySelector(selector);
const API_URL = '/api/ask';
const DOCS_INGEST_URL = '/api/docs/ingest';
const DOCS_QUERY_URL = '/api/docs/query';

// --- DOM Elements ---
const themeBtn = qs('#themeBtn');
const askBtn = qs('#askBtn');
const questionEl = qs('#question');
const kEl = qs('#kValue');
const statusEl = qs('#statusEl');
const answerBox = qs('#answerBox');
const answerMeta = qs('#answerMeta');
const historyBox = qs('#historyBox');
const historyTools = qs('#historyTools');
const clearHistoryBtn = qs('#clearHistoryBtn');
const exampleListEl = qs('#exampleList');
const docTextEl = qs('#docText');
const addDocBtn = qs('#addDocBtn');
const refreshDocsBtn = qs('#refreshDocsBtn');
const docsListEl = qs('#docsList');

statusEl.hidden = true;

// --- Theme Management ---
const applyTheme = (theme) => {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('theme', theme);
  themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
  themeBtn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
};
const storedTheme = localStorage.getItem('theme');
const preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
applyTheme(storedTheme || preferredTheme);
themeBtn.addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

// --- Example Questions ---
exampleListEl.addEventListener('click', (e) => {
    if (e.target.tagName === 'LI') {
        questionEl.value = e.target.textContent;
        questionEl.focus();
    }
});
// --- Floating label for textarea ---
questionEl.addEventListener('input', () => {
  questionEl.setAttribute('data-has-content', questionEl.value.trim() ? "1" : "");
});
// --- Event Listeners ---
questionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});
askBtn.addEventListener('click', async () => {
  const question = questionEl.value.trim();
  if (!question) return;
  const k = parseInt(kEl.value) || 3;

  askBtn.disabled = true;
  statusEl.hidden = false;
  answerBox.textContent = 'Asking the oracle...';
  answerMeta.innerHTML = '';

  let answer = '';
  let sources = [];
  let errorMsg = '';

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k })
    });

    if (!response.ok) {
      let errorData;
      try { errorData = await response.json(); } catch (e) { errorData = { detail: await response.text() || `HTTP error ${response.status}`}; }
      const message = errorData.detail || (typeof errorData === 'string' ? errorData : 'Unknown error');
      throw new Error(message, {cause: response.status});
    }

    const json = await response.json();
    answer = json.answer || '(No answer received)';
    sources = Array.isArray(json.sources) ? json.sources : [];
  } catch (err) {
    errorMsg = `Error: ${err.message}`;
  } finally {
    statusEl.hidden = true;
    askBtn.disabled = false;

    if (errorMsg) {
      answerBox.textContent = errorMsg;
    } else {
      answerBox.textContent = answer;
      renderAnswerMeta(answer, sources);
      pushToHistory(question, answer, sources);
      questionEl.value = '';
      questionEl.focus();
    }
  }
});
clearHistoryBtn.addEventListener('click', () => {
  if (confirm('Are you sure you want to clear the history?')) {
    historyBox.innerHTML = '';
    historyBox.hidden = true;
    historyTools.hidden = true;
    answerBox.textContent = 'The answer will appear here…';
    answerMeta.innerHTML = '';
  }
});

// --- Helper Functions ---
function renderAnswerMeta(answerText, sources) {
  answerMeta.innerHTML = '';
  if (sources.length > 0) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'source-ids';
    sourcesDiv.innerHTML = 'Sources: ';
    sources.forEach((src, i) => {
      if (src.document) {
        const codeEl = document.createElement('code');
        codeEl.textContent = `#${src.document.id}`;
        codeEl.title = src.document.content.length > 60 ? src.document.content.slice(0,60) + '…' : src.document.content;
        sourcesDiv.appendChild(codeEl);
        if (i < sources.length - 1) sourcesDiv.append(', ');
      }
    });
    answerMeta.appendChild(sourcesDiv);
  }
}
function pushToHistory(question, answer, sources) {
  historyTools.hidden = false;
  historyBox.hidden = false;
  const itemDiv = document.createElement('div');
  itemDiv.className = 'history-item';
  const qPara = document.createElement('p');
  qPara.innerHTML = `<strong>Q:</strong> ${escapeHTML(question)}`;
  itemDiv.appendChild(qPara);
  const aPara = document.createElement('p');
  aPara.innerHTML = `<strong>A:</strong> ${escapeHTML(answer)}`;
  itemDiv.appendChild(aPara);
  if (sources.length > 0) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'source-ids';
    sourcesDiv.innerHTML = 'Sources: ';
    sources.forEach((src, i) => {
      if (src.document) {
        const codeEl = document.createElement('code');
        codeEl.textContent = `#${src.document.id}`;
        codeEl.title = src.document.content.length > 60 ? src.document.content.slice(0,60) + '…' : src.document.content;
        sourcesDiv.appendChild(codeEl);
        if (i < sources.length - 1) sourcesDiv.append(', ');
      }
    });
    itemDiv.appendChild(sourcesDiv);
  }
  historyBox.prepend(itemDiv);
  historyBox.scrollTop = 0;
}
function escapeHTML(str) {
  const p = document.createElement('p');
  p.textContent = str;
  return p.innerHTML;
}

addDocBtn.addEventListener('click', async () => {
  const txt = (docTextEl.value || '').trim();
  if (!txt) return;
  addDocBtn.disabled = true;
  addDocBtn.textContent = 'Adding...';
  try {
    const r = await fetch(DOCS_INGEST_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ texts: [txt] })
    });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    docTextEl.value = '';
    await loadDocs();
    toast(`✅ Ingested ${j.count} document${j.count===1?'':'s'}`);
  } catch (e) {
    toast(`❌ Ingest error: ${e.message||e}`, true);
  } finally {
    addDocBtn.disabled = false;
    addDocBtn.textContent = 'Add';
  }
});

refreshDocsBtn.addEventListener('click', loadDocs);

async function loadDocs() {
  try {
    const r = await fetch(DOCS_QUERY_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({limit: 100, offset: 0})
    });
    if (!r.ok) throw new Error(await r.text());
    const docs = await r.json();
    renderDocs(docs);
  } catch (e) {
    docsListEl.innerHTML = `<span style="color:crimson">Failed to load docs: ${escapeHTML(String(e))}</span>`;
  }
}

function documentPreviewBody(doc) {
  const content = doc.content || '';
  const md = doc.metadata;
  // docs_ingest emits this exact header. Require its matching metadata and
  // chunk identity so canonical frontmatter and ordinary prose remain intact.
  const fields = ['source', 'input_index', 'chunk_index', 'chunk_start_char',
    'chunk_end_char', 'chunker_version', 'embedding_model', 'dedup_sha256', 'parent_doc_id'];
  if (!md || !fields.every(key => Object.hasOwn(md, key)) ||
      !/^[a-f0-9]{64}$/.test(md.dedup_sha256) ||
      doc.external_id !== `chunk:${md.dedup_sha256}` ||
      md.parent_doc_id !== `${md.source}:text=${md.input_index}` ||
      !['input_index', 'chunk_index', 'chunk_start_char', 'chunk_end_char'].every(
        key => Number.isInteger(md[key]) && md[key] >= 0) ||
      !fields.every(key => !/[\r\n]/.test(String(md[key])))) return content;
  const header = fields.map(key => {
    const title = key.split('_').map(part => part[0].toUpperCase() + part.slice(1)).join('_');
    return `${title}: ${md[key]}`;
  }).join('\n') + '\n\n';
  return content.startsWith(header) ? content.slice(header.length) : content;
}

function renderDocs(docs) {
  if (!Array.isArray(docs) || docs.length === 0) {
    docsListEl.innerHTML = `<em style="color:var(--subtle-text)">No documents found.</em>`;
    return;
  }
  const frag = document.createDocumentFragment();
  docs.forEach(d => {
    const div = document.createElement('div');
    div.style.borderBottom = '1px dashed var(--border)';
    div.style.padding = '.35rem 0';
    const p = document.createElement('div');
    const body = documentPreviewBody(d);
    const preview = body.slice(0, 160).replace(/\s+/g,' ').trim();
    const id = document.createElement('strong');
    id.textContent = `#${d.id}`;
    p.append(id, ' — ', preview, body.length > 160 ? '…' : '');
    div.appendChild(p);
    frag.appendChild(div);
  });
  docsListEl.innerHTML = '';
  docsListEl.appendChild(frag);
}

function toast(msg, isErr=false){
  answerMeta.innerHTML = `<span style="color:${isErr?'crimson':'var(--accent)'}">${escapeHTML(msg)}</span>`;
  setTimeout(()=>{answerMeta.innerHTML='';}, 2200);
}

// auto-load at startup
loadDocs();
