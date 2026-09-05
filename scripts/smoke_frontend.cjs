// Existing workspace Playwright/Chromium; override PLAYWRIGHT_MODULE elsewhere.
// Run with `make smoke-frontend`. All data and service state live under /tmp.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn, execFileSync } = require('node:child_process');
const { test, before, after } = require('node:test');
const repo = path.resolve(__dirname, '..');
const { chromium, expect } = require(process.env.PLAYWRIGHT_MODULE || path.resolve(
  repo, '../../../web/python-lair-2026/node_modules/@playwright/test',
));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const listen = server => new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', () => resolve(server.address().port));
});
let root, browser, configPath;

before(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), 'rag-frontend-'));
  configPath = path.join(root, 'config.yaml');
  await fs.writeFile(configPath, JSON.stringify({
    data_dir: path.join(root, 'data'), sqlite_url: `sqlite:///${root}/data/app.db`,
    index_path: path.join(root, 'data/index.npz'), id_map_path: path.join(root, 'data/id_map.json'),
    openai_api_key: null,
  }));
  browser = await chromium.launch({ headless: true });
});
after(async () => { if (browser) await browser.close(); });

async function staticPage(docs) {
  const page = await browser.newPage();
  const files = { '/': ['index.html', 'text/html'], '/assets/styles.css': ['styles.css', 'text/css'],
    '/assets/app.js': ['app.js', 'application/javascript'] };
  await page.route('**/*', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (files[pathname]) {
      const [name, contentType] = files[pathname];
      return route.fulfill({ status: 200, contentType,
        body: await fs.readFile(path.join(repo, 'src/local_rag_backend/frontend', name)) });
    }
    if (pathname === '/api/docs/query') return route.fulfill({ json: docs });
    return route.fulfill({ status: 404, body: '' });
  });
  await page.goto('http://smoke.test/', { waitUntil: 'networkidle' });
  return page;
}

test('accepted Elasticsearch IDs and HTML previews remain literal text', async () => {
  const docs = JSON.parse(execFileSync('uv', ['run', '--project', repo, '--frozen', '--no-sync',
    'python', path.join(repo, 'tests/support/frontend_query_fixture.py')], {
    cwd: root, env: { ...process.env, RAG_CONFIG_PATH: configPath, UV_CACHE_DIR: '/tmp/uv-cache' },
    encoding: 'utf8',
  }));
  const page = await staticPage(docs);
  try {
    assert.equal(await page.evaluate(() => document.documentElement.dataset.domProbe), undefined);
    assert.equal(await page.locator('#docsList img, #docsList b').count(), 0);
    assert.deepEqual(await page.locator('#docsList strong').allTextContents(), docs.map(d => `#${d.id}`));
    await expect(page.locator('#docsList')).toContainText('Preview: <b>literal HTML</b>');
  } finally { await page.close(); }
});

test('preview strips only a verified ingestion header and measures its body', async () => {
  const metadata = { source: 'api:/docs/ingest', input_index: 0, chunk_index: 0,
    chunk_start_char: 0, chunk_end_char: 10, chunker_version: 'chars_v1',
    embedding_model: 'none', dedup_sha256: 'a'.repeat(64), parent_doc_id: 'api:/docs/ingest:text=0' };
  const header = Object.entries(metadata).map(([key, value]) =>
    `${key.split('_').map(part => part[0].toUpperCase() + part.slice(1)).join('_')}: ${value}`).join('\n');
  const ingested = body => ({ id: 'ingested', external_id: `chunk:${metadata.dedup_sha256}`,
    metadata, content: `${header}\n\n${body}` });
  const plain = content => ({ id: 'canonical', content });
  const docs = [ingested('Short body'), ingested('x'.repeat(161)), plain('Canonical body'),
    plain('Title: legitimate\nAuthor: Ada\n\nKeep this header'),
    plain('---\ntitle: canonical frontmatter\n---\nBody'), plain('<b>HTML: literal</b>'), plain(''),
    plain(`${header}\n\nCanonical technical prose`),
    { ...ingested('Keep unknown header'), content: `Unknown: extra\n${header}\n\nBody` },
    { ...ingested('No separator'), content: header },
  ];
  const page = await staticPage(docs);
  try {
    const rows = await page.locator('#docsList > div').allTextContents();
    assert.equal(rows[0], '#ingested — Short body');
    assert.equal(rows[1], `#ingested — ${'x'.repeat(160)}…`);
    for (let i = 2; i < docs.length; i++) {
      const content = docs[i].content;
      assert.equal(rows[i], `#${docs[i].id} — ${content.slice(0, 160).replace(/\s+/g, ' ').trim()}${content.length > 160 ? '…' : ''}`);
    }
    assert.equal(await page.locator('#docsList b').count(), 0);
  } finally { await page.close(); }
});

test('real HTTP and SQLite add/list/query, status and history lifecycle', async () => {
  const text = 'Comet 731 catalog code is BLUE-731. Temporary local browser smoke.';
  const question = 'What is the comet 731 catalog code?';
  const answer = 'The comet 731 catalog code is BLUE-731.';
  const llmCalls = [], requests = [], errors = [];
  let serverLogs = '', rag, mock, page;
  try {
    mock = http.createServer(async (req, res) => {
      let body = '';
      for await (const part of req) body += part;
      res.setHeader('Content-Type', 'application/json');
      if (req.method === 'POST' && req.url === '/api/generate') {
        llmCalls.push(JSON.parse(body));
        await sleep(800);
        res.end(JSON.stringify({ response: answer }));
      } else if (req.url === '/api/tags') {
        res.end(JSON.stringify({ models: [{ name: 'local-smoke' }] }));
      } else { res.statusCode = 404; res.end('{}'); }
    });
    const mockPort = await listen(mock);
    const reservation = net.createServer();
    const port = await listen(reservation);
    await new Promise(resolve => reservation.close(resolve));
    const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
    await fs.writeFile(configPath, JSON.stringify({ ...config,
      app_host: '127.0.0.1', app_port: port, debug: false, log_level: 'WARNING',
      persistence_backend: 'local_split', search_backend: 'local_split', retrieval_mode: 'sparse',
      embedding_cache_db_path: path.join(root, 'data/embedding-cache.sqlite3'),
      faq_csv: path.join(root, 'data/unused-faq.csv'), eval_dataset_path: path.join(root, 'data/unused-eval.jsonl'),
      openrouter_enabled: false, ollama_enabled: true, ollama_model: 'local-smoke',
      ollama_base_url: `http://127.0.0.1:${mockPort}`, ollama_request_timeout: 10,
      mutation_recovery_enabled: false, api_key: null, public_bind_requires_api_key: true,
      enable_monitoring: false,
    }));
    rag = spawn('uv', ['run', '--project', repo, '--frozen', '--no-sync', 'rag-server'], {
      cwd: root, env: { ...process.env, RAG_CONFIG_PATH: configPath, UV_CACHE_DIR: '/tmp/uv-cache' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    rag.stdout.on('data', b => serverLogs += String(b));
    rag.stderr.on('data', b => serverLogs += String(b));
    const base = `http://127.0.0.1:${port}`;
    let ready = false;
    for (let i = 0; i < 200; i++) {
      if (rag.exitCode !== null) throw Error(`RAG exited ${rag.exitCode}: ${serverLogs}`);
      try { if ((await fetch(base)).ok) { ready = true; break; } } catch {}
      await sleep(200);
    }
    assert(ready, `RAG startup timeout: ${serverLogs}`);
    page = await browser.newPage();
    page.on('pageerror', error => errors.push(String(error)));
    page.on('request', req => {
      if (req.url().startsWith(base + '/api/')) requests.push({ method: req.method(),
        path: new URL(req.url()).pathname, payload: req.postDataJSON() });
    });
    await page.goto(base, { waitUntil: 'networkidle' });
    const status = page.locator('#statusEl'), history = page.locator('#historyTools');
    await expect(status).toBeHidden();
    assert.equal(await status.evaluate(el => getComputedStyle(el).display), 'none');
    await expect(history).toBeHidden();
    assert.equal(await history.evaluate(el => getComputedStyle(el).display), 'none');
    await expect(page.locator('#docsList')).toContainText('No documents found');
    await page.locator('#docText').fill(text);
    const ingestion = page.waitForResponse(base + '/api/docs/ingest');
    await page.locator('#addDocBtn').click();
    const response = await ingestion;
    assert.equal(response.status(), 200);
    const ingested = await response.json();
    assert.equal(ingested.count, 1);
    await expect(page.locator('#docsList')).toContainText(text.toLowerCase());
    await expect(page.locator('#docText')).toHaveValue('');
    await page.locator('#refreshDocsBtn').click();
    await page.reload({ waitUntil: 'networkidle' });
    await expect(page.locator('#docsList')).toContainText(ingested.ids[0]);
    await page.locator('#question').fill(question);
    await page.locator('#kValue').fill('1');
    const askPromise = page.waitForResponse(base + '/api/ask');
    await page.locator('#askBtn').click();
    await expect(status).toBeVisible();
    await expect(status).toContainText('Processing');
    const ask = await askPromise;
    assert.equal(ask.status(), 200);
    const result = await ask.json();
    assert.equal(result.answer, answer);
    assert.equal(result.sources.length, 1);
    assert(result.sources[0].document.content.includes(text.toLowerCase()));
    await expect(page.locator('#answerBox')).toHaveText(answer);
    await expect(status).toBeHidden();
    await expect(history).toBeVisible();
    assert.equal(llmCalls.length, 1);
    assert(llmCalls[0].prompt.includes(text.toLowerCase()));
    assert(llmCalls[0].prompt.includes(question));
    page.once('dialog', dialog => dialog.accept());
    await page.locator('#clearHistoryBtn').click();
    await expect(history).toBeHidden();
    await expect(page.locator('#historyBox')).toBeHidden();
    await page.route('**/api/ask', async route => {
      await sleep(800);
      await route.fulfill({ status: 503, json: { detail: 'Deliberate smoke failure' } });
    });
    await page.locator('#question').fill('Error state');
    await page.locator('#askBtn').click();
    await expect(status).toBeVisible();
    await expect(page.locator('#answerBox')).toContainText('Deliberate smoke failure');
    await expect(status).toBeHidden();
    await expect(page.locator('#askBtn')).toBeEnabled();
    await expect(history).toBeHidden();
    assert.deepEqual(errors, []);
    const queries = requests.filter(req => req.path === '/api/docs/query');
    assert(queries.length >= 3);
    assert(queries.every(req => req.method === 'POST' && req.payload.limit === 100 && req.payload.offset === 0));
    assert(requests.every(req => req.path !== '/api/docs'));
    await fs.writeFile(path.join(root, 'summary.json'), JSON.stringify({ status: 'passed',
      ingested, requests, errors, limitation: 'Real Chromium/HTTP/SQLite/retrieval; local Ollama double and deliberate error response.' }, null, 2));
    console.log(`Frontend smoke evidence: ${root}`);
  } finally {
    if (page) await page.close();
    if (rag && rag.exitCode === null) {
      rag.kill('SIGTERM');
      await Promise.race([new Promise(resolve => rag.once('exit', resolve)), sleep(5000)]);
      if (rag.exitCode === null) rag.kill('SIGKILL');
    }
    if (mock) await new Promise(resolve => mock.close(resolve));
    await fs.writeFile(path.join(root, 'server.log'), serverLogs);
  }
});
