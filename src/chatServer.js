/**
 * 本地静态页 + /api/chat（一次性 JSON）与 /api/chat/stream（SSE 流式，推荐）。
 * 流式：每个字一条 data，最后一条 done；出错也会以 data: {"error":...} 结束。
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { generateContinuation } from './generate.js';
import { readHfModelDir } from './io/hfModelDir.js';
import { readMiniGPTFile } from './io/miniGptIO.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const PUBLIC = path.join(ROOT, 'public');

const defaultModelPath = () => path.join(ROOT, 'out/export/hf-style');

/** @type {{ model: import('./model/MiniGPT.js').MiniGPT, tok: object, cfg: object } | null} */
let modelCache = null;

const loadModel = () => {
  if (modelCache) return modelCache;
  const p = process.env.MODEL_PATH
    ? path.resolve(process.cwd(), process.env.MODEL_PATH)
    : defaultModelPath();
  if (!existsSync(p)) {
    throw new Error(`找不到模型: ${p}（先 npm run build 或设置 MODEL_PATH）`);
  }
  const st = statSync(p);
  const loaded = st.isDirectory() ? readHfModelDir(p) : readMiniGPTFile(p);
  modelCache = { model: loaded.model, tok: loaded.tok, cfg: loaded.cfg };
  return modelCache;
};

const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });

const sendJson = (res, status, obj) => {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(JSON.stringify(obj));
};

/**
 * 从 POST JSON 解析出生成参数（/api/chat 与 /api/chat/stream 共用）
 * @param {object} body
 */
const parseGenerateBody = (body) => {
  const prompt = typeof body.prompt === 'string' ? body.prompt : '';
  const maxNewTokens = Math.min(512, Math.max(0, parseInt(body.maxNewTokens ?? 64, 10) || 64));
  const temperature = body.greedy ? 0 : parseFloat(body.temperature ?? 0.75);
  const topK = Math.max(0, parseInt(body.topK ?? 50, 10) || 0);
  const topP = body.topP !== undefined ? parseFloat(body.topP) : 0.92;
  const repetitionPenalty = body.repetitionPenalty !== undefined ? parseFloat(body.repetitionPenalty) : 1.12;
  const repetitionWindow = Math.min(256, Math.max(8, parseInt(body.repetitionWindow ?? 48, 10) || 48));
  const greedy = !!body.greedy;
  const seed = body.seed !== undefined && body.seed !== '' ? parseInt(body.seed, 10) >>> 0 : undefined;
  return {
    prompt: prompt || '在',
    opts: {
      maxNewTokens,
      temperature,
      topK,
      topP: Number.isNaN(topP) ? 0.92 : topP,
      repetitionPenalty: Number.isNaN(repetitionPenalty) ? 1.12 : repetitionPenalty,
      repetitionWindow,
      greedy,
      seed,
    },
  };
};

/** 写 SSE 一行；尽量减小缓冲（本地 TCP_NODELAY + 反代不缓冲头） */
const sseWrite = (res, obj) => {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host}`);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  if (req.method === 'GET' && url.pathname === '/') {
    const htmlPath = path.join(PUBLIC, 'index.html');
    if (!existsSync(htmlPath)) {
      res.writeHead(404);
      res.end('Missing public/index.html');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(readFileSync(htmlPath, 'utf8'));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/app.js') {
    const jsPath = path.join(PUBLIC, 'app.js');
    if (!existsSync(jsPath)) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/javascript; charset=utf-8' });
    res.end(readFileSync(jsPath, 'utf8'));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/styles.css') {
    const cssPath = path.join(PUBLIC, 'styles.css');
    if (!existsSync(cssPath)) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/css; charset=utf-8' });
    res.end(readFileSync(cssPath, 'utf8'));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/chat') {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      sendJson(res, 400, { error: 'Invalid JSON' });
      return;
    }
    try {
      const { model, tok } = loadModel();
      const { prompt, opts } = parseGenerateBody(body);
      const result = await generateContinuation(model, tok, prompt, opts);
      sendJson(res, 200, {
        prompt: result.prompt,
        generated: result.generated,
        fullText: result.fullText,
        seedUsed: result.seedUsed,
      });
    } catch (e) {
      sendJson(res, 500, { error: (e instanceof Error ? e.message : String(e)) || 'error' });
    }
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/chat/stream') {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      sendJson(res, 400, { error: 'Invalid JSON' });
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'X-Accel-Buffering': 'no',
      'X-Content-Type-Options': 'nosniff',
    });
    res.socket?.setNoDelay?.(true);
    res.write(': stream\n\n');

    try {
      const { model, tok } = loadModel();
      const { prompt, opts } = parseGenerateBody(body);

      const result = await generateContinuation(model, tok, prompt, {
        ...opts,
        yieldEachToken: true,
        onToken: (ch) => {
          sseWrite(res, { token: ch });
        },
      });
      sseWrite(res, { done: true, seedUsed: result.seedUsed, prompt: result.prompt });
      res.end();
    } catch (e) {
      const msg = (e instanceof Error ? e.message : String(e)) || 'error';
      if (!res.writableEnded) {
        sseWrite(res, { error: msg });
        res.end();
      }
    }
    return;
  }

  res.writeHead(404);
  res.end();
});

const port = parseInt(process.env.PORT ?? '3847', 10);
server.listen(port, () => {
  console.log(`Chat UI: http://localhost:${port}/`);
  console.log(`Model: ${process.env.MODEL_PATH || defaultModelPath()}`);
});
