/**
 * 从公开仓库拉取中文语料（UTF-8），抽取字符串后合并到 data/corpus/downloaded_mixed_zh.txt。
 * 请自行遵守各来源许可；可编辑下方 URL 列表。
 *
 * 用法: node scripts/fetch-corpora.mjs
 * 环境: OUT=path/to/out.txt 覆盖输出路径
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

const DEFAULT_SOURCES = [
  {
    label: '水墨唐诗（chinese-poetry）',
    url: 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E6%B0%B4%E5%A2%A8%E5%94%90%E8%AF%97/shuimotangshi.json',
  },
  {
    label: '诗经（chinese-poetry）',
    url: 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E8%AF%97%E7%BB%8F/shijing.json',
  },
  {
    label: '中文 NLP 语料索引说明（brightmart README）',
    url: 'https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/README.md',
  },
];

const ZH_RE = /[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/;

const collectStrings = (obj, out) => {
  if (typeof obj === 'string') {
    const t = obj.trim();
    if (t.length >= 2 && ZH_RE.test(t)) out.push(t);
  } else if (Array.isArray(obj)) {
    for (const x of obj) collectStrings(x, out);
  } else if (obj && typeof obj === 'object') {
    for (const v of Object.values(obj)) collectStrings(v, out);
  }
};

const textFromPayload = (body, url) => {
  const lower = url.toLowerCase();
  if (lower.includes('.json') || body.trimStart().startsWith('{') || body.trimStart().startsWith('[')) {
    try {
      const parsed = JSON.parse(body);
      const parts = [];
      collectStrings(parsed, parts);
      return parts.join('\n');
    } catch {
      return body;
    }
  }
  return body;
};

const main = async () => {
  const outRel = process.env.OUT ?? 'data/corpus/downloaded_mixed_zh.txt';
  const outAbs = path.isAbsolute(outRel) ? outRel : path.join(ROOT, outRel);
  mkdirSync(path.dirname(outAbs), { recursive: true });

  const rawDir = path.join(ROOT, 'data/corpus/downloaded/raw');
  mkdirSync(rawDir, { recursive: true });

  let merged = `【网络混合语料 · 自动生成】\n生成时间（UTC）：${new Date().toISOString()}\n\n`;

  for (const s of DEFAULT_SOURCES) {
    try {
      const res = await fetch(s.url, {
        headers: { 'User-Agent': 'mini-llm-from-scratch-corpus-fetch/1.0' },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = Buffer.from(await res.arrayBuffer());
      const slug = String(s.label).replace(/\W+/g, '_').slice(0, 48);
      writeFileSync(path.join(rawDir, `${slug}.bin`), buf);
      const body = buf.toString('utf8');
      const text = textFromPayload(body, s.url);
      console.log('[fetch]', s.label, '→ 抽出字符约', [...text].length);
      merged += `\n\n【${s.label}】\n\n${text}`;
    } catch (e) {
      console.warn('[fetch] 跳过（失败）', s.label, (e && e.message) || e);
    }
  }

  writeFileSync(outAbs, merged.trim() + '\n', 'utf8');
  console.log('[fetch] 已写入', outAbs, '· 总字符约', [...merged].length);
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
