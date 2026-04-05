/**
 * 从已有 checkpoint 重新写出全部导出位置（不重新训练）。
 * 默认优先读 out/export/json-single/model.mgpt.json，否则读 out/export/hf-style。
 *
 * SOURCE_JSON=path/to/model.mgpt.json SOURCE_HF=path/to/hf-dir 可指定来源。
 */

import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { readHfModelDir, writeHfModelDir } from '../src/io/hfModelDir.js';
import { readMiniGPTFile, writeMiniGPTFile } from '../src/io/miniGptIO.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

const resolve = (p) => path.join(ROOT, p);

const load = () => {
  const envJson = process.env.SOURCE_JSON && path.isAbsolute(process.env.SOURCE_JSON)
    ? process.env.SOURCE_JSON
    : process.env.SOURCE_JSON
      ? resolve(process.env.SOURCE_JSON)
      : null;
  const envHf = process.env.SOURCE_HF && path.isAbsolute(process.env.SOURCE_HF)
    ? process.env.SOURCE_HF
    : process.env.SOURCE_HF
      ? resolve(process.env.SOURCE_HF)
      : null;

  if (envJson && existsSync(envJson)) {
    console.log('[export-all] 来源（单文件）', envJson);
    return readMiniGPTFile(envJson);
  }
  if (envHf && existsSync(envHf)) {
    console.log('[export-all] 来源（HF 目录）', envHf);
    return readHfModelDir(envHf);
  }

  const defJson = resolve('out/export/json-single/model.mgpt.json');
  const defHf = resolve('out/export/hf-style');
  if (existsSync(defJson)) {
    console.log('[export-all] 来源（单文件）', defJson);
    return readMiniGPTFile(defJson);
  }
  if (existsSync(defHf)) {
    console.log('[export-all] 来源（HF 目录）', defHf);
    return readHfModelDir(defHf);
  }

  throw new Error(
    '找不到可加载的模型：请先有 out/export/json-single/model.mgpt.json 或 out/export/hf-style，或设置 SOURCE_JSON / SOURCE_HF',
  );
};

const { model, cfg, tok } = load();

const targets = [
  { kind: 'hf', path: resolve('out/export/hf-style'), label: 'HF 默认导出' },
  { kind: 'hf', path: resolve('out/hf-model'), label: 'HF 兼容路径 out/hf-model' },
  { kind: 'json', path: resolve('out/export/json-single/model.mgpt.json'), label: '单文件 JSON（export 下）' },
  { kind: 'json', path: resolve('out/model.mgpt.json'), label: '单文件 JSON（out 根目录）' },
];

for (const t of targets) {
  if (t.kind === 'hf') {
    mkdirSync(t.path, { recursive: true });
    writeHfModelDir(model, cfg, tok, t.path);
  } else {
    mkdirSync(path.dirname(t.path), { recursive: true });
    writeMiniGPTFile(model, cfg, tok, t.path);
  }
  console.log('[export-all] 已写入', t.label, '→', t.path);
}

console.log('[export-all] 全部完成。');
