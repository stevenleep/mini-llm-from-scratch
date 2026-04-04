/**
 * =============================================================================
 * 这个脚本干什么（人话）
 * =============================================================================
 * 从磁盘加载已经训练好并导出的模型，给你一段话当前缀，猜「下一个字最可能是谁」。
 * 猜法很笨：只看最后一个位置，在所有字里选**分数最高的那个**（不搞随机抽样）。
 *
 * 第一个参数可以是：
 *   - HF 风格文件夹（默认训练导出在 out/export/hf-style/）：config、vocab、model.safetensors 或 model.bin 等
 *   - 单文件：默认在 out/export/json-single/model.mgpt.json（旧式一个大 JSON）
 * 第二个参数是提示前缀，不写就默认「在」。
 * =============================================================================
 */

import { statSync } from 'node:fs';
import { readHfModelDir } from './io/hfModelDir.js';
import { readMiniGPTFile } from './io/miniGptIO.js';

const path = process.argv[2];
const prompt = process.argv[3] ?? '在';

if (!path) {
  console.error('用法: node src/infer.js <文件夹 或 单个.json> [提示文字]');
  process.exit(1);
}

const st = statSync(path);
const { model, tok } = st.isDirectory() ? readHfModelDir(path) : readMiniGPTFile(path);

// 把提示变成编号 → 跑一遍前向 → 得到每个位置、每个字的分数表
const ids = tok.encode(prompt);
const logits = model.forward(ids);
const T = logits.rows;
const V = logits.cols;
const last = T - 1;
let best = 0;
let bestv = -Infinity;
const rowOff = last * V;
// 只看最后一个字那一行：哪个字的分数最大，就猜哪个
for (let j = 0; j < V; j++) {
  const v = logits.data[rowOff + j];
  if (v > bestv) {
    bestv = v;
    best = j;
  }
}
console.log('提示:', prompt);
console.log('下一字（分数最高的那个）:', tok.itos[best], `（编号 ${best}）`);
