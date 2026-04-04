/**
 * =============================================================================
 * 续写生成（人话）
 * =============================================================================
 * 从提示出发逐步预测下一字。除温度 / top-k 外，增加：
 * - 重复惩罚：最近出现过的字在 logits 上被压低，减少「啊啊啊」式循环。
 * - Top-p（nucleus）：只在累积概率达 p 的最小集合里抽样，句子往往更顺。
 * =============================================================================
 */

import { mulberry32 } from './tensor/Tensor.js';

const softmax1d = (logits) => {
  const n = logits.length;
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
  const out = new Float32Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(logits[i] - mx);
    s += out[i];
  }
  for (let i = 0; i < n; i++) out[i] /= s;
  return out;
};

/** 只保留分数最高的 k 个。 */
const applyTopK = (logits, k) => {
  if (k <= 0 || k >= logits.length) return;
  const V = logits.length;
  const order = Array.from({ length: V }, (_, i) => i);
  order.sort((a, b) => logits[b] - logits[a]);
  const keep = new Set(order.slice(0, k));
  for (let i = 0; i < V; i++) {
    if (!keep.has(i)) logits[i] = -1e30;
  }
};

/**
 * Nucleus sampling：按概率质量从高往低累加到 ≥ p，只保留这一截（HFW 常用）。
 * @param {Float32Array} logits
 * @param {number} p 例如 0.92；≥1 或 ≤0 表示不裁剪
 */
const applyTopP = (logits, p) => {
  if (p >= 1 || p <= 0) return;
  const V = logits.length;
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => logits[b] - logits[a]);
  const probs = softmax1d(logits);
  let cum = 0;
  const keep = new Set();
  for (let rank = 0; rank < V; rank++) {
    const i = idx[rank];
    cum += probs[i];
    keep.add(i);
    if (cum >= p) break;
  }
  for (let i = 0; i < V; i++) {
    if (!keep.has(i)) logits[i] = -1e30;
  }
};

/**
 * 最近窗口里出现过的 token：正值 logits 除以 penalty，负值乘以 penalty（与常见 HF 实现一致）。
 */
const applyRepetitionPenalty = (logits, tokenIds, penalty) => {
  if (penalty <= 1 || !tokenIds.length) return;
  const seen = new Set(tokenIds);
  for (const id of seen) {
    if (id < 0 || id >= logits.length) continue;
    if (logits[id] < 0) logits[id] *= penalty;
    else logits[id] /= penalty;
  }
};

const sampleMultinomial = (probs, rng) => {
  let u = rng();
  if (u <= 0) u = 1e-12;
  if (u >= 1) u = 1 - 1e-12;
  for (let i = 0; i < probs.length; i++) {
    u -= probs[i];
    if (u <= 0) return i;
  }
  return probs.length - 1;
};

/**
 * @param {Float32Array} logitsRow
 * @param {object} options
 * @param {number[]} [options.recentTokenIds] 参与重复惩罚的最近若干 token id
 * @param {number} [options.repetitionPenalty] 默认 1.12
 * @param {number} [options.topP] nucleus，默认 0.92；1 关闭
 */
export const sampleNextId = (logitsRow, options) => {
  const {
    temperature = 0.8,
    topK = 40,
    topP = 0.92,
    greedy = false,
    rng,
    recentTokenIds = [],
    repetitionPenalty = 1.12,
  } = options;
  const V = logitsRow.length;
  const work = new Float32Array(V);
  for (let j = 0; j < V; j++) work[j] = logitsRow[j];

  if (greedy || temperature <= 0) {
    // 与采样路径一致：贪心也要先压低「最近出现过的字」，否则会反复 argmax 到「。」「好」等高分符号。
    applyRepetitionPenalty(work, recentTokenIds, repetitionPenalty);
    let best = 0;
    let bestv = -Infinity;
    for (let j = 0; j < V; j++) {
      if (work[j] > bestv) {
        bestv = work[j];
        best = j;
      }
    }
    return best;
  }

  for (let j = 0; j < V; j++) work[j] /= temperature;
  applyRepetitionPenalty(work, recentTokenIds, repetitionPenalty);
  applyTopK(work, topK);
  applyTopP(work, topP);
  const probs = softmax1d(work);
  return sampleMultinomial(probs, rng);
};

/**
 * @param {import('./model/MiniGPT.js').MiniGPT} model
 * @param {{ encode: (s: string) => Uint32Array, itos: string[] }} tok
 * @param {string} prompt
 * @param {object} [opt]
 * @param {boolean} [opt.yieldEachToken] 为 true 且提供 onToken 时，每字后 await 一次事件循环，让 SSE/终端能真正逐块到达（否则整段在同一次 tick 内算完，看起来像同步）。
 */
export async function generateContinuation(model, tok, prompt, opt = {}) {
  const maxNewTokens = opt.maxNewTokens ?? 64;
  const temperature = opt.temperature ?? 0.75;
  const topK = opt.topK ?? 50;
  const topP = opt.topP !== undefined ? opt.topP : 0.92;
  const repetitionPenalty = opt.repetitionPenalty !== undefined ? opt.repetitionPenalty : 1.12;
  const repetitionWindow = opt.repetitionWindow ?? 48;
  const greedy = opt.greedy ?? false;
  const onToken = opt.onToken ?? null;
  const yieldEachToken = !!opt.yieldEachToken && !!onToken;
  const seed = opt.seed !== undefined ? opt.seed >>> 0 : (Date.now() ^ (Math.random() * 0x100000000)) >>> 0;
  const rng = mulberry32(seed);

  if (prompt.length === 0) {
    throw new Error('提示不能为空，至少给一个字符，模型才知道从哪接');
  }

  /** @type {number[]} */
  const ids = Array.from(tok.encode(prompt));
  const seqLen = model.seqLen;
  let generated = '';

  for (let step = 0; step < maxNewTokens; step++) {
    const window = ids.length > seqLen ? ids.slice(-seqLen) : ids;
    const input = new Uint32Array(window);
    const logits = model.forward(input);
    const T = logits.rows;
    const V = logits.cols;
    const last = T - 1;
    const rowOff = last * V;
    const row = new Float32Array(V);
    for (let j = 0; j < V; j++) row[j] = logits.data[rowOff + j];

    const recentForPenalty = ids.slice(Math.max(0, ids.length - repetitionWindow));
    const nextId = sampleNextId(row, {
      temperature,
      topK,
      topP,
      greedy,
      rng,
      recentTokenIds: recentForPenalty,
      repetitionPenalty,
    });
    ids.push(nextId);
    const ch = tok.itos[nextId];
    generated += ch;
    if (onToken) onToken(ch);
    if (yieldEachToken) {
      await new Promise((resolve) => setImmediate(resolve));
    }
  }

  return { prompt, generated, fullText: prompt + generated, seedUsed: seed, ids };
}
