/**
 * =============================================================================
 * 这个文件在干什么（人话）
 * =============================================================================
 * 把「整个模型」塞进**一个** JSON 文件里：配置数字、字表、每张权重表变成 Base64 文本。
 * 好处：拷一个文件就走。坏处：大模型会巨肥，也不如 safetensors 正规。
 * 想和网上常见发布方式一致，请用 hfModelDir.js 那种「一个文件夹 + model.safetensors」。
 * =============================================================================
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { tokenizerFromItos } from '../data/charTokenizer.js';
import { MiniGPT, mulberry32 } from '../model/MiniGPT.js';
import { prod } from '../tensor/size.js';

/** 1 = 仅主干权重；2 = config 含 loraRank / loraAlpha，权重序列含 LoRA 矩阵（若秩>0）。 */
const FORMAT_VERSION = 2;

/** 一张 Tensor → JSON 里能存的一小块：形状 + Base64（里面是 float32 原始字节）。 */
const tensorToWeightEntry = (t) => ({
  shape: [t.rows, t.cols],
  data: Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength).toString('base64'),
});

/** 把 JSON 里那一小块还原，写进已经分配好的那张 Tensor 里。 */
const loadTensorFromEntry = (t, entry) => {
  const [r, c] = entry.shape;
  if (t.rows !== r || t.cols !== c) {
    throw new Error(`形状不一致: 模型里是 [${t.rows},${t.cols}]，文件里是 [${r},${c}]`);
  }
  const n = prod(entry.shape);
  const buf = Buffer.from(entry.data, 'base64');
  if (buf.byteLength !== n * 4) {
    throw new Error(`权重字节数不对: 应该是 ${n * 4}，实际是 ${buf.byteLength}`);
  }
  t.data.set(new Float32Array(buf.buffer, buf.byteOffset, n));
};

/**
 * 在内存里拼出一个「能 JSON.stringify 的大对象」：版本号、配置、字表、所有权重块。
 */
export const serializeMiniGPT = (model, cfg, tok) => {
  const params = model.parameters();
  return {
    formatVersion: FORMAT_VERSION,
    kind: 'mini-llm-from-scratch',
    config: {
      vocabSize: cfg.vocabSize,
      seqLen: cfg.seqLen,
      dModel: cfg.dModel,
      nHeads: cfg.nHeads,
      nLayers: cfg.nLayers,
      dFf: cfg.dFf,
      loraRank: cfg.loraRank ?? 0,
      loraAlpha: cfg.loraAlpha ?? 16,
    },
    itos: tok.itos,
    weights: params.map((t) => tensorToWeightEntry(t)),
  };
};

/**
 * 把上面那种对象读回来：先对版本号和词表长度做检查，再 new 模型，按顺序把每张表填满。
 */
export const deserializeMiniGPT = (payload) => {
  const fv = payload.formatVersion;
  if (fv !== 1 && fv !== 2) {
    throw new Error(`不认识 formatVersion: ${payload.formatVersion}`);
  }
  const { config, itos, weights } = payload;
  if (!itos || !weights || !config) {
    throw new Error('缺了 itos、weights 或 config 里的某一块');
  }
  if (itos.length !== config.vocabSize) {
    throw new Error(`字表长度 ${itos.length} 和 config.vocabSize ${config.vocabSize} 不一致`);
  }
  const cfg = {
    vocabSize: config.vocabSize,
    seqLen: config.seqLen,
    dModel: config.dModel,
    nHeads: config.nHeads,
    nLayers: config.nLayers,
    dFf: config.dFf,
    loraRank: fv >= 2 ? config.loraRank ?? 0 : 0,
    loraAlpha: fv >= 2 ? config.loraAlpha ?? 16 : 16,
  };
  const tok = tokenizerFromItos(itos);
  const model = new MiniGPT(cfg, mulberry32(0));
  const params = model.parameters();
  if (params.length !== weights.length) {
    throw new Error(`文件里权重块数是 ${weights.length}，当前代码造出来的模型要 ${params.length} 块`);
  }
  for (let i = 0; i < params.length; i++) {
    loadTensorFromEntry(params[i], weights[i]);
  }
  return { model, cfg, tok };
};

/** 序列化并写入一个路径；没有文件夹会先建（只建父目录）。 */
export const writeMiniGPTFile = (model, cfg, tok, filePath) => {
  const payload = serializeMiniGPT(model, cfg, tok);
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, JSON.stringify(payload), 'utf8');
};

/** 读整个 JSON 文件再反序列化。 */
export const readMiniGPTFile = (filePath) => {
  const text = readFileSync(filePath, 'utf8');
  /** @type {ReturnType<typeof serializeMiniGPT>} */
  const payload = JSON.parse(text);
  return deserializeMiniGPT(payload);
};
