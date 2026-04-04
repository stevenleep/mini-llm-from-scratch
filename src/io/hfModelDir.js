/**
 * =============================================================================
 * 这个文件在干什么（人话）
 * =============================================================================
 * 默认训练脚本会把这一整包写到 out/export/hf-style/（与单文件 JSON 的目录分开）。
 *
 * 网上很多开源模型是一个文件夹，里面有：
 *   config.json          —— 模型多宽、多少层、词表多大这类「结构说明」
 *   tokenizer_config.json —— 分词器说明（我们是按字切，很简单）
 *   vocab.json           —— 第几号对应哪个字（一行一个意思，用 JSON 数组存）
 *   model.safetensors    —— 权重（SafeTensors 标准格式，带 JSON 头）
 *   model.bin            —— 同一份权重的「纯 float32 拼接」，无头；配 model.bin.manifest.json 才知道每段在哪
 *
 * 权重里每一张表的名字，故意起得和 GPT-2 那套有点像（什么 wte、wpe、h.0.attn…），
 * 方便你对照 Hugging Face 文档；但我们的网络结构和官方 GPT-2 并不一样，别指望能直接
 * 丢进 transformers 里一键加载，只是「文件摆法」和「起名习惯」往那边靠。
 * =============================================================================
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tokenizerFromItos } from '../data/charTokenizer.js';
import { MiniGPT, mulberry32 } from '../model/MiniGPT.js';
import { decodeRawFp32BinToLoaded, writeRawFp32BinFiles } from './rawFp32Bin.js';
import { decodeSafetensors, encodeSafetensors } from './safetensors.js';

/**
 * 把内存里的模型拆成「字符串名字 → 那张 Tensor」，方便按名字写进 safetensors。
 */
export const collectNamedStateDict = (model) => {
  /** @type {Record<string, import('../tensor/Tensor.js').Tensor>} */
  const d = {};
  d['transformer.wte.weight'] = model.tokEmb;
  d['transformer.wpe.weight'] = model.posEmb;
  d['lm_head.weight'] = model.lmHead.weight;
  d['lm_head.bias'] = model.lmHead.bias;
  for (let i = 0; i < model.blocks.length; i++) {
    const b = model.blocks[i];
    const p = `transformer.h.${i}`;
    d[`${p}.attn.c_attn.weight`] = b.attn.qkv.weight;
    d[`${p}.attn.c_attn.bias`] = b.attn.qkv.bias;
    d[`${p}.attn.c_proj.weight`] = b.attn.proj.weight;
    d[`${p}.attn.c_proj.bias`] = b.attn.proj.bias;
    d[`${p}.mlp.c_fc.weight`] = b.ff.fc1.weight;
    d[`${p}.mlp.c_fc.bias`] = b.ff.fc1.bias;
    d[`${p}.mlp.c_proj.weight`] = b.ff.fc2.weight;
    d[`${p}.mlp.c_proj.bias`] = b.ff.fc2.bias;
  }
  return d;
};

/** 把文件里读出来的一串数，按形状填进模型里已有的一张表（行、列必须对得上）。 */
const loadTensorInto = (t, data, shape) => {
  const [r, c] = shape;
  if (t.rows !== r || t.cols !== c) {
    throw new Error(`形状对不上: 模型里是 [${t.rows},${t.cols}]，文件里是 [${r},${c}]`);
  }
  if (data.length !== r * c) throw new Error('格子数和形状说的不一致');
  t.data.set(data);
};

/** 按「名字」把磁盘上的表一张张塞回模型；缺一张或类型不是 float32 就报错。 */
const applyNamedStateDict = (model, loaded) => {
  const expected = collectNamedStateDict(model);
  for (const name of Object.keys(expected)) {
    const entry = loaded[name];
    if (!entry) throw new Error(`文件里少了这张表: ${name}`);
    if (entry.dtype !== 'F32') throw new Error(`${name}: 只认 F32（32 位浮点），文件里是 ${entry.dtype}`);
    loadTensorInto(expected[name], entry.data, entry.shape);
  }
};

/**
 * 把我们训练用的 cfg 转成 config.json 里那种字段名（网上常见的是 vocab_size、n_layer 这种下划线写法）。
 */
export const hfConfigFromTraining = (cfg) => ({
  model_type: 'mini_gpt',
  architectures: ['MiniGPT'],
  vocab_size: cfg.vocabSize,
  n_positions: cfg.seqLen,
  n_embd: cfg.dModel,
  n_layer: cfg.nLayers,
  n_head: cfg.nHeads,
  n_inner: cfg.dFf,
  torch_dtype: 'float32',
});

/** 把 config.json 读出来的对象再变回我们代码里用的 cfg（驼峰那一套）。 */
export const trainingConfigFromHf = (hfc) => ({
  vocabSize: hfc.vocab_size,
  seqLen: hfc.n_positions,
  dModel: hfc.n_embd,
  nLayers: hfc.n_layer,
  nHeads: hfc.n_head,
  dFf: hfc.n_inner,
});

/**
 * 往一个目录里写：config、分词器说明、词表、model.safetensors，以及 model.bin + model.bin.manifest.json（纯二进制那份）。
 */
export const writeHfModelDir = (model, cfg, tok, dir) => {
  mkdirSync(dir, { recursive: true });
  const hfc = hfConfigFromTraining(cfg);
  writeFileSync(join(dir, 'config.json'), `${JSON.stringify(hfc, null, 2)}\n`, 'utf8');
  writeFileSync(
    join(dir, 'tokenizer_config.json'),
    `${JSON.stringify(
      {
        tokenizer_class: 'CharacterTokenizer',
        char_level: true,
        model_max_length: cfg.seqLen,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  writeFileSync(join(dir, 'vocab.json'), `${JSON.stringify(tok.itos, null, 2)}\n`, 'utf8');

  const state = collectNamedStateDict(model);
  /** @type {Record<string, { data: Float32Array, shape: number[] }>} */
  const named = {};
  for (const [name, t] of Object.entries(state)) {
    named[name] = { data: t.data, shape: [t.rows, t.cols] };
  }
  const bin = encodeSafetensors(named);
  writeFileSync(join(dir, 'model.safetensors'), bin);
  writeRawFp32BinFiles(dir, state);
};

/**
 * 从一个目录读回模型：优先读 model.safetensors；若没有，再试 model.bin + model.bin.manifest.json。
 */
export const readHfModelDir = (dir) => {
  const hfc = JSON.parse(readFileSync(join(dir, 'config.json'), 'utf8'));
  if (hfc.model_type !== 'mini_gpt') {
    throw new Error(`只认 model_type 为 mini_gpt 的配置，当前是: ${hfc.model_type}`);
  }
  const itos = JSON.parse(readFileSync(join(dir, 'vocab.json'), 'utf8'));
  if (!Array.isArray(itos)) throw new Error('vocab.json 应该是一个字符串数组（第几号是什么字）');
  const cfg = trainingConfigFromHf(hfc);
  if (itos.length !== cfg.vocabSize) {
    throw new Error(`vocab.json 里有 ${itos.length} 个字，config 里说词表大小是 ${cfg.vocabSize}，对不上`);
  }
  const tok = tokenizerFromItos(itos);
  const model = new MiniGPT(cfg, mulberry32(0));
  const safePath = join(dir, 'model.safetensors');
  const binPath = join(dir, 'model.bin');
  const manPath = join(dir, 'model.bin.manifest.json');
  let loaded;
  if (existsSync(safePath)) {
    loaded = decodeSafetensors(readFileSync(safePath));
  } else if (existsSync(binPath) && existsSync(manPath)) {
    const manifest = JSON.parse(readFileSync(manPath, 'utf8'));
    loaded = decodeRawFp32BinToLoaded(readFileSync(binPath), manifest);
  } else {
    throw new Error('目录里既没有 model.safetensors，也没有 model.bin + model.bin.manifest.json');
  }
  applyNamedStateDict(model, loaded);
  return { model, cfg, tok, hfConfig: hfc };
};
