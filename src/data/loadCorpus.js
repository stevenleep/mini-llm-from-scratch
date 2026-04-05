/**
 * 从文件读训练语料；找不到文件时用内置短文兜底。
 * CORPUS_PATH：相对当前工作目录的路径；不设则默认 data/corpus/playful_zh.txt
 *
 * 身份语料（可选）：
 * 若存在 data/corpus/identity_zh.txt，会合并身份片段（教「你是谁」→「我是李小烨」等）。
 * 默认把身份段放在主语料前面（IDENTITY_PREPEND=1）：随机训练窗口大多落在身份段，
 * 避免「身份只在文件末尾 → 全进验证集、训练集几乎采不到」的问题。设 IDENTITY_PREPEND=0 则改回「接在主语料后面」。
 * 不需要身份时设 SKIP_IDENTITY_CORPUS=1。
 * IDENTITY_REPEAT：身份片段重复几遍再合并（默认 6；设 1 可减轻重复）；例如 IDENTITY_REPEAT=10 更重身份。
 * 网络语料：若存在 data/corpus/downloaded_mixed_zh.txt（可先运行 npm run corpus:fetch），会接在「主文件 + 身份」之后合并。
 * SKIP_DOWNLOAD_CORPUS=1：不追加该文件。
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

/** 默认追加的身份片段（相对项目根 / 当前工作目录） */
const DEFAULT_IDENTITY_REL = 'data/corpus/identity_zh.txt';

const BUILTIN = `
在最小实现里，一条训练样本是：输入长度为 T 的 token 序列，
让模型在每个位置预测下一个 token。反向传播从平均交叉熵开始，
沿计算图回到词嵌入与所有线性层。这里没有 LayerNorm，仅用于演示核心数据流。
`.trim();

/**
 * @returns {string}
 */
export const loadCorpus = () => {
  const rel = process.env.CORPUS_PATH ?? 'data/corpus/playful_zh.txt';
  const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);
  if (!existsSync(abs)) {
    console.warn('[语料] 未找到文件，使用内置短文:', abs);
    return BUILTIN;
  }
  const text = readFileSync(abs, 'utf8').trim();
  if (!text) {
    console.warn('[语料] 文件为空，使用内置短文');
    return BUILTIN;
  }
  console.log('[语料] 主文件', abs, '· 字符数', [...text].length);
  let merged = text;
  const skipId = process.env.SKIP_IDENTITY_CORPUS === '1';
  const idRel = process.env.IDENTITY_CORPUS_PATH ?? DEFAULT_IDENTITY_REL;
  const idAbs = path.isAbsolute(idRel) ? idRel : path.join(process.cwd(), idRel);
  if (!skipId && existsSync(idAbs)) {
    const idText = readFileSync(idAbs, 'utf8').trim();
    if (idText) {
      const repeat = Math.min(100, Math.max(1, parseInt(process.env.IDENTITY_REPEAT ?? '6', 10) || 1));
      const idMerged = Array.from({ length: repeat }, () => idText).join('\n\n');
      const prepend = process.env.IDENTITY_PREPEND !== '0';
      merged = prepend ? `${idMerged}\n\n${text}` : `${text}\n\n${idMerged}`;
      console.log(
        prepend ? '[语料] 已前置身份片段' : '[语料] 已追加身份片段',
        idAbs,
        '· 单份字符数',
        [...idText].length,
        '· 重复',
        repeat,
        '次 · 身份段总字符',
        [...idMerged].length,
        prepend ? '· 顺序：身份→主语料' : '· 顺序：主语料→身份',
      );
    }
  } else if (!skipId) {
    console.log('[语料] 未找到身份片段（可选）', idAbs, '· 可跳过本提示：设 SKIP_IDENTITY_CORPUS=1');
  }

  const skipDl = process.env.SKIP_DOWNLOAD_CORPUS === '1';
  const dlRel = process.env.DOWNLOAD_CORPUS_PATH ?? 'data/corpus/downloaded_mixed_zh.txt';
  const dlAbs = path.isAbsolute(dlRel) ? dlRel : path.join(process.cwd(), dlRel);
  if (!skipDl && existsSync(dlAbs)) {
    const dlText = readFileSync(dlAbs, 'utf8').trim();
    if (dlText) {
      merged = `${merged}\n\n${dlText}`;
      console.log('[语料] 已追加网络下载混合段', dlAbs, '· 字符数', [...dlText].length);
    }
  } else if (!skipDl) {
    console.log('[语料] 无网络混合文件（可选）', dlAbs, '· 可先 npm run corpus:fetch');
  }

  const len = [...merged].length;
  console.log('[语料] 合并后总字符数', len);
  return merged;
};
