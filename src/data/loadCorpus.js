/**
 * 从文件读训练语料；找不到文件时用内置短文兜底。
 * CORPUS_PATH：相对当前工作目录的路径；不设则默认 data/corpus/playful_zh.txt
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

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
  const len = [...text].length;
  console.log('[语料] 已加载', abs, '· 字符数', len);
  return text;
};
