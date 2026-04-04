/**
 * =============================================================================
 * 本文件里的英文缩写 / 变量名是什么意思
 * =============================================================================
 * Causal           因果的：只能看「当前及过去」，不许看「未来」。
 * Attention        注意力：按相关程度混合各位置的信息。
 * Multi-Head       多头：同一套机制复制多份并行，最后再拼起来。
 * Self-            自注意力：Q/K/V 都来自同一段输入 x，不是来自另一个序列。
 * dModel           每个位置向量的宽度（和 config 里一致）。
 * nHeads           头数；headDim = dModel/nHeads 是每个头的宽度。
 * headDim          每个头内部的向量长度。
 * scale            缩放因子：1/√headDim，防止点积太大导致后面「分变比例」时太尖。
 * qkv              把 Q、K、V 三块拼成一次线性输出（Query/Key/Value 的缩写）。
 * proj             projection 投影：多头拼完后再线性混合一次。
 * _mask            下划线开头常表示「内部用」；mask=掩码：挡住不允许看的位置。
 * T                常表示时间步长度，这里即序列有几个字。
 * d                当前张量的列数，这里等于 dModel。
 * hd               head dimension 的缩写。
 * h                常用作 head 的循环变量或头数。
 * qh/kh/vh         第 h 个头的 Q/K/V（h 是 head 的首字母）。
 * scores           相似度打分表（未缩放/未掩码前也可叫 logits 一类的东西）。
 * scaled           缩放后的 scores。
 * masked           加上掩码后的分数。
 * attn             attention 注意力权重（每行和为 1 的比例）。
 * merged           多头拼成一条后的张量。
 *
 * =============================================================================
 */

import {
  add,
  causalAttentionMask,
  concatHeads,
  matmul,
  mulScalar,
  narrowCols,
  softmaxRows,
  transpose,
} from '../tensor/ops.js';
import { Linear } from './Linear.js';

/**
 * 因果多头自注意力（Causal Multi-Head Self-Attention）
 *
 * 【一句话目的】
 * 让每个位置的向量，根据「和自己及前面所有位置」的关系，重新混合信息；
 * 不能看后面的位置，否则训练时等于「偷看未来」，和实际生成一句话时的情况不一致。
 *
 * 【Q、K、V 直觉】
 * - Q（Query）：我现在这个位置「想查什么」。
 * - K（Key）：每个位置「提供什么标签让别人来匹配」。
 * - V（Value）：真正被混进来的内容。
 * 分数 = Q 与 K 的相似度；分数高 → 在 softmax 后权重大 → 从 V 里多取那份信息。
 *
 * 【为什么要除以 sqrt(headDim)】
 * 点积的方差随维度变大而变大，softmax 会变得极尖（几乎 one-hot），梯度不稳定。
 * 缩放让分数量级稳定一些。
 *
 * 【为什么要掩码】
 * 对位置 j > i 的分数加上极大负数，softmax 后权重≈0，等价于「禁止看未来」。
 *
 * 【为什么要多头】
 * 一组 QKV 只能学一种「关系」；多头像多组并行专家，各自关注不同模式，再拼起来。
 *
 * 【为什么要最后 proj 再线性一次】
 * 多头拼接后维度虽对，但各头之间仍需要可学习的混合，否则只是简单拼接。
 *
 * 【dModel 必须整除 nHeads】
 * 每个头维度 headDim = dModel/nHeads，必须分得匀；否则切列会乱。
 */
export class CausalAttention {
  constructor(dModel, nHeads, rng) {
    if (dModel % nHeads !== 0) throw new Error('dModel 必须整除 nHeads');
    this.nHeads = nHeads;
    this.headDim = dModel / nHeads;
    this.scale = 1 / Math.sqrt(this.headDim);
    /** 一次线性从 x 得到 [Q,K,V] 拼在最后一维，省三次单独矩阵乘（实现紧凑） */
    this.qkv = new Linear(dModel, 3 * dModel, rng);
    this.proj = new Linear(dModel, dModel, rng);
    /**
     * 掩码只依赖序列长度 T；同长度可复用，少分配内存。
     * 若 T 变化（例如不同长度样本），要重建掩码。
     */
    /** @type {import('../tensor/Tensor.js').Tensor | null} */
    this._mask = null;
  }

  /**
   * @param {import('../tensor/Tensor.js').Tensor} x 形状 [T, dModel]，T 为当前序列长度
   * @returns {import('../tensor/Tensor.js').Tensor} 同形状，已融合上下文
   */
  forward(x) {
    const T = x.rows;
    const d = x.cols;
    const hd = this.headDim;
    const h = this.nHeads;

    if (!this._mask || this._mask.shape[0] !== T) {
      this._mask = causalAttentionMask(T);
    }

    const qkv = this.qkv.forward(x);
    const heads = [];

    for (let head = 0; head < h; head++) {
      const qh = narrowCols(qkv, head * hd, hd);
      const kh = narrowCols(qkv, d + head * hd, hd);
      const vh = narrowCols(qkv, 2 * d + head * hd, hd);

      const scores = matmul(qh, transpose(kh));
      const scaled = mulScalar(scores, this.scale);
      const masked = add(scaled, this._mask);
      const attn = softmaxRows(masked);
      heads.push(matmul(attn, vh));
    }

    const merged = concatHeads(heads);
    return this.proj.forward(merged);
  }

  parameters() {
    return [...this.qkv.parameters(), ...this.proj.parameters()];
  }
}
