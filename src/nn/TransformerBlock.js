/**
 * =============================================================================
 * 本文件里的英文词是什么意思
 * =============================================================================
 * TransformerBlock  Transformer 的一层：一块「注意力 + 前馈」的组合。
 * Block            块：深度学习里常把一层重复单元叫 block。
 * attn             attention 的缩写：注意力子层。
 * ff               feed-forward 的缩写：前馈子层。
 * x/y/z/w          常用作中间变量名：表示一层层变换后的张量（字母无特殊含义，只是接力）。
 *
 * =============================================================================
 */

import { add } from '../tensor/ops.js';
import { CausalAttention } from './CausalAttention.js';
import { FeedForward } from './FeedForward.js';
import { LayerNorm } from './LayerNorm.js';

/**
 * Transformer 的一层（一个 Block）
 *
 * 【LayerNorm】每层两个：注意力前、FFN 前（Pre-LN）。用来稳定每层的数值范围，
 * 训练通常更稳、对学习率不那么敏感（仍要配合足够数据与合理步数）。
 *
 * 【子层顺序：Pre-LN + 先注意力再 FFN】
 * h = x + attn(LN(x))；h = h + FFN(LN(h))。
 *
 * 【残差】直观理解：子层学的是「在原来向量上打补丁」，而不是从零再造。
 */
export class TransformerBlock {
  /**
   * @param {{ loraRank?: number, loraAlpha?: number }} [loraOpts]
   */
  constructor(dModel, nHeads, dFf, rng, loraOpts = {}) {
    this.ln1 = new LayerNorm(dModel, rng);
    this.ln2 = new LayerNorm(dModel, rng);
    this.attn = new CausalAttention(dModel, nHeads, rng, loraOpts);
    this.ff = new FeedForward(dModel, dFf, rng, loraOpts);
  }

  forward(x) {
    const y = this.attn.forward(this.ln1.forward(x));
    const z = add(x, y);
    const w = this.ff.forward(this.ln2.forward(z));
    return add(z, w);
  }

  parameters() {
    return [...this.ln1.parameters(), ...this.ln2.parameters(), ...this.attn.parameters(), ...this.ff.parameters()];
  }

  loraParameters() {
    return [...this.attn.loraParameters(), ...this.ff.loraParameters()];
  }

  freezeBaseForLoRA() {
    this.ln1.freezeParameters();
    this.ln2.freezeParameters();
    this.attn.freezeBaseForLoRA();
    this.ff.freezeBaseForLoRA();
  }
}
