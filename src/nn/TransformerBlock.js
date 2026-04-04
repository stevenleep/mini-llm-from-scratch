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

/**
 * Transformer 的一层（一个 Block）
 *
 * 【标准论文里还有 LayerNorm】
 * 这里为了代码最短，先省略 Norm；真实模型几乎都有，用来稳定每层的数值范围。
 * 没有 Norm 时，训练有时会更难、更依赖学习率，但足够演示「数据怎么流」。
 *
 * 【子层顺序：先注意力再 FFN】
 * 这是最常见的一种堆叠顺序（Pre-LN / Post-LN 会变，但都是「注意力 + FFN」两块）。
 *
 * 【残差 x + f(x) 解决什么问题】
 * 若网络很深，信号直接乘很多层会衰减或爆炸；加一条「跳过连接」让原始信息也能传到后面，
 * 梯度更好回传，训练更稳。直观理解：f(x) 学的是「在原来向量上打补丁」，而不是从零再造。
 *
 * 【这里用两次 add】
 * z = x + attn(x) 后，z 已经包含上下文；再在 z 上做 FFN，最后再加回 z 上。
 */
export class TransformerBlock {
  constructor(dModel, nHeads, dFf, rng) {
    this.attn = new CausalAttention(dModel, nHeads, rng);
    this.ff = new FeedForward(dModel, dFf, rng);
  }

  forward(x) {
    const y = this.attn.forward(x);
    const z = add(x, y);
    const w = this.ff.forward(z);
    return add(z, w);
  }

  parameters() {
    return [...this.attn.parameters(), ...this.ff.parameters()];
  }
}
