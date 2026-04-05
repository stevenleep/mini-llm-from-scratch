/**
 * =============================================================================
 * 本文件里的英文缩写是什么意思
 * =============================================================================
 * FeedForward      前馈：每个位置单独做两层线性，不直接和别的位置交互（和 Attention 分工）。
 * FFN              Feed-Forward Network 的缩写，同上。
 * fc1 / fc2        fully connected 全连接第 1 层 / 第 2 层（fc 常表示全连接）。
 * dModel           模型宽度：输入输出向量长度。
 * dFf              中间层宽度：先放大到多宽（Ff 表示 feed-forward 隐层）。
 * gelu             一种弯曲激活函数的名字（GELU），不用记英文，理解成「非线性弯一下」即可。
 *
 * =============================================================================
 */

import { gelu } from '../tensor/ops.js';
import { Linear } from './Linear.js';

/**
 * 前馈网络 FFN（每个位置「单独」处理，不看别的位置）
 *
 * 【和注意力的分工】
 * 注意力负责「谁和谁相关、信息怎么流动」；FFN 负责「每个位置拿到混合后的向量，
 * 在自己内部做更复杂的非线性变换」——像每个字单独进一个小 MLP 加工。
 *
 * 【为什么是 dModel → dFf → dModel】
 * 中间先「变宽」再变回原宽，给网络容量：宽层能表示更复杂的局部函数；若一直不变宽，
 * 表达能力会受限（仍取决于具体任务）。
 *
 * 【为什么必须有 GELU】
 * 若只有两层线性堆叠，数学上仍等价于一层线性（矩阵相乘可合并），没有弯曲就学不深。
 * 非线性把「不同维度组合」拆开，才能拟合复杂模式。
 *
 * 【dFf 通常 ≥ dModel】
 * 常见做法是把中间层加宽（如 4 倍）；这里用较小数字只是让 CPU 跑得动。
 */
export class FeedForward {
  /**
   * @param {{ loraRank?: number, loraAlpha?: number }} [loraOpts]
   */
  constructor(dModel, dFf, rng, loraOpts = {}) {
    this.fc1 = new Linear(dModel, dFf, rng, loraOpts);
    this.fc2 = new Linear(dFf, dModel, rng, loraOpts);
  }

  forward(x) {
    return this.fc2.forward(gelu(this.fc1.forward(x)));
  }

  parameters() {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }

  loraParameters() {
    return [...this.fc1.loraParameters(), ...this.fc2.loraParameters()];
  }

  freezeBaseForLoRA() {
    this.fc1.freezeBaseForLoRA();
    this.fc2.freezeBaseForLoRA();
  }
}
