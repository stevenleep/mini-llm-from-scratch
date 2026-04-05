/**
 * LayerNorm（层归一化）
 *
 * 【作用】对每个位置上的向量，在「特征维」上减均值、除方差，再学一组 γ、β 缩放平移，
 * 让每层的数值范围更稳，训练更不容易炸、对学习率不那么敏感（与 TransformerBlock 里注释一致）。
 *
 * 【Pre-LN】本项目的 Block 里采用「先 LayerNorm 再子层再残差」的顺序（常见解码器写法）。
 */

import { Tensor } from '../tensor/Tensor.js';
import { layerNorm } from '../tensor/ops.js';

export class LayerNorm {
  /**
   * @param {number} dModel 特征维宽度（与 dModel 一致）
   * @param {() => number} rng 未使用，保留与 Linear 等构造函数签名风格一致
   * @param {number} [eps]
   */
  constructor(dModel, rng, eps = 1e-5) {
    void rng;
    this.eps = eps;
    this.gamma = Tensor.ones([1, dModel], true);
    this.beta = Tensor.zeros([1, dModel], true);
  }

  /**
   * @param {import('../tensor/Tensor.js').Tensor} x [T, dModel]
   */
  forward(x) {
    return layerNorm(x, this.gamma, this.beta, this.eps);
  }

  parameters() {
    return [this.gamma, this.beta];
  }

  /** 只训 LoRA 时一并冻结 Norm 参数。 */
  freezeParameters() {
    this.gamma.requiresGrad = false;
    this.beta.requiresGrad = false;
  }
}
