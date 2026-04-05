/**
 * =============================================================================
 * 本文件里的英文词是什么意思
 * =============================================================================
 * Linear           线性层：这里指「全连接层」，公式大致是 输出 = 输入 × 权重表 + 偏置。
 * weight           权重：要学的变换表 W。
 * bias             偏置：每个输出维度上加的一串常数 b（先全零，训练会学出来）。
 * inF / outF       in Features / out Features：输入向量长度、输出向量长度；F 常表示 feature。
 * forward          前向：从输入 x 算出输出（推理或训练时「从前往后算」）。
 * parameters       参数：所有要学的权重列表，训练时一起更新。
 * rng              random number generator：随机数函数，只用来初始化 weight。
 *
 * =============================================================================
 */

import { Tensor } from '../tensor/Tensor.js';
import { add, addBias, matmul, mulScalar } from '../tensor/ops.js';

/**
 * Linear（全连接层 / 仿射层）
 *
 * 【数学在干什么】
 * 对每个位置上的向量 x，做 y = x @ W + b。W 像一张「从输入维度映射到输出维度」的变换表，
 * b 是每个输出维度上的平移，让直线不必过原点。
 *
 * 【为什么要这一层】
 * 神经网络若只有嵌入，没有线性变换，就无法学习「哪些维度组合起来有意义」；
 * 多层线性+非线性叠加，才能逼近复杂函数。
 *
 * 【权重与偏置怎么初始化】
 * W 随机（否则对称死锁）；b 全零很常见——先不引入偏置噪声，训练会自己学出需要的偏移。
 *
 * 【形状】
 * x 是 [任意行数, inF]，输出 [相同行数, outF]。每一行独立做同样的线性变换（同一套 W、b）。
 */
export class Linear {
  /**
   * @param {number} inF  输入向量长度
   * @param {number} outF 输出向量长度
   * @param {() => number} rng 随机数，只用于初始化 W
   * @param {{ loraRank?: number, loraAlpha?: number }} [loraOpts] LoRA：低秩适配 ΔW≈A@B，只训 A/B 时可冻结 weight/bias（见 train.js 的 LORA_ONLY）。
   */
  constructor(inF, outF, rng, loraOpts = {}) {
    const { loraRank = 0, loraAlpha = 16 } = loraOpts;
    this.loraRank = loraRank;
    this.loraAlpha = loraAlpha;
    this.weight = Tensor.randn([inF, outF], rng, true);
    this.bias = Tensor.zeros([1, outF], true);
    if (loraRank > 0) {
      this.loraA = Tensor.randn([inF, loraRank], rng, true);
      this.loraB = Tensor.zeros([loraRank, outF], true);
    } else {
      this.loraA = null;
      this.loraB = null;
    }
  }

  forward(x) {
    const base = addBias(matmul(x, this.weight), this.bias);
    if (!this.loraA || !this.loraB) return base;
    const mid = matmul(x, this.loraA);
    const delta = matmul(mid, this.loraB);
    const scale = this.loraAlpha / this.loraRank;
    return add(base, mulScalar(delta, scale));
  }

  /** 冻结全连接「主干」，只训 LoRA 时用。 */
  freezeBaseForLoRA() {
    this.weight.requiresGrad = false;
    this.bias.requiresGrad = false;
  }

  baseParameters() {
    return [this.weight, this.bias];
  }

  loraParameters() {
    return this.loraA && this.loraB ? [this.loraA, this.loraB] : [];
  }

  /** 训练时收集所有要更新的参数，方便统一做一步梯度下降。 */
  parameters() {
    return [...this.baseParameters(), ...this.loraParameters()];
  }
}
