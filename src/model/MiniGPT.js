/**
 * =============================================================================
 * 本文件里的英文词 / 缩写是什么意思
 * =============================================================================
 * MiniGPT          Mini=小型的，GPT=一类「只解码器」的语言模型架构的通称；这里指教学用小模型。
 * cfg              config 配置对象：从 train 传进来的旋钮集合。
 * vocabSize        词表大小：有多少个不同编号（字）。
 * seqLen           支持的最大序列长度（位置表也只准备这么多行）。
 * dModel           模型宽度：每个位置向量长度。
 * nHeads / nLayers / dFf   与 config 相同：头数、层数、前馈隐层宽度。
 * tokEmb           token embedding：词嵌入表，编号 → 向量。
 * posEmb           position embedding：位置嵌入表，第几个位置 → 要加上的向量。
 * blocks           多层 TransformerBlock 组成的数组。
 * lmHead           language model head：语言模型输出头，把隐藏向量变成「对每个字的打分」。
 * tokenIds         token 的复数 id：一串编号；token 在此项目里基本等于「字」。
 * T                序列长度（几个 token）。
 * h                hidden 的常用简写：这里表示「当前层算完后的隐藏表示」。
 * pos              position：取出来的位置编码片段。
 * b                循环变量：某一层 block。
 * p                parameters 的简写：收集所有要学的张量。
 * mulberry32       一种随机数算法的人名代号，不用记；知道是「可复现随机数」即可。
 *
 * =============================================================================
 */

import { Tensor, mulberry32 } from '../tensor/Tensor.js';
import { add, embed, narrowRows } from '../tensor/ops.js';
import { Linear } from '../nn/Linear.js';
import { TransformerBlock } from '../nn/TransformerBlock.js';

/**
 * MiniGPT：整段「猜下一个字」的模型（名字不用记也行）
 *
 * 【和「调用顺序.txt」的关系】
 * train.js 里会调用本类的 forward；下面 forward 里的编号与「调用顺序.txt」第 3 节一致。
 *
 * 【几块表里装的是什么】
 * - tokEmb：每个「字的编号」对应一行小数（像字典：编号 → 一串特征）。
 * - posEmb：第 1 个位置、第 2 个位置…各有一行小数（让模型知道「第几个字」，而不只是「哪些字」）。
 * - blocks：你设了几层，就重复几次「大家互相看 + 每个字自己加工」。
 * - lmHead：把每个位置的小向量，变成「对词表里每个字的打分」。
 *
 * 【seqLen】
 * 位置表只准备了这么长的「第几个字」；输入长度不能超过它，否则没有对应的位置行可加。
 */
export class MiniGPT {
  constructor(cfg, rng) {
    const { vocabSize, seqLen, dModel, nHeads, nLayers, dFf } = cfg;
    this.seqLen = seqLen;
    this.dModel = dModel;
    this.tokEmb = Tensor.randn([vocabSize, dModel], rng, true);
    this.posEmb = Tensor.randn([seqLen, dModel], rng, true);
    /** @type {import('../nn/TransformerBlock.js').TransformerBlock[]} */
    this.blocks = [];
    for (let i = 0; i < nLayers; i++) {
      this.blocks.push(new TransformerBlock(dModel, nHeads, dFf, rng));
    }
    this.lmHead = new Linear(dModel, vocabSize, rng);
  }

  /**
   * 从编号到「每个位置、每个可能字的打分」
   * @param {Uint32Array | number[]} tokenIds 长度 T，且 T <= seqLen
   *
   * 【内部顺序（与 调用顺序.txt 第 3 节一致）】
   * ① embed：编号 → 查字典 → 每字一行向量
   * ② narrowRows + add：取出「第几个字」的信息，加到向量上
   * ③ 每一层 block：注意力（大家看前面）→ 加回 → 前馈（每个字自己弯一弯）→ 加回
   * ④ lmHead：每行向量 → 对每个可能字的打分
   */
  forward(tokenIds) {
    const T = tokenIds.length;
    if (T > this.seqLen) throw new Error(`序列长度 ${T} 超过 seqLen ${this.seqLen}`);
    let h = embed(this.tokEmb, tokenIds);
    const pos = narrowRows(this.posEmb, 0, T);
    h = add(h, pos);
    for (const b of this.blocks) {
      h = b.forward(h);
    }
    return this.lmHead.forward(h);
  }

  parameters() {
    const p = [this.tokEmb, this.posEmb, ...this.lmHead.parameters()];
    for (const b of this.blocks) {
      p.push(...b.parameters());
    }
    return p;
  }
}

export { mulberry32 };
