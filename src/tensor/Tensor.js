/**
 * =============================================================================
 * 本文件里的英文词 / 字段是什么意思
 * =============================================================================
 * Tensor           张量：这里可理解为「带行列形状的数字表」；名字来自数学/框架习惯。
 * Float32Array     JS 内置类型：存 32 位浮点小数的一长串；不必记英文名，即「小数格子数组」。
 * shape            形状：[行, 列]。
 * requiresGrad     require gradient：是否需要为这张表计算「该怎么改」。
 * grad             gradient 梯度：每个格子该怎么动才能让最后的「错多少」变小。
 * _prev            前一个/父节点：这张表由哪些表算出来（下划线表示偏内部实现）。
 * _backward        往回传的规则函数。
 * rows / cols      行数 / 列数快捷访问。
 * zeroGrad         把梯度清零。
 * backward         从标量「错多少」往回传梯度。
 * topo             topological 拓扑序：按依赖排好的顺序列表。
 * visited          访问标记：防止重复遍历。
 * static zeros / randn   静态方法：全零表 / 随机表。
 * mulberry32       随机数算法名，可忽略。
 * seed             种子：固定随机起点。
 * randn1           生成一个近似正态分布的随机数（内部辅助函数）。
 *
 * =============================================================================
 */

import { assertShape2D, prod } from './size.js';

/**
 * Tensor：装「一张数字表格」的类（权重、中间结果、最后的「错多少」都可以放这里）
 *
 * 【先读哪】若你关心「程序先调用谁后调用谁」，请看上一级目录里的 调用顺序.txt。
 *
 * 【表里怎么存】
 * 人脑想的是「几行几列」，程序里为了省内存，把格子按行顺序排成「一长串小数」。
 * 这一长串在 JS 里用一种内置的数组类型来存（名字不用记）；你只要知道：都是小数格子。
 *
 * 【shape 两行数字】
 * 表示「这一长串按几行几列折回去」。行数不对齐就会算错位，所以构造时会检查。
 *
 * 【requiresGrad】
 * 这张表要不要参与「往回算该怎么改」。写死的常数表不需要；要学的权重需要。
 *
 * 【grad】
 * 训练时，在「往回算」之后，每个格子对应「这个数稍微动一点，错多少会怎么变」。
 *
 * 【_prev / _backward】
 * 前向：每算一次新表，就记下它是用哪些旧表算出来的（_prev），
 *      并记下「若知道新表每个格子该怎么改，旧表每个格子该怎么改」（_backward）。
 * 反向：从「错多少」这个数开始，按相反顺序执行这些步骤，把「该怎么改」传回去。
 *
 * 【为什么「错多少」只能是一个数才能往回传】
 * 优化一般是「把一个数变小」。若同时有很多个错，要先合成一个（这里交叉熵已经帮你平均了）。
 */

export class Tensor {
  /**
   * @param data  一长串小数格子（程序里用的那种数组即可）
   * @param shape [行数, 列数]
   * @param requiresGrad  这张表要不要学、要不要算「该怎么改」
   */
  constructor(data, shape, requiresGrad = false) {
    assertShape2D(data.length, shape[0], shape[1]);
    this.data = data;
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    /** 和 data 同样多个格子；「往回算」之后才有意义 */
    /** @type {Float32Array | null} */
    this.grad = null;
    /** 「往回算」时：把「新表每个格子该怎么改」传回「旧表」 */
    /** @type {(() => void) | null} */
    this._backward = null;
    /** 这张新表是直接由哪些旧表算出来的（方便往回传） */
    /** @type {Tensor[]} */
    this._prev = [];
  }

  get rows() {
    return this.shape[0];
  }

  get cols() {
    return this.shape[1];
  }

  /**
   * 若你多次手动累加「该怎么改」，需要清零时用；本训练里每次往回算会重新分配一块，不一定用到。
   */
  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  /**
   * 往回传「该怎么改」：从「错多少」这一个数出发，一直传回所有要学的表。
   *
   * 【什么时候调用】整次「从输入算到错多少」完成之后，每个训练小步里只调一次。
   *
   * 【里面两件事】
   * 1) 先把所有参与计算的表排成一条线：谁只依赖谁，保证先算清后面的再传前面的。
   * 2) 再按相反顺序，把「该怎么改」从「错多少」一层层传回去。
   *
   * 【为什么先把「错多少」自己的「该怎么改」设成 1】
   * 这是链条起点：后面每一步都在这个基础上乘上去。
   */
  backward() {
    if (this.data.length !== 1) {
      throw new Error('backward() 仅支持标量张量（例如平均 loss）');
    }
    const topo = [];
    const visited = new Set();
    const build = (v) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);

    for (const t of topo) {
      if (t.requiresGrad) {
        t.grad = new Float32Array(t.data.length);
      }
    }
    this.grad[0] = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      if (v._backward) v._backward();
    }
  }

  /** 全零表：偏置常用，一开始不引入随机噪声。 */
  static zeros(shape, requiresGrad = false) {
    return new Tensor(new Float32Array(prod(shape)), shape, requiresGrad);
  }

  /**
   * 随机填表当初始权重。
   * 【问题】全 0 会卡住；太大太小都不好。
   * 【做法】随机小数再缩一缩（经验上好用，不必先懂公式）。
   */
  static randn(shape, rng, requiresGrad = false) {
    const n = prod(shape);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = randn1(rng) * Math.sqrt(2 / shape[0]);
    return new Tensor(data, shape, requiresGrad);
  }
}

/**
 * 给一个「种子数」，以后每次调用返回 0～1 之间的小数，用来随机初始化。
 * 【问题】若用系统自带的随机且不能固定种子，两次运行结果不好对比。
 */
export const mulberry32 = (seed) => {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** 把两个 0～1 随机数变成「像钟形曲线那样」的一个随机数，用来填初始权重。 */
const randn1 = (rng) => {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};
