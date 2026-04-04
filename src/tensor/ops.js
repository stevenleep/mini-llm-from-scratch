/**
 * =============================================================================
 * 本文件里的函数名 / 英文词是什么意思（按字母相关分组）
 * =============================================================================
 * transpose        转置：行列互换。
 * matmul           matrix multiply：矩阵乘法。
 * add              逐格相加（两张表形状相同）。
 * addBias          add bias：加偏置（每一行加同一串 b）。
 * mul              multiply：逐格相乘（不是矩阵乘）。
 * mulScalar        multiply scalar：每个数乘同一个常数。
 * softmaxRows      softmax 按行：每一行变成「非负且和为 1」的比例（像概率）。
 * gelu             一种激活函数名字，可当成「弯一下」。
 * embed            embedding：嵌入/查表，用编号从 emb 表里取行。
 * narrowCols       narrow 取窄：只取连续几列。
 * narrowRows       只取连续几行。
 * concatHeads      concatenate heads：把多个「头」的输出在最后一维拼起来。
 * causalAttentionMask  causal 因果：mask 掩码；合起来=因果注意力用的掩码表。
 * crossEntropyMean cross entropy 交叉熵：分类里常用的「错多少」；mean 表示对位置取平均。
 *
 * 常见参数名：
 * a, b             通用输入张量；matmul 里 a、b 表示左右矩阵。
 * x                常用输入张量记号。
 * emb              embedding 嵌入表。
 * idx / indices    index 索引：编号序列。
 * heads            多个头的输出数组。
 * logits           模型最后一层前：每个类别的原始分（未变概率）。
 * targets          目标：正确答案的编号序列。
 * T, V             常表示序列长度 T、词表大小 V（vocab size）。
 * NEG_INF          negative infinity：极大负数，用来屏蔽不允许的位置。
 * requiresGrad     是否需要梯度。
 * _prev / _backward  计算图：父节点、往回传函数。
 *
 * =============================================================================
 * ops.js —— 各种「从旧表算出新表」的规则，以及「往回传该怎么改」时对应的每一步
 * =============================================================================
 *
 * 【先看哪】整体先调用谁：请看上一级目录里的「调用顺序.txt」第 3 节；这里是被调用的「零件」。
 *
 * 【为什么单独成文件】
 * 模型文件只写「先注意力再前馈」这种结构；具体乘、加、查表写在这里，方便你对照「算新表」的每一步。
 *
 * 【约定（不用记名词）】
 * - 每张表用 [行数, 列数] 说明；格子在程序里按行排成一长串小数。
 * - 每个函数算出新表返回，不偷偷改传进来的旧表——否则「往回传该怎么改」会断掉。
 * - 只有标记了「要学的表」才会在往回传时累加「该怎么改」。
 *
 * 【下面出现的「极大负数」】
 * 加在「不允许去看的位置」上，后面一步会把一长串分变成比例时，那些位置比例接近 0。
 * 用极大负数而不用「无穷」，只是编程里好写、好算。
 */

import { Tensor } from './Tensor.js';
import { assertShape2D } from './size.js';

const NEG_INF = -1e9;

/**
 * transpose：行列互换。
 * 【问题】注意力里要算 Q 与 K 的点积，形状要对齐，常需要 K^T。
 * 【结果】若输入 [m,n]，输出 [n,m]。
 */
export const transpose = (a) => {
  const [m, n] = a.shape;
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      out[j * m + i] = a.data[i * n + j];
    }
  }
  const t = new Tensor(out, [n, m], a.requiresGrad);
  t._prev = [a];
  t._backward = () => {
    if (!a.requiresGrad) return;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        a.grad[i * n + j] += t.grad[j * m + i];
      }
    }
  };
  return t;
};

/**
 * matmul：矩阵乘法 C = A @ B。
 * 【形状】A 为 [m,k]，B 为 [k,n]，得 [m,n]。中间维度 k 必须一致，否则「列乘行」对不上。
 * 【用在哪】Linear 的 x@W；注意力里 scores = Q@K^T、attn@V。
 * 【不新建 Tensor、直接改 data 会怎样】反向传播不知道前向干了什么，梯度无法传回权重。
 */
export const matmul = (a, b) => {
  const [m, k1] = a.shape;
  const [k2, n] = b.shape;
  if (k1 !== k2) throw new Error(`matmul: inner ${k1} vs ${k2}`);
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      const row = i * k1;
      for (let k = 0; k < k1; k++) {
        s += a.data[row + k] * b.data[k * n + j];
      }
      out[i * n + j] = s;
    }
  }
  const requiresGrad = a.requiresGrad || b.requiresGrad;
  const t = new Tensor(out, [m, n], requiresGrad);
  t._prev = [a, b];
  t._backward = () => {
    if (a.requiresGrad) {
      for (let i = 0; i < m; i++) {
        for (let k = 0; k < k1; k++) {
          let s = 0;
          for (let j = 0; j < n; j++) {
            s += t.grad[i * n + j] * b.data[k * n + j];
          }
          a.grad[i * k1 + k] += s;
        }
      }
    }
    if (b.requiresGrad) {
      for (let k = 0; k < k1; k++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let i = 0; i < m; i++) {
            s += a.data[i * k1 + k] * t.grad[i * n + j];
          }
          b.grad[k * n + j] += s;
        }
      }
    }
  };
  return t;
};

/**
 * add：两张相同形状的表逐格相加。
 * 【用在哪】残差 x+f(x)、位置编码加到嵌入上、掩码加到注意力分数上。
 * 【梯度】加法把上游梯度原样分给两个输入（像「两条路各承担一份责任」）。
 */
export const add = (a, b) => {
  const [r1, c1] = a.shape;
  const [r2, c2] = b.shape;
  if (r1 !== r2 || c1 !== c2) throw new Error('add: shape mismatch');
  const n = r1 * c1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] + b.data[i];
  const requiresGrad = a.requiresGrad || b.requiresGrad;
  const t = new Tensor(out, a.shape, requiresGrad);
  t._prev = [a, b];
  t._backward = () => {
    if (a.requiresGrad) {
      for (let i = 0; i < n; i++) a.grad[i] += t.grad[i];
    }
    if (b.requiresGrad) {
      for (let i = 0; i < n; i++) b.grad[i] += t.grad[i];
    }
  };
  return t;
};

/**
 * addBias：每一行加上同一个偏置向量 b（b 形状 [1, 列数]）。
 * 【问题】若把 b 复制成很多行再 add，浪费内存；数学上完全等价于「每行加同一向量」。
 * 【偏置梯度】对多行累加：因为每一行都对 b 的同一位有贡献。
 */
export const addBias = (x, b) => {
  const [rows, cols] = x.shape;
  if (b.shape[0] !== 1 || b.shape[1] !== cols) {
    throw new Error('addBias: 期望 b 为 [1, cols]');
  }
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] = x.data[i * cols + j] + b.data[j];
    }
  }
  const requiresGrad = x.requiresGrad || b.requiresGrad;
  const t = new Tensor(out, x.shape, requiresGrad);
  t._prev = [x, b];
  t._backward = () => {
    if (x.requiresGrad) {
      for (let i = 0; i < out.length; i++) x.grad[i] += t.grad[i];
    }
    if (b.requiresGrad) {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          b.grad[j] += t.grad[i * cols + j];
        }
      }
    }
  };
  return t;
};

/**
 * mul：同形状逐格相乘（不是矩阵乘）。
 * 【注意】和 matmul 完全不同；若混淆会得到错误形状或错误梯度。
 */
export const mul = (a, b) => {
  const n = a.data.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] * b.data[i];
  const requiresGrad = a.requiresGrad || b.requiresGrad;
  const t = new Tensor(out, a.shape, requiresGrad);
  t._prev = [a, b];
  t._backward = () => {
    if (a.requiresGrad) {
      for (let i = 0; i < n; i++) a.grad[i] += t.grad[i] * b.data[i];
    }
    if (b.requiresGrad) {
      for (let i = 0; i < n; i++) b.grad[i] += t.grad[i] * a.data[i];
    }
  };
  return t;
};

/**
 * mulScalar：每个数都乘同一个常数 s。
 * 【问题】若手写循环改 scores.data *= s，会破坏计算图，反向不知道发生了什么。
 * 【用在哪】注意力里除以 sqrt(head_dim)，缩放点积大小。
 */
export const mulScalar = (a, s) => {
  const n = a.data.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] * s;
  const t = new Tensor(out, a.shape, a.requiresGrad);
  t._prev = [a];
  t._backward = () => {
    if (a.requiresGrad) {
      for (let i = 0; i < n; i++) a.grad[i] += t.grad[i] * s;
    }
  };
  return t;
};

/**
 * softmaxRows：每一行单独做 softmax，使该行非负且和为 1（可看成概率）。
 * 【问题】直接 exp 可能溢出：先减该行最大值，数学上等价但数值稳定。
 * 【用在哪】注意力权重：第 i 行表示「第 i 个位置对各个 key 的分配比例」。
 */
export const softmaxRows = (x) => {
  const [rows, cols] = x.shape;
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    let max = -Infinity;
    for (let j = 0; j < cols; j++) {
      const v = x.data[i * cols + j];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const e = Math.exp(x.data[i * cols + j] - max);
      out[i * cols + j] = e;
      sum += e;
    }
    for (let j = 0; j < cols; j++) out[i * cols + j] /= sum;
  }
  const t = new Tensor(out, x.shape, x.requiresGrad);
  t._prev = [x];
  t._backward = () => {
    if (!x.requiresGrad) return;
    for (let i = 0; i < rows; i++) {
      let dot = 0;
      for (let j = 0; j < cols; j++) {
        dot += out[i * cols + j] * t.grad[i * cols + j];
      }
      for (let j = 0; j < cols; j++) {
        const y = out[i * cols + j];
        x.grad[i * cols + j] += y * (t.grad[i * cols + j] - dot);
      }
    }
  };
  return t;
};

/**
 * GELU：带一点弯曲的激活函数，夹在「全通过」和「压掉」之间。
 * 【问题】若只有线性层堆叠，等价于一层线性，表达能力很弱；必须加非线性。
 * 【为什么用 GELU 而不是 ReLU】Transformer 里常用，平滑一点；教学上可替换对比。
 */
export const gelu = (x) => {
  const n = x.data.length;
  const out = new Float32Array(n);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < n; i++) {
    const v = x.data[i];
    const tt = v + 0.044715 * v * v * v;
    out[i] = 0.5 * v * (1 + Math.tanh(c * tt));
  }
  const res = new Tensor(out, x.shape, x.requiresGrad);
  res._prev = [x];
  res._backward = () => {
    if (!x.requiresGrad) return;
    for (let i = 0; i < n; i++) {
      const v = x.data[i];
      const tt = v + 0.044715 * v * v * v;
      const th = Math.tanh(c * tt);
      const sech2 = 1 - th * th;
      const dt = c * sech2 * (1 + 3 * 0.044715 * v * v);
      const dg = 0.5 * (1 + th) + 0.5 * v * dt;
      x.grad[i] += dg * res.grad[i];
    }
  };
  return res;
};

/**
 * embed：查表。emb 第 id 行就是编号 id 对应的向量。
 * 【问题】神经网络不能直接吃「字符」，要先变成固定长度向量；嵌入表就是可学习的「字典」。
 * 【反向】同一 id 在序列里出现多次，梯度要累加到 emb 的同一行（像多人投票）。
 */
export const embed = (emb, idx) => {
  const [vocab, d] = emb.shape;
  const tlen = idx.length;
  const out = new Float32Array(tlen * d);
  for (let i = 0; i < tlen; i++) {
    const id = idx[i];
    if (id < 0 || id >= vocab) throw new Error(`bad token ${id}`);
    const off = id * d;
    const row = i * d;
    for (let j = 0; j < d; j++) out[row + j] = emb.data[off + j];
  }
  const tensor = new Tensor(out, [tlen, d], emb.requiresGrad);
  tensor._prev = [emb];
  tensor._backward = () => {
    if (!emb.requiresGrad) return;
    for (let i = 0; i < tlen; i++) {
      const id = idx[i];
      const row = i * d;
      const off = id * d;
      for (let j = 0; j < d; j++) emb.grad[off + j] += tensor.grad[row + j];
    }
  };
  return tensor;
};

/**
 * narrowCols：从大矩阵里取连续的几列（复制出来，不是视图）。
 * 【用在哪】多头注意力里把 QKV 宽向量切成每个头自己的 Q、K、V。
 */
export const narrowCols = (x, start, width) => {
  const [T, C] = x.shape;
  if (start + width > C) throw new Error('narrowCols: 越界');
  const out = new Float32Array(T * width);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < width; j++) {
      out[i * width + j] = x.data[i * C + start + j];
    }
  }
  const t = new Tensor(out, [T, width], x.requiresGrad);
  t._prev = [x];
  t._backward = () => {
    if (!x.requiresGrad) return;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < width; j++) {
        x.grad[i * C + start + j] += t.grad[i * width + j];
      }
    }
  };
  return t;
};

/**
 * narrowRows：取连续的几行。
 * 【用在哪】位置编码表存了最长 seqLen 行，当前序列只有 T 个 token，就取前 T 行。
 */
export const narrowRows = (x, rowStart, rowCount) => {
  const [, C] = x.shape;
  const out = new Float32Array(rowCount * C);
  for (let i = 0; i < rowCount; i++) {
    for (let j = 0; j < C; j++) {
      out[i * C + j] = x.data[(rowStart + i) * C + j];
    }
  }
  const t = new Tensor(out, [rowCount, C], x.requiresGrad);
  t._prev = [x];
  t._backward = () => {
    if (!x.requiresGrad) return;
    for (let i = 0; i < rowCount; i++) {
      for (let j = 0; j < C; j++) {
        x.grad[(rowStart + i) * C + j] += t.grad[i * C + j];
      }
    }
  };
  return t;
};

/**
 * concatHeads：把多个头的输出在最后一维拼成更宽的向量。
 * 【问题】每个头各自算完还是窄的，要拼回 dModel 宽，再交给后面的 Linear。
 */
export const concatHeads = (heads) => {
  const T = heads[0].rows;
  const hd = heads[0].cols;
  const H = heads.length;
  const d = H * hd;
  const out = new Float32Array(T * d);
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < hd; j++) {
        out[i * d + h * hd + j] = heads[h].data[i * hd + j];
      }
    }
  }
  const requiresGrad = heads.some((t) => t.requiresGrad);
  const t = new Tensor(out, [T, d], requiresGrad);
  t._prev = heads;
  t._backward = () => {
    for (let h = 0; h < H; h++) {
      if (!heads[h].requiresGrad) continue;
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < hd; j++) {
          heads[h].grad[i * hd + j] += t.grad[i * d + h * hd + j];
        }
      }
    }
  };
  return t;
};

/**
 * causalAttentionMask：上三角（未来位置）加极大负数，softmax 后权重≈0。
 * 【问题】生成文本时不能看到未来 token，否则训练和实际使用不一致（数据泄漏）。
 * 【为什么 requiresGrad 为 false】掩码是固定规则，不参与学习。
 */
export const causalAttentionMask = (T) => {
  const data = new Float32Array(T * T);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      data[i * T + j] = j > i ? NEG_INF : 0;
    }
  }
  return new Tensor(data, [T, T], false);
};

/**
 * crossEntropyMean：「打分」怎么变成「错多少」，以及「统计」怎么做——具体全在本函数里。
 *
 * 【输入是什么】
 * - logits：形状 [T, V]。T = 句子里有几个「猜下一个字」的位置；V = 词表里有多少个字。
 *   第 i 行、第 j 列 = 模型在「第 i 个位置」对「编号为 j 的字」打的 raw 分（未归一，可正可负）。
 * - targets：长度 T。targets[i] = 第 i 个位置上，课文里真正的下一个字的编号（标准答案，只有一个整数）。
 *
 * 【「打分」和「概率」的关系（每一行单独做）】
 * 人不能直接拿 raw 分比「有多确信」，所以先把一行 V 个分变成 V 个「比例」，加起来=1，像概率：
 *   1) 找这一行最大的分 max（防止指数爆掉，数学上结果不变）。
 *   2) 每个分先减 max，再 exp（变成正的放大系数），得到 exps[j]。
 *   3) sum = 所有 exps 相加。
 *   4) 第 j 个字的比例 = exps[j] / sum。这就是 softmax（软最大）：分高的比例大。
 *
 * 【「错多少」怎么统计（仍是对每个位置 i）】
 * 标准答案只有一个字 ti = targets[i]。理想情况：ti 那一格比例 = 1，其余 = 0。
 * 用 -log(正确答案那一格的比例) 表示「有多离谱」：比例越接近 1，-log 越接近 0；越接近 0，-log 越大。
 * 这叫交叉熵里的一项。代码里：loss += -Math.log(exps[ti] / sum)。
 *
 * 【整句的「错多少」】
 * 对 i = 0..T-1 每个位置都加一项，再除以 T：loss /= T。即「平均每个位置错多少」。
 *
 * 【backward 里在干什么（往回传怎么改打分）】
 * 对每个位置再算一遍 probs[j]（同上），然后对每个 j：logits 的梯度 += (probs[j] - 正确答案是否为 j)。
 * 直观：若某字分太高但不是答案，梯度会把分往下拉；正确答案分不够则往上推。
 */
export const crossEntropyMean = (logits, targets) => {
  const [T, V] = logits.shape;
  assertShape2D(logits.data.length, T, V);
  if (targets.length !== T) throw new Error('crossEntropy: len mismatch');

  let loss = 0;
  for (let i = 0; i < T; i++) {
    // 本行在 logits 里从第几个格子开始（一行有 V 个格子）
    const row = i * V;
    // 本行最大的 raw 分，用来数值稳定（减 max 再 exp，避免溢出）
    let max = -Infinity;
    for (let j = 0; j < V; j++) {
      if (logits.data[row + j] > max) max = logits.data[row + j];
    }
    let sum = 0;
    const exps = new Float32Array(V);
    for (let j = 0; j < V; j++) {
      exps[j] = Math.exp(logits.data[row + j] - max);
      sum += exps[j];
    }
    const ti = targets[i];
    // 正确答案 ti 那一格，在 softmax 之后的「比例」是 exps[ti]/sum；取负对数作为本位置损失
    loss += -Math.log(exps[ti] / sum);
  }
  // 对 T 个位置取平均：「平均每个字位错多少」
  loss /= T;

  const L = new Tensor(new Float32Array([loss]), [1, 1], logits.requiresGrad);
  L._prev = [logits];
  L._backward = () => {
    if (!logits.requiresGrad) return;
    const g = L.grad[0] / T;
    for (let i = 0; i < T; i++) {
      let max = -Infinity;
      const row = i * V;
      for (let j = 0; j < V; j++) {
        if (logits.data[row + j] > max) max = logits.data[row + j];
      }
      let sum = 0;
      const probs = new Float32Array(V);
      for (let j = 0; j < V; j++) {
        probs[j] = Math.exp(logits.data[row + j] - max);
        sum += probs[j];
      }
      for (let j = 0; j < V; j++) probs[j] /= sum;
      const ti = targets[i];
      for (let j = 0; j < V; j++) {
        const p = probs[j];
        const indicator = j === ti ? 1 : 0;
        logits.grad[row + j] += g * (p - indicator);
      }
    }
  };
  return L;
};
