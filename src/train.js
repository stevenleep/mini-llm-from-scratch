/**
 * =============================================================================
 * 本文件里的英文词 / 变量名是什么意思（全部写在这里）
 * =============================================================================
 * import           从别的文件「拉进来」用；不是英文生词，是 JS 语法关键字。
 * defaultConfig    默认配置：从 config.js 来的那一组旋钮。
 * buildCharTokenizer   建立分词器（见 data/charTokenizer.js）。
 * MiniGPT          小语言模型类（见 model/MiniGPT.js）。
 * mulberry32       随机数生成函数名（可忽略），用来可复现随机。
 * crossEntropyMean 交叉熵平均：把「打分」和「正确答案」比成一个「错多少」的数。
 * CORPUS           语料：一整段用来训练的例文字符串。
 * sgdStep          stochastic gradient descent step：随机梯度下降一步，真改权重。
 * params           parameters 缩写：要更新的参数列表。
 * lr               learning rate 学习率。
 * p                循环里常表示「一个参数张量」。
 * requiresGrad     require gradient：这个张量要不要算「该怎么改」。
 * grad             gradient 梯度：每个格子该怎么改。
 * main             主函数：程序入口习惯叫 main。
 * tok              tokenizer 分词器对象。
 * data             整段文章编成的一长串编号。
 * cfg              config 配置。
 * model            模型实例。
 * rng / dataRng    两套随机数；rng 初始化权重，dataRng 随机截取文章。
 * step             第几步训练循环。
 * maxStart         随机起点能取的最大下标。
 * start            本次随机选中的起点下标。
 * chunk            切出来的一小段编号。
 * input / target   输入序列 / 目标（答案）序列。
 * logits           模型原始输出分（未变概率）；可理解成「打分表」。
 * loss             损失：错多少。
 * backward         往回传梯度（算每个参数该怎么改）。
 * console.log      在终端打印一行字。
 * writeHfModelDir  把模型存成「一个文件夹」：config、词表、分词器说明、model.safetensors，以及纯二进制的 model.bin + 清单（和网上常见摆法类似）。
 * writeMiniGPTFile 把模型存成「一个大 JSON」，老格式，单文件好拷贝。
 * EXPORT_DIR       环境变量：只改「HF 风格文件夹」写到哪里，默认 ./out/export/hf-style。
 * EXPORT_PATH      环境变量：只改「单文件 JSON」写到哪里，默认 ./out/export/json-single/model.mgpt.json。
 *                  两种产物分开放：都在 out/export/ 下，hf-style 与 json-single 各一档，不混在一个目录里。
 *                  训练正常跑完后会两种格式各导出一份（除非设 SKIP_EXPORT=1 不写盘）。
 * SKIP_EXPORT      环境变量：设为 1 时跳过导出（只想快速试跑、不落盘时）。
 *
 * =============================================================================
 * train.js —— 直接运行这个文件，就会按「调用顺序.txt」里的大顺序练模型
 * =============================================================================
 *
 * 【这一文件在整条流水线里的位置】
 * 最外层：负责读配置、造字表、造模型、循环「取一段字 → 算错多少 → 往回改 → 真的改表」。
 *
 * 【input 和 target 为什么错开一位】
 * 每个位置练的是：「看到目前为止的字，猜下一个」。
 * 所以「标准答案」要比「输入」整体往后挪一位（同长度里，第 i 个答案 = 原文里紧跟输入第 i 个字后面的那个字）。
 *
 * 【两个小随机】
 * 一个用来「第一次给表里填随机小数」；一个用来「从长文章里随机起点截取」。
 * 分开是为了以后你做实验时，不会把「换取样方式」和「换初始表」搅在一起。
 *
 * 【seqLen 用 min 截断】
 * 若示例文章比配置里「一次看的字数」还短，就只能取更短的窗口，否则切不出「多 1 个」用来做答案。
 *
 * =============================================================================
 * 「打分」和「错多少」具体在代码哪一段？（逐句对应）
 * =============================================================================
 * ① model.forward(input) 产出 logits：
 *    最后一层在 MiniGPT 里是 lmHead（Linear），把每个位置的向量变成「词表里每个字一列」的 raw 分。
 *    中间从编号到向量、混合上下文，都在 MiniGPT 与各 nn 里完成。
 * ② crossEntropyMean(logits, target) 产出 loss（一个数）：
 *    具体「先把每行分变成比例，再和 target 比，再平均」的算术，全在 tensor/ops.js 的 crossEntropyMean 里，
 *    该函数上方有逐行中文注释，对着 for 循环读即可。
 * ③ loss.backward()：根据错多少，往回算每张权重表每个格子该怎么动（tensor/Tensor.js + ops 里各 _backward）。
 * ④ sgdStep：真的改权重。
 */

import { defaultConfig } from './config.js';
import { buildCharTokenizer } from './data/charTokenizer.js';
import { MiniGPT, mulberry32 } from './model/MiniGPT.js';
import { crossEntropyMean } from './tensor/ops.js';
import { writeHfModelDir } from './io/hfModelDir.js';
import { writeMiniGPTFile } from './io/miniGptIO.js';

/** 默认：HF 风格整包（config / vocab / tokenizer_config / safetensors / model.bin …），单独一档 */
const DEFAULT_EXPORT_HF_DIR = './out/export/hf-style';
/** 默认：单文件 JSON 整包，放在 json-single 子目录里，不和 hf-style 混放 */
const DEFAULT_EXPORT_JSON_PATH = './out/export/json-single/model.mgpt.json';

/**
 * CORPUS（读作「科普斯」，英文原意是「语料库」）
 * —— 这里指：用来训练的一整段示例文章（字符串）。
 * 程序会从里面统计有哪些字、给每个字编号，再反复截取小段来练「猜下一个字」。
 * 你换成自己的长文章也可以，只要还是普通文本。
 */
const CORPUS = `
在最小实现里，一条训练样本是：输入长度为 T 的 token 序列，
让模型在每个位置预测下一个 token。反向传播从平均交叉熵开始，
沿计算图回到词嵌入与所有线性层。这里没有 LayerNorm，仅用于演示核心数据流。
`.trim();

/**
 * sgdStep：名字里 sgd 是「随机梯度下降」的英文缩写（一种最简单的改权重方式）。
 * 作用：根据「每个格子该怎么改」，真的把表里的数减掉一小步。
 * lr：learning rate（学习率）的缩写，管「每一步迈多大」。
 */
const sgdStep = (params, lr) => {
  for (const p of params) {
    if (!p.requiresGrad || !p.grad) continue;
    for (let i = 0; i < p.data.length; i++) {
      p.data[i] -= lr * p.grad[i];
    }
  }
};

/**
 * 主流程（和「调用顺序.txt」第 1、2 节一一对应）：
 * 建字表 → 编码文章 → 建模型 → 循环：取样 → forward → 算错多少 → 往回算 → 改表
 */
const main = () => {
  /** tok：tokenizer 的缩写 = 分词/编码器；这里负责「字 ↔ 编号」。 */
  const tok = buildCharTokenizer(CORPUS);

  /** data：把 CORPUS 整段文章编成「一长串编号」，供后面随机截取。 */
  const data = tok.encode(CORPUS);

  /** cfg：config 的缩写 = 配置；把默认旋钮和本语料的词表大小、实际可用的 seqLen 合在一起。 */
  const cfg = {
    ...defaultConfig,
    vocabSize: tok.vocabSize,
    seqLen: Math.min(defaultConfig.seqLen, data.length - 1),
  };

  /** rng：random number generator，随机数发生器；用来给模型里各张表填初始随机小数。 */
  const rng = mulberry32(cfg.seed);
  const model = new MiniGPT(cfg, rng);
  /** dataRng：另一套随机数，只负责「从文章里随机选起点」，和上面分开，方便做实验对比。 */
  const dataRng = mulberry32(cfg.seed + 999);

  console.log('词表大小', cfg.vocabSize, 'seqLen', cfg.seqLen, '步数', cfg.steps);

  for (let step = 0; step < cfg.steps; step++) {
    const maxStart = data.length - cfg.seqLen - 1;
    const start = Math.floor(dataRng() * (maxStart + 1));
    /** chunk：从整段编号里切出「窗口长度 + 1」的一小段（多 1 个才能错位成答案）。 */
    const chunk = data.subarray(start, start + cfg.seqLen + 1);
    /** input：喂给模型的输入编号；target：每个位置对应的「下一个正确字」的编号。 */
    const input = chunk.subarray(0, cfg.seqLen);
    const target = chunk.subarray(1, cfg.seqLen + 1);

    console.log('--------------------------------');
    console.log("  ")
    console.log('input -->', input, tok.decode(input));
    
    console.log('--------------------------------');
    console.log("  ")
    console.log('target -->', target, tok.decode(target));

    // ① 打分表 logits[T×词表大小]：第 i 行 = 「看到 input 里第 0..i 个字后，对下一个字」给每个候选字的 raw 分（见 MiniGPT.forward → lmHead）
    const logits = model.forward(input);
    // ② 统计错多少：在 ops.js 的 crossEntropyMean 里，每行 softmax 成比例，再对「正确答案那一格」取 -log，最后对 T 个位置取平均得到 loss
    const loss = crossEntropyMean(logits, target);
    // ③ 往回算每个权重「该怎么改」
    loss.backward();
    // ④ 真的改
    sgdStep(model.parameters(), cfg.lr);

    if (step % 40 === 0 || step === cfg.steps - 1) {
      console.log(`step ${step}  loss ${loss.data[0].toFixed(4)}`);
    }
  }

  console.log('完成。可继续：增大语料、加 LayerNorm、Adam、多 batch。');

  // 两种导出：out/export/hf-style（常见目录结构）与 out/export/json-single（单文件）。可用 EXPORT_DIR / EXPORT_PATH 覆盖；SKIP_EXPORT=1 不写盘。
  if (process.env.SKIP_EXPORT === '1') {
    console.log('已跳过导出（SKIP_EXPORT=1）。');
  } else {
    const hfDir = process.env.EXPORT_DIR ?? DEFAULT_EXPORT_HF_DIR;
    const jsonPath = process.env.EXPORT_PATH ?? DEFAULT_EXPORT_JSON_PATH;
    writeHfModelDir(model, cfg, tok, hfDir);
    writeMiniGPTFile(model, cfg, tok, jsonPath);
    console.log('已导出 → HF 风格目录:', hfDir);
    console.log('         单文件 JSON :', jsonPath);
  }
};

main();
