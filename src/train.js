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
 * CORPUS_PATH      语料文件路径（相对项目根目录）；默认 data/corpus/playful_zh.txt。
 * FUN_TRAIN        设为 1 时使用 funTrainingPreset（更多步数、更长 seqLen、略降 lr）。
 * VERBOSE          设为 1 时每一步打印 input/target（很慢，仅调试）。
 * STEPS            覆盖训练步数（正整数），方便快速试跑，例如 STEPS=50。
 * GRAD_CLIP        梯度裁剪上限，默认 1；设为 0 表示不裁剪。
 * VAL_FRACTION     验证集占语料 token 比例（默认 0.08，尾部划出；语料过短则自动关闭验证）。
 * VAL_SAMPLES      验证时在验证集上随机采样的窗口数（默认 24，固定种子下可复现）。
 * METRICS_CSV      训练指标追加写入的 CSV 路径（默认 ./out/train_metrics.csv）；SKIP_METRICS=1 关闭。
 * LORA_RANK        大于 0 时为每个 Linear 挂 LoRA 低秩适配器；常与已训练主干配合，或从头联合训练。
 * LORA_ALPHA       LoRA 缩放系数（默认 16），越大适配器影响越大。
 * LORA_ONLY        设为 1 时只更新 LoRA 矩阵，冻结 tokEmb/posEmb 与各 Linear 的 weight/bias（须 LORA_RANK>0）。
 * MEGA_TRAIN       设为 1 时使用 megaTrainingPreset（超长步数，见 src/config.js）；优先于 FUN_TRAIN。
 * ULTIMATE_TRAIN   设为 1 时使用 ultimateTrainingPreset（大模型 + 长上下文 + 超多步，见 src/config.js）；优先于 MEGA / FUN。
 * OPTIMIZER        adam 使用 Adam；不设则 SGD。极致档建议 adam。
 * COSINE_LR        设为 1 时学习率按步数余弦降到 0（乘在 cfg.lr 上）。
 * LR_WARMUP        正整数：前若干步从 0 线性升到 cfg.lr，再进入余弦（或常数 lr）；0 表示关闭。
 * RESUME_FROM      指向 .mgpt.json：从该 checkpoint 加载权重续训（须与当前语料词表、结构一致）。
 * STEP_OFFSET      非负整数：已完成的训练步数，用于续训时学习率调度从全局步继续；默认 0。
 * TOTAL_STEPS      可选正整数：覆盖「调度总步数」（默认 STEP_OFFSET + 本 run 的 STEPS），用于手动对齐余弦总长。
 * CHECKPOINT_EVERY 正整数：每隔多少**全局步**写一次单文件 checkpoint；0 关闭。默认目录见 CHECKPOINT_DIR。
 * CHECKPOINT_DIR   checkpoint 与 latest.mgpt.json 所在目录，默认 ./out/checkpoints。
 * EARLY_STOP_PATIENCE 正整数：有验证集时，连续多少次**打日志** val_loss 未刷新最优则提前结束；0 关闭。
 *                  续训时 Adam 的动量状态不会从文件恢复，续训相当于 Adam 热重启；需要可改 SGD 或接受此行为。
 * SKIP_DOWNLOAD_CORPUS 设为 1 时不合并 data/corpus/downloaded_mixed_zh.txt。
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

import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import path from 'node:path';

import { defaultConfig, funTrainingPreset, megaTrainingPreset, ultimateTrainingPreset } from './config.js';
import { buildCharTokenizer } from './data/charTokenizer.js';
import { loadCorpus } from './data/loadCorpus.js';
import { MiniGPT, mulberry32 } from './model/MiniGPT.js';
import { crossEntropyMean } from './tensor/ops.js';
import { writeHfModelDir } from './io/hfModelDir.js';
import { readMiniGPTFile, writeMiniGPTFile } from './io/miniGptIO.js';

/** 默认：HF 风格整包（config / vocab / tokenizer_config / safetensors / model.bin …），单独一档 */
const DEFAULT_EXPORT_HF_DIR = './out/export/hf-style';
/** 默认：单文件 JSON 整包，放在 json-single 子目录里，不和 hf-style 混放 */
const DEFAULT_EXPORT_JSON_PATH = './out/export/json-single/model.mgpt.json';

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

/** Adam 一阶/二阶矩；按张量实例存，避免字符串化 key。 */
const adamState = new WeakMap();

/**
 * Adam 更新（与常见默认 β1=0.9, β2=0.999, ε=1e-8 一致）。
 */
const adamStep = (params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) => {
  for (const p of params) {
    if (!p.requiresGrad || !p.grad) continue;
    let st = adamState.get(p);
    if (!st) {
      st = { t: 0, m: new Float32Array(p.data.length), v: new Float32Array(p.data.length) };
      adamState.set(p, st);
    }
    st.t += 1;
    const t = st.t;
    const m = st.m;
    const v = st.v;
    for (let i = 0; i < p.data.length; i++) {
      const g = p.grad[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      const mhat = m[i] / (1 - beta1 ** t);
      const vhat = v[i] / (1 - beta2 ** t);
      p.data[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }
};

/** 梯度整体范数超过 maxNorm 时按比例缩小，训练更稳、loss 不那么乱跳。GRAD_CLIP=0 关闭。 */
/**
 * 当前步学习率：可选线性预热，再可选余弦衰减到 0（与 cfg.lr 相乘的峰值在预热结束时达到）。
 */
const learningRateAtStep = (step, steps, baseLr, warmup, useCosine) => {
  if (steps <= 1) return baseLr;
  const last = steps - 1;
  if (warmup > 0 && step < warmup) {
    return baseLr * ((step + 1) / warmup);
  }
  if (!useCosine) return baseLr;
  if (warmup > 0) {
    if (last <= warmup) {
      return baseLr * 0.5 * (1 + Math.cos(Math.PI));
    }
    const prog = (step - warmup) / (last - warmup);
    return baseLr * (0.5 * (1 + Math.cos(Math.PI * prog)));
  }
  const prog = step / last;
  return baseLr * (0.5 * (1 + Math.cos(Math.PI * prog)));
};

const clipGradNorm = (params, maxNorm) => {
  if (!maxNorm || maxNorm <= 0) return;
  let sum = 0;
  for (const p of params) {
    if (!p.grad) continue;
    for (let i = 0; i < p.grad.length; i++) sum += p.grad[i] * p.grad[i];
  }
  const norm = Math.sqrt(sum);
  if (norm <= maxNorm || norm === 0) return;
  const scale = maxNorm / norm;
  for (const p of params) {
    if (!p.grad) continue;
    for (let i = 0; i < p.grad.length; i++) p.grad[i] *= scale;
  }
};

/**
 * 从尾部划出一段 token 作验证集；语料太短则返回整段训练、无验证。
 * @param {Uint32Array} data
 * @param {number} seqLen
 * @param {number} valFraction 0~0.5
 */
const splitTrainVal = (data, seqLen, valFraction) => {
  const minW = seqLen + 1;
  if (data.length < 2 * minW) {
    return { train: data, val: null };
  }
  let valLen = Math.floor(data.length * valFraction);
  valLen = Math.max(minW, valLen);
  const trainLen = data.length - valLen;
  if (trainLen < minW) {
    return { train: data, val: null };
  }
  return {
    train: data.subarray(0, trainLen),
    val: data.subarray(trainLen),
  };
};

/**
 * 在 buffer 上随机采样若干窗口，平均交叉熵（不做 backward，用于验证集）。
 */
const meanLossSampled = (model, buf, seqLen, nSamples, rng) => {
  const maxStart = buf.length - seqLen - 1;
  if (maxStart < 0) return null;
  const n = Math.min(Math.max(1, nSamples), maxStart + 1);
  let total = 0;
  for (let s = 0; s < n; s++) {
    const start = Math.floor(rng() * (maxStart + 1));
    const chunk = buf.subarray(start, start + seqLen + 1);
    const input = chunk.subarray(0, seqLen);
    const target = chunk.subarray(1, seqLen + 1);
    const logits = model.forward(input);
    const loss = crossEntropyMean(logits, target);
    total += loss.data[0];
  }
  return total / n;
};

/**
 * 追加一行指标（step, train_loss, val_loss, val_ppl）；文件不存在或为空时写表头。
 */
/** checkpoint 里的 config 与当前训练 cfg 在结构字段上须一致（不含 lr/steps 等训练旋钮）。 */
const assertArchCompatible = (fileCfg, trainCfg) => {
  const keys = ['vocabSize', 'seqLen', 'dModel', 'nHeads', 'nLayers', 'dFf', 'loraRank', 'loraAlpha'];
  for (const k of keys) {
    const a = fileCfg[k] ?? 0;
    const b = trainCfg[k] ?? 0;
    if (a !== b) {
      throw new Error(`checkpoint 与当前训练结构不一致: ${k} 文件=${a} 当前=${b}`);
    }
  }
};

const appendMetricsCsv = (csvPath, step, trainLoss, valLoss, valPpl) => {
  const dir = path.dirname(csvPath);
  mkdirSync(dir, { recursive: true });
  const needHeader = !existsSync(csvPath) || statSync(csvPath).size === 0;
  if (needHeader) {
    appendFileSync(csvPath, 'step,train_loss,val_loss,val_ppl\n', 'utf8');
  }
  const v = valLoss == null ? '' : String(valLoss);
  const p = valPpl == null ? '' : String(valPpl);
  appendFileSync(csvPath, `${step},${trainLoss},${v},${p}\n`, 'utf8');
};

/**
 * 主流程（和「调用顺序.txt」第 1、2 节一一对应）：
 * 建字表 → 编码文章 → 建模型 → 循环：取样 → forward → 算错多少 → 往回算 → 改表
 */
const main = () => {
  const CORPUS = loadCorpus();
  const corpusSha16 = createHash('sha256').update(CORPUS).digest('hex').slice(0, 16);

  const useUltimate = process.env.ULTIMATE_TRAIN === '1';
  const useMega = process.env.MEGA_TRAIN === '1';
  const useFun = process.env.FUN_TRAIN === '1';
  let baseCfg = { ...defaultConfig };
  if (useUltimate) {
    baseCfg = { ...defaultConfig, ...ultimateTrainingPreset };
  } else if (useMega) {
    baseCfg = { ...defaultConfig, ...megaTrainingPreset };
  } else if (useFun) {
    baseCfg = { ...defaultConfig, ...funTrainingPreset };
  }
  if (process.env.STEPS) {
    const n = parseInt(process.env.STEPS, 10);
    if (!Number.isNaN(n) && n > 0) baseCfg.steps = n;
  }
  if (useUltimate) {
    console.log(
      '[训练] ULTIMATE_TRAIN=1：大模型 dModel',
      baseCfg.dModel,
      '层',
      baseCfg.nLayers,
      'dFf',
      baseCfg.dFf,
      'seqLen',
      baseCfg.seqLen,
      'lr',
      baseCfg.lr,
      'steps',
      baseCfg.steps,
    );
  } else if (useMega) {
    console.log('[训练] MEGA_TRAIN=1：超长步数 / 上下文，lr=', baseCfg.lr, 'steps=', baseCfg.steps);
  } else if (useFun) {
    console.log('[训练] FUN_TRAIN=1：加长步数 / 上下文，lr=', baseCfg.lr, 'steps=', baseCfg.steps);
  }

  /** tok：tokenizer 的缩写 = 分词/编码器；这里负责「字 ↔ 编号」。 */
  const tok = buildCharTokenizer(CORPUS);

  /** data：把 CORPUS 整段文章编成「一长串编号」，供后面随机截取。 */
  const data = tok.encode(CORPUS);

  /** cfg：config 的缩写 = 配置；把默认旋钮和本语料的词表大小、实际可用的 seqLen 合在一起。 */
  const loraRank = Math.max(0, parseInt(process.env.LORA_RANK ?? String(baseCfg.loraRank ?? 0), 10) || 0);
  const loraAlpha = parseFloat(process.env.LORA_ALPHA ?? String(baseCfg.loraAlpha ?? 16)) || 16;
  const loraOnly = process.env.LORA_ONLY === '1';

  const cfg = {
    ...baseCfg,
    vocabSize: tok.vocabSize,
    seqLen: Math.min(baseCfg.seqLen, data.length - 1),
    loraRank,
    loraAlpha,
    loraOnly,
  };

  if (loraOnly && loraRank <= 0) {
    throw new Error('LORA_ONLY=1 时需要 LORA_RANK>0（先在主干上训练或加载 checkpoint，再挂 LoRA 适配器）');
  }
  if (loraRank > 0) {
    console.log(
      '[训练] LoRA 秩',
      loraRank,
      'alpha',
      loraAlpha,
      loraOnly ? '· 仅更新 LoRA（LORA_ONLY=1）' : '· 与主干一起更新',
    );
  }

  const valFraction = Math.min(0.5, Math.max(0, parseFloat(process.env.VAL_FRACTION ?? '0.08')));
  const valSamples = Math.max(1, parseInt(process.env.VAL_SAMPLES ?? '24', 10) || 24);
  const { train: trainData, val: valData } = splitTrainVal(data, cfg.seqLen, valFraction);
  if (valData) {
    console.log(
      '[验证] 尾部 hold-out · VAL_FRACTION=',
      valFraction,
      '· train tokens',
      trainData.length,
      '· val tokens',
      valData.length,
      '· VAL_SAMPLES=',
      valSamples,
    );
  } else {
    console.log('[验证] 语料过短，跳过 hold-out（仅用训练采样）');
  }

  const metricsCsv = process.env.METRICS_CSV ?? './out/train_metrics.csv';
  const skipMetrics = process.env.SKIP_METRICS === '1';
  if (!skipMetrics) {
    console.log('[指标] CSV →', path.resolve(metricsCsv), '（SKIP_METRICS=1 可关闭）');
  }

  console.log(
    '[repro] corpus_sha256_16=',
    corpusSha16,
    'seed=',
    cfg.seed,
    'steps=',
    cfg.steps,
    'seqLen=',
    cfg.seqLen,
    'val_fraction=',
    valFraction,
  );

  const stepOffset = Math.max(0, parseInt(process.env.STEP_OFFSET ?? '0', 10) || 0);
  const computedEnd = stepOffset + cfg.steps;
  const envTotalSteps = parseInt(process.env.TOTAL_STEPS ?? '', 10);
  const scheduleTotal =
    !Number.isNaN(envTotalSteps) && envTotalSteps > 0 ? Math.max(computedEnd, envTotalSteps) : computedEnd;

  const resumePath = (process.env.RESUME_FROM ?? process.env.LOAD_CHECKPOINT ?? '').trim();
  /** rng：random number generator；仅从头训练时用于初始化权重。 */
  let model;
  if (resumePath) {
    if (!existsSync(resumePath)) {
      throw new Error(`找不到 RESUME_FROM / LOAD_CHECKPOINT: ${resumePath}`);
    }
    const loaded = readMiniGPTFile(resumePath);
    assertArchCompatible(loaded.cfg, cfg);
    if (JSON.stringify(loaded.tok.itos) !== JSON.stringify(tok.itos)) {
      throw new Error('checkpoint 词表与当前语料生成的词表不一致，无法续训');
    }
    model = loaded.model;
    console.log(
      '[续训] 已加载',
      path.resolve(resumePath),
      '· STEP_OFFSET=',
      stepOffset,
      '· 调度总步数',
      scheduleTotal,
      '（本段',
      cfg.steps,
      '步）',
    );
    if (stepOffset === 0) {
      console.warn('[续训] 未设 STEP_OFFSET：学习率调度从第 0 步算起；从中间 checkpoint 继续时请设为已训练全局步数');
    }
  } else {
    const rng = mulberry32(cfg.seed);
    model = new MiniGPT(cfg, rng);
  }

  /** 默认同训主干 + LoRA；LORA_ONLY=1 时只更新 LoRA 与嵌入无关的适配矩阵。 */
  let trainParams = model.parameters();
  if (cfg.loraOnly) {
    model.freezeEmbeddings();
    model.freezeBaseLinearForLoRA();
    trainParams = model.loraParameters();
  }
  /** dataRng：另一套随机数，只负责「从文章里随机选起点」，和上面分开，方便做实验对比。 */
  const dataRng = mulberry32(cfg.seed + 999);

  const useAdam = process.env.OPTIMIZER === 'adam';
  const useCosineLr = process.env.COSINE_LR === '1';
  const lrWarmup = Math.max(0, parseInt(process.env.LR_WARMUP ?? '0', 10) || 0);
  if (useAdam) console.log('[优化] OPTIMIZER=adam');
  if (useCosineLr) console.log('[优化] COSINE_LR=1：余弦学习率');
  if (lrWarmup > 0) {
    console.log('[优化] LR_WARMUP=', lrWarmup, useCosineLr ? '（先线性预热，再余弦衰减）' : '（先线性预热，再保持峰值 lr）');
  }
  if (resumePath && useAdam) {
    console.warn('[续训] Adam 动量未保存在 checkpoint 中，续训相当于 Adam 重新累计动量');
  }

  const checkpointEvery = Math.max(0, parseInt(process.env.CHECKPOINT_EVERY ?? '0', 10) || 0);
  const checkpointDir = process.env.CHECKPOINT_DIR ?? './out/checkpoints';
  if (checkpointEvery > 0) {
    mkdirSync(checkpointDir, { recursive: true });
    console.log('[checkpoint] 每', checkpointEvery, '全局步写入', path.resolve(checkpointDir));
  }

  const earlyStopPatience = Math.max(0, parseInt(process.env.EARLY_STOP_PATIENCE ?? '0', 10) || 0);
  if (earlyStopPatience > 0 && !valData) {
    console.warn('[早停] 无验证集（语料过短或 VAL_FRACTION），已忽略 EARLY_STOP_PATIENCE');
  }

  console.log('词表大小', cfg.vocabSize, 'seqLen', cfg.seqLen, '步数', cfg.steps);

  let bestValLoss = Infinity;
  let valStagnantLogs = 0;

  for (let step = 0; step < cfg.steps; step++) {
    const maxStart = trainData.length - cfg.seqLen - 1;
    const start = Math.floor(dataRng() * (maxStart + 1));
    /** chunk：从整段编号里切出「窗口长度 + 1」的一小段（多 1 个才能错位成答案）。 */
    const chunk = trainData.subarray(start, start + cfg.seqLen + 1);
    /** input：喂给模型的输入编号；target：每个位置对应的「下一个正确字」的编号。 */
    const input = chunk.subarray(0, cfg.seqLen);
    const target = chunk.subarray(1, cfg.seqLen + 1);

    if (process.env.VERBOSE === '1') {
      console.log('--------------------------------');
      console.log('input -->', tok.decode(input));
      console.log('target -->', tok.decode(target));
    }

    // ① 打分表 logits[T×词表大小]：第 i 行 = 「看到 input 里第 0..i 个字后，对下一个字」给每个候选字的 raw 分（见 MiniGPT.forward → lmHead）
    const logits = model.forward(input);
    // ② 统计错多少：在 ops.js 的 crossEntropyMean 里，每行 softmax 成比例，再对「正确答案那一格」取 -log，最后对 T 个位置取平均得到 loss
    const loss = crossEntropyMean(logits, target);
    // ③ 往回算每个权重「该怎么改」
    loss.backward();
    const clip = process.env.GRAD_CLIP === '0' ? 0 : parseFloat(process.env.GRAD_CLIP ?? '1');
    clipGradNorm(trainParams, clip);
    const globalStep = stepOffset + step;
    const stepLr = learningRateAtStep(globalStep, scheduleTotal, cfg.lr, lrWarmup, useCosineLr);
    if (useAdam) {
      adamStep(trainParams, stepLr);
    } else {
      sgdStep(trainParams, stepLr);
    }

    if (checkpointEvery > 0 && (globalStep + 1) % checkpointEvery === 0) {
      const cpPath = path.join(checkpointDir, `checkpoint-step-${globalStep + 1}.mgpt.json`);
      writeMiniGPTFile(model, cfg, tok, cpPath);
      writeMiniGPTFile(model, cfg, tok, path.join(checkpointDir, 'latest.mgpt.json'));
      console.log('[checkpoint] 已写入', cpPath);
    }

    const logEvery = Math.max(40, Math.floor(cfg.steps / 20));
    if (step % logEvery === 0 || step === cfg.steps - 1) {
      let vLoss = null;
      let ppl = null;
      if (valData) {
        const valRng = mulberry32((cfg.seed + 424242) >>> 0);
        vLoss = meanLossSampled(model, valData, cfg.seqLen, valSamples, valRng);
        if (vLoss !== null) ppl = Math.exp(vLoss);
      }
      const extra =
        vLoss !== null && ppl !== null
          ? `  val_loss ${vLoss.toFixed(4)}  val_ppl ${ppl.toFixed(2)}`
          : '';
      console.log(`step ${globalStep}  train_loss ${loss.data[0].toFixed(4)}${extra}`);
      if (!skipMetrics) {
        appendMetricsCsv(metricsCsv, globalStep, loss.data[0], vLoss, ppl);
      }
      if (earlyStopPatience > 0 && valData && vLoss !== null) {
        if (vLoss < bestValLoss - 1e-6) {
          bestValLoss = vLoss;
          valStagnantLogs = 0;
        } else {
          valStagnantLogs += 1;
        }
        if (valStagnantLogs >= earlyStopPatience) {
          console.log(
            '[早停] 验证 loss 连续',
            earlyStopPatience,
            '次日志未改善 · 最佳 val_loss',
            bestValLoss.toFixed(6),
          );
          break;
        }
      }
    }
  }

  console.log('完成。可继续：RESUME_FROM、CHECKPOINT_EVERY、EARLY_STOP_PATIENCE、ULTIMATE_TRAIN、Adam/余弦/LR_WARMUP。');

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
