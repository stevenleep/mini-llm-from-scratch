/**
 * =============================================================================
 * 这个脚本干什么（人话）
 * =============================================================================
 * 加载训练导出的模型，按提示「续写」若干字：可贪心（总选分数最高的字），也可按温度 + top-k 随机抽，
 * 更像聊天/创作。默认流式打印续写；可用 --no-stream 整段打完再输出。
 *
 * 第一个参数：模型路径（HF 风格文件夹，或单个 .json 整包）。
 * 第二个参数：提示文字（可省略，默认「在」）。
 * 其余用参数名：
 *   --max-new-tokens / -n   续写多少字，默认 64
 *   --temperature / -t      越大越随机，默认 0.8；配合 --greedy 或 <=0 则贪心
 *   --top-k / -k            只从分数最高的 k 个字里抽，默认 50；0 表示不限制
 *   --top-p                 nucleus 采样（0~1，默认 0.92；1 表示不用）
 *   --repetition-penalty    重复惩罚（≥1，默认 1.12，减轻单字循环）
 *   --seed                  随机种子，同一模型同一提示可复现
 *   --greedy / -g           贪心，不看温度
 *   --no-stream             不打流式，整段输出
 *   --help / -h             说明
 * =============================================================================
 */

import { statSync } from 'node:fs';
import { generateContinuation } from './generate.js';
import { alignIdentityPrompt } from './promptAlign.js';
import { readHfModelDir } from './io/hfModelDir.js';
import { readMiniGPTFile } from './io/miniGptIO.js';

const printHelp = () => {
  console.log(`
用法: node src/infer.js <模型路径> [提示文字] [选项]

选项:
  -n, --max-new-tokens <数>   续写字数（默认 64）
  -t, --temperature <数>    采样温度（默认 0.8；越小越稳）
  -k, --top-k <数>           只在最高的 k 个字里采样（默认 50；0=不截断）
      --top-p <0~1>          nucleus 采样（默认 0.92）
      --repetition-penalty <≥1>  重复惩罚（默认 1.12）
      --seed <数>            随机种子（可选，便于复现）
  -g, --greedy               贪心解码（每步取 argmax）
      --no-stream            关闭流式，整段打印续写
  -h, --help                 显示本说明

示例:
  node src/infer.js ./out/export/hf-style "在最小" -n 30 -t 0.9
  node src/infer.js ./out/export/json-single/model.mgpt.json 在 -g -n 5
`);
};

/** 是不是「选项」而不是提示正文（避免把 -n 当成提示） */
const isOptionToken = (s) => {
  if (!s) return false;
  if (s.startsWith('--')) return true;
  return ['-n', '-t', '-k', '-g', '-s', '-h'].includes(s) || s === '--no-stream';
};

const parseArgs = (argv) => {
  const out = {
    modelPath: null,
    prompt: '在',
    maxNewTokens: 64,
    temperature: 0.75,
    topK: 50,
    topP: 0.92,
    repetitionPenalty: 1.12,
    /** @type {number | undefined} */
    seed: undefined,
    greedy: false,
    stream: true,
    help: false,
  };
  const args = argv.slice(2);
  if (args.length === 0) {
    out.help = true;
    return out;
  }
  if (args[0] === '-h' || args[0] === '--help') {
    out.help = true;
    return out;
  }
  out.modelPath = args.shift();
  if (args.length > 0 && !isOptionToken(args[0])) {
    out.prompt = args.shift();
  }
  const take = () => {
    const v = args.shift();
    if (v === undefined) throw new Error('参数缺值');
    return v;
  };
  while (args.length) {
    const a = args.shift();
    if (a === '-h' || a === '--help') out.help = true;
    else if (a === '-n' || a === '--max-new-tokens') out.maxNewTokens = parseInt(take(), 10);
    else if (a === '-t' || a === '--temperature') out.temperature = parseFloat(take());
    else if (a === '-k' || a === '--top-k') out.topK = parseInt(take(), 10);
    else if (a === '--seed') out.seed = parseInt(take(), 10) >>> 0;
    else if (a === '--top-p') out.topP = parseFloat(take());
    else if (a === '--repetition-penalty') out.repetitionPenalty = parseFloat(take());
    else if (a === '-g' || a === '--greedy') out.greedy = true;
    else if (a === '--no-stream') out.stream = false;
    else if (a === '-s' || a === '--stream') out.stream = true;
    else throw new Error(`不认识参数: ${a}（用 --help 查看用法）`);
  }
  if (out.help) return out;
  if (!out.modelPath) throw new Error('请提供模型路径');
  if (Number.isNaN(out.maxNewTokens) || out.maxNewTokens < 0) throw new Error('max-new-tokens 须为非负整数');
  if (Number.isNaN(out.temperature)) throw new Error('temperature 须为数字');
  if (Number.isNaN(out.topK) || out.topK < 0) throw new Error('top-k 须为 >=0 的整数');
  return out;
};

const main = async () => {
  let opts;
  try {
    opts = parseArgs(process.argv);
  } catch (e) {
    console.error((e instanceof Error ? e.message : e) || String(e));
    printHelp();
    process.exit(1);
  }
  if (opts.help || !opts.modelPath) {
    printHelp();
    process.exit(opts.help ? 0 : 1);
  }

  const st = statSync(opts.modelPath);
  const { model, tok } = st.isDirectory() ? readHfModelDir(opts.modelPath) : readMiniGPTFile(opts.modelPath);

  const promptUsed = alignIdentityPrompt(opts.prompt);
  if (promptUsed !== opts.prompt) {
    console.log('[提示] 已对齐训练语料:', opts.prompt, '→', promptUsed);
  }

  const genOpts = {
    maxNewTokens: opts.maxNewTokens,
    temperature: opts.greedy ? 0 : opts.temperature,
    topK: opts.topK,
    topP: opts.topP,
    repetitionPenalty: opts.repetitionPenalty,
    seed: opts.seed,
    greedy: opts.greedy,
    yieldEachToken: opts.stream,
    onToken: opts.stream ? (ch) => process.stdout.write(ch) : null,
  };

  if (opts.stream) {
    console.log('提示:', promptUsed);
    process.stdout.write('续写: ');
  }

  const result = await generateContinuation(model, tok, promptUsed, genOpts);

  if (opts.stream) {
    process.stdout.write('\n');
    if (!opts.greedy) console.error('（seed', result.seedUsed, '，同提示同种子可复现）');
  } else {
    console.log('提示:', promptUsed);
    console.log('续写:', result.generated);
    if (!opts.greedy) console.log('（随机种子', result.seedUsed, '，同提示同种子可复现）');
  }
};

main().catch((e) => {
  console.error((e instanceof Error ? e.message : e) || String(e));
  process.exit(1);
});
