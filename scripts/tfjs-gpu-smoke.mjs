/**
 * 用 TensorFlow.js 做一次矩阵乘，打印当前后端名称。
 * 用于在 **JavaScript** 栈上验证 TF 是否可用（GPU / Node 原生绑定 / 纯 JS CPU）。
 *
 * 安装（任选其一；勿同时安装 tfjs-node-gpu 与 tfjs-node）：
 *   npm install @tensorflow/tfjs-node-gpu   # NVIDIA + CUDA，可能用 GPU
 *   npm install @tensorflow/tfjs-node       # Node 原生绑定（需编译成功）
 *   npm install @tensorflow/tfjs            # 纯 JS，无原生模块，最易安装
 *
 * 运行：
 *   node scripts/tfjs-gpu-smoke.mjs
 *   SIZE=2048 node scripts/tfjs-gpu-smoke.mjs
 */

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const loadTf = () => {
  const candidates = [
    ['@tensorflow/tfjs-node-gpu', 'tfjs-node-gpu'],
    ['@tensorflow/tfjs-node', 'tfjs-node'],
    ['@tensorflow/tfjs', 'tfjs (pure JS)'],
  ];
  for (const [moduleName, label] of candidates) {
    try {
      return { tf: require(moduleName), pkg: label };
    } catch {
      /* try next */
    }
  }
  return null;
};

const main = async () => {
  const loaded = loadTf();
  if (!loaded) {
    console.error('未找到 TensorFlow.js。请在项目根目录执行其一：');
    console.error('  npm install @tensorflow/tfjs-node-gpu   # GPU（CUDA）');
    console.error('  npm install @tensorflow/tfjs-node       # Node 原生');
    console.error('  npm install @tensorflow/tfjs            # 纯 JS（无需编译）');
    process.exit(1);
  }

  const { tf, pkg } = loaded;
  await tf.ready();

  const backend = tf.getBackend();
  console.log('已加载:', pkg);
  console.log('当前后端:', backend);
  if (pkg === 'tfjs-node-gpu') {
    console.log('说明: 后端为 tensorflow 且环境正确时可用 GPU；否则检查 CUDA/cuDNN。');
  }

  const size = Math.min(4096, Math.max(64, parseInt(process.env.SIZE ?? '1024', 10) || 1024));
  const a = tf.randomNormal([size, size]);
  const b = tf.randomNormal([size, size]);

  let warm = tf.matMul(a, b);
  await warm.data();
  warm.dispose();

  const t0 = performance.now();
  const c = tf.matMul(a, b);
  await c.data();
  const ms = performance.now() - t0;
  console.log(`基准: matMul(${size},${size}) x (${size},${size}) ≈ ${ms.toFixed(2)} ms`);

  a.dispose();
  b.dispose();
  c.dispose();
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
