# JavaScript 与 GPU 加速

[English](GPU.md)

本仓库的**训练循环**（`src/train.js`）使用的是 **CPU 上的自实现张量**（`src/tensor/*.js`）。**当前代码里没有 GPU 执行路径**。

若要在**仍使用 JavaScript**的前提下用 GPU 训练，一般需要接入能把重计算交给本机 GPU 库的运行时：

| 方案 | 运行环境 | 说明 |
| :--- | :--- | :--- |
| **`@tensorflow/tfjs-node-gpu`** | Node.js + NVIDIA CUDA/cuDNN | 较成熟；绑定 TensorFlow C API；**必须**安装匹配的 CUDA/cuDNN，常见组合是 Linux + NVIDIA |
| **`@tensorflow/tfjs-node`** | Node CPU | API 类似，无 GPU |
| **WebGPU 后端**（`@tensorflow/tfjs-backend-webgpu`） | 浏览器（或实验性 Node WebGPU） | 适合演示；大模型训练能力通常仍弱于桌面 CUDA |
| **自写 WebGPU / WASM 算子** | 自定义 | 自由度最高；工程量大（矩阵乘、反向等都要对齐） |

**为什么不能在本仓库里「加一个开关就 GPU」：** 现在的 `Tensor`、自动微分和各类算子都是基于 `Float32Array` 的 JS 实现。要上 GPU，通常只有两类路：

1. 用 **TensorFlow.js**（等）**重写**模型与训练步，把前向/反向交给框架；或  
2. 只把**底层算子**（至少 large matmul）换成 GPU 核函数，并保证与现有 `backward` 一致——相当于做一套**新后端**，工作量仍很大。

**若目标是「严肃 GPU 训练」：** 常见做法是 **PyTorch / JAX** 在 GPU 上训，本仓库当作**结构参考**；或按同样超参在 **TensorFlow.js + tfjs-node-gpu** 里重实现 `MiniGPT`。

**若目标是「验证 JS 能调用 GPU」：** 可单独写一个**小脚本**，只检测 `tfjs-node-gpu` 能否加载并在 GPU 上跑一次 `matmul`，再逐步迁移各层——**不要**指望在不大改架构的情况下，给现有 `Tensor.js` 逐行打补丁就完成。

更多英文说明见 [GPU.md](GPU.md)。

---

## 本仓库自带的 JS 自检（TensorFlow.js）

核心训练仍为零依赖；若要**在 JS 里验证**本机能否用 TF Node 后端（GPU 或 CPU 绑定），可使用：

```bash
# 任选其一（勿同时装 tfjs-node-gpu 与 tfjs-node）
npm install @tensorflow/tfjs-node-gpu   # NVIDIA + CUDA，可能用 GPU
# 或
npm install @tensorflow/tfjs-node       # Node 原生（需本地编译成功）
# 或（最易成功，纯 JS、较慢）
npm install @tensorflow/tfjs

npm run gpu:smoke
# 可选：SIZE=2048 npm run gpu:smoke
```

脚本路径：`scripts/tfjs-gpu-smoke.mjs`。按顺序尝试 **gpu → node → 纯 tfjs**，打印 **backend** 并对随机矩阵做 `matMul` 计时。  
纯 JS 包会多一行 TensorFlow 提示，可忽略；**Linux + NVIDIA + CUDA** 下用 `tfjs-node-gpu` 才有机会走 GPU。
