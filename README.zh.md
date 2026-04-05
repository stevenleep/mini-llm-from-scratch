# mini-llm-from-scratch

[![Node.js](https://img.shields.io/badge/node-%3E%3D18-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![无 npm 依赖](https://img.shields.io/badge/npm%20依赖-无-555555)]()

[English](README.md) | **简体中文**

本项目实现了一个**零第三方 npm 依赖**、**字符级**、**仅解码器**的类 GPT Transformer（`MiniGPT`），配套轻量自动微分与张量算子。支持从纯文本训练、以接近 Hugging Face 目录结构或单文件 JSON 导出权重、命令行续写，以及基于 **Server-Sent Events（SSE）** 的**本地**网页界面。全部计算在本地进程完成，无需外部推理 API。

---

## 价值与适用场景

| 面向人群 | 能带来的价值 |
| :--- | :--- |
| **学习者与讲师** | **端到端可读**：嵌入、注意力、LayerNorm、前馈、损失、反向传播、SGD/Adam、解码策略均在 JavaScript 中实现，可用调试器逐步跟踪；理解 Transformer 训练不必依赖不透明原生算子。 |
| **研究者与实验者** | **可复现的小型实验**：语料哈希（`corpus_sha256_16`）、固定随机种子、尾部验证集与追加写入的 **metrics CSV**，便于对比实验、写实验笔记或课程报告时给出**可核查依据**。 |
| **重视环境可移植性的开发者** | **核心路径零 npm 依赖**，仅需 **Node.js ≥ 18**。默认 CPU 训练/推理，无需先配 CUDA 或复杂 Python 环境即可在笔记本、机房或 CI 中跑通，适合教学与快速验证想法。 |
| **重视工具链与归档的用户** | **导出形态贴近常见习惯**：HF 风格目录（`config.json`、词表、`model.safetensors` 等）与**单文件 JSON**，便于 diff、备份或与现有脚本对接。 |
| **注重隐私与离线场景** | **默认不经过任何云端推理 API**：数据与计算留在本机；适合离线演示、内网环境或机构对数据出境有要求的场景（仍须遵守贵方合规要求）。 |

**明确不承诺的事项：** 本仓库**不以**对标商业大模型的流畅度、事实准确性或模型规模**为目标**，也**不**提供生产级推理服务（鉴权、弹性扩缩、运维监控等）。它是用于**学习与受控实验**的透明实现，**不能**直接替代成熟对话产品或基座模型服务。

---

## 目录

- [价值与适用场景](#价值与适用场景)
- [功能概览](#功能概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [npm 脚本](#npm-脚本)
- [配置说明](#配置说明)
- [可复现性](#可复现性)
- [仓库结构](#仓库结构)
- [路线图](#路线图)
- [局限与说明](#局限与说明)
- [贡献指南](#贡献指南)
- [安全说明](#安全说明)
- [许可证](#许可证)
- [引用](#引用)

---

## 功能概览

| 模块 | 说明 |
| :--- | :--- |
| **模型** | 词嵌入与位置嵌入、多层 Transformer 块（Pre-LN）、因果自注意力、前馈层、语言模型头；线性层可选 **LoRA** 低秩适配 |
| **自动微分** | 轻量 `Tensor` 与算子（如矩阵乘、softmax、交叉熵）；训练循环支持 **SGD** 与 **Adam** |
| **训练** | 随机滑窗、可选梯度裁剪、尾部验证集、指标 CSV；支持 **checkpoint**、**续训**、**余弦学习率**、**预热**、**早停** |
| **导出** | 常见 HF 风格目录（`config.json`、分词器与词表、`model.safetensors`、可选 `model.bin` + 清单）及**单文件 JSON** |
| **解码** | 温度、top-k、top-p、重复惩罚；贪心路径在 argmax 前施加重复惩罚 |
| **网页** | `public/` 静态资源；`POST /api/chat/stream` 以 SSE 流式返回 token |

完整行为清单（接口、环境变量、文件职责）见 **[docs/FEATURES.zh.md](docs/FEATURES.zh.md)**（[English](docs/FEATURES.md)）。

---

## 环境要求

- **Node.js ≥ 18**（ES 模块）
- **无需 `npm install`**：核心训练与推理仅使用 Node.js 标准库

可选：TensorFlow.js Node 相关包，用于 GPU 相关实验（**[docs/GPU.zh.md](docs/GPU.zh.md)**）。

---

## 快速开始

### 训练

默认语料：`data/corpus/playful_zh.txt`（`CORPUS_PATH` 可覆盖）。若存在 `data/corpus/identity_zh.txt`，会在主语料**之后**拼接（用于身份类提示等）。不需要时设置 `SKIP_IDENTITY_CORPUS=1`；自定义路径用 `IDENTITY_CORPUS_PATH`。

```bash
npm run train
```

**预设**（详见 `src/config.js`）：

| 命令 | 作用 |
| :--- | :--- |
| `npm run train:fun` | 相对默认更长的步数与上下文等 |
| `npm run train:mega` | 长跑预设（约 1 万步，`megaTrainingPreset`） |
| `npm run train:ultimate` | 更大模型（`ultimateTrainingPreset`），默认 **Adam + 余弦 + 预热**，默认 **5000** 步 |
| `npm run train:ultimate:long` | 同上但 **20000** 步并延长预热 |

可选：拉取混合中文语料至 `data/corpus/downloaded_mixed_zh.txt`（训练时默认合并，除非 `SKIP_DOWNLOAD_CORPUS=1`）：

```bash
npm run corpus:fetch
```

训练产物（checkpoint、导出、日志）默认写入 `out/`（一般由 `.gitignore` 忽略）。

### 命令行推理

默认流式输出；`--no-stream` 在结束时一次性输出。

```bash
node src/infer.js ./out/export/hf-style "你好" -n 40
node src/infer.js ./out/export/hf-style "你好" --no-stream
```

完整参数见 `node src/infer.js --help`。

### 网页界面

```bash
npm run ui
```

浏览器访问 **http://localhost:3847/**。默认加载 `out/export/hf-style`（`MODEL_PATH` 可覆盖）。结束占用 **3847** 端口的进程：

```bash
npm run stop:ui
```

---

## npm 脚本

| 脚本 | 说明 |
| :--- | :--- |
| `train` | `node src/train.js` |
| `train:fun` | `FUN_TRAIN=1 node src/train.js` |
| `train:mega` | `MEGA_TRAIN=1 node src/train.js` |
| `train:lora` | `LORA_RANK=8 LORA_ONLY=1 node src/train.js` |
| `train:ultimate` | Ultimate 预设 + Adam + 余弦 + 预热（默认 5000 步） |
| `train:ultimate:long` | Ultimate + `STEPS=20000` 与更长预热 |
| `corpus:fetch` | 下载并生成 `data/corpus/downloaded_mixed_zh.txt` |
| `export:all` | 自 checkpoint 多路径导出（见脚本注释） |
| `build` | 等同于 `train` |
| `infer` / `chat` | `node src/infer.js` |
| `ui` | `node src/chatServer.js` |
| `stop:ui` / `restart:ui` | 释放或重启 `3847` 端口 |
| `gpu:smoke` | 可选：安装 tfjs-node* 后的矩阵乘自检 |

---

## 配置说明

### 训练与导出

| 变量 | 作用 |
| :--- | :--- |
| `CORPUS_PATH` | 主语料路径（默认 `data/corpus/playful_zh.txt`） |
| `FUN_TRAIN` | `1` → 合并 `funTrainingPreset` |
| `MEGA_TRAIN` | `1` → 合并 `megaTrainingPreset`（与 `FUN_TRAIN` 同时设时优先本项） |
| `ULTIMATE_TRAIN` | `1` → 合并 `ultimateTrainingPreset`（优先于 MEGA / FUN） |
| `OPTIMIZER` | `adam` 为 Adam，否则 SGD |
| `COSINE_LR` | `1` → 按调度余弦衰减学习率 |
| `LR_WARMUP` | 正整数 → 线性预热后再余弦或常数 lr |
| `RESUME_FROM` / `LOAD_CHECKPOINT` | `.mgpt.json` 续训（须与词表与结构一致） |
| `STEP_OFFSET` | 已完成的**全局**步数（续训时对齐学习率） |
| `TOTAL_STEPS` | 可选；调度总长为 `max(本段步数+STEP_OFFSET, TOTAL_STEPS)` |
| `CHECKPOINT_EVERY` | 每 **N** 个全局步保存（`0` 关闭） |
| `CHECKPOINT_DIR` | checkpoint 目录（默认 `./out/checkpoints`） |
| `EARLY_STOP_PATIENCE` | 验证连续若干次日志未改善则停止（`0` 关闭）。**Adam 动量不会从 checkpoint 恢复** |
| `SKIP_DOWNLOAD_CORPUS` | `1` → 不合并下载混合语料 |
| `DOWNLOAD_CORPUS_PATH` | 混合语料路径 |
| `STEPS` | 覆盖训练总步数 |
| `SKIP_EXPORT` | `1` → 不写出导出 |
| `EXPORT_DIR` / `EXPORT_PATH` | HF 目录与单文件 JSON 路径 |
| `VERBOSE` | `1` → 详细日志 |
| `GRAD_CLIP` | 全局梯度裁剪（默认 `1`；`0` 关闭） |
| `VAL_FRACTION` | 尾部验证比例（默认 `0.08`） |
| `VAL_SAMPLES` | 每次日志在验证集上采样窗口数（默认 `24`） |
| `METRICS_CSV` / `SKIP_METRICS` | 指标 CSV 与开关 |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_ONLY` | LoRA（见 `src/config.js` 与 `src/train.js` 注释） |

### 推理与 UI

| 变量 | 作用 |
| :--- | :--- |
| `MODEL_PATH` | `chatServer.js` 加载的目录或 `.json` |
| `PORT` | HTTP 端口（默认 `3847`） |

---

## 可复现性

训练开始会打印 **`[repro]`** 行，包含：

- **`corpus_sha256_16`**：读入内存后的 UTF-8 语料字符串的 SHA-256 前 16 个十六进制字符。
- **`seed`、`steps`、`seqLen`、`val_fraction`**：本次有效超参。

**训练脚本不调用 Git。** 论文或实验笔记中请自行记录 `git rev-parse --short HEAD`（或 CI 中的 `CI_COMMIT_SHA`），并与 `train_metrics.csv` 一并归档。

验证使用 token 序列的**尾部留出**，且验证窗口采样 RNG **固定**，因而在相同代码、语料字节与环境变量下，`val_loss` / `val_ppl` 可重复。

---

## 仓库结构

```
LICENSE            MIT 许可证全文
CONTRIBUTING.md    贡献说明
SECURITY.md        本地演示服务的安全预期
docs/              扩展文档（功能清单、GPU 说明等）
scripts/           辅助脚本（语料拉取、导出、可选 GPU 自检）
src/
  config.js        超参数预设
  train.js         训练、优化器、导出
  infer.js         命令行
  chatServer.js    静态页与 HTTP/SSE
  generate.js      解码与续写
  model/ nn/ tensor/ io/ data/
data/corpus/       示例与可选语料（UTF-8 文本）
public/            前端静态资源
out/               训练输出（默认 gitignore）
```

---

## 路线图

以 GitHub 任务列表跟踪；顺序不代表优先级。

- [ ] 架构说明短文：张量前向/反向数据流，指向 `train.js`、`tensor/ops.js`
- [ ] 可选子词分词路径；字符级仍为默认教学路径
- [x] 尾部验证、`val_loss` / `val_ppl`、指标 CSV
- [ ] 自动化测试：核心算子、分词往返、形状不变量
- [ ] 在保持可读前提下探索 WASM 等加速
- [ ] UI：默认导出路径缺失时的更清晰报错
- [ ] 仅推理分发所需最小文件说明

---

## 局限与说明

- **规模与数据：** 小模型与有限语料易产生重复或不通顺输出；解码技巧只能缓解，无法替代模型与数据规模。
- **字符级建模：** Unicode 场景下词表较大，长程结构通常弱于子词模型。
- **CPU 训练：** 核心实现为纯 JS 张量栈；GPU 路径需 TensorFlow.js 等（**[docs/GPU.zh.md](docs/GPU.zh.md)**）。
- **本地演示服务：** 内置 HTTP 服务仅供**本机或可信环境**试用（**[SECURITY.md](SECURITY.md)**）。

---

## 贡献指南

请参阅 **[CONTRIBUTING.md](CONTRIBUTING.md)**。

---

## 安全说明

本地 Web 服务的预期与风险提示见 **[SECURITY.md](SECURITY.md)**。

---

## 许可证

本项目以 **MIT License** 发布，全文见 **[LICENSE](LICENSE)**。

---

## 引用

若用于教学或研究，注明仓库 URL 与提交哈希即可；无强制引用格式。
