# mini-llm-from-scratch

[English](README.md) | **中文**

用 **纯 JavaScript** 实现的**零依赖**、**字符级**、类 GPT **仅解码器** Transformer。支持本地训练、权重导出、命令行续写与带 **SSE 流式** 的小型网页界面；计算全部在本机完成，不依赖外部推理服务。

| | |
| :--- | :--- |
| **运行环境** | Node.js 18+（ES 模块），仅标准库 |
| **定位** | 教学与科研原型；**不是**生产级对话或事实问答系统 |
| **用途** | 理解小型 Transformer 的前向/反向与续写解码；可改配置、换语料对比现象。**不可**与商用大模型对标 |

---

## 功能概览

**已实现行为的完整清单**（接口、环境变量、文件职责）：**[docs/FEATURES.zh.md](docs/FEATURES.zh.md)**（[English](docs/FEATURES.md)）。

| 模块 | 说明 |
| :--- | :--- |
| 模型 | `MiniGPT`：词嵌入 + 位置嵌入、多层 Transformer 块、语言模型头 |
| 自动微分 | 轻量张量与算子（矩阵乘、softmax、交叉熵、SGD） |
| 训练 | 文本上随机滑窗；可选梯度裁剪；词表由语料构建 |
| 导出 | HF 风格目录（`config.json`、词表、`model.safetensors`、可选 `model.bin` + 清单）及单文件 JSON |
| 解码 | 温度、top-k、top-p、重复惩罚；贪心路径在 argmax 前施加重复惩罚 |
| 网页 | `public/` 静态资源；`POST /api/chat/stream`（SSE）；异步生成并按 token 让出事件循环以实现真实流式 |

---

## 环境要求

- Node.js **≥ 18**
- 无需 `npm install`，无第三方依赖

---

## 快速开始

### 训练

默认语料：`data/corpus/playful_zh.txt`（可用 `CORPUS_PATH` 覆盖）。

```bash
npm run train
```

加长训练预设（更多步数、更长上下文、学习率等见 `src/config.js`）：

```bash
npm run train:fun
```

产物写入 `out/`（默认被 Git 忽略）。

### 命令行推理

默认流式输出到标准输出；`--no-stream` 为整段结束后一次性打印。

```bash
node src/infer.js ./out/export/hf-style "你好" -n 40
node src/infer.js ./out/export/hf-style "你好" --no-stream
```

解码选项见 `node src/infer.js --help`。

### 网页界面

```bash
npm run ui
```

浏览器打开 `http://localhost:3847/`。默认加载 `out/export/hf-style`（可用 `MODEL_PATH` 覆盖）。

结束占用 `3847` 端口的进程：

```bash
npm run stop:ui
```

---

## npm 脚本

| 脚本 | 说明 |
| :--- | :--- |
| `train` | `node src/train.js` |
| `train:fun` | `FUN_TRAIN=1 node src/train.js` |
| `build` | 与 `train` 相同 |
| `infer` / `chat` | `node src/infer.js`（传入模型路径与参数） |
| `ui` | `node src/chatServer.js` |
| `stop:ui` | 结束监听 `3847` 端口的进程 |
| `restart:ui` | 先 `stop:ui` 再启动 `chatServer` |

---

## 配置（环境变量）

### 训练与导出

| 变量 | 作用 |
| :--- | :--- |
| `CORPUS_PATH` | 语料文件路径（默认 `data/corpus/playful_zh.txt`） |
| `FUN_TRAIN` | 设为 `1` 时合并 `funTrainingPreset`（`src/config.js`） |
| `STEPS` | 正整数，覆盖训练总步数 |
| `SKIP_EXPORT` | `1` 时不写出导出文件 |
| `EXPORT_DIR` | HF 风格导出目录（默认 `out/export/hf-style` 一带） |
| `EXPORT_PATH` | 单文件 JSON 路径 |
| `VERBOSE` | `1` 时打印详细训练日志 |
| `GRAD_CLIP` | 梯度裁剪上界（默认 `1`；`0` 关闭） |
| `VAL_FRACTION` | 语料**尾部**留出作验证的比例（默认 `0.08`；语料过短则关闭验证） |
| `VAL_SAMPLES` | 每次打日志时在验证段上采样的窗口数（默认 `24`；固定种子以利复现 `val_loss` / `val_ppl`） |
| `METRICS_CSV` | 指标 CSV 路径（默认 `./out/train_metrics.csv`） |
| `SKIP_METRICS` | `1` 时不写指标 CSV |

### 推理与 UI

| 变量 | 作用 |
| :--- | :--- |
| `MODEL_PATH` | `chatServer.js` 加载的模型目录或 `.json`（默认 `out/export/hf-style`） |
| `PORT` | HTTP 端口（默认 `3847`） |

---

## 可复现性（学术向小模型）

训练开始会打印 **`[repro]`** 行，包含：

- **`corpus_sha256_16`**：读入内存后的语料字符串（UTF-8）的 SHA-256 前 16 个十六进制字符；需相同文件内容与 `CORPUS_PATH` 才能对齐。
- **`seed` / `steps` / `seqLen` / `val_fraction`**：本次运行的有效超参。

**训练脚本不会调用 Git**；若需在论文或实验笔记中标注代码版本，请自行在终端执行 `git rev-parse --short HEAD`（或 CI 中的 `CI_COMMIT_SHA`），并与保存的 `train_metrics.csv` 一并记录。

与控制台同频率向 **`METRICS_CSV`** 追加行，列为 `step`, `train_loss`, `val_loss`, `val_ppl`。验证使用语料**尾部留出**的 token 序列，且验证窗口采样 RNG **固定**，因而在相同代码、语料与环境下，各步的 `val_*` 可重复。

复现他人实验时建议记录：Node 版本、Git 提交（手动）、语料文件及其哈希、完整环境变量、`train_metrics.csv` 归档。

---

## 仓库结构

```
docs/
  FEATURES.md      已实现功能（英文）
  FEATURES.zh.md   已实现功能（中文）
  GPU.md           JS 与 GPU（英文）
  GPU.zh.md        JS 与 GPU（中文）
README.md          本说明（英文）
README.zh.md       本说明（中文）
src/
  config.js        超参数
  train.js         训练与导出
  infer.js         命令行
  chatServer.js    静态页与 HTTP/SSE
  generate.js      续写与解码
  model/ nn/ tensor/ io/ data/
data/corpus/       文本语料
public/            浏览器前端
out/               训练输出（默认 gitignore）
```

---

## 路线图

以下使用 GitHub 任务列表语法，可在 PR 中勾选。顺序不代表优先级。

- [ ] **架构说明** — 张量前向/反向数据流短文，指向 `train.js`、`tensor/ops.js`。
- [ ] **子词分词** — 可选 BPE / SentencePiece 风格路径；字符级仍为默认教学路径。
- [x] **评估** — 尾部留出验证集、`val_loss` / `val_ppl`、指标 CSV（见「可复现性」）。
- [ ] **自动化测试** — 核心算子数值、分词往返、前向形状等。
- [ ] **训练** — 学习率调度；在保持可读前提下探索 WASM 等加速。
- [ ] **网页** — 开发时可选模型路径；默认导出路径缺失时的更清晰报错。
- [ ] **分发** — 仅推理所需最小文件说明；可选 `npm pack` 流程。

---

## 局限

- 模型小、数据少时，输出易重复或不通顺；解码技巧只能缓解，不能替代规模。
- 字符级建模在 Unicode 场景下词表膨胀，长程结构弱于子词模型。
- 自带 HTTP 服务仅供本机实验（无鉴权、未做公网加固）。
- **训练仅在 CPU 上运行**（自实现 JS 张量）。若要在 JS 里用 GPU，通常需 TensorFlow.js + CUDA（`tfjs-node-gpu`）或重写算子；说明见 **[docs/GPU.zh.md](docs/GPU.zh.md)**（[English](docs/GPU.md)）。

---

## 许可

仓库未附带 `LICENSE` 文件；`package.json` 中 `"private": true`。再分发前请自行添加许可条款。
