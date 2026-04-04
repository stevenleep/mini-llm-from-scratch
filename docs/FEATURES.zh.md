# 已实现功能说明

[English](FEATURES.md) | **中文**

本文档列出本仓库**当前已实现**的行为、接口与主要源码位置，便于对照代码与撰写实验记录。**未**列入项视为尚未实现或仅存在于 Roadmap。

---

## 1. 核心栈

| 模块 | 实现要点 | 主要源码 |
| :--- | :--- | :--- |
| 运行环境 | Node.js ≥ 18、ES 模块、**无 npm 依赖** | `package.json` |
| 分词 | 由训练语料构建**字符级**词表 | `src/data/charTokenizer.js`、`src/data/loadCorpus.js` |
| 张量 / 自动微分 | 二维张量、SGD、由标量 loss 调用 `backward()` | `src/tensor/Tensor.js`、`src/tensor/ops.js` |
| 模型 | 仅解码器 Transformer：嵌入、因果自注意力块、FFN、语言模型头 | `src/model/MiniGPT.js`、`src/nn/*.js` |

---

## 2. 训练（`src/train.js`）

| 功能 | 行为说明 |
| :--- | :--- |
| 训练目标 | 下一字符预测：`input` 长度 `T`，`target` 相对输入错位一位 |
| 优化 | 对 `crossEntropyMean` 的结果做 `backward()`，再对 `MiniGPT` 参数做 SGD |
| 数据采样 | 仅在**训练**段上随机起点（`mulberry32(cfg.seed + 999)`） |
| 梯度裁剪 | 全局 L2 范数裁剪（默认 `1`）；`GRAD_CLIP=0` 关闭 |
| 验证集 | **尾部留出**：语料 token 流末尾 `VAL_FRACTION`（默认 `0.08`）；在验证段上采 `VAL_SAMPLES`（默认 `24`）个窗口计算 `val_loss`、`val_ppl`；验证采样 RNG 固定 |
| 日志 | 周期性输出 `train_loss` 及可选 `val_*`；`[repro]` 行含 `corpus_sha256_16`、`seed`、`steps`、`seqLen`、`val_fraction` |
| 指标文件 | 向 `METRICS_CSV`（默认 `./out/train_metrics.csv`）追加 CSV；`SKIP_METRICS=1` 不写 |
| 导出 | 除非 `SKIP_EXPORT=1`：HF 风格目录 + 单文件 JSON 整包 | `src/io/hfModelDir.js`、`src/io/miniGptIO.js` |
| 配置预设 | `src/config.js` 中 `defaultConfig`；`FUN_TRAIN=1` 时合并 `funTrainingPreset` |

---

## 3. 导出产物

| 格式 | 内容 |
| :--- | :--- |
| HF 风格目录 | `config.json`、分词器相关 JSON、`model.safetensors`、可选 `model.bin` + 清单 | 由 `EXPORT_DIR` 指定 |
| 单文件 JSON | 模型 + 分词器合一 | 由 `EXPORT_PATH` 指定 |

---

## 4. 推理与解码（`src/generate.js`、`src/infer.js`）

| 功能 | 说明 |
| :--- | :--- |
| 续写 | `generateContinuation`：自回归至多 `maxNewTokens` 步 |
| 采样 | 温度、top-k、top-p（核采样）、对最近窗口的重复惩罚 |
| 贪心 | 在**重复惩罚之后**再 argmax，减轻句号等符号死循环 |
| 流式（CLI） | 默认启用流式（`yieldEachToken`）；`--no-stream` 为整段输出 |
| 命令行 | `node src/infer.js <模型路径> [提示] [选项]`，详见 `--help` |

---

## 5. HTTP 服务（`src/chatServer.js`）

| 方法 / 路径 | 用途 |
| :--- | :--- |
| `GET /` | 返回 `public/index.html` |
| `GET /app.js`、`GET /styles.css` | 静态前端资源 |
| `POST /api/chat` | JSON 请求体 → 一次性返回整段续写 |
| `POST /api/chat/stream` | **SSE**：`data: {token}`，结束 `{done, seedUsed, …}`；错误 `{error}`；异步生成且按字让出事件循环 |
| 环境变量 | `MODEL_PATH`、`PORT`（默认 `3847`） |

**浏览器界面**（`public/`）：提示与解码参数表单；续写**仅**通过 `/api/chat/stream`（fetch + SSE 解析）。

---

## 6. 可复现性（训练侧）

- **语料指纹**：对读入内存后的 UTF-8 文本做 SHA-256，取十六进制前 16 位（不是仅路径）。
- **验证指标**：相同代码、语料与环境下，各日志步的 `val_loss` / `val_ppl` 可复现（验证 RNG 固定）。
- **`train.js` 不调用 Git**；若论文或实验记录需要 commit，请在终端自行执行 `git rev-parse --short HEAD` 等并一并保存。

---

## 7. 本仓库刻意不包含

- 子词 / BPE 分词（见 Roadmap）。
- Adam、LayerNorm、多卡、混合精度等（Roadmap 或注释层面）。
- **内置 `Tensor` 运行时的 GPU 执行**（当前仅 CPU）；若要坚持 JS 又想试 GPU，可先跑 **`npm run gpu:smoke`**（需另装 TensorFlow.js Node，见 [GPU.zh.md](GPU.zh.md)），与主训练无关。
- 面向公网的生产级 API 安全（当前仅本地实验）。

后续计划见 [README § Roadmap](../README.md#roadmap) 或 [README.zh § 路线图](../README.zh.md#路线图)。
