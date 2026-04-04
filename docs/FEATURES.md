# Implemented features

**English** | [中文版](FEATURES.zh.md)

**中文说明：** 本文档列出仓库**当前已实现**的行为与入口，便于对照代码与写实验记录。未列入项视为未实现或仅存在于 Roadmap。

---

## 1. Core stack

| Area | Implementation | Source (primary) |
| :--- | :--- | :--- |
| Runtime | Node.js ≥ 18, ES modules, **no npm dependencies** | `package.json` |
| Tokenization | **Character-level** vocabulary from training corpus | `src/data/charTokenizer.js`, `src/data/loadCorpus.js` |
| Tensor / autograd | 2D tensors, SGD, `backward()` from scalar loss | `src/tensor/Tensor.js`, `src/tensor/ops.js` |
| Model | Decoder-only Transformer: embeddings, causal self-attention blocks, FFN, LM head | `src/model/MiniGPT.js`, `src/nn/*.js` |

---

## 2. Training (`src/train.js`)

| Feature | Behavior |
| :--- | :--- |
| Objective | Next-character prediction: `input` length `T`, `target` shifted by one |
| Optimization | SGD on `MiniGPT` parameters after `crossEntropyMean` + `backward()` |
| Data sampling | Random start positions on **training** slice only (`mulberry32(cfg.seed + 999)`) |
| Gradient clipping | Global L2 clip (default `1`); `GRAD_CLIP=0` disables |
| Validation | **Tail hold-out**: last `VAL_FRACTION` (default `0.08`) of token stream; `val_loss`, `val_ppl` on `VAL_SAMPLES` (default `24`) windows; fixed RNG seed for val sampler |
| Logging | Periodic `train_loss` + optional `val_*`; `[repro]` line: `corpus_sha256_16`, `seed`, `steps`, `seqLen`, `val_fraction` |
| Metrics file | Append CSV to `METRICS_CSV` (default `./out/train_metrics.csv`); `SKIP_METRICS=1` off |
| Export | Unless `SKIP_EXPORT=1`: HF-style dir + single JSON bundle | `src/io/hfModelDir.js`, `src/io/miniGptIO.js` |
| Config presets | `src/config.js`: `defaultConfig`; `FUN_TRAIN=1` merges `funTrainingPreset` |

---

## 3. Export artifacts

| Format | Contents |
| :--- | :--- |
| HF-style directory | `config.json`, tokenizer JSONs, `model.safetensors`, optional `model.bin` + manifest | `EXPORT_DIR` |
| Single JSON | Full model + tokenizer in one file | `EXPORT_PATH` |

---

## 4. Inference & decoding (`src/generate.js`, `src/infer.js`)

| Feature | Notes |
| :--- | :--- |
| Continuation | `generateContinuation`: autoregressive steps up to `maxNewTokens` |
| Sampling | Temperature, top-k, top-p (nucleus), repetition penalty over recent window |
| Greedy | Argmax after **repetition penalty** (avoids punctuation / token loops) |
| Streaming (CLI) | Default: `--stream` behavior via `yieldEachToken`; `--no-stream` for batch print |
| CLI | `node src/infer.js <model> [prompt] [options]` — see `--help` |

---

## 5. HTTP server (`src/chatServer.js`)

| Method / path | Purpose |
| :--- | :--- |
| `GET /` | Serves `public/index.html` |
| `GET /app.js`, `GET /styles.css` | Static UI assets |
| `POST /api/chat` | JSON body → full continuation in one response |
| `POST /api/chat/stream` | **SSE**: `data: {token}`, then `{done, seedUsed, …}`; errors as `{error}`; async generation with per-token yields |
| Env | `MODEL_PATH`, `PORT` (default `3847`) |

**Browser UI** (`public/`): form for prompt and decoding knobs; uses **only** `/api/chat/stream` for generation (fetch + SSE reader).

---

## 6. Reproducibility hooks (training)

- Corpus fingerprint: first 16 hex chars of SHA-256 of loaded UTF-8 string (not the file path alone).
- Validation metrics: same code + corpus + env → same `val_loss` / `val_ppl` at each logged step (fixed val RNG).
- Git is **not** invoked by `train.js`; record commit hash manually if needed.

---

## 7. Out of scope (by design here)

- Subword / BPE tokenizer (roadmap).
- Adam, LayerNorm, multi-GPU, mixed precision (roadmap or comments only).
- **GPU execution** for the built-in `Tensor` runtime (CPU-only). Optional **`npm run gpu:smoke`** (TensorFlow.js Node, separate install) verifies JS+GPU/CPU bindings; see [GPU.md](GPU.md).
- Production API security (local experimentation only).

For planned work, see [README.md § Roadmap](../README.md#roadmap).
