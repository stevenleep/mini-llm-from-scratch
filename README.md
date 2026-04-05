# mini-llm-from-scratch

**English** | [简体中文](README.zh.md)

Minimal, **zero-dependency** implementation of a character-level decoder-only Transformer (GPT-style) in JavaScript. Training, weight export, CLI decoding, and a local web UI with Server-Sent Events streaming are supported; all computation runs on the host machine without external services.

| | |
| :--- | :--- |
| **Runtime** | Node.js 18+ (ES modules), standard library only |
| **Scope** | Education and research prototyping; not intended for production chat or factual QA |
| **用途** | 教学与实验：理解小型 Transformer 的前向/反向与续写解码；可通过配置与语料对比行为。**非**商用对话或基座模型替代品 |

---

## Features

**Full inventory of current behavior** (endpoints, env vars, file roles): **[docs/FEATURES.md](docs/FEATURES.md)** · [中文版 / Chinese](docs/FEATURES.zh.md)

| Component | Description |
| :--- | :--- |
| Model | `MiniGPT`: token and position embeddings, stacked Transformer blocks, language-model head |
| Autograd | Lightweight tensors and ops (matrix multiply, softmax, cross-entropy, SGD) |
| Training | Random sliding windows over text; optional gradient clipping; character vocabulary from corpus |
| Export | Hugging Face–style directory (`config.json`, tokenizer files, `model.safetensors`, optional `model.bin` + manifest) and single-file JSON bundle |
| Decoding | Temperature, top-k, top-p (nucleus), repetition penalty; greedy decoding applies repetition penalty before argmax |
| HTTP UI | Static assets under `public/`; `POST /api/chat/stream` (SSE); incremental delivery via async generation and per-token scheduling |

---

## Requirements

- Node.js **≥ 18**
- No `npm install`; dependencies are not used

---

## Getting started

### Train

Default corpus: `data/corpus/playful_zh.txt` (override with `CORPUS_PATH`). If `data/corpus/identity_zh.txt` exists, its text is **appended** after the main corpus (optional identity Q&A patterns, e.g. “who are you” → “我是李小烨”). Disable with `SKIP_IDENTITY_CORPUS=1`; custom path with `IDENTITY_CORPUS_PATH`.

```bash
npm run train
```

Extended preset (more steps, longer context, tuned learning rate):

```bash
npm run train:fun
```

Very long run (default ~10k steps; see `megaTrainingPreset` in `src/config.js`):

```bash
npm run train:mega
```

**Maximum-effort preset** (larger model, longer context, default **5000** steps in `ultimateTrainingPreset`; defaults to **Adam + cosine LR**; slow and larger exports):

```bash
npm run train:ultimate
```

Equivalent to `ULTIMATE_TRAIN=1 OPTIMIZER=adam COSINE_LR=1 LR_WARMUP=500 node src/train.js` (warmup then cosine). For a longer run: `STEPS=20000 npm run train:ultimate`. Tune `LR_WARMUP` or `OPTIMIZER` / `COSINE_LR` as needed.

Download public Chinese text and merge into `data/corpus/downloaded_mixed_zh.txt` (auto-appended when training unless `SKIP_DOWNLOAD_CORPUS=1`):

```bash
npm run corpus:fetch
```

Checkpoints and exports are written under `out/` (ignored by Git by default).

### CLI inference

Streams tokens to stdout by default; use `--no-stream` for a single block at the end.

```bash
node src/infer.js ./out/export/hf-style "你好" -n 40
node src/infer.js ./out/export/hf-style "你好" --no-stream
```

Run `node src/infer.js --help` for decoding options.

### Web UI

```bash
npm run ui
```

Open `http://localhost:3847/`. Default model path: `out/export/hf-style` (override with `MODEL_PATH`).

Stop the server bound to port `3847`:

```bash
npm run stop:ui
```

---

## npm scripts

| Script | Description |
| :--- | :--- |
| `train` | `node src/train.js` |
| `train:fun` | `FUN_TRAIN=1 node src/train.js` |
| `train:mega` | `MEGA_TRAIN=1 node src/train.js` (very long preset) |
| `train:ultimate` | `ULTIMATE_TRAIN=1` + Adam + cosine LR (default 5000 steps) |
| `train:ultimate:long` | Same with `STEPS=20000` and `LR_WARMUP=2000` (long run) |
| `corpus:fetch` | Fetch web sources → `data/corpus/downloaded_mixed_zh.txt` |
| `export:all` | Re-export checkpoints to multiple paths (see script) |
| `build` | Alias for `train` |
| `infer` / `chat` | `node src/infer.js` (pass model path and flags) |
| `ui` | `node src/chatServer.js` |
| `stop:ui` | Terminate process listening on port `3847` |
| `restart:ui` | `stop:ui` then start `chatServer` |
| `gpu:smoke` | Optional: after installing `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu`, runs `scripts/tfjs-gpu-smoke.mjs` (see [docs/GPU.md](docs/GPU.md)) |

---

## Configuration

### Training and export

| Variable | Effect |
| :--- | :--- |
| `CORPUS_PATH` | Corpus file path (default: `data/corpus/playful_zh.txt`) |
| `FUN_TRAIN` | Set to `1` to apply `funTrainingPreset` in `src/config.js` |
| `MEGA_TRAIN` | Set to `1` to apply `megaTrainingPreset` (more steps; wins over `FUN_TRAIN` if both set) |
| `ULTIMATE_TRAIN` | Set to `1` to apply `ultimateTrainingPreset` (large model; wins over MEGA / FUN) |
| `OPTIMIZER` | `adam` for Adam; otherwise SGD |
| `COSINE_LR` | `1` for cosine decay of learning rate over the run |
| `LR_WARMUP` | If set to a positive integer, linearly ramp LR from 0 to the configured peak over that many steps first (then cosine if `COSINE_LR=1`, else constant) |
| `RESUME_FROM` / `LOAD_CHECKPOINT` | Path to a `.mgpt.json` checkpoint to continue training (must match current corpus vocabulary and architecture) |
| `STEP_OFFSET` | Global steps already completed before this run (for LR schedule when resuming; default `0`) |
| `TOTAL_STEPS` | Optional; LR schedule length is `max(STEPS + STEP_OFFSET, TOTAL_STEPS)` when `TOTAL_STEPS` is set |
| `CHECKPOINT_EVERY` | If set to a positive integer, write `checkpoint-step-*.mgpt.json` and `latest.mgpt.json` under `CHECKPOINT_DIR` every that many **global** steps; `0` disables |
| `CHECKPOINT_DIR` | Checkpoint directory (default `./out/checkpoints`) |
| `EARLY_STOP_PATIENCE` | With a validation set, stop early after this many **log intervals** without `val_loss` improvement; `0` disables (Adam state is not restored from checkpoints) |
| `SKIP_DOWNLOAD_CORPUS` | Set to `1` to skip merging `data/corpus/downloaded_mixed_zh.txt` |
| `DOWNLOAD_CORPUS_PATH` | Path to downloaded mix (default `data/corpus/downloaded_mixed_zh.txt`) |
| `STEPS` | Positive integer; overrides training step count |
| `SKIP_EXPORT` | Set to `1` to skip writing export artifacts |
| `EXPORT_DIR` | HF-style export directory (default under `out/export/hf-style`) |
| `EXPORT_PATH` | Single JSON bundle path |
| `VERBOSE` | Set to `1` for verbose training logs |
| `GRAD_CLIP` | Global norm clip (default `1`; `0` disables clipping) |
| `VAL_FRACTION` | Fraction of corpus tokens held out at the **tail** for validation (default `0.08`; disabled if the corpus is too short) |
| `VAL_SAMPLES` | Number of random windows sampled on the validation slice per log (default `24`; RNG seed fixed for reproducible `val_loss` / `val_ppl`) |
| `METRICS_CSV` | Append-only CSV path for metrics (default `./out/train_metrics.csv`) |
| `SKIP_METRICS` | Set to `1` to disable writing the metrics CSV |

### Inference and UI

| Variable | Effect |
| :--- | :--- |
| `MODEL_PATH` | Model directory or `.json` file for `chatServer.js` (default: `out/export/hf-style`) |
| `PORT` | HTTP port for `chatServer.js` (default: `3847`) |

---

## Reproducibility (academic small-model runs)

Training prints a **`[repro]`** line with:

- **`corpus_sha256_16`**: first 16 hex chars of SHA-256 over the **loaded** corpus string (after UTF-8 read); use the same file bytes and `CORPUS_PATH` to match.
- **`seed` / `steps` / `seqLen` / `val_fraction`**: effective hyperparameters for that run.

The trainer does **not** invoke Git (no dependency on `.git` or the `git` binary). To attach a source revision to a paper or lab note, run `git rev-parse --short HEAD` yourself (or read `CI_COMMIT_SHA` in CI) and write it next to the saved `train_metrics.csv`.

Metrics rows (same intervals as console logs) are appended to **`METRICS_CSV`** with columns `step`, `train_loss`, `val_loss`, `val_ppl`. Validation uses a **tail hold-out** of the token sequence and a **fixed** RNG seed for the validation window sampler so repeated runs with the same code, corpus, and config yield the same `val_*` numbers at each step.

To replicate a reported experiment: record Node.js version, Git commit (manually), corpus file and its hash, full env (or export `env | sort`), and archive `train_metrics.csv` next to the paper or note.

---

## Repository layout

```
docs/
  FEATURES.md     Implemented features (EN)
  FEATURES.zh.md  Implemented features (中文)
  GPU.md          JS + GPU training options (EN)
  GPU.zh.md       JS + GPU 说明（中文）
scripts/
  tfjs-gpu-smoke.mjs   Optional TF.js matmul smoke test (after npm install tfjs-node*)
README.zh.md      This readme (简体中文)
src/
  config.js       Hyperparameters
  train.js        Training loop and export
  infer.js        Command-line interface
  chatServer.js   Static UI and REST/SSE endpoints
  generate.js     Continuation and decoding
  model/ nn/ tensor/ io/ data/
data/corpus/      Plain-text corpora
public/           Browser UI
out/              Training outputs (gitignored)
```

---

## Roadmap

The list below uses [GitHub-flavored Markdown task lists](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#task-lists). Toggle `- [ ]` / `- [x]` in pull requests or commits to record progress. Ordering does not imply priority.

- [ ] **Architecture note** — Short document describing tensor flow through forward and backward passes, with pointers to `train.js` and `tensor/ops.js`.
- [ ] **Subword tokenizer** — Optional BPE- or SentencePiece-style path behind a flag; character-level remains the default teaching path.
- [x] **Evaluation** — Hold-out tail slice, `val_loss` / `val_ppl`, logged metrics CSV (see **Reproducibility**).
- [ ] **Automated tests** — Numerical checks for core ops, tokenizer round-trip, and forward-pass shape invariants.
- [ ] **Training** — Learning-rate schedule; explore faster numerics (e.g. WASM) only if clarity is preserved.
- [ ] **Web UI** — Configurable model path in development builds; improved error when the default export path is missing.
- [ ] **Distribution** — Document minimal file set for inference-only use; optional `npm pack` workflow.

---

## Limitations

- Small capacity and limited training data yield repetitive or incoherent text; decoding heuristics reduce artifacts but do not replace model scale.
- Character-level modeling implies large vocabularies for diverse Unicode text and weaker long-range structure than subword models.
- The bundled HTTP server is for local experimentation only (no authentication, no hardening for untrusted networks).
- **Training runs on CPU only** (custom JS tensors). GPU training in the JavaScript ecosystem usually means TensorFlow.js with CUDA (`tfjs-node-gpu`) or rewriting ops; see **[docs/GPU.md](docs/GPU.md)** ([中文](docs/GPU.zh.md)).

---

## License

No license file is shipped in this repository. `package.json` marks the package as `"private": true`. Add an explicit `LICENSE` before redistribution.
