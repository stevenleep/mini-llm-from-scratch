# mini-llm-from-scratch

Minimal, **zero-dependency** implementation of a character-level decoder-only Transformer (GPT-style) in JavaScript. Training, weight export, CLI decoding, and a local web UI with Server-Sent Events streaming are supported; all computation runs on the host machine without external services.

| | |
| :--- | :--- |
| **Runtime** | Node.js 18+ (ES modules), standard library only |
| **Scope** | Education and research prototyping; not intended for production chat or factual QA |
| **用途** | 教学与实验：理解小型 Transformer 的前向/反向与续写解码；可通过配置与语料对比行为。**非**商用对话或基座模型替代品 |

---

## Features

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

Default corpus: `data/corpus/playful_zh.txt` (override with `CORPUS_PATH`).

```bash
npm run train
```

Extended preset (more steps, longer context, tuned learning rate):

```bash
npm run train:fun
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
| `build` | Alias for `train` |
| `infer` / `chat` | `node src/infer.js` (pass model path and flags) |
| `ui` | `node src/chatServer.js` |
| `stop:ui` | Terminate process listening on port `3847` |
| `restart:ui` | `stop:ui` then start `chatServer` |

---

## Configuration

### Training and export

| Variable | Effect |
| :--- | :--- |
| `CORPUS_PATH` | Corpus file path (default: `data/corpus/playful_zh.txt`) |
| `FUN_TRAIN` | Set to `1` to apply `funTrainingPreset` in `src/config.js` |
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

---

## License

No license file is shipped in this repository. `package.json` marks the package as `"private": true`. Add an explicit `LICENSE` before redistribution.
