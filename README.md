# mini-llm-from-scratch

[![Node.js](https://img.shields.io/badge/node-%3E%3D18-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No dependencies](https://img.shields.io/badge/dependencies-none-555555)]()

**English** | [简体中文](README.zh.md)

A **dependency-free**, **character-level**, decoder-only Transformer (GPT-style language model) implemented in **JavaScript** with a small autograd stack. It supports training from plain text, exporting weights in a Hugging Face–style layout or a single JSON bundle, command-line continuation, and a **local** web UI with **Server-Sent Events (SSE)** streaming. All computation runs on the host; no external inference API is required.

---

## Value proposition

| Audience | What you gain |
| :--- | :--- |
| **Learners & instructors** | A **readable, end-to-end** codebase: embeddings, attention, LayerNorm, FFN, loss, backward, SGD/Adam, and decoding—all in plain JavaScript you can trace in a debugger. No black-box native kernels required to understand the training loop. |
| **Researchers & tinkerers** | **Reproducible** small-model runs: corpus hash (`corpus_sha256_16`), fixed RNG seeds, tail hold-out validation, and append-only **metrics CSV** so experiments can be compared and reported with concrete artifacts. |
| **Practitioners who value portability** | **Zero npm dependencies** for core train/infer: only Node.js ≥ 18. Clone the repo and run without fighting CUDA wheels or Python envs for the default CPU path—useful for laptops, CI, and teaching labs. |
| **Tooling-oriented users** | **Interop-friendly exports**: Hugging Face–style directories (`config.json`, tokenizer files, `model.safetensors`, optional `model.bin`) plus a **single-file JSON** bundle—familiar layout for diffing, archiving, or piping into other tooling. |
| **Privacy-conscious or offline workflows** | **Data never leaves the machine** by default: no cloud API keys, no vendor lock-in for inference; suitable for demos on air-gapped or policy-restricted machines (still follow your org’s rules). |

**Explicit non-goals:** This project does **not** aim to match the fluency, factual reliability, or scale of frontier LLMs, nor to provide production-grade serving (authentication, scaling, SLOs). It is a **transparent substrate** for learning and controlled experiments—not a drop-in product backend.

---

## Table of contents

- [Value proposition](#value-proposition)
- [Features](#features)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [npm scripts](#npm-scripts)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## Features

| Area | Details |
| :--- | :--- |
| **Model** | `MiniGPT`: token and position embeddings, stacked Transformer blocks (with Pre-LN), causal self-attention, feed-forward layers, language-model head; optional **LoRA** adapters on linear layers |
| **Autograd** | Lightweight `Tensor` type and ops (e.g. matmul, softmax, cross-entropy); **SGD** or **Adam** in the training loop |
| **Training** | Random sliding windows; optional gradient clipping; tail hold-out validation; metrics CSV; optional **checkpoints**, **resume**, **cosine LR**, **warmup**, **early stopping** |
| **Export** | Directory compatible with common HF layouts (`config.json`, tokenizer files, `model.safetensors`, optional `model.bin` + manifest) and a portable **single-file JSON** format |
| **Decoding** | Temperature, top-k, top-p (nucleus), repetition penalty; greedy path applies repetition penalty before argmax |
| **Web UI** | Static assets under `public/`; `POST /api/chat/stream` returns SSE token streams |

For a full inventory of flags, endpoints, and file roles, see **[docs/FEATURES.md](docs/FEATURES.md)** ([中文](docs/FEATURES.zh.md)).

---

## Requirements

- **Node.js ≥ 18** (ES modules)
- **No `npm install`** for the core library—training and inference use only the Node.js standard library

Optional: TensorFlow.js Node packages for GPU-related experiments (**[docs/GPU.md](docs/GPU.md)**).

---

## Quick start

### Train

Default corpus: `data/corpus/playful_zh.txt` (override with `CORPUS_PATH`). If `data/corpus/identity_zh.txt` exists, it is concatenated **after** the main corpus for optional identity-style prompts. Disable with `SKIP_IDENTITY_CORPUS=1`; set `IDENTITY_CORPUS_PATH` for a custom file.

```bash
npm run train
```

**Presets** (see `src/config.js`):

| Command | Role |
| :--- | :--- |
| `npm run train:fun` | More steps and longer context than the default |
| `npm run train:mega` | Long preset (~10k steps in `megaTrainingPreset`) |
| `npm run train:ultimate` | Larger model (`ultimateTrainingPreset`), **Adam + cosine LR + warmup**; default **5000** steps |
| `npm run train:ultimate:long` | Same as ultimate with **20000** steps and longer warmup |

Fetch optional mixed Chinese text into `data/corpus/downloaded_mixed_zh.txt` (merged automatically during training unless `SKIP_DOWNLOAD_CORPUS=1`):

```bash
npm run corpus:fetch
```

Artifacts (checkpoints, exports, logs) go under `out/` (gitignored by default).

### CLI inference

Streaming to stdout is the default; `--no-stream` prints once at the end.

```bash
node src/infer.js ./out/export/hf-style "Hello" -n 40
node src/infer.js ./out/export/hf-style "Hello" --no-stream
```

Run `node src/infer.js --help` for decoding options.

### Web UI

```bash
npm run ui
```

Open **http://localhost:3847/**. Default model path: `out/export/hf-style` (override with `MODEL_PATH`). Stop the process listening on port **3847**:

```bash
npm run stop:ui
```

---

## npm scripts

| Script | Description |
| :--- | :--- |
| `train` | `node src/train.js` |
| `train:fun` | `FUN_TRAIN=1 node src/train.js` |
| `train:mega` | `MEGA_TRAIN=1 node src/train.js` |
| `train:lora` | `LORA_RANK=8 LORA_ONLY=1 node src/train.js` |
| `train:ultimate` | Ultimate preset + Adam + cosine LR + warmup (default 5000 steps) |
| `train:ultimate:long` | Ultimate preset with `STEPS=20000` and extended warmup |
| `corpus:fetch` | Download sources → `data/corpus/downloaded_mixed_zh.txt` |
| `export:all` | Re-export from checkpoints (see script header) |
| `build` | Alias for `train` |
| `infer` / `chat` | `node src/infer.js` |
| `ui` | `node src/chatServer.js` |
| `stop:ui` / `restart:ui` | Free or restart `3847` |
| `gpu:smoke` | Optional TF.js matmul check (requires installing tfjs-node*) |

---

## Configuration

### Training and export

| Variable | Effect |
| :--- | :--- |
| `CORPUS_PATH` | Main corpus file (default `data/corpus/playful_zh.txt`) |
| `FUN_TRAIN` | `1` → merge `funTrainingPreset` |
| `MEGA_TRAIN` | `1` → merge `megaTrainingPreset` (overrides `FUN_TRAIN` if both set) |
| `ULTIMATE_TRAIN` | `1` → merge `ultimateTrainingPreset` (overrides MEGA / FUN) |
| `OPTIMIZER` | `adam` for Adam; otherwise SGD |
| `COSINE_LR` | `1` → cosine decay of learning rate over the schedule |
| `LR_WARMUP` | Positive integer → linear LR warmup before cosine or constant LR |
| `RESUME_FROM` / `LOAD_CHECKPOINT` | Path to `.mgpt.json` checkpoint (must match vocabulary and architecture) |
| `STEP_OFFSET` | Global steps already completed before this run (for LR schedule when resuming) |
| `TOTAL_STEPS` | Optional; schedule length is `max(STEPS + STEP_OFFSET, TOTAL_STEPS)` when set |
| `CHECKPOINT_EVERY` | Save checkpoint every *N* global steps (`0` disables) |
| `CHECKPOINT_DIR` | Checkpoint directory (default `./out/checkpoints`) |
| `EARLY_STOP_PATIENCE` | Stop after this many log intervals without `val_loss` improvement (`0` disables). Adam moments are **not** restored from checkpoints |
| `SKIP_DOWNLOAD_CORPUS` | `1` → do not merge `downloaded_mixed_zh.txt` |
| `DOWNLOAD_CORPUS_PATH` | Path to downloaded mix (default `data/corpus/downloaded_mixed_zh.txt`) |
| `STEPS` | Override total training steps |
| `SKIP_EXPORT` | `1` → skip writing export artifacts |
| `EXPORT_DIR` / `EXPORT_PATH` | HF-style dir and single JSON path |
| `VERBOSE` | `1` → verbose training logs |
| `GRAD_CLIP` | Global gradient norm clip (default `1`; `0` disables) |
| `VAL_FRACTION` | Tail fraction for validation (default `0.08`; disabled if corpus is too short) |
| `VAL_SAMPLES` | Validation windows per log line (default `24`) |
| `METRICS_CSV` / `SKIP_METRICS` | Metrics CSV path and toggle |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_ONLY` | LoRA adapters (see `src/config.js` and training header in `src/train.js`) |

### Inference and UI

| Variable | Effect |
| :--- | :--- |
| `MODEL_PATH` | Model directory or `.json` for `chatServer.js` |
| `PORT` | HTTP port (default `3847`) |

---

## Reproducibility

Each training run prints a **`[repro]`** line:

- **`corpus_sha256_16`**: first 16 hex characters of SHA-256 over the **loaded** UTF-8 corpus string.
- **`seed`**, **`steps`**, **`seqLen`**, **`val_fraction`**: effective hyperparameters.

The trainer does **not** invoke Git. For papers or lab notes, record `git rev-parse --short HEAD` (or `CI_COMMIT_SHA`) manually alongside `train_metrics.csv`.

Validation uses a **tail hold-out** on token sequences and a **fixed** RNG seed for validation windows, so `val_loss` / `val_ppl` are repeatable given identical code, corpus bytes, and environment.

---

## Repository layout

```
LICENSE            MIT license text
CONTRIBUTING.md    Contribution guidelines
SECURITY.md        Expectations for the local demo server
docs/              Extended documentation (features, GPU notes)
scripts/           Auxiliary scripts (corpus fetch, export, optional GPU smoke)
src/
  config.js        Hyperparameter presets
  train.js         Training loop, optimizers, export
  infer.js         CLI
  chatServer.js    Static UI + REST/SSE
  generate.js      Decoding
  model/ nn/ tensor/ io/ data/
data/corpus/       Example and optional corpora (UTF-8 text)
public/            Web UI assets
out/               Training outputs (default: gitignored)
```

---

## Roadmap

Tracked as GitHub task lists; order is not a strict priority.

- [ ] Short architecture note: tensor flow through forward/backward, with pointers into `train.js` and `tensor/ops.js`
- [ ] Optional subword tokenizer path (BPE/SentencePiece-style); character-level remains the default teaching path
- [x] Hold-out validation, `val_loss` / `val_ppl`, metrics CSV
- [ ] Automated tests for core ops, tokenizer round-trip, shape invariants
- [ ] Optional faster numerics (e.g. WASM) without sacrificing readability
- [ ] UI: clearer errors when the default export path is missing
- [ ] Document minimal file set for inference-only redistribution

---

## Limitations

- **Capacity and data:** Small models and limited corpora produce repetitive or incoherent text; decoding heuristics mitigate artifacts but do not replace scale.
- **Character-level modeling:** Large Unicode vocabularies and weaker long-range structure compared to subword models.
- **CPU training:** Core training uses a pure JS tensor stack. GPU paths in JS typically require TensorFlow.js or similar (**[docs/GPU.md](docs/GPU.md)**).
- **Local demo server:** The bundled HTTP server is for **local** experimentation only (**[SECURITY.md](SECURITY.md)**).

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Security

See **[SECURITY.md](SECURITY.md)** for expectations around the local web server.

---

## License

This project is released under the **MIT License** — see **[LICENSE](LICENSE)**.

---

## Citation

If you use this repository for research or teaching, a plain reference to the project URL and commit hash is sufficient; there is no mandatory citation format.
