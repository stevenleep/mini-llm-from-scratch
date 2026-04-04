# GPU acceleration and JavaScript

[简体中文](GPU.zh.md)

This repository’s **training loop** (`src/train.js`) uses a **CPU-only** tensor runtime (`src/tensor/*.js`). There is **no GPU path in the current code**.

To train with a **GPU while staying in JavaScript**, you typically use a runtime that delegates heavy ops to native GPU libraries:

| Approach | Where it runs | Notes |
| :--- | :--- | :--- |
| **`@tensorflow/tfjs-node-gpu`** | Node.js + NVIDIA CUDA/cuDNN | Mature; binds to TensorFlow C API; **requires** matching CUDA stack and often a Linux/NVIDIA setup |
| **`@tensorflow/tfjs-node`** | Node CPU | Same API, no GPU |
| **WebGPU backend** (`@tensorflow/tfjs-backend-webgpu`) | Browser (or experimental Node WebGPU) | Good for demos; training large models is still limited vs CUDA |
| **Rewrite hot ops** | Custom WebGPU / WASM | Full control; large engineering cost (matmul + backward) |

**Why this repo does not “flip a switch” for GPU:** the autograd system, `Tensor` class, and all ops are implemented in plain JavaScript on `Float32Array`. A GPU backend would require either:

1. **Reimplementing** the model and training step in **TensorFlow.js** (or JAX/PyTorch via export), or  
2. **Replacing** low-level ops (at least matrix multiply and softmax) with GPU kernels while keeping backward rules consistent—essentially a new backend.

**Practical path for production-style GPU training:** use **PyTorch / JAX** on GPU and treat this repo as a **reference** for architecture and data flow, or reimplement the same `MiniGPT` config in TensorFlow.js with `tfjs-node-gpu`.

**Practical path for “JS + GPU” experiments:** start a **separate** small script that only checks `tfjs-node-gpu` loads and runs one `matmul` on GPU, then gradually port layers—not by patching `Tensor.js` line by line.

See [GPU.zh.md](GPU.zh.md) for the same content in Chinese.

---

## Bundled JS smoke test (TensorFlow.js)

Core training stays dependency-free. To **verify** TF.js Node (GPU or CPU binding) on your machine:

```bash
# Pick one (do not install tfjs-node-gpu and tfjs-node together)
npm install @tensorflow/tfjs-node-gpu   # NVIDIA + CUDA, may use GPU
# or
npm install @tensorflow/tfjs-node       # native binding (must build)
# or (easiest, pure JS, slowest)
npm install @tensorflow/tfjs

npm run gpu:smoke
# optional: SIZE=2048 npm run gpu:smoke
```

Script: `scripts/tfjs-gpu-smoke.mjs` — tries **gpu → node → pure `@tensorflow/tfjs`**, prints **backend**, times a random `matMul`. Pure JS may print a TF.js hint to install `tfjs-node`; ignore or install the native package. Linux + NVIDIA + CUDA + `tfjs-node-gpu` is the path to GPU.
