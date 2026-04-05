# Contributing

Thank you for your interest in improving this project. The codebase is intended to stay **readable and dependency-free**; changes that preserve those goals are especially welcome.

## How to contribute

1. **Open an issue** first for substantial features (new tokenizer modes, training backends, or large refactors) so design can be discussed briefly.
2. **Fork the repository**, create a branch from `main`, and open a **pull request** with a clear description of the change and how you tested it.
3. Keep pull requests **focused**: one logical change per PR is easier to review than a bundle of unrelated edits.

## Development setup

- **Node.js ≥ 18** (ES modules). No `npm install` is required for the core project.
- Run training: `npm run train` (or presets such as `npm run train:fun`).
- Run inference: `node src/infer.js <model-path> "prompt" -n 32`.
- Optional GPU smoke test: install `@tensorflow/tfjs-node` or `@tensorflow/tfjs-node-gpu`, then `npm run gpu:smoke` (see `docs/GPU.md`).

## Code style

- Match existing naming, comment style, and file layout.
- Prefer small, well-scoped diffs over drive-by refactors in unrelated files.
- If you add user-visible behavior, update **README.md** and **README.zh.md** (and `docs/FEATURES*.md` when the feature inventory changes).

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (see `LICENSE`).
