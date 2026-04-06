# Alignment

Alignment training playground for supervised fine-tuning, expert iteration,
and GRPO-style reinforcement learning on reasoning datasets.

This repository is adapted from the
[`assignment5-alignment`](https://github.com/YYZhang2025/Stanford-CS336/tree/main/assignment5-alignment)
subproject in `YYZhang2025/Stanford-CS336`. The upstream handout PDFs are kept
here as reference material:

- [Alignment handout](./cs336_spring2025_assignment5_alignment.pdf)
- [Safety / RLHF supplement](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

## Setup

This project uses `uv` for environment and dependency management.

1. Create the environment and install dependencies. `flash-attn` is installed
   in a second pass because it has stricter build requirements.

```sh
uv sync --python 3.11 --no-install-package flash-attn
uv sync --python 3.11
```

2. Run the test suite:

```sh
uv run pytest
```

3. Smoke-check a CLI entrypoint:

```sh
uv run python train_sft.py --help
```

## Notes

- Python `>=3.11,<3.13` is required.
- Training-oriented dependencies such as `flash-attn`, `vllm`, and GPU-backed
  PyTorch may need machine-specific setup before end-to-end training runs.
- On `macOS x86_64`, this repo skips `flash-attn` and `vllm` during dependency
  resolution so the CPU-side code path and tests can still be exercised.
- The tests directory includes adapter coverage and lightweight fixtures for
  verifying the local implementation.

## Verification Status

Verified locally on `macOS x86_64` with Python `3.11.15`.

- `uv sync --python 3.11 --no-install-package flash-attn`: passed
- `uv sync --python 3.11`: passed
- `uv run python -c "import train_sft; print('train_sft-import-ok')"`: passed
- `uv run pytest`: partial pass

Current known test blockers on this machine:

- Some tests expect a local model snapshot at `./models/Qwen2.5-Math-1.5B`.
- `tests/test_dpo.py` still depends on an unimplemented adapter in
  `tests/adapters.py`.
- A few data and masking tests currently fail against the checked-in code path.
