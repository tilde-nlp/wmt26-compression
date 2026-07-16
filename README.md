# WMT26 compression task

This repo contains source code for **Tilde** submission of wmt26 compression shared task

## Models

All models are built on top of TildeOpen15B-64k, trained for the cs-de language pair, and served via vLLM in docker (no python environment needed). Each submission folder is self-contained with its own `setup.sh` (pulls the docker image, downloads the model from HF) and `run.sh` (runs inference). See the per-submission READMEs for details.

| Submission | Model (HF, under [TildeAI](https://huggingface.co/TildeAI)) | Params | Size | Compression | Note |
| --- | --- | --- | --- | --- | --- |
| [baseline](submissions/baseline/README.md) | TildeOpen15B-64k-wmt26-compression-task-cs-de | 15B | 30.3 GB | None (SFT + GRPO baseline) | |
| [15B-GPTQ-nvfp4a16](submissions/15B-GPTQ-nvfp4a16/README.md) | TildeOpen15B-64k-wmt26-compression-task-cs-de-fp4a16 | 15B | 10.1 GB | GPTQ PTQ, static NVFP4 weights, BF16 activations | **PRIMARY** |
| [15B-RTN-fp8-dyn](submissions/15B-RTN-fp8-dyn/README.md) | TildeOpen15B-64k-wmt26-compression-task-cs-de-fp8-dyn | 15B | 16.3 GB | RTN PTQ, FP8 weights, dynamic per-token FP8 activations | |
| [8B-distill](submissions/8B-distill/README.md) | TildeOpen8B-64k-wmt26-compression-task-cs-de | 8B | 16.3 GB | Distilled from baseline (Nvidia-NeMo) + GRPO | |
| [8B-distill-GPTQ-nvfp4a16](submissions/8B-distill-GPTQ-nvfp4a16/README.md) | TildeOpen8B-64k-wmt26-compression-task-cs-de-fp4a16 | 8B | 5.7 GB | Distillation from baseline + GPTQ PTQ, static NVFP4 weights, BF16 activations | |
| [8B-distill-RTN-fp8-dyn](submissions/8B-distill-RTN-fp8-dyn/README.md) | TildeOpen8B-64k-wmt26-compression-task-cs-de-fp8-dyn | 8B | 8.9 GB | Distillation from baseline + RTN PTQ, FP8 weights, dynamic per-token FP8 activations | |

All 8B models are distilled from the baseline model; the quantized 8B variants are quantized from the resulting 8B-distill model. The quantized 15B variants are quantized directly from the baseline.

## Usage

Inside any submission folder:

```
bash setup.sh
```

Downloads the model to a cache folder, `workdir/models` inside the submission folder by default (override with the `MODEL_CACHE` env var), and symlinks it to `workdir/model`.

```
bash run.sh --lang-pair ces-deu \
            --batch-size 1 \
            --input input.txt \
            --output output.txt \
            [--enforce-eager true|false] \
            [--max-model-len <=65536]
```

Requires docker permissions. See the submission READMEs for optional parameters and runtime notes.
 