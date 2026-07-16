# 8B-distill

- Model: TildeOpen8B-64k-wmt26-compression-task-cs-de
- Parent model: TildeOpen15B-64k-wmt26-compression-task-cs-de
- Params: 8B
- Size: 16.3 GB
- Supported languages: cs-de
- Quantization: None
- Quantization serialization: None
- HF repo: [TildeAI/TildeOpen8B-64k-wmt26-compression-task-cs-de](https://huggingface.co/TildeAI/TildeOpen8B-64k-wmt26-compression-task-cs-de)
- Training: Distilled from *Parent model* using [Nvidia-NEMO](https://github.com/nvidia-nemo) + further GRPO
- Runtime: vLLM

## Setup

``` 
bash setup.sh 
```
Requires docker permissions. No python environment is needed.

- Pulls the following docker image: vllm/vllm-openai:v0.24.0-x86_64-cu129-ubuntu2404
- Downloads the model from HF repo via the docker to a cache folder (default: `workdir/models` next to the script; override with the `MODEL_CACHE` env var)
- Links to workdir/model

## Run
```
bash run.sh --lang-pair ces-deu \
            --batch-size 1 \
            --input input.txt \
            --output output.txt \
            [--enforce-eager true|false] \
            [--max-model-len <=65536]
```

Runs via vLLM. 90% of VRAM is reserved for vLLM allocations, such as KV-cache.

**Note**: transformers > 5 do not allow for hidden_size to not be exactly divisble by num_attention_heads for LLama class models, which is the case TildeOpen family of 15B and 8B models. This is a simple validation check, that we disable via ```sitecustomize.py``` during runtime.

### Optional parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--enforce-eager` | `true` | Enforces eager mode. For very large datasets and long benchmarks, we recommend setting `--enforce-eager false` to benefit from pre-compiled CUDA graphs. |
| `--max-model-len` | `8192` | Maximum model length. Increase if necessary; the model supports up to 65536 context size. |



