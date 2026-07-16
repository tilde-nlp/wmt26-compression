#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODEL_DIR=${MODEL_DIR:-"$ROOT/workdir/model"}

lang_pair= batch_size= input= output=
extra=()
while (($#)); do
  case "$1" in
    --lang-pair)  lang_pair=$2;  shift 2 ;;
    --batch-size) batch_size=$2; shift 2 ;;
    --input)      input=$2;      shift 2 ;;
    --output)     output=$2;     shift 2 ;;
    *)            extra+=("$1"); shift ;;
  esac
done

[[ -n $lang_pair && -n $batch_size && -n $input && -n $output ]] || {
  echo "required: --lang-pair PAIR --batch-size N --input FILE --output FILE" >&2
  exit 2
}
[[ -f $input ]] || { echo "input not found: $input" >&2; exit 2; }
[[ -d $MODEL_DIR ]] || { echo "model not found: $MODEL_DIR" >&2; exit 2; }

IMAGE="vllm/vllm-openai:v0.24.0-x86_64-cu129-ubuntu2404"
input=$(realpath -- "$input")
model=$(realpath -- "$MODEL_DIR")
mkdir -p -- "$(dirname -- "$output")"
output_dir=$(realpath -- "$(dirname -- "$output")")
output_name=$(basename -- "$output")

gpus=(--gpus all)
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] || gpus=(--gpus "device=$CUDA_VISIBLE_DEVICES")

docker run --rm \
  "${gpus[@]}" \
  --ipc=host \
  --network=none \
  --env PYTHONPATH=/app \
  --env ALLOW_LLAMA_DECOUPLED_HEAD_DIM=1 \
  --volume "$ROOT/inference.py:/app/inference.py:ro" \
  --volume "$ROOT/sitecustomize.py:/app/sitecustomize.py:ro" \
  --volume "$model:/model:ro" \
  --volume "$input:/input.txt:ro" \
  --volume "$output_dir:/output" \
  --entrypoint python3 \
  "$IMAGE" /app/inference.py \
    --lang-pair "$lang_pair" \
    --batch-size "$batch_size" \
    --input /input.txt \
    --output "/output/$output_name" \
    "${extra[@]}" \
  1>&2

