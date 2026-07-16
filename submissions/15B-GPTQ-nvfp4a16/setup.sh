#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
IMAGE="vllm/vllm-openai:v0.24.0-x86_64-cu129-ubuntu2404"
MODEL_ID="TildeAI/TildeOpen15B-64k-wmt26-compression-task-cs-de-fp4a16"
MODEL_CACHE=${MODEL_CACHE:-"$ROOT/workdir/models"}
MODEL_LINK="$ROOT/workdir/model"


docker pull "$IMAGE"
mkdir -p "$MODEL_CACHE" "$(dirname -- "$MODEL_LINK")"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env MODEL_ID="$MODEL_ID" \
  --volume "$MODEL_CACHE:/models" \
  --entrypoint python3 \
  "$IMAGE" -c '
import os
from pathlib import Path
from huggingface_hub import snapshot_download
model_id = os.environ["MODEL_ID"]
target = Path("/models") / model_id
snapshot_download(repo_id=model_id, local_dir=target)
print(target)
'

MODEL_SOURCE=$(realpath -- "$MODEL_CACHE/$MODEL_ID")
[[ -f "$MODEL_SOURCE/config.json" ]] || { echo "model download is incomplete" >&2; exit 1; }
rm -f -- "$MODEL_LINK"  # removes an old symlink; refuses to remove a real directory
ln -s "$MODEL_SOURCE" "$MODEL_LINK"
printf '%s\n' "$IMAGE" > "$ROOT/.docker-image"

