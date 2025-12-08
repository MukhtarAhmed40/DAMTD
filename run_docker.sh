#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="damtd_artifact:latest"
if [ "${1:-}" = "build" ]; then
  docker build -t "$IMAGE_NAME" "$ROOT_DIR"
  exit 0
elif [ "${1:-}" = "run" ]; then
  docker run --rm -it -v "$ROOT_DIR":/workspace --name damtd_artifact_run "$IMAGE_NAME" bash -lc "cd /workspace && bash run_all.sh"
  exit 0
else
  echo "Usage: bash run_docker.sh [build|run]"
  exit 1
fi
