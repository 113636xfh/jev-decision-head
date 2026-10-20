#!/usr/bin/env bash
# One complete build recipe, implemented by verify.py + manifest.json. Never run/install the binary.
# GGML_DIR=/path/to/pinned/llama.cpp ./patches/reproduce-build.sh /tmp/strata-repro-NEW
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
[ $# -eq 1 ] || { echo 'usage: GGML_DIR=/path/to/llama.cpp reproduce-build.sh <new-or-empty-dir>' >&2; exit 2; }
: "${GGML_DIR:?set GGML_DIR to the pinned llama.cpp source cache}"
args=(--work-dir "$1" --build cuda --ggml-dir "$GGML_DIR"
      --cuda-compiler "${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
      --cuda-arch "${CUDA_ARCH:-70}" --jobs "${JOBS:-1}")
# A local CLEAN pinned checkout avoids network; otherwise the verifier clones and pins upstream.
if [ -n "${SOURCE:-}" ]; then args+=(--source "$SOURCE"); fi
if [ -n "${UPSTREAM:-}" ]; then args+=(--upstream "$UPSTREAM"); fi
exec "$PYTHON" "$HERE/verify.py" "${args[@]}"
